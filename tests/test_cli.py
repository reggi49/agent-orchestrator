import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_system.project_loader import ProjectNotFoundError, load_project


def setup_cli_test_project(proj_dir: Path):
    (proj_dir / "backend").mkdir(parents=True, exist_ok=True)
    (proj_dir / "backend" / "go.mod").write_text("module example.com/cli_app\n\ngo 1.22\n", encoding="utf-8")
    (proj_dir / "README.md").write_text("# CLI App\n", encoding="utf-8")


PYTHON_BIN = sys.executable


def test_34_cli_init_default_is_dry_run(tmp_path: Path):
    setup_cli_test_project(tmp_path)
    res = subprocess.run(
        [PYTHON_BIN, "main.py", "init", "--project", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert res.returncode == 0
    assert "DRY RUN" in res.stdout
    assert not (tmp_path / ".agent").exists()


def test_35_cli_init_apply_creates_protocol(tmp_path: Path):
    setup_cli_test_project(tmp_path)
    res = subprocess.run(
        [PYTHON_BIN, "main.py", "init", "--project", str(tmp_path), "--apply"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert res.returncode == 0
    assert "INITIALIZATION APPLIED" in res.stdout
    assert (tmp_path / ".agent" / "manifest.yaml").is_file()
    assert (tmp_path / ".agent" / "policy.yaml").is_file()
    assert (tmp_path / ".agent" / "workflow.yaml").is_file()


def test_36_cli_inspect_succeeds_without_llm(tmp_path: Path):
    setup_cli_test_project(tmp_path)
    # Init first
    subprocess.run(
        [PYTHON_BIN, "main.py", "init", "--project", str(tmp_path), "--apply"],
        cwd=Path(__file__).parent.parent,
    )
    # Inspect
    res = subprocess.run(
        [PYTHON_BIN, "main.py", "inspect", "--project", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert res.returncode == 0
    assert "PROJECT INSPECTION REPORT" in res.stdout
    assert "PORTABILITY STATUS: PORTABLE" in res.stdout.upper()


def test_37_cli_inspect_fails_clearly_if_not_init(tmp_path: Path):
    setup_cli_test_project(tmp_path)
    res = subprocess.run(
        [PYTHON_BIN, "main.py", "inspect", "--project", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert res.returncode != 0
    assert "manifest.yaml" in res.stdout or "manifest.yaml" in res.stderr
    assert "init" in res.stdout.lower() or "init" in res.stderr.lower()


def test_38_cli_plan_rejects_uninitialized_project(tmp_path: Path):
    setup_cli_test_project(tmp_path)
    res = subprocess.run(
        [
            PYTHON_BIN,
            "main.py",
            "plan",
            "--project",
            str(tmp_path),
            "--request",
            "Test request",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert res.returncode != 0
    assert "manifest.yaml" in res.stdout or "manifest.yaml" in res.stderr


def test_39_cli_inspect_requires_no_api_key(tmp_path: Path):
    setup_cli_test_project(tmp_path)
    subprocess.run(
        [PYTHON_BIN, "main.py", "init", "--project", str(tmp_path), "--apply"],
        cwd=Path(__file__).parent.parent,
    )

    # Run with empty environment for LLM keys
    clean_env = os.environ.copy()
    clean_env.pop("LLM_API_KEY", None)
    clean_env.pop("OPENAI_API_KEY", None)

    res = subprocess.run(
        [PYTHON_BIN, "main.py", "inspect", "--project", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=clean_env,
    )
    assert res.returncode == 0
    assert "PORTABILITY STATUS: PORTABLE" in res.stdout.upper()


def test_40_cli_init_requires_no_api_key(tmp_path: Path):
    setup_cli_test_project(tmp_path)

    clean_env = os.environ.copy()
    clean_env.pop("LLM_API_KEY", None)
    clean_env.pop("OPENAI_API_KEY", None)

    res = subprocess.run(
        [PYTHON_BIN, "main.py", "init", "--project", str(tmp_path), "--apply"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=clean_env,
    )
    assert res.returncode == 0
    assert (tmp_path / ".agent" / "manifest.yaml").is_file()


def test_41_cli_ceo_requires_project(tmp_path: Path):
    res = subprocess.run(
        [PYTHON_BIN, "main.py", "ceo"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert res.returncode != 0
    assert "--project" in res.stderr or "required" in res.stderr.lower()


def test_42_cli_ceo_initializes_program_and_reports(tmp_path: Path):
    setup_cli_test_project(tmp_path)
    subprocess.run(
        [PYTHON_BIN, "main.py", "init", "--project", str(tmp_path), "--apply"],
        cwd=Path(__file__).parent.parent,
        check=True,
    )

    clean_env = os.environ.copy()
    clean_env.pop("LLM_API_KEY", None)
    clean_env.pop("OPENAI_API_KEY", None)

    res = subprocess.run(
        [
            PYTHON_BIN,
            "main.py",
            "ceo",
            "--project",
            str(tmp_path),
            "--request",
            "Bangun CLI module",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=clean_env,
    )
    assert res.returncode == 0
    assert "CEO MODE PROGRAM DASHBOARD" in res.stdout
    assert "READY" in res.stdout
    assert (tmp_path / ".agent" / "programs").is_dir()
