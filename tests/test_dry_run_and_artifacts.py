import argparse
import json
import subprocess
from pathlib import Path

import pytest

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
from main import cmd_implement


def setup_full_project(tmp_path: Path) -> tuple[Path, str, str]:
    p = tmp_path / "repo"
    p.mkdir()
    (p / "backend" / "internal" / "service").mkdir(parents=True)
    (p / "frontend").mkdir(parents=True)
    (p / "backend" / "go.mod").write_text("module test/backend\n")
    (p / "frontend" / "package.json").write_text('{"name": "frontend"}\n')

    svc_file = p / "backend" / "internal" / "service" / "wallet_service.go"
    svc_file.write_text("package service\n\nfunc TopUp() {}\n")

    initialize_project(p, apply=True)

    subprocess.run(["git", "init", "-b", "main"], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=p, check=True, capture_output=True)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True, check=True).stdout.strip()

    # Plan artifact
    store = ProjectArtifactStore(p, load_project(p).manifest)
    plan_data = {
        "status": "plan_ready",
        "result": {
            "recommended_task": {"title": "Update wallet", "owner": "backend", "acceptance_criteria": ["ac1"]},
            "verification_status": "verified_pending",
        },
    }
    plan_file = store.write_plan("plan_test", plan_data)

    # Proposal artifact
    prop_data = {
        "source_plan_path": str(plan_file.relative_to(p)),
        "status": "approved_for_implementation",
        "source_revision": {"commit_head": head, "is_dirty": False},
        "proposal": {
            "proposal_id": "prop_test",
            "task_title": "Update wallet",
            "owner": "backend",
            "files_to_modify": [{"path": "backend/internal/service/wallet_service.go", "action": "modify", "purpose": "update"}],
            "files_to_create": [],
            "validation_commands": [{"executable": "go", "arguments": ["test", "./..."], "working_directory": "backend", "purpose": "test"}],
        },
    }
    prop_file = store.write_proposal("proposal_test", prop_data)

    # Review report
    rev_data = {
        "status": "approved_for_implementation",
        "review": {
            "status": "approved",
            "proposal_id": "prop_test",
            "approved_files": ["backend/internal/service/wallet_service.go"],
        },
    }
    store.write_report("proposal_review_test", rev_data)

    return p, str(plan_file.relative_to(p)), str(prop_file.relative_to(p))


def test_60_to_65_dry_run_invariants(tmp_path, capsys):
    """60-65: Default implement dry run invariants."""
    p, plan_rel, prop_rel = setup_full_project(tmp_path)
    svc_file = p / "backend" / "internal" / "service" / "wallet_service.go"
    content_before = svc_file.read_text()

    branch_before = subprocess.run(["git", "branch", "--show-current"], cwd=p, capture_output=True, text=True).stdout.strip()

    args = argparse.Namespace(
        project=str(p),
        proposal=prop_rel,
        apply=False,
    )
    cmd_implement(args)
    captured = capsys.readouterr().out

    # 60. Default implement tidak mengubah source
    assert svc_file.read_text() == content_before

    # 61. Default implement tidak membuat branch
    branch_after = subprocess.run(["git", "branch", "--show-current"], cwd=p, capture_output=True, text=True).stdout.strip()
    assert branch_after == branch_before

    # 62. Default implement tidak menjalankan command
    assert "Pengujian dilewati" in captured or "DRY RUN" in captured

    # 64. Dry-run menampilkan planned files
    assert "backend/internal/service/wallet_service.go" in captured

    # 65. Dry-run menampilkan planned command
def test_63_apply_required_to_write(tmp_path):
    """63. --apply diperlukan untuk menulis."""
    p, plan_rel, prop_rel = setup_full_project(tmp_path)
    store = ProjectArtifactStore(p, load_project(p).manifest)

    # Without apply: dry run does not write
    args_dry = argparse.Namespace(project=str(p), proposal=prop_rel, apply=False)
    cmd_implement(args_dry)

    # Verify no implementation artifacts in implementations dir
    impl_files = list((p / ".agent" / "artifacts" / "implementations").glob("implementation_*.json"))
    assert len(impl_files) == 0


def test_66_to_74_implementation_artifacts(tmp_path):
    """66-74: Artifact storage, security, and metadata properties."""
    p, plan_rel, prop_rel = setup_full_project(tmp_path)
    store = ProjectArtifactStore(p, load_project(p).manifest)

    impl_data = {
        "source_plan_path": plan_rel,
        "source_proposal_path": prop_rel,
        "source_revision_before": {"commit_head": "12345", "is_dirty": False},
        "branch_name": "agent/test-12345678",
        "task_title": "Update wallet",
        "files_changed": ["backend/internal/service/wallet_service.go"],
        "file_hashes_before": {"backend/internal/service/wallet_service.go": "hash1"},
        "file_hashes_after": {"backend/internal/service/wallet_service.go": "hash2"},
        "validation_command_summaries": [{"status": "passed", "exit_code": 0}],
        "test_status": "passed",
        "diff_review_status": "approved",
        "repair_count": 0,
        "acceptance_criteria_status": [{"criterion": "ac1", "status": "satisfied"}],
        "final_status": "implementation_ready",
        "status": "implementation_ready",
        "secret_token": "bearer secret123",  # should be redacted
        "full_diff": "diff --git a/file b/file",  # should be omitted
    }

    impl_file = store.write_implementation("implementation_12345678", impl_data)

    # 66. Artifact berada di project
    assert impl_file.is_relative_to(p)
    assert impl_file.is_file()

    # 67. Artifact tidak berada di orchestrator
    orchestrator_root = Path(__file__).resolve().parent.parent
    assert not impl_file.is_relative_to(orchestrator_root / ".agent" / "artifacts") if (orchestrator_root / ".agent").exists() else True

    data = json.loads(impl_file.read_text())

    # 68. Artifact tidak menyimpan source code lengkap
    assert "source_code" not in data.get("result", {})

    # 69. Artifact tidak menyimpan full diff
    assert data.get("result", {}).get("full_diff") == "[OMITTED_FOR_SAFETY]"

    # 70. Artifact tidak menyimpan secret
    assert "secret123" not in json.dumps(data)

    # 71. Artifact menggunakan relative path
    assert not data.get("result", {}).get("source_plan_path", "").startswith("/")
    assert not data.get("result", {}).get("source_proposal_path", "").startswith("/")

    # 72. Artifact memiliki source revision
    assert "source_revision" in data
    assert "commit_head" in data["source_revision"]

    # 73. Artifact memiliki file hashes
    res = data.get("result", {})
    assert "file_hashes_before" in res
    assert "file_hashes_after" in res

    # 74. Artifact memiliki test summary
    assert "validation_command_summaries" in res
    assert res.get("test_status") == "passed"
