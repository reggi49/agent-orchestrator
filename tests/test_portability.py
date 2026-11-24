import json
import shutil
from pathlib import Path

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.project_initializer import initialize_project
from agent_system.project_inspector import inspect_project
from agent_system.project_loader import load_project


def setup_portable_project(base_dir: Path):
    (base_dir / "backend").mkdir(parents=True, exist_ok=True)
    (base_dir / "backend" / "go.mod").write_text("module example.com/app\n\ngo 1.22\n", encoding="utf-8")
    (base_dir / "README.md").write_text("# Test App\n", encoding="utf-8")
    initialize_project(base_dir, apply=True)


def test_29_project_can_be_moved(tmp_path: Path):
    orig_dir = tmp_path / "original_proj"
    new_dir = tmp_path / "moved_proj"
    setup_portable_project(orig_dir)

    shutil.copytree(orig_dir, new_dir)
    assert (new_dir / ".agent" / "manifest.yaml").is_file()


def test_30_loader_works_after_move(tmp_path: Path):
    orig_dir = tmp_path / "orig"
    moved_dir = tmp_path / "moved"
    setup_portable_project(orig_dir)
    shutil.copytree(orig_dir, moved_dir)

    validated = load_project(moved_dir)
    assert validated.root == moved_dir
    assert validated.manifest.project.name == "moved" or validated.manifest.project.name == "orig"


def test_31_no_reference_to_old_path(tmp_path: Path):
    orig_dir = tmp_path / "orig"
    moved_dir = tmp_path / "moved"
    setup_portable_project(orig_dir)
    shutil.copytree(orig_dir, moved_dir)

    report = inspect_project(moved_dir)
    assert report.portability_status == "portable"
    for yml in ["manifest.yaml", "policy.yaml", "workflow.yaml"]:
        text = (moved_dir / ".agent" / yml).read_text(encoding="utf-8")
        assert str(orig_dir) not in text


def test_32_artifact_written_to_moved_copy(tmp_path: Path):
    orig_dir = tmp_path / "orig"
    moved_dir = tmp_path / "moved"
    setup_portable_project(orig_dir)
    shutil.copytree(orig_dir, moved_dir)

    validated = load_project(moved_dir)
    store = ProjectArtifactStore(moved_dir, validated.manifest)
    plan_path = store.write_plan("moved_plan", {"task": "verify portability"})

    assert plan_path.is_file()
    assert str(moved_dir) in str(plan_path)
    assert str(orig_dir) not in str(plan_path)


def test_33_original_project_unmodified(tmp_path: Path):
    orig_dir = tmp_path / "orig"
    moved_dir = tmp_path / "moved"
    setup_portable_project(orig_dir)

    orig_before = {p: p.stat().st_mtime for p in orig_dir.rglob("*")}

    shutil.copytree(orig_dir, moved_dir)
    validated = load_project(moved_dir)
    store = ProjectArtifactStore(moved_dir, validated.manifest)
    store.write_plan("moved_plan", {"test": True})

    orig_after = {p: p.stat().st_mtime for p in orig_dir.rglob("*")}
    assert orig_before == orig_after
