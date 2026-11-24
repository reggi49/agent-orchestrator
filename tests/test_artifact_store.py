import json
from pathlib import Path
import pytest

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.path_safety import PathSafetyError
from agent_system.project_detector import detect_project
from agent_system.project_initializer import initialize_project


def setup_initialized_project(project_dir: Path):
    (project_dir / "backend").mkdir(parents=True, exist_ok=True)
    (project_dir / "backend" / "go.mod").write_text("module example.com/app\n\ngo 1.22\n", encoding="utf-8")
    initialize_project(project_dir, apply=True)
    manifest = detect_project(project_dir)
    return ProjectArtifactStore(project_dir, manifest)


def test_21_plan_written_to_plans_dir(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    plan_path = store.write_plan("plan_001", {"goal": "migrate auth"})
    assert plan_path.is_file()
    assert plan_path.parent.name == "plans"
    assert "plan_001.json" in str(plan_path)

    data = json.loads(plan_path.read_text(encoding="utf-8"))
    assert data["artifact_type"] == "plan"
    assert data["result"]["goal"] == "migrate auth"


def test_22_report_written_to_reports_dir(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    report_path = store.write_report("report_001", {"summary": "all clear"})
    assert report_path.is_file()
    assert report_path.parent.name == "reports"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["artifact_type"] == "report"


def test_23_decision_written_to_decisions_dir(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    dec_path = store.write_decision("dec_001", {"decision": "proceed"})
    assert dec_path.is_file()
    assert dec_path.parent.name == "decisions"
    data = json.loads(dec_path.read_text(encoding="utf-8"))
    assert data["artifact_type"] == "decision"


def test_24_path_traversal_rejected(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    with pytest.raises((PathSafetyError, ValueError)):
        store.write_plan("../escape_plan", {"data": "test"})


def test_25_write_outside_artifact_root_rejected(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    with pytest.raises((PathSafetyError, ValueError)):
        store.write_plan("/tmp/evil_plan", {"data": "test"})


def test_26_secret_fields_redacted(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    payload = {
        "api_key": "sk-secret12345",
        "nested": {"token": "my-secret-token", "safe_val": 42},
        "auth_header": "Bearer abcdef123456",
        "prompt": "You are a secret agent ...",
    }
    plan_path = store.write_plan("secret_test", payload)
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    
    assert data["result"]["api_key"] == "[REDACTED]"
    assert data["result"]["nested"]["token"] == "[REDACTED]"
    assert data["result"]["nested"]["safe_val"] == 42
    assert "sk-secret" not in json.dumps(data)
    assert data["result"]["prompt"] == "[OMITTED_FOR_SAFETY]"


def test_27_atomic_write_completed(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    plan_path = store.write_plan("atomic_test", {"item": "ok"})
    assert plan_path.is_file()
    # tmp dir shouldn't leave un-renamed artifacts
    tmp_files = list((tmp_path / ".agent" / "tmp").glob("tmp_*_atomic_test.json"))
    assert len(tmp_files) == 0


def test_28_relative_evidence_paths(tmp_path: Path):
    store = setup_initialized_project(tmp_path)
    payload = {
        "evidence_paths": ["backend/go.mod", "README.md"],
        "info": "evidence test",
    }
    plan_path = store.write_plan("evidence_test", payload)
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    assert data["evidence_paths"] == ["backend/go.mod", "README.md"]
    for p in data["evidence_paths"]:
        assert not p.startswith("/")
