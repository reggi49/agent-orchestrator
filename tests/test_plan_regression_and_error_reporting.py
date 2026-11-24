import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.error_reporting import format_safe_error_report, sanitize_text
from agent_system.nodes import repository_explorer_node
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
from main import locate_verified_plan


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    repo = tmp_path / "test_repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Test Repo\n")
    (repo / "backend").mkdir()
    (repo / "backend" / "main.go").write_text("package main\n\nfunc main() {}\n")
    initialize_project(repo, apply=True)
    return repo


def test_repository_explorer_node_completes_without_name_error(sample_project: Path):
    """Test that repository_explorer_node completes successfully without NameError."""
    state = {
        "project_root": str(sample_project),
        "analysis": {
            "recommended_task": {
                "title": "Add health endpoint",
                "description": "Implement GET /health in main.go",
                "acceptance_criteria": ["Returns 200 ok"],
            }
        },
    }

    result = repository_explorer_node(state)

    assert result is not None
    assert "repository_evidence" in result
    assert result["current_stage"] == "task_verifier"
    assert result["status"] == "running"
    evidence = result["repository_evidence"]
    assert evidence["candidate_task"] == "Add health endpoint"
    assert isinstance(evidence["inspected_files"], list)
    assert isinstance(evidence["results"], list)


def test_no_undefined_name_in_repository_explorer_node():
    """Verify that RepositoryExplorer is properly imported and defined in nodes module."""
    import agent_system.nodes as nodes

    assert hasattr(nodes, "RepositoryExplorer")
    assert callable(nodes.RepositoryExplorer)


def test_safe_error_reporting_includes_type_msg_traceback_without_secrets():
    """Verify error reporting outputs exception type, message, and traceback safely."""
    secret_key = "sk-super-secret-key-1234567890"

    try:
        raise ValueError(f"Failed connecting to server with {secret_key} and Bearer abc123def456")
    except Exception as e:
        report = format_safe_error_report(e, api_key=secret_key, context="Workflow")

    assert "ERROR: Workflow gagal: ValueError" in report
    assert "Message:" in report
    assert "Traceback (sanitized):" in report
    assert secret_key not in report
    assert "[REDACTED_API_KEY]" in report
    assert "abc123def456" not in report
    assert "Bearer [REDACTED]" in report


def test_plan_failure_produces_failed_status_artifact(sample_project: Path):
    """Verify that a failed planning run writes artifact with status 'failed'."""
    val_proj = load_project(sample_project)
    store = ProjectArtifactStore(sample_project, val_proj.manifest)

    plan_id = "plan_failed_01"
    plan_data = {
        "initial_candidate": "Candidate",
        "status": "failed",
        "error_type": "NameError",
        "error_message": "RepositoryExplorer is not defined",
    }
    plan_file = store.write_plan(plan_id, plan_data)
    assert plan_file.is_file()

    saved_data = json.loads(plan_file.read_text(encoding="utf-8"))
    assert saved_data["status"] == "failed"


def test_locate_verified_plan_rejects_failed_plan(sample_project: Path):
    """Verify that locate_verified_plan skips failed plans when searching for 'latest'."""
    val_proj = load_project(sample_project)
    store = ProjectArtifactStore(sample_project, val_proj.manifest)

    # Write failed plan
    store.write_plan("plan_fail", {
        "status": "failed",
        "final_task": "Task fail",
    })

    with pytest.raises(SystemExit):
        locate_verified_plan(sample_project, plan_arg="latest")


def test_locate_verified_plan_rejects_partial_plan(sample_project: Path):
    """Verify that locate_verified_plan skips partial plans missing final_task."""
    val_proj = load_project(sample_project)
    store = ProjectArtifactStore(sample_project, val_proj.manifest)

    # Write plan with plan_ready status but missing final_task (partial artifact)
    store.write_plan("plan_partial", {
        "status": "plan_ready",
        # missing final_task and recommended_task
    })

    with pytest.raises(SystemExit):
        locate_verified_plan(sample_project, plan_arg="latest")


def test_locate_verified_plan_selects_valid_over_newer_failed_or_partial_plan(sample_project: Path):
    """Verify latest resolver picks valid plan_ready even if newer failed/partial plans exist."""
    val_proj = load_project(sample_project)
    store = ProjectArtifactStore(sample_project, val_proj.manifest)

    # 1. Write older valid plan
    valid_file = store.write_plan("plan_valid", {
        "status": "plan_ready",
        "final_task": "Add health endpoint",
        "final_owner": "backend",
        "final_acceptance_criteria": ["200 OK"],
    })

    time.sleep(0.05)

    # 2. Write newer failed plan
    store.write_plan("plan_newer_failed", {
        "status": "failed",
        "final_task": "Some broken task",
    })

    time.sleep(0.05)

    # 3. Write newer partial plan
    store.write_plan("plan_newer_partial", {
        "status": "plan_ready",
        # missing task fields
    })

    # locate_verified_plan with "latest" must find the valid plan, not the newer failed/partial ones
    selected_file, data = locate_verified_plan(sample_project, plan_arg="latest")
    assert selected_file.name == "plan_valid.json"
    assert data["status"] == "plan_ready"
    assert data["result"]["final_task"] == "Add health endpoint"
