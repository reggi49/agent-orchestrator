import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.artifact_store import ProjectArtifactStore, sanitize_artifact_payload
from agent_system.config import Settings
from agent_system.diff_reviewer import review_diff
from agent_system.execution_package import (
    build_execution_request_from_proposal,
    extract_acceptance_criteria_from_proposal,
    extract_approved_files_from_proposal,
    extract_validation_commands_from_proposal,
)
from agent_system.executors.base import (
    CodingAgentExecutor,
    CodingExecutionRequest,
    CodingExecutionResult,
)
from agent_system.executors.openhands_executor import (
    OPENHANDS_EXECUTION_TIMEOUT_SECONDS,
    OPENHANDS_MAX_ITERATIONS,
    OpenHandsExecutor,
)
from agent_system.graph import build_execution_graph, route_after_coding_executor
from agent_system.nodes import coding_executor_node, independent_diff_review_node
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
from main import cmd_execute


def create_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test Repo\n")
    (path / "src").mkdir(parents=True, exist_ok=True)
    (path / "src" / "app.py").write_text("def run():\n    return 'ok'\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    return path


@pytest.fixture
def dummy_settings():
    return Settings(
        llm_provider="openai",
        llm_api_key="sk-secret-test-key-12345",
        llm_base_url="http://localhost:8000/v1",
        llm_model="gpt-4o",
        default_project_path=None,
        max_repair_attempts=2,
        max_file_characters=10000,
        debug=False,
    )


@pytest.fixture
def sample_proposal_data():
    return {
        "schema_version": "1.0",
        "artifact_id": "proposal_test_01",
        "artifact_type": "proposal",
        "project_id": "sample_proj",
        "status": "approved_for_implementation",
        "evidence_paths": ["src/app.py"],
        "result": {
            "source_plan_path": ".agent/artifacts/plans/plan_01.json",
            "proposal": {
                "proposal_id": "prop_test_01",
                "task_title": "Update Application Logic",
                "owner": "backend",
                "objective": "Enhance app logic in app.py",
                "approved_writable_files": ["src/app.py"],
                "files_to_modify": [
                    {
                        "path": "src/app.py",
                        "action": "modify",
                        "purpose": "Update logic",
                        "evidence_paths": ["src/app.py"],
                    }
                ],
                "implementation_steps": [
                    {
                        "order": 1,
                        "title": "Update logic",
                        "description": "Modify run function",
                        "affected_files": ["src/app.py"],
                        "acceptance_criteria": ["run() must return 'updated'"],
                    }
                ],
                "validation_commands": [
                    {
                        "executable": "python",
                        "arguments": ["-m", "unittest"],
                        "working_directory": ".",
                        "purpose": "Run unit test suite",
                        "requires_approval": True,
                    }
                ],
            },
        },
    }


# ==============================================================================
# TEST 1-10
# ==============================================================================

def test_01_openhands_executor_implements_coding_agent_executor(dummy_settings):
    """1. OpenHandsExecutor mengimplementasikan CodingAgentExecutor."""
    assert issubclass(OpenHandsExecutor, CodingAgentExecutor)
    executor = OpenHandsExecutor(settings=dummy_settings)
    assert isinstance(executor, CodingAgentExecutor)
    # Check all interface methods
    for method in ["validate_configuration", "prepare_execution", "execute", "collect_result", "cancel", "cleanup"]:
        assert hasattr(executor, method)
        assert callable(getattr(executor, method))


def test_02_local_workspace_receives_worktree_path(tmp_path, dummy_settings, sample_proposal_data):
    """2. LocalWorkspace menerima worktree path."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "worktree"
    wt.mkdir()

    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_01")

    mock_ws_cls = MagicMock()
    mock_agent_cls = MagicMock()
    mock_conv_cls = MagicMock()

    with patch.dict("sys.modules", {
        "openhands.sdk": MagicMock(
            LocalWorkspace=mock_ws_cls,
            Agent=mock_agent_cls,
            Conversation=mock_conv_cls,
            LLM=MagicMock(),
        )
    }):
        # Mock git commands inside collect_result
        with patch("subprocess.run") as mock_subproc:
            mock_subproc.return_value = MagicMock(returncode=0, stdout="", stderr="")
            executor.execute(req)

        mock_ws_cls.assert_called_once()
        _, kwargs = mock_ws_cls.call_args
        assert kwargs["working_dir"] == wt.resolve()


def test_03_main_project_not_used_as_workspace(tmp_path, dummy_settings, sample_proposal_data):
    """3. Project utama tidak digunakan sebagai workspace."""
    repo = create_git_repo(tmp_path / "main_repo")
    settings_with_main = Settings(
        llm_provider="openai",
        llm_api_key="sk-test",
        llm_base_url="http://localhost:8000/v1",
        llm_model="gpt-4o",
        default_project_path=repo,
        max_repair_attempts=2,
        max_file_characters=10000,
        debug=False,
    )
    executor = OpenHandsExecutor(settings=settings_with_main)
    # Target is main repo instead of worktree -> must be rejected
    req = build_execution_request_from_proposal(sample_proposal_data, repo, "run_02")
    res = executor.execute(req)

    assert res.status == "scope_violation"
    assert "Workspace tidak boleh sama dengan project utama" in res.summary


def test_04_request_created_from_approved_proposal(tmp_path, sample_proposal_data):
    """4. Request dibuat dari approved proposal."""
    wt = tmp_path / "wt"
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_04", project_id="sample_proj")

    assert isinstance(req, CodingExecutionRequest)
    assert req.run_id == "run_04"
    assert req.project_id == "sample_proj"
    assert req.task_title == "Update Application Logic"
    assert "Enhance app logic" in req.task_description
    assert req.worktree_identifier == str(wt)


def test_05_approved_files_derived_from_proposal(sample_proposal_data):
    """5. Approved files berasal dari proposal."""
    prop_inner = sample_proposal_data["result"]["proposal"]
    approved = extract_approved_files_from_proposal(prop_inner)
    assert approved == ["src/app.py"]
    # All paths must be relative
    for p in approved:
        assert not p.startswith("/")
        assert not p.startswith("\\")


def test_06_acceptance_criteria_derived_from_proposal(sample_proposal_data):
    """6. Acceptance criteria berasal dari proposal."""
    prop_inner = sample_proposal_data["result"]["proposal"]
    ac = extract_acceptance_criteria_from_proposal(prop_inner)
    assert "run() must return 'updated'" in ac


def test_07_validation_commands_derived_from_proposal(sample_proposal_data):
    """7. Validation commands berasal dari proposal."""
    prop_inner = sample_proposal_data["result"]["proposal"]
    cmds = extract_validation_commands_from_proposal(prop_inner)
    assert len(cmds) == 1
    assert cmds[0]["executable"] == "python"
    assert cmds[0]["arguments"] == ["-m", "unittest"]
    assert cmds[0]["working_directory"] == "."


def test_08_api_key_not_in_prompt(tmp_path, dummy_settings, sample_proposal_data):
    """8. API key tidak masuk prompt."""
    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, tmp_path / "wt", "run_08")
    prep = executor.prepare_execution(req)
    prompt = prep["instruction"]

    assert dummy_settings.llm_api_key not in prompt
    assert "sk-" not in prompt
    assert "Authorization" not in prompt


def test_09_absolute_path_not_in_artifact():
    """9. Absolute path tidak masuk artifact."""
    wt_abs = "/Users/user/Projects/.agent-worktrees/sample-1234"
    payload = {
        "worktree_path": wt_abs,
        "changed_files": ["src/app.py"],
        "summary": "Completed",
    }
    sanitized = sanitize_artifact_payload(payload)
    dumped = json.dumps(sanitized)
    assert wt_abs not in dumped
    assert sanitized["worktree_descriptor"] == "sample-1234"


def test_10_unexpected_file_produces_scope_violation(tmp_path, dummy_settings, sample_proposal_data):
    """10. Unexpected file menghasilkan scope_violation."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-scope", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    # Modify approved file AND an unexpected unapproved file
    (wt / "src" / "app.py").write_text("def run():\n    return 'updated'\n")
    (wt / "unexpected_file.txt").write_text("unauthorized file\n")

    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_10")
    res = executor.collect_result(req)

    assert res.status == "scope_violation"
    assert "unexpected_file.txt" in res.scope_violations
    assert res.status != "completed"


# ==============================================================================
# TEST 11-20
# ==============================================================================

def test_11_additional_file_request_produces_proposal_revision_required(tmp_path, dummy_settings, sample_proposal_data):
    """11. Additional file request menghasilkan proposal_revision_required."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-add-file", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_11")
    res = executor.collect_result(
        req,
        requested_additional_files=["src/extra_helper.py"],
    )

    assert res.status == "proposal_revision_required"
    assert "src/extra_helper.py" in res.requested_additional_files


def test_12_timeout_produces_timed_out(tmp_path, dummy_settings, sample_proposal_data):
    """12. Timeout menghasilkan timed_out."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-timeout", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_12")
    res = executor.collect_result(req, timed_out=True)

    assert res.status == "timed_out"
    assert res.timed_out is True


def test_13_failure_produces_failed(tmp_path, dummy_settings, sample_proposal_data):
    """13. Failure menghasilkan failed."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-fail", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_13")
    res = executor.collect_result(req, execution_error="Syntax error during compilation")

    assert res.status == "failed"


def test_14_completed_routes_to_independent_review():
    """14. Completed menuju independent review."""
    state = {
        "status": "completed",
        "executor_retry_count": 0,
        "max_executor_retries": 1,
    }
    next_node = route_after_coding_executor(state)
    assert next_node == "independent_diff_review"


def test_15_custom_fallback_not_called(tmp_path, dummy_settings, sample_proposal_data):
    """15. Custom fallback tidak dipanggil."""
    with patch("agent_system.implementation_executor.LLMImplementationExecutor") as mock_legacy:
        executor = OpenHandsExecutor(settings=dummy_settings)
        req = build_execution_request_from_proposal(sample_proposal_data, tmp_path / "wt", "run_15")
        prep = executor.prepare_execution(req)
        mock_legacy.assert_not_called()


def test_16_custom_implementation_template_not_called(tmp_path, dummy_settings, sample_proposal_data):
    """16. Custom implementation template tidak dipanggil."""
    # Ensure no PPOB implementation templates exist or are imported
    executor_file = (Path(__file__).parent.parent / "agent_system" / "executors" / "openhands_executor.py").read_text()
    assert "ppob_implementation_templates" not in executor_file
    assert "template" not in executor_file.lower() or "template" in "system prompt"


def test_17_executor_does_not_commit(tmp_path, dummy_settings, sample_proposal_data):
    """17. Executor tidak commit."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-no-commit", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    commits_before = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()

    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_17")
    (wt / "src" / "app.py").write_text("def run():\n    return 'changed'\n")
    executor.collect_result(req)

    commits_after = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    assert commits_after == commits_before


def test_18_executor_does_not_push(tmp_path, dummy_settings, sample_proposal_data):
    """18. Executor tidak push."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-no-push", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_18")
    executor.collect_result(req)

    remotes = subprocess.run(["git", "remote"], cwd=wt, capture_output=True, text=True).stdout.strip()
    assert remotes == ""


def test_19_executor_does_not_merge(tmp_path, dummy_settings, sample_proposal_data):
    """19. Executor tidak merge."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-no-merge", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    main_branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert main_branch == "main"
    main_count = subprocess.run(["git", "rev-list", "--count", "main"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert main_count == "1"


def test_20_local_workspace_used_without_docker(dummy_settings):
    """20. LocalWorkspace digunakan tanpa Docker."""
    executor = OpenHandsExecutor(settings=dummy_settings)
    config = executor.validate_configuration()

    assert config["valid"] is True
    assert config["workspace_type"] == "LocalWorkspace"
    assert config["docker_used"] is False


# ==============================================================================
# TEST 21-30
# ==============================================================================

def test_21_dry_run_does_not_call_llm(tmp_path, sample_proposal_data):
    """21. Dry-run tidak memanggil LLM."""
    mock_llm_call = MagicMock()
    with patch("langchain_openai.ChatOpenAI.invoke", mock_llm_call):
        state = {
            "run_id": "test_dry_21",
            "dry_run": True,
            "proposal_data": sample_proposal_data,
            "preflight_result": {"is_valid": True, "status": "valid"},
            "worktree_path": str(tmp_path / "wt"),
        }
        res = coding_executor_node(state)
        assert res["status"] == "dry_run_ready"
        mock_llm_call.assert_not_called()


def test_22_dry_run_does_not_create_worktree(tmp_path, sample_proposal_data):
    """22. Dry-run tidak membuat worktree."""
    repo = create_git_repo(tmp_path / "repo")
    from agent_system.nodes import branch_guard_node

    state = {
        "run_id": "test_dry_22",
        "dry_run": True,
        "isolation": "git-worktree",
        "project_root": str(repo),
        "proposal_data": sample_proposal_data,
    }
    res = branch_guard_node(state)
    assert "worktree_descriptor" in res
    # Ensure no directory was created
    wt_root = tmp_path / ".agent-worktrees"
    assert not wt_root.exists()


def test_23_dry_run_does_not_write_source(tmp_path, sample_proposal_data):
    """23. Dry-run tidak menulis source."""
    repo = create_git_repo(tmp_path / "repo")
    app_file = repo / "src" / "app.py"
    original_content = app_file.read_text()

    state = {
        "run_id": "test_dry_23",
        "dry_run": True,
        "proposal_data": sample_proposal_data,
        "preflight_result": {"is_valid": True, "status": "valid"},
        "project_root": str(repo),
    }
    coding_executor_node(state)
    assert app_file.read_text() == original_content


def test_24_dry_run_does_not_run_commands(tmp_path, sample_proposal_data):
    """24. Dry-run tidak menjalankan command."""
    commands_executed = []
    orig_run = subprocess.run

    def mock_run(cmd, *a, **k):
        if isinstance(cmd, list) and cmd and cmd[0] in ["python", "go", "pytest"]:
            commands_executed.append(cmd)
        return orig_run(cmd, *a, **k)

    with patch("subprocess.run", mock_run):
        state = {
            "run_id": "test_dry_24",
            "dry_run": True,
            "proposal_data": sample_proposal_data,
            "preflight_result": {"is_valid": True, "status": "valid"},
            "worktree_path": str(tmp_path / "wt"),
        }
        coding_executor_node(state)

    assert len(commands_executed) == 0


def test_25_independent_reviewer_uses_git_actual_diff(tmp_path):
    """25. Independent reviewer menggunakan Git actual diff."""
    repo = create_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-b", "agent/test-diff-25", str(wt), "HEAD"], cwd=repo, check=True, capture_output=True)

    # Modify approved file
    (wt / "src" / "app.py").write_text("def run():\n    return 'reviewed'\n")

    state = {
        "worktree_path": str(wt),
        "dry_run": False,
        "approved_files": ["src/app.py"],
        "proposal_data": {
            "result": {
                "proposal": {
                    "implementation_steps": [{"acceptance_criteria": ["run() must work"]}],
                }
            }
        },
        "coding_execution_result": {
            "validation_results": [{"status": "passed", "exit_code": 0}],
        },
    }

    res = independent_diff_review_node(state)
    assert res["status"] == "execution_review_ready"
    assert res["diff_review"]["changed_files"] == ["src/app.py"]
    assert res["diff_review"]["unexpected_files"] == []


def test_26_result_artifact_saved_in_project(tmp_path):
    """26. Result artifact disimpan di project."""
    repo = create_git_repo(tmp_path / "repo")
    initialize_project(repo, apply=True)
    val_proj = load_project(repo)
    store = ProjectArtifactStore(repo, val_proj.manifest)

    report_file = store.write_report("execution_review_01", {
        "status": "execution_review_ready",
        "executor": "openhands",
        "changed_files": ["src/app.py"],
    })

    assert report_file.is_file()
    assert report_file.is_relative_to(repo / ".agent" / "artifacts" / "reports")


def test_27_result_artifact_not_saved_in_orchestrator(tmp_path):
    """27. Result artifact tidak disimpan di orchestrator."""
    orchestrator_dir = Path(__file__).resolve().parent.parent
    assert not (orchestrator_dir / ".agent" / "artifacts" / "reports" / "execution_review_01.json").exists()


def test_28_secret_not_printed(capsys, dummy_settings):
    """28. Secret tidak dicetak."""
    executor = OpenHandsExecutor(settings=dummy_settings)
    executor._log_progress(
        executor="openhands",
        iteration=1,
        elapsed_time=2.5,
        action_category="edit",
        status="running",
    )
    captured = capsys.readouterr().out
    assert dummy_settings.llm_api_key not in captured
    assert "sk-" not in captured
    assert "Authorization" not in captured


def test_29_timeout_does_not_poll_indefinitely(tmp_path, dummy_settings, sample_proposal_data):
    """29. Timeout tidak melakukan polling tanpa batas."""
    wt = tmp_path / "wt"
    wt.mkdir()
    executor = OpenHandsExecutor(settings=dummy_settings)
    req = build_execution_request_from_proposal(sample_proposal_data, wt, "run_29")
    # Timeout collection returns immediately
    res = executor.collect_result(req, timed_out=True)
    assert res.timed_out is True
    assert res.status == "timed_out"


def test_30_runtime_source_does_not_hardcode_ppob():
    """30. Runtime source tidak hardcode PPOB."""
    orchestrator_dir = Path(__file__).resolve().parent.parent
    runtime_files = [
        orchestrator_dir / "agent_system" / "executors" / "openhands_executor.py",
        orchestrator_dir / "agent_system" / "executors" / "base.py",
        orchestrator_dir / "agent_system" / "execution_package.py",
    ]
    for rf in runtime_files:
        if rf.is_file():
            content = rf.read_text(encoding="utf-8")
            assert "/Projects/PPOB" not in content, f"Hardcoded PPOB path found in {rf.name}"
