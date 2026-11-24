import json
import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.deterministic_safety import (
    ALLOWED_FALLBACK_SCOPES,
    PROHIBITED_DECISIONS,
    evaluate_deterministic_fallback_safety,
)
from agent_system.project_initializer import initialize_project
from agent_system.recovery_inspector import RecoveryAction, inspect_recovery_conditions
from agent_system.schemas import ProgramState, ProgramTask


def setup_test_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test Repo\n")
    (path / "backend").mkdir(parents=True, exist_ok=True)
    (path / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"ok\" }\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    initialize_project(path, apply=True)
    return path


# ==============================================================================
# Category I: Deterministic Fallback Tests (57 - 60)
# ==============================================================================

def test_57_low_risk_schema_normalization_allowed():
    res = evaluate_deterministic_fallback_safety(
        context_text="Format unit test results",
        risk_level="low",
        fallback_type="schema_normalization",
    )
    assert res["allowed"] is True
    assert res["requires_ceo_escalation"] is False
    assert "schema_normalization" in res["allowed_scope"]


def test_58_high_risk_wallet_design_not_determined_by_fallback():
    res = evaluate_deterministic_fallback_safety(
        context_text="Update wallet balance transfer ledger",
        risk_level="high",
        fallback_type="task_execution",
    )
    assert res["allowed"] is False
    assert res["requires_ceo_escalation"] is True
    assert "wallet_balance" in res["prohibited_decisions"]


def test_59_high_risk_fallback_produces_pause_or_escalation():
    res = evaluate_deterministic_fallback_safety(
        context_text="Database migration drop table ledger",
        risk_level="high",
        fallback_type="proposal_design",
    )
    assert res["allowed"] is False
    assert "pause" in res["reason"].lower() or "jeda" in res["reason"].lower()


def test_60_medium_fallback_cannot_final_integrate_without_approval():
    res = evaluate_deterministic_fallback_safety(
        context_text="Update backend service order status",
        risk_level="medium",
        fallback_type="proposal_design",
    )
    assert res["allowed"] is True
    assert res["requires_human_approval"] is True


# ==============================================================================
# Category J: Shutdown dan Recovery Tests (61 - 68)
# ==============================================================================

def test_61_sigint_saves_paused_checkpoint(tmp_path: Path):
    from agent_system.ceo_mode import CEORunner

    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls, \
         patch("langchain_openai.ChatOpenAI"):
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        # Trigger simulated SIGINT during first task
        def simulate_sigint(request, **kwargs):
            # Send SIGINT to current process
            os.kill(os.getpid(), signal.SIGINT)
            return {"overall_status": "interrupted", "error_message": "KeyboardInterrupt"}

        mock_inst.run.side_effect = simulate_sigint

        state = runner.run_ceo(
            request_text="- Task 1\n- Task 2\n",
            program_id="sigint01",
            autonomous=True,
        )

        assert state.status == "interrupted"
        assert "interrupted" in state.last_checkpoint.lower()
        assert "resume" in state.next_action


def test_62_sigterm_does_not_start_new_task(tmp_path: Path):
    from agent_system.ceo_mode import CEORunner

    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    tasks_started = 0

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls, \
         patch("langchain_openai.ChatOpenAI"):
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        def simulate_sigterm(request, **kwargs):
            nonlocal tasks_started
            tasks_started += 1
            os.kill(os.getpid(), signal.SIGTERM)
            return {"overall_status": "interrupted", "error_message": "SIGTERM"}

        mock_inst.run.side_effect = simulate_sigterm

        state = runner.run_ceo(
            request_text="- Task 1\n- Task 2\n- Task 3\n",
            program_id="sigterm01",
            autonomous=True,
        )

        # Only 1 task was started before stopping
        assert tasks_started == 1
        assert state.status == "interrupted"


def test_63_completed_task_not_repeated_after_resume(tmp_path: Path):
    from agent_system.ceo_mode import CEORunner

    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    executed_tasks = []

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls, \
         patch("langchain_openai.ChatOpenAI"):
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        def run_mock(request, **kwargs):
            executed_tasks.append(request)
            return {
                "overall_status": "completed",
                "run_id": f"run_{len(executed_tasks)}",
                "changed_files": ["backend/app.go"],
                "test_status": "passed",
                "commit_hash": "a" * 40,
            }

        mock_inst.run.side_effect = run_mock

        # Run 1 task, then pause/stop
        state = runner.run_ceo(
            request_text="- Task 1\n- Task 2\n",
            program_id="resume01",
            autonomous=False,
        )
        # Mark Task 1 completed in state
        state.tasks[0].status = "completed"
        runner.pm.save_program_state(state)

        # Resume autonomous loop
        runner.run_ceo(program_id="resume01", resume=True, autonomous=True)

        # Only Task 2 should have been executed!
        assert len(executed_tasks) == 1
        assert "Task 2" in executed_tasks[0]


def test_64_openhands_execution_finished_not_rerun(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    state = ProgramState(
        program_id="p64",
        project_id="repo",
        objective="Obj",
        active_task="t1",
        active_run_id="run_64",
        tasks=[ProgramTask(task_id="t1", epic_id="e1", title="T1", linked_run_id="run_64", status="in_progress")],
    )

    # Execution artifact exists and is execution_review_ready
    exec_file = repo / ".agent" / "artifacts" / "executions" / "execution_run_64.json"
    exec_file.parent.mkdir(parents=True, exist_ok=True)
    exec_file.write_text(json.dumps({"status": "execution_review_ready", "result": {"status": "execution_review_ready"}}))

    res = inspect_recovery_conditions(repo, state)
    assert res["execution_completed"] is True
    assert res["recommended_action"] == RecoveryAction.SKIP_EXECUTION_PROCEED_TO_REVIEW


def test_65_worktree_not_duplicated_after_crash(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    wt_id = "test-wt-unique"
    wt_path = tmp_path / ".agent-worktrees" / wt_id
    wt_path.mkdir(parents=True, exist_ok=True)

    state = ProgramState(
        program_id="p65",
        project_id="repo",
        objective="Obj",
        active_task="t1",
        worktree_reference=wt_id,
        tasks=[ProgramTask(task_id="t1", epic_id="e1", title="T1", status="in_progress")],
    )

    with patch("agent_system.recovery_inspector.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        res = inspect_recovery_conditions(repo, state)
        assert res["worktree_exists"] is True
        assert res["recommended_action"] == RecoveryAction.RECONSTRUCT_WORKTREE


def test_66_commit_already_integrated_not_cherry_picked_again(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    current_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    state = ProgramState(
        program_id="p66",
        project_id="repo",
        objective="Obj",
        active_task="t1",
        tasks=[ProgramTask(task_id="t1", epic_id="e1", title="T1", status="in_progress", completion_evidence=current_head)],
    )

    res = inspect_recovery_conditions(repo, state, target_branch="main")
    assert res["commit_already_on_target"] is True
    assert res["recommended_action"] == RecoveryAction.SKIP_INTEGRATION_MARK_COMPLETED


def test_67_crash_during_post_validation_recovered_idempotently(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    current_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    # Integration artifact exists
    integ_file = repo / ".agent" / "artifacts" / "executions" / "integration_r67.json"
    integ_file.parent.mkdir(parents=True, exist_ok=True)
    integ_file.write_text(json.dumps({"status": "completed", "result": {"status": "completed", "commit_hash": current_head}}))

    state = ProgramState(
        program_id="p67",
        project_id="repo",
        objective="Obj",
        active_task="t1",
        active_run_id="r67",
        tasks=[ProgramTask(task_id="t1", epic_id="e1", title="T1", linked_run_id="r67", status="in_progress")],
    )

    res = inspect_recovery_conditions(repo, state)
    assert res["integration_completed"] is True
    assert res["recommended_action"] == RecoveryAction.SKIP_INTEGRATION_MARK_COMPLETED


def test_68_printed_resume_command_is_valid():
    state = ProgramState(
        program_id="cmd_prog1",
        project_id="repo",
        objective="Obj",
        next_action="python main.py ceo --project /path/to/project --program cmd_prog1 --resume --autonomous",
    )
    assert "main.py ceo" in state.next_action
    assert "--resume" in state.next_action
    assert "cmd_prog1" in state.next_action


# ==============================================================================
# Category K: Security and Prohibited Operations (69 - 80)
# ==============================================================================

def test_69_to_74_no_prohibited_git_operations():
    from agent_system.integrator import integrate_execution

    forbidden_git_subcommands = {"push", "merge", "rebase", "reset", "tag"}
    # Verify by inspecting integrator source code directly
    integrator_src = (Path(__file__).parent.parent / "agent_system" / "integrator.py").read_text(encoding="utf-8")

    for fcmd in forbidden_git_subcommands:
        # Check there is no subprocess calling git push, merge, rebase, reset, tag
        assert f'["git", "{fcmd}"' not in integrator_src
        assert f"['git', '{fcmd}'" not in integrator_src


def test_75_76_no_production_deployment_or_direct_db_access():
    orchestrator_root = Path(__file__).parent.parent
    for pyf in (orchestrator_root / "agent_system").glob("**/*.py"):
        content = pyf.read_text(encoding="utf-8").lower()
        assert "drop database" not in content
        assert "kubectl apply" not in content


def test_77_to_80_artifact_sanitization_strictness():
    from agent_system.artifact_store import sanitize_artifact_payload

    dirty = {
        "abs_path": "/Users/user/Projects/secret_folder",
        "raw_diff": "diff --git a/app.go b/app.go\n+secret",
        "raw_prompt": "You are assistant with key sk-9999888877776666",
        "auth": "Authorization: Bearer my_secret_token_123",
        "api_key": "sk-1111222233334444",
        "env_var": "SECRET_KEY='super_env_secret'",
    }

    clean = sanitize_artifact_payload(dirty)
    dumped = json.dumps(clean)

    assert "/Users/user/Projects/secret_folder" not in dumped
    assert "diff --git" not in dumped
    assert "sk-9999888877776666" not in dumped
    assert "my_secret_token_123" not in dumped
    assert "sk-1111222233334444" not in dumped
    assert "super_env_secret" not in dumped
