import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.ceo_mode import CEORunner, ProgressHeartbeat
from agent_system.gateway_resilience import (
    CircuitBreaker,
    CircuitState,
    FailureCategory,
    classify_failure,
)
from agent_system.program_manager import ProgramManager
from agent_system.project_initializer import initialize_project
from agent_system.recovery_inspector import reconstruct_misclassified_approval_states
from agent_system.risk_classifier import (
    classify_final_risk,
    classify_post_execution_risk,
    classify_pre_execution_risk,
)
from agent_system.schemas import ProgramEpic, ProgramState, ProgramTask


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


@pytest.fixture(autouse=True)
def mock_langchain_openai():
    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value.invoke.side_effect = RuntimeError("Offline mock")
        yield mock_chat


# ==============================================================================
# 1. human_required becomes awaiting_ceo_input
# ==============================================================================
def test_01_human_required_becomes_awaiting_ceo_input(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        # Runner returns awaiting_human_approval at proposal stage
        mock_inst.run.return_value = {
            "overall_status": "awaiting_human_approval",
            "approval_stage": "proposal",
            "run_id": "run_hr_01",
            "proposal_artifact_id": "proposal_hr_01",
            "error_message": "Proposal review status 'human_required'.",
        }

        state = runner.run_ceo(
            request_text="- Task 1: Check something\n",
            program_id="prog_hr1",
            autonomous=True,
        )

        assert state.status == "awaiting_ceo_input"
        assert state.current_phase == "proposal_approval"
        assert state.tasks[0].status == "awaiting_input"
        assert state.tasks[0].approval_stage == "proposal"
        assert state.approval_checkpoint is not None
        assert state.approval_checkpoint["approval_stage"] == "proposal"
        assert "resume" in state.next_action
        assert "--approve" in state.next_action


# ==============================================================================
# 2. human_required does not increment retry count
# ==============================================================================
def test_02_human_required_does_not_increment_retry_count(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        mock_inst.run.return_value = {
            "overall_status": "awaiting_human_approval",
            "approval_stage": "proposal",
            "run_id": "run_hr_02",
            "error_message": "human_required",
        }

        state = runner.run_ceo(
            request_text="- Task 1: Check\n",
            program_id="prog_hr2",
            autonomous=True,
        )

        assert state.tasks[0].attempt_count == 0
        assert state.tasks[0].status == "awaiting_input"


# ==============================================================================
# 3. human_required not classified as transient gateway
# ==============================================================================
def test_03_human_required_not_classified_as_transient_gateway():
    cat1 = classify_failure(Exception("Proposal review tidak disetujui (status: human_required)."))
    assert cat1 == FailureCategory.APPROVAL_REQUIRED
    assert cat1 != FailureCategory.TRANSIENT_GATEWAY

    cat2 = classify_failure(Exception("Task failed: human_required approval needed"))
    assert cat2 == FailureCategory.APPROVAL_REQUIRED


# ==============================================================================
# 4. human_required does not open circuit breaker
# ==============================================================================
def test_04_human_required_does_not_open_circuit_breaker(tmp_path: Path):
    cb = CircuitBreaker(tmp_path, provider_id="test_prov", failure_threshold=2)
    cb.record_failure(FailureCategory.APPROVAL_REQUIRED)
    cb.record_failure(FailureCategory.APPROVAL_REQUIRED)
    cb.record_failure(FailureCategory.APPROVAL_REQUIRED)

    assert cb.state == CircuitState.CLOSED
    assert cb.is_available() is True


# ==============================================================================
# 5. resume --approve from proposal gate continues to execution
# ==============================================================================
def test_05_resume_approve_from_proposal_gate_continues_to_execution(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        mock_inst.run.return_value = {
            "overall_status": "awaiting_human_approval",
            "approval_stage": "proposal",
            "run_id": "run_prop_gate",
            "proposal_artifact_id": "proposal_gate_01",
        }

        # Step 1: Pauses at proposal approval gate
        state = runner.run_ceo(
            request_text="- Task 1: Proposal Gate Task\n",
            program_id="p_gate01",
            autonomous=True,
        )
        assert state.status == "awaiting_ceo_input"
        assert state.current_phase == "proposal_approval"

        # Step 2: Resume with --approve
        mock_inst.resume.return_value = {
            "overall_status": "completed",
            "changed_files": ["backend/app.go"],
            "test_status": "passed",
            "commit_hash": "c" * 40,
        }

        resumed = runner.run_ceo(
            program_id="p_gate01",
            resume=True,
            approve=True,
            autonomous=True,
        )

        assert resumed.status == "completed"
        assert resumed.tasks[0].status == "completed"
        assert resumed.tasks[0].completion_evidence == "c" * 40


# ==============================================================================
# 6. resume --approve from execution gate continues to integration
# ==============================================================================
def test_06_resume_approve_from_execution_gate_continues_to_integration(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        mock_inst.run.return_value = {
            "overall_status": "awaiting_human_approval",
            "approval_stage": "execution",
            "run_id": "run_exec_gate",
            "execution_artifact_id": "execution_gate_01",
        }

        state = runner.run_ceo(
            request_text="- Task 1: Exec Gate Task\n",
            program_id="e_gate01",
            autonomous=True,
        )
        assert state.status == "awaiting_ceo_input"
        assert state.current_phase == "execution_approval"

        mock_inst.resume.return_value = {
            "overall_status": "completed",
            "changed_files": ["backend/app.go"],
            "test_status": "passed",
            "commit_hash": "e" * 40,
        }

        resumed = runner.run_ceo(
            program_id="e_gate01",
            resume=True,
            approve=True,
            autonomous=True,
        )

        assert resumed.status == "completed"
        assert resumed.tasks[0].status == "completed"
        assert resumed.tasks[0].completion_evidence == "e" * 40


# ==============================================================================
# 7. approval does not create duplicate plan/proposal/run/worktree
# ==============================================================================
def test_07_approval_does_not_create_duplicate_plan_proposal_run_worktree(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    run_calls = []

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst

        def mock_run(request, **kwargs):
            run_calls.append(request)
            return {
                "overall_status": "awaiting_human_approval",
                "approval_stage": "proposal",
                "run_id": "run_single_id",
                "proposal_artifact_id": "proposal_single_id",
            }

        mock_inst.run.side_effect = mock_run

        state = runner.run_ceo(
            request_text="- Task 1\n",
            program_id="single_id_p",
            autonomous=True,
        )
        assert len(run_calls) == 1

        mock_inst.resume.return_value = {
            "overall_status": "completed",
            "changed_files": ["backend/app.go"],
            "test_status": "passed",
            "commit_hash": "f" * 40,
        }

        runner.run_ceo(
            program_id="single_id_p",
            resume=True,
            approve=True,
            autonomous=True,
        )

        # runner.run should NOT have been called again (no new plan/proposal/run)
        assert len(run_calls) == 1
        mock_inst.resume.assert_called_once_with(
            run_arg="run_single_id",
            approve=True,
            reject=False,
            request_changes=False,
            apply=True,
            note="",
        )


# ==============================================================================
# 8. approve without pending approval is rejected fail-closed
# ==============================================================================
def test_08_approve_without_pending_approval_rejected(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    pm = ProgramManager(repo)
    state = pm.initialize_program("- Task 1\n", program_id="no_appr01")
    state.status = "ready"
    pm.save_program_state(state)

    with pytest.raises(ValueError, match="FAIL-CLOSED: Tidak ada pending approval"):
        runner.run_ceo(
            program_id="no_appr01",
            resume=True,
            approve=True,
            autonomous=False,
        )


# ==============================================================================
# 9. scheduler recognizes awaiting_input before declaring blocked
# ==============================================================================
def test_09_scheduler_recognizes_awaiting_input_before_declaring_blocked(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    pm = ProgramManager(repo)
    runner = CEORunner(repo, verbose=False)

    state = pm.initialize_program("- Task 1\n", program_id="sched_awt01")
    state.tasks[0].status = "awaiting_input"
    state.tasks[0].approval_stage = "proposal"
    pm.save_program_state(state)

    # Calling autonomous loop when a task is awaiting_input
    state_out = runner._run_autonomous_loop(state, approval_policy="safe-auto")
    assert state_out.status == "awaiting_ceo_input"
    assert "blocked" not in state_out.status
    assert "No ready tasks available and not all tasks completed." not in state_out.blockers


# ==============================================================================
# 10. legacy failed state from human_required is reconstructed
# ==============================================================================
def test_10_legacy_failed_state_from_human_required_is_reconstructed(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    state = ProgramState(
        program_id="0dd89460_mock",
        project_id="repo",
        objective="Observability",
        status="blocked",
        tasks=[
            ProgramTask(
                task_id="task_01",
                epic_id="epic_01",
                title="Task 1",
                status="failed",
                attempt_count=3,
                linked_run_id="8cffa92e",
            ),
            ProgramTask(
                task_id="task_02",
                epic_id="epic_01",
                title="Task 2",
                status="blocked",
                dependencies=["task_01"],
            ),
        ],
        blockers=[
            "Task task_01 failed after 3 attempts: Proposal review tidak disetujui (status: human_required).",
            "No ready tasks available and not all tasks completed.",
        ],
        retry_state={"failure_class": "TRANSIENT_GATEWAY", "attempt": 3},
    )

    reconstructed = reconstruct_misclassified_approval_states(repo, state)

    assert reconstructed is True
    assert state.tasks[0].status == "awaiting_input"
    assert state.tasks[0].approval_stage == "proposal"
    assert state.tasks[0].attempt_count == 3  # not silently reset
    assert state.status == "awaiting_ceo_input"
    assert state.current_phase == "proposal_approval"
    assert state.approval_checkpoint is not None
    assert state.approval_checkpoint["task_id"] == "task_01"


# ==============================================================================
# 11. derived approval blockers cleaned after recovery
# ==============================================================================
def test_11_derived_approval_blockers_cleaned_after_recovery(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    state = ProgramState(
        program_id="p11",
        project_id="repo",
        objective="Obj",
        tasks=[ProgramTask(task_id="task_01", epic_id="e1", title="T1", status="failed")],
        blockers=[
            "Task task_01 failed after 3 attempts: Proposal review tidak disetujui (status: human_required).",
            "No ready tasks available and not all tasks completed.",
            "Unrelated genuine environmental blocker.",
        ],
    )

    reconstruct_misclassified_approval_states(repo, state)

    assert len(state.blockers) == 1
    assert state.blockers[0] == "Unrelated genuine environmental blocker."


# ==============================================================================
# 12. observability task not high risk despite repo containing wallet
# ==============================================================================
def test_12_observability_task_not_high_risk_despite_repo_containing_wallet():
    # Negative constraint mentioning wallet, but task is observability
    pre_info = classify_pre_execution_risk(
        request="Tingkatkan observability. Jangan mengubah business logic wallet, transaksi, auth.",
        task_title="Tambahkan endpoint GET /ping",
        task_desc="Endpoint /ping untuk liveness probe. Constraints: jangan mengubah wallet.",
        expected_writable_files=["backend/internal/handler/http/health_handler.go", "backend/internal/handler/http/health_handler_test.go"],
    )

    assert pre_info["level"] == "low"
    assert pre_info["level"] != "high"


# ==============================================================================
# 13. actual diff touching wallet escalates post risk to high
# ==============================================================================
def test_13_actual_diff_touching_wallet_escalates_post_risk_to_high():
    pre_info = {"level": "low"}
    post_info = classify_post_execution_risk(
        actual_changed_files=["backend/internal/service/wallet_service.go"],
        test_status="passed",
    )
    assert post_info["level"] == "high"

    final_info = classify_final_risk(pre_info, post_info)
    assert final_info["level"] == "high"
    assert final_info["escalated"] is True


# ==============================================================================
# 14. output progress is flushed
# ==============================================================================
def test_14_output_progress_is_flushed(tmp_path: Path, capsys):
    repo = setup_test_git_repo(tmp_path / "repo")
    pm_state = ProgramState(
        program_id="dash01",
        project_id="repo",
        objective="Observability",
        status="awaiting_ceo_input",
        active_task="task_01",
        approval_checkpoint={"approval_stage": "proposal"},
    )
    runner = CEORunner(repo, verbose=False)
    runner._print_ceo_summary(pm_state)

    out = capsys.readouterr().out
    assert "Approval      : proposal" in out
    assert "Status        : AWAITING_CEO_INPUT" in out


# ==============================================================================
# 15. heartbeat appears during long operation
# ==============================================================================
def test_15_heartbeat_appears_during_long_operation(capsys):
    heartbeat = ProgressHeartbeat(message="[CEO] Testing heartbeat", interval=0.04)
    heartbeat.start()
    time.sleep(0.12)
    heartbeat.stop()

    captured = capsys.readouterr().out
    assert "[CEO] Testing heartbeat" in captured or "[LLM] Menunggu respons" in captured
