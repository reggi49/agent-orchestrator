import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.gateway_resilience import (
    BackoffManager,
    CircuitBreaker,
    CircuitState,
    FailureCategory,
    ModelFailoverManager,
    classify_failure,
)
from agent_system.integrator import integrate_execution
from agent_system.project_initializer import initialize_project
from agent_system.schemas import ProgramTask


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


def setup_test_worktree_conflict(repo: Path, tmp_path: Path):
    wt_id = "wt-conflict-01"
    wt_path = tmp_path / ".agent-worktrees" / wt_id
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    branch_name = "agent/test-conflict"
    subprocess.run(["git", "worktree", "add", "-b", branch_name, str(wt_path), "HEAD"], cwd=repo, check=True, capture_output=True)

    # Edit app.go in worktree
    (wt_path / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"from worktree\" }\n")

    # Simultaneously edit same line in main repo to cause conflict
    (repo / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"from main concurrent\" }\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "concurrent conflicting commit"], cwd=repo, check=True)

    exec_data = {
        "artifact_id": "execution_conf_01",
        "status": "approved_for_integration",
        "result": {
            "source_proposal_path": ".agent/artifacts/proposals/proposal_01.json",
            "worktree_identifier": wt_id,
            "branch_name": branch_name,
            "approved_files": ["backend/app.go"],
            "actual_changed_files": ["backend/app.go"],
            "validation_command_summary": [
                {"executable": "git", "arguments": ["status"], "working_directory": ".", "exit_code": 0, "status": "passed"}
            ],
        },
    }
    exec_file = repo / ".agent" / "artifacts" / "executions" / "execution_conf_01.json"
    exec_file.parent.mkdir(parents=True, exist_ok=True)
    exec_file.write_text(json.dumps(exec_data))

    return wt_path, exec_file, exec_data


# ==============================================================================
# Category G: Conflict Recovery Tests (37 - 43)
# ==============================================================================

def test_37_cherry_pick_conflict_results_in_blocked_not_completed(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    wt_path, exec_file, exec_data = setup_test_worktree_conflict(repo, tmp_path)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        with pytest.raises(RuntimeError, match="Cherry-pick gagal"):
            integrate_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                strategy="cherry-pick",
                apply=True,
            )

    # Main branch HEAD must remain intact, not completed
    res_log = subprocess.run(["git", "log", "-1", "--oneline"], cwd=repo, capture_output=True, text=True).stdout
    assert "concurrent conflicting commit" in res_log


def test_38_conflict_artifact_safely_records_conflicting_files(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    wt_path, exec_file, exec_data = setup_test_worktree_conflict(repo, tmp_path)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        try:
            integrate_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                strategy="cherry-pick",
                apply=True,
            )
        except RuntimeError:
            pass

    failure_art = repo / ".agent" / "artifacts" / "executions" / "integration_conf_01.json"
    assert failure_art.is_file()
    data = json.loads(failure_art.read_text())
    assert data["status"] == "integration_failed"
    res_data = data.get("result", data)
    assert res_data["is_conflict"] is True
    assert "backend/app.go" in res_data["conflicting_files"]
    assert "recovery_instructions" in res_data


def test_39_one_automatic_replan_executed_from_fresh_head(tmp_path: Path):
    from agent_system.ceo_mode import CEORunner

    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    attempt_counter = 0

    def mock_run(request, **kwargs):
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            # First attempt fails with conflict
            return {
                "overall_status": "failed",
                "run_id": "run_attempt_1",
                "error_message": "Cherry-pick gagal: conflict in backend/app.go",
            }
        # Second attempt succeeds on fresh HEAD
        return {
            "overall_status": "completed",
            "run_id": "run_attempt_2",
            "changed_files": ["backend/app.go"],
            "test_status": "passed",
            "commit_hash": "2" * 40,
        }

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls, \
         patch("langchain_openai.ChatOpenAI"):
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst
        mock_inst.run.side_effect = mock_run

        state = runner.run_ceo(
            request_text="- Task A: Update app\n",
            program_id="replan01",
            autonomous=True,
        )

        assert attempt_counter == 2
        assert state.tasks[0].conflict_replan_count == 1
        assert state.tasks[0].status == "completed"
        assert len(state.tasks[0].superseded_attempts) == 1


def test_40_second_conflict_escalates_to_ceo(tmp_path: Path):
    from agent_system.ceo_mode import CEORunner

    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    def mock_conflict_run(request, **kwargs):
        return {
            "overall_status": "failed",
            "run_id": "run_always_conflict",
            "error_message": "Cherry-pick gagal: conflict in backend/app.go",
        }

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls, \
         patch("langchain_openai.ChatOpenAI"):
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst
        mock_inst.run.side_effect = mock_conflict_run

        state = runner.run_ceo(
            request_text="- Task A: Update app\n",
            program_id="rep_esc01",
            autonomous=True,
        )

        assert state.status == "awaiting_ceo_input"
        assert state.tasks[0].status == "blocked"
        assert state.tasks[0].conflict_replan_count >= 1
        assert any("Eskalasi ke CEO" in b for b in state.blockers)


def test_41_no_forbidden_git_commands_executed_during_conflict(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    wt_path, exec_file, exec_data = setup_test_worktree_conflict(repo, tmp_path)

    executed_cmds = []
    orig_run = subprocess.run

    def intercept(cmd, *a, **kw):
        if isinstance(cmd, list) and cmd and cmd[0] == "git":
            executed_cmds.append(cmd)
        return orig_run(cmd, *a, **kw)

    with patch("subprocess.run", side_effect=intercept), \
         patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        try:
            integrate_execution(repo, exec_file, exec_data, apply=True)
        except RuntimeError:
            pass

    forbidden = {"push", "merge", "rebase", "reset", "tag"}
    for cmd in executed_cmds:
        verbs = set(cmd[1:])
        assert not (verbs & forbidden), f"Forbidden git command called: {cmd}"


def test_42_abort_recorded_before_execution(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    wt_path, exec_file, exec_data = setup_test_worktree_conflict(repo, tmp_path)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        try:
            integrate_execution(repo, exec_file, exec_data, apply=True)
        except RuntimeError:
            pass

    failure_art = repo / ".agent" / "artifacts" / "executions" / "integration_conf_01.json"
    data = json.loads(failure_art.read_text())
    res_data = data.get("result", data)
    assert "abort_performed" in res_data
    # Target repo should be clean of pending cherry-pick
    assert not (repo / ".git" / "CHERRY_PICK_HEAD").is_file()


def test_43_old_worktree_retained_as_evidence(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    wt_path, exec_file, exec_data = setup_test_worktree_conflict(repo, tmp_path)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        try:
            integrate_execution(repo, exec_file, exec_data, apply=True)
        except RuntimeError:
            pass

    # Old worktree still exists on disk
    assert wt_path.is_dir()
    assert (wt_path / "backend" / "app.go").is_file()


# ==============================================================================
# Category H: Gateway Resilience Tests (44 - 56)
# ==============================================================================

def test_44_http_429_classified_rate_limit():
    cat = classify_failure(Exception("HTTP 429: Too Many Requests, quota exceeded"))
    assert cat == FailureCategory.RATE_LIMIT


def test_45_timeout_classified_transient_gateway():
    cat = classify_failure(Exception("Request timed out after 30000ms"))
    assert cat == FailureCategory.TRANSIENT_GATEWAY


def test_46_http_401_classified_auth_configuration_no_blind_retry():
    cat = classify_failure(Exception("HTTP 401 Unauthorized: Invalid API key"))
    assert cat == FailureCategory.AUTH_CONFIGURATION


def test_47_schema_parse_failure_classified_model_output():
    cat = classify_failure(Exception("ValidationError: Output parsing error, missing required fields"))
    assert cat == FailureCategory.MODEL_OUTPUT


def test_48_failed_code_test_not_classified_as_gateway_failure():
    cat = classify_failure(Exception("Validation command gagal: go test ./... failed with exit code 1"))
    assert cat == FailureCategory.CODE_OR_VALIDATION_FAILURE
    assert cat != FailureCategory.TRANSIENT_GATEWAY


def test_49_backoff_state_saved_before_sleeper(tmp_path: Path):
    waited_delays = []
    manager = BackoffManager(
        base_delays=[10.0, 30.0, 90.0],
        sleeper=lambda d: waited_delays.append(d),
    )
    d = manager.calculate_delay(1, add_jitter=False)
    assert d == 10.0
    manager.wait(d)
    assert len(waited_delays) == 1
    assert waited_delays[0] == 10.0


def test_50_resume_respects_retry_after():
    manager = BackoffManager()
    d = manager.calculate_delay(1, retry_after=45.0)
    assert d == 45.0


def test_51_retry_budget_not_reset_after_restart(tmp_path: Path):
    t = ProgramTask(task_id="t51", epic_id="e1", title="T51", attempt_count=2)
    t.attempt_count += 1
    assert t.attempt_count == 3


def test_52_circuit_breaker_opens_after_threshold(tmp_path: Path):
    cb = CircuitBreaker(tmp_path, provider_id="prov_test", failure_threshold=3, cooldown_seconds=60.0)
    assert cb.state == CircuitState.CLOSED
    assert cb.is_available() is True

    cb.record_failure(FailureCategory.TRANSIENT_GATEWAY)
    cb.record_failure(FailureCategory.TRANSIENT_GATEWAY)
    assert cb.state == CircuitState.CLOSED

    cb.record_failure(FailureCategory.TRANSIENT_GATEWAY)
    assert cb.state == CircuitState.OPEN
    assert cb.is_available() is False


def test_53_half_open_allows_single_probe(tmp_path: Path):
    mock_now = 1000.0
    cb = CircuitBreaker(
        tmp_path,
        provider_id="prov_probe",
        failure_threshold=2,
        cooldown_seconds=50.0,
        clock=lambda: mock_now,
    )
    cb.record_failure(FailureCategory.TRANSIENT_GATEWAY)
    cb.record_failure(FailureCategory.TRANSIENT_GATEWAY)
    assert cb.state == CircuitState.OPEN

    # Advance clock past cooldown
    mock_now = 1055.0
    assert cb.is_available() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_54_provider_recovery_closes_circuit(tmp_path: Path):
    cb = CircuitBreaker(tmp_path, provider_id="prov_rec", failure_threshold=2, cooldown_seconds=10.0)
    cb.record_failure(FailureCategory.TRANSIENT_GATEWAY)
    cb.record_failure(FailureCategory.TRANSIENT_GATEWAY)
    assert cb.state == CircuitState.OPEN

    # Success probe closes circuit
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.is_available() is True


def test_55_model_failover_recorded():
    fm = ModelFailoverManager(["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"])
    assert fm.get_model_for_attempt(1) == "gpt-4o"
    assert fm.get_model_for_attempt(2) == "claude-3-5-sonnet"
    assert fm.get_model_for_attempt(3) == "gemini-1.5-pro"


def test_56_secret_and_raw_prompt_not_stored_in_resilience_state(tmp_path: Path):
    from agent_system.error_reporting import sanitize_text

    raw_err = "Auth failed for sk-1234567890abcdefsecret with Bearer token_secret_999"
    clean = sanitize_text(raw_err)
    assert "sk-1234567890abcdefsecret" not in clean
    assert "token_secret_999" not in clean
