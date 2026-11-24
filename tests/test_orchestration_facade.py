import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.orchestration_runner import (
    OrchestrationRunner,
    locate_run_checkpoint,
)
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
from agent_system.risk_classifier import classify_risk, evaluate_approval_policy


def setup_test_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    (path / "backend").mkdir(parents=True, exist_ok=True)
    (path / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"ok\" }\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    initialize_project(path, apply=True)
    return path


def mock_successful_plan_state(initial_state: dict) -> dict:
    return {
        **initial_state,
        "status": "plan_ready",
        "analysis": {
            "recommended_task": {
                "title": initial_state.get("user_request", "Test Task"),
                "owner": "backend",
                "description": "Task description",
                "acceptance_criteria": ["Test passes"],
            }
        },
        "task_verification": {"status": "verified"},
        "repository_evidence": {"results": [{"path": "backend/app.go"}]},
    }


def mock_successful_propose_state(initial_state: dict) -> dict:
    task_title = initial_state.get("plan_data", {}).get("final_task", "Test Task")
    return {
        **initial_state,
        "status": "approved_for_implementation",
        "implementation_proposal": {
            "proposal_id": f"prop_{initial_state['run_id']}",
            "task_title": task_title,
            "owner": "backend",
            "files_to_modify": [{"path": "backend/app.go", "action": "modify", "purpose": "update"}],
            "tests_to_create": [{"path": "backend/app_test.go", "test_type": "unit"}],
            "approved_writable_files": ["backend/app.go", "backend/app_test.go"],
            "validation_commands": [
                {"executable": "git", "arguments": ["status"], "working_directory": ".", "purpose": "test", "requires_approval": True}
            ],
            "evidence_paths": ["backend/app.go"],
        },
        "proposal_review": {
            "status": "approved",
            "approved_files": ["backend/app.go", "backend/app_test.go"],
            "approved_writable_files": ["backend/app.go", "backend/app_test.go"],
            "summary": "Proposal approved",
        },
    }


def mock_successful_exec_state(initial_state: dict, wt_path: Path) -> dict:
    # Ensure worktree directory exists and contains changes
    wt_path.mkdir(parents=True, exist_ok=True)
    test_file = wt_path / "backend" / "app_test.go"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("package main\n\nimport \"testing\"\nfunc TestApp(t *testing.T) {}\n")

    return {
        **initial_state,
        "status": "execution_review_ready",
        "worktree_path": str(wt_path),
        "preflight_result": {"status": "preflight_passed", "source_revision": {"commit_head": "HEAD", "is_dirty": False}},
        "coding_execution_result": {
            "status": "completed",
            "iterations": 2,
            "changed_files": ["backend/app_test.go"],
            "actual_changed_files": ["backend/app_test.go"],
            "approved_files": ["backend/app.go", "backend/app_test.go"],
            "validation_results": [
                {"executable": "git", "arguments": ["status"], "working_directory": ".", "exit_code": 0, "status": "passed"}
            ],
            "scope_violations": [],
            "requested_additional_files": [],
        },
        "diff_review": {
            "status": "approved",
            "changed_files": ["backend/app_test.go"],
            "unexpected_files": [],
        },
    }


# ==============================================================================
# 1. Happy path LOW risk single command
# ==============================================================================
def test_01_happy_path_low_risk_single_command(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            # create real git worktree
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return mock_successful_exec_state(state, wt_dir)

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )

        assert res["overall_status"] == "completed"
        assert res["risk_level"] == "low"
        assert res["commit_hash"] is not None
        assert len(res["commit_hash"]) == 40
        assert (repo / "backend" / "app_test.go").is_file()


# ==============================================================================
# 2. MEDIUM risk stops at awaiting_human_approval
# ==============================================================================
def test_02_medium_risk_stops_at_awaiting_human_approval(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            # production file modified -> medium risk
            wt_dir.mkdir(parents=True, exist_ok=True)
            (wt_dir / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"updated\" }\n")
            return {
                **state,
                "status": "execution_review_ready",
                "worktree_path": str(wt_dir),
                "preflight_result": {"status": "preflight_passed", "source_revision": {"commit_head": "HEAD", "is_dirty": False}},
                "coding_execution_result": {
                    "status": "completed",
                    "iterations": 1,
                    "changed_files": ["backend/app.go"],
                    "approved_files": ["backend/app.go"],
                    "validation_results": [
                        {"executable": "git", "arguments": ["status"], "working_directory": ".", "exit_code": 0, "status": "passed"}
                    ],
                },
                "diff_review": {"status": "approved", "changed_files": ["backend/app.go"], "unexpected_files": []},
            }

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Perbarui endpoint service dan tambahkan test readiness",
            apply=True,
            approval_policy="safe-auto",
        )

        assert res["overall_status"] == "awaiting_human_approval"
        assert res["risk_level"] == "medium"
        assert res["approval_requirement"] == "human_approval_required"
        assert "resume" in res["next_action"]
        assert res["commit_hash"] is None


# ==============================================================================
# 3. HIGH risk cannot auto-approve
# ==============================================================================
def test_03_high_risk_cannot_auto_approve(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            (wt_dir / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"wallet\" }\n")
            return {
                **state,
                "status": "execution_review_ready",
                "worktree_path": str(wt_dir),
                "preflight_result": {"status": "preflight_passed", "source_revision": {"commit_head": "HEAD", "is_dirty": False}},
                "coding_execution_result": {
                    "status": "completed",
                    "iterations": 1,
                    "changed_files": ["backend/app.go"],
                    "approved_files": ["backend/app.go"],
                    "validation_results": [
                        {"executable": "git", "arguments": ["status"], "working_directory": ".", "exit_code": 0, "status": "passed"}
                    ],
                },
                "diff_review": {"status": "approved", "changed_files": ["backend/app.go"], "unexpected_files": []},
            }

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Implementasikan update saldo wallet dan transaksi pembayaran",
            apply=True,
            approval_policy="safe-auto",
        )

        assert res["overall_status"] == "awaiting_human_approval"
        assert res["risk_level"] == "high"
        assert res["approval_requirement"] == "human_approval_required"
        assert res["commit_hash"] is None


# ==============================================================================
# 4. Resume from approval gate
# ==============================================================================
def test_04_resume_from_approval_gate(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            (wt_dir / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"medium\" }\n")
            return {
                **state,
                "status": "execution_review_ready",
                "worktree_path": str(wt_dir),
                "preflight_result": {"status": "preflight_passed", "source_revision": {"commit_head": "HEAD", "is_dirty": False}},
                "coding_execution_result": {
                    "status": "completed",
                    "iterations": 1,
                    "changed_files": ["backend/app.go"],
                    "approved_files": ["backend/app.go"],
                    "validation_results": [
                        {"executable": "git", "arguments": ["status"], "working_directory": ".", "exit_code": 0, "status": "passed"}
                    ],
                },
                "diff_review": {"status": "approved", "changed_files": ["backend/app.go"], "unexpected_files": []},
            }

        mock_beg.return_value.invoke = exec_side_effect

        # Step 1: Pauses at awaiting_human_approval
        res_initial = runner.run(
            request="Perbarui endpoint service dan tambahkan test readiness",
            apply=True,
            approval_policy="safe-auto",
        )
        run_id = res_initial["run_id"]
        assert res_initial["overall_status"] == "awaiting_human_approval"

        # Step 2: Resume with --approve --apply
        res_resumed = runner.resume(
            run_arg=run_id,
            approve=True,
            apply=True,
            note="Approved by senior engineer",
        )

        assert res_resumed["overall_status"] == "completed"
        assert res_resumed["commit_hash"] is not None


# ==============================================================================
# 5. Resume does not repeat OpenHands execution
# ==============================================================================
def test_05_resume_does_not_repeat_openhands_execution(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    exec_call_count = 0

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            nonlocal exec_call_count
            exec_call_count += 1
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            (wt_dir / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"medium\" }\n")
            return {
                **state,
                "status": "execution_review_ready",
                "worktree_path": str(wt_dir),
                "preflight_result": {"status": "preflight_passed", "source_revision": {"commit_head": "HEAD", "is_dirty": False}},
                "coding_execution_result": {
                    "status": "completed",
                    "iterations": 1,
                    "changed_files": ["backend/app.go"],
                    "approved_files": ["backend/app.go"],
                    "validation_results": [
                        {"executable": "git", "arguments": ["status"], "working_directory": ".", "exit_code": 0, "status": "passed"}
                    ],
                },
                "diff_review": {"status": "approved", "changed_files": ["backend/app.go"], "unexpected_files": []},
            }

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Perbarui endpoint service dan tambahkan test readiness",
            apply=True,
            approval_policy="safe-auto",
        )
        assert exec_call_count == 1
        assert res["overall_status"] == "awaiting_human_approval"

        # Resume from gate
        runner.resume(run_arg=res["run_id"], approve=True, apply=True)
        # Execution graph must not have been invoked again!
        assert exec_call_count == 1


# ==============================================================================
# 6. Crash after proposal can be resumed
# ==============================================================================
def test_06_crash_after_proposal_can_be_resumed(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        # First run crashes during execution
        mock_beg.return_value.invoke = MagicMock(side_effect=RuntimeError("Simulated execution crash"))

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )
        assert res["overall_status"] == "interrupted"
        assert res["proposal_artifact_id"] is not None

        # Fix crash for resume
        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return mock_successful_exec_state(state, wt_dir)

        mock_beg.return_value.invoke = exec_side_effect

        res_resumed = runner.resume(run_arg=res["run_id"], apply=True)
        assert res_resumed["overall_status"] == "completed"


# ==============================================================================
# 7. Crash after execution does not create second worktree
# ==============================================================================
def test_07_crash_after_execution_does_not_create_second_worktree(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    worktrees_created = 0

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg, \
         patch("agent_system.orchestration_runner.integrate_execution") as mock_integ:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            nonlocal worktrees_created
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            worktrees_created += 1
            return mock_successful_exec_state(state, wt_dir)

        mock_beg.return_value.invoke = exec_side_effect
        # Crash during integration
        mock_integ.side_effect = RuntimeError("Temporary integration lock crash")

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )
        assert res["overall_status"] == "interrupted"
        assert worktrees_created == 1

        # Resume after integration fix
        mock_integ.side_effect = None
        mock_integ.return_value = {
            "mode": "apply",
            "status": "completed",
            "commit_hash": "a" * 40,
            "integration_artifact_file": repo / ".agent/artifacts/executions/integration_mock.json",
        }
        res_resumed = runner.resume(run_arg=res["run_id"], apply=True)
        assert worktrees_created == 1
        assert res_resumed["overall_status"] == "completed"


# ==============================================================================
# 8. Artifact relationship uses same run_id
# ==============================================================================
def test_08_artifact_relationship_uses_same_run_id(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return mock_successful_exec_state(state, wt_dir)

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )

        short_id = res["run_id"]
        assert res["plan_artifact_id"] == f"plan_{short_id}"
        assert res["proposal_artifact_id"] == f"proposal_{short_id}"
        assert res["execution_artifact_id"] == f"execution_{short_id}"
        assert res["review_artifact_id"] == f"human_execution_review_{short_id}"
        assert res["integration_artifact_id"] == f"integration_{short_id}"


# ==============================================================================
# 9. Dry-run does not call OpenHands and does not modify source
# ==============================================================================
def test_09_dry_run_does_not_modify_source(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.executors.openhands_executor.OpenHandsExecutor.execute") as mock_oh:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=False,  # dry-run
            approval_policy="safe-auto",
        )

        assert res["overall_status"] == "dry_run_ready"
        mock_oh.assert_not_called()
        res_git = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
        # Only runtime artifacts in .agent, source is clean
        lines = [l for l in res_git.stdout.splitlines() if not l.strip().endswith(".agent") and ".agent/" not in l]
        assert len(lines) == 0


# ==============================================================================
# 10. Apply reaches completed
# ==============================================================================
def test_10_apply_reaches_completed(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return mock_successful_exec_state(state, wt_dir)

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )

        assert res["overall_status"] == "completed"
        assert res["commit_hash"] is not None


# ==============================================================================
# 11. Failed tests cannot be integrated
# ==============================================================================
def test_11_failed_tests_cannot_be_integrated(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return {
                **state,
                "status": "failed",
                "worktree_path": str(wt_dir),
                "coding_execution_result": {
                    "status": "failed",
                    "validation_results": [{"status": "failed", "exit_code": 1}],
                },
            }

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )

        assert res["overall_status"] == "failed"
        assert res["commit_hash"] is None


# ==============================================================================
# 12. Scope violation cannot be integrated
# ==============================================================================
def test_12_scope_violation_cannot_be_integrated(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return {
                **state,
                "status": "scope_violation",
                "worktree_path": str(wt_dir),
                "coding_execution_result": {
                    "status": "scope_violation",
                    "scope_violations": ["unexpected_file.go"],
                },
            }

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )

        assert res["overall_status"] == "scope_violation"
        assert res["commit_hash"] is None


# ==============================================================================
# 13. Sanitized output and artifacts
# ==============================================================================
def test_13_sanitized_output_and_artifacts(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return mock_successful_exec_state(state, wt_dir)

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Tambahkan automated test untuk health handler with secret sk-1234567890abcdef and Bearer token123",
            apply=True,
            approval_policy="safe-auto",
        )

        _, checkpoint = locate_run_checkpoint(repo, res["run_id"])
        checkpoint_str = json.dumps(checkpoint)

        assert "sk-1234567890abcdef" not in checkpoint_str
        assert "token123" not in checkpoint_str


# ==============================================================================
# 14. Status command reports all fields
# ==============================================================================
def test_14_status_command_reports_all_fields(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    wt_root = tmp_path / ".agent-worktrees"
    runner = OrchestrationRunner(repo, verbose=False)

    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=wt_root), \
         patch("agent_system.orchestration_runner.build_graph") as mock_bg, \
         patch("agent_system.orchestration_runner.build_proposal_graph") as mock_bpg, \
         patch("agent_system.orchestration_runner.build_execution_graph") as mock_beg:

        mock_bg.return_value.invoke = lambda state, config=None: mock_successful_plan_state(state)
        mock_bpg.return_value.invoke = lambda state, config=None: mock_successful_propose_state(state)

        def exec_side_effect(state, config=None):
            short_id = state["run_id"]
            wt_name = f"{repo.name}-openhands-{short_id}"
            wt_dir = wt_root / wt_name
            branch = state["branch_name"]
            subprocess.run(["git", "worktree", "add", "-b", branch, str(wt_dir), "HEAD"], cwd=repo, check=True, capture_output=True)
            return mock_successful_exec_state(state, wt_dir)

        mock_beg.return_value.invoke = exec_side_effect

        res = runner.run(
            request="Tambahkan automated test untuk health handler",
            apply=True,
            approval_policy="safe-auto",
        )

        status_info = runner.get_status(res["run_id"])

        assert status_info["run_id"] == res["run_id"]
        assert status_info["current_stage"] == "completed"
        assert status_info["overall_status"] == "completed"
        assert status_info["risk_level"] == "low"
        assert status_info["approval_requirement"] == "none"
        assert status_info["plan_artifact_id"] is not None
        assert status_info["proposal_artifact_id"] is not None
        assert status_info["execution_artifact_id"] is not None
        assert status_info["review_artifact_id"] is not None
        assert status_info["integration_artifact_id"] is not None
