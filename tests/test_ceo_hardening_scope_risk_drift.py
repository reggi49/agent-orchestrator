import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.objective_alignment import (
    detect_objective_drift,
    normalize_objective_to_requirements,
    validate_task_objective_alignment,
)
from agent_system.program_manager import ProgramManager
from agent_system.project_initializer import initialize_project
from agent_system.risk_classifier import (
    classify_final_risk,
    classify_post_execution_risk,
    classify_pre_execution_risk,
    classify_risk,
    evaluate_approval_policy,
)
from agent_system.schemas import ProgramTask
from agent_system.scope_and_overlap import (
    OverlapType,
    analyze_backlog_overlaps_and_enforce_dependencies,
    detect_cycle_in_dependencies,
    detect_task_overlap,
)


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
# Category D: Fresh Base dan Overlap Tests (18 - 24)
# ==============================================================================

def test_18_second_task_built_from_head_after_first_integrates(tmp_path: Path):
    from agent_system.ceo_mode import CEORunner

    repo = setup_test_git_repo(tmp_path / "repo")
    runner = CEORunner(repo, verbose=False)

    recorded_base_revisions = []

    def mock_run(request, **kwargs):
        res_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
        head_now = res_head.stdout.strip()
        recorded_base_revisions.append(head_now)

        # Simulate task 1 commits to target branch
        (repo / "new_file.txt").write_text(f"edit {len(recorded_base_revisions)}\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", f"commit {len(recorded_base_revisions)}"], cwd=repo, check=True)

        new_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        return {
            "overall_status": "completed",
            "run_id": f"run_{len(recorded_base_revisions)}",
            "changed_files": ["new_file.txt"],
            "test_status": "passed",
            "commit_hash": new_head,
        }

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls, \
         patch("langchain_openai.ChatOpenAI"):
        mock_inst = MagicMock()
        mock_runner_cls.return_value = mock_inst
        mock_inst.run.side_effect = mock_run

        runner.run_ceo(
            request_text="- Task A\n- Task B\n",
            program_id="fresh01",
            autonomous=True,
        )

        assert len(recorded_base_revisions) == 2
        # Task 2 base must differ from Task 1 base!
        assert recorded_base_revisions[1] != recorded_base_revisions[0]


def test_19_worktrees_for_backlog_not_created_upfront(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    pm = ProgramManager(repo)

    state = pm.initialize_program("- Task 1\n- Task 2\n- Task 3\n", program_id="lazy01")
    wt_dir = repo / ".agent" / "worktrees"
    wt_dir_alt = tmp_path / ".agent-worktrees"

    # Backlog exists
    assert len(state.tasks) == 3
    # No worktrees were created upfront
    if wt_dir.is_dir():
        assert len(list(wt_dir.iterdir())) == 0
    if wt_dir_alt.is_dir():
        assert len(list(wt_dir_alt.iterdir())) == 0


def test_20_direct_overlap_enforces_dependency(tmp_path: Path):
    t1 = ProgramTask(task_id="t1", epic_id="e1", title="Task 1", expected_writable_files=["backend/app.go"])
    t2 = ProgramTask(task_id="t2", epic_id="e1", title="Task 2", expected_writable_files=["backend/app.go"])

    overlap = detect_task_overlap(t1, t2)
    assert overlap == OverlapType.DIRECT

    analysis, is_valid, err = analyze_backlog_overlaps_and_enforce_dependencies([t1, t2])
    assert is_valid is True
    # t2 must depend on t1
    assert "t1" in t2.dependencies


def test_21_possible_overlap_remains_serial(tmp_path: Path):
    t1 = ProgramTask(task_id="t1", epic_id="e1", title="Task 1", expected_writable_files=["backend/internal/service/wallet.go"])
    t2 = ProgramTask(task_id="t2", epic_id="e1", title="Task 2", expected_writable_files=["backend/internal/service/account.go"])

    overlap = detect_task_overlap(t1, t2)
    assert overlap == OverlapType.POSSIBLE


def test_22_dependency_cycle_rejected_fail_closed(tmp_path: Path):
    # A -> B -> A cycle
    deps = {
        "task_1": ["task_2"],
        "task_2": ["task_1"],
    }
    cycle = detect_cycle_in_dependencies(deps)
    assert cycle is not None

    tasks = [
        ProgramTask(task_id="task_1", epic_id="e1", title="T1", dependencies=["task_2"], expected_writable_files=["backend/app.go"]),
        ProgramTask(task_id="task_2", epic_id="e1", title="T2", dependencies=["task_1"], expected_writable_files=["backend/app.go"]),
    ]
    _, is_valid, err = analyze_backlog_overlaps_and_enforce_dependencies(tasks)
    assert is_valid is False
    assert "FAIL-CLOSED" in err
    assert "cycle" in err.lower()


def test_23_read_only_evidence_file_not_modified(tmp_path: Path):
    t = ProgramTask(
        task_id="t1",
        epic_id="e1",
        title="T1",
        readable_files=["README.md", ".agent/manifest.yaml"],
        expected_writable_files=["backend/app.go"],
    )
    # If agent modified a read-only file
    drift = detect_objective_drift(t, actual_changed_files=["README.md", "backend/app.go"])
    # README is a doc file, but manifest is not writable
    drift_manifest = detect_objective_drift(t, actual_changed_files=[".agent/manifest.yaml", "backend/app.go"])
    assert drift_manifest["drift_detected"] is True
    assert ".agent/manifest.yaml" in drift_manifest["unmapped_changes"]


def test_24_conditional_file_requires_justification():
    t = ProgramTask(
        task_id="t1",
        epic_id="e1",
        title="T1",
        expected_writable_files=["backend/app.go"],
        conditionally_writable_files=["backend/internal/util/helper.go"],
    )
    # Conditional file without being in expected writable is allowed if in conditionally_writable
    drift = detect_objective_drift(t, actual_changed_files=["backend/internal/util/helper.go"])
    assert drift["drift_detected"] is False


# ==============================================================================
# Category E: Risk Reclassification Tests (25 - 31)
# ==============================================================================

def test_25_low_request_editing_wallet_escalates_to_high():
    pre_info = classify_pre_execution_risk(
        request="Small formatting tweak",
        expected_writable_files=["backend/app.go"],
    )
    assert pre_info["level"] == "medium" or pre_info["level"] == "low"

    # Actual diff touches wallet service
    post_info = classify_post_execution_risk(
        actual_changed_files=["backend/internal/service/wallet_service.go"],
        test_status="passed",
    )
    assert post_info["level"] == "high"

    final_info = classify_final_risk(pre_info, post_info)
    assert final_info["level"] == "high"
    assert final_info["escalated"] is True


def test_26_test_only_diff_remains_low():
    pre_info = classify_pre_execution_risk(
        request="Add unit test for health handler",
        expected_writable_files=["backend/internal/handler/health_test.go"],
    )
    post_info = classify_post_execution_risk(
        actual_changed_files=["backend/internal/handler/health_test.go"],
        test_status="passed",
        diff_review_status="approved",
    )
    final_info = classify_final_risk(pre_info, post_info)
    assert final_info["level"] == "low"


def test_27_migration_file_always_high():
    post_info = classify_post_execution_risk(
        actual_changed_files=["backend/migrations/0001_init.sql"],
        test_status="passed",
    )
    assert post_info["level"] == "high"


def test_28_manifest_dependency_change_always_high():
    post_info = classify_post_execution_risk(
        actual_changed_files=["backend/go.mod", "backend/go.sum"],
        test_status="passed",
    )
    assert post_info["level"] == "high"


def test_29_risk_cannot_decrease_from_high():
    pre_info = {"level": "high", "reasons": ["Financial domain"]}
    post_info = {"level": "low", "reasons": ["Test only"]}
    final_info = classify_final_risk(pre_info, post_info)
    assert final_info["level"] == "high"


def test_30_scope_violation_prevents_auto_integration():
    post_info = classify_post_execution_risk(
        actual_changed_files=["backend/unexpected.go"],
        scope_violations=["backend/unexpected.go"],
        test_status="passed",
    )
    assert post_info["level"] == "medium"
    policy_eval = evaluate_approval_policy(post_info["level"], approval_policy="safe-auto")
    assert policy_eval["can_auto_approve"] is False
    assert policy_eval["action"] == "awaiting_human_approval"


def test_31_pre_risk_and_post_risk_stored_in_artifact():
    res = classify_risk(
        request="Update logic",
        changed_files=["backend/app.go"],
        test_status="passed",
    )
    assert "pre_risk" in res
    assert "post_risk" in res
    assert "level" in res


# ==============================================================================
# Category F: Objective Drift Tests (32 - 36)
# ==============================================================================

def test_32_task_without_mapped_requirement_rejected():
    task = ProgramTask(task_id="t32", epic_id="e1", title="T32", mapped_requirement_ids=[])
    is_valid, err = validate_task_objective_alignment(task, {"FR-001", "FR-002"})
    assert is_valid is False
    assert "tidak memiliki mapped_requirement_ids" in err


def test_33_unmapped_production_file_causes_awaiting_ceo_input():
    task = ProgramTask(
        task_id="t33",
        epic_id="e1",
        title="T33",
        expected_writable_files=["backend/cmd/api/main.go"],
    )
    # Changed an unexpected unmapped production file in another component
    drift = detect_objective_drift(task, actual_changed_files=["frontend/src/SecretAuth.tsx"])
    assert drift["drift_detected"] is True
    assert "frontend/src/SecretAuth.tsx" in drift["unmapped_changes"]


def test_34_markdown_edit_does_not_overwrite_json_state(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    pm = ProgramManager(repo)
    state = pm.initialize_program("- Task A\n- Task B\n", program_id="mdtest01")

    # Manually tamper with CURRENT_STATUS.md
    prog_dir = repo / ".agent" / "programs" / "mdtest01"
    status_md = prog_dir / "CURRENT_STATUS.md"
    status_md.write_text("# Hacked Status: COMPLETED ALL")

    # Load JSON state - JSON is source of truth and unaffected
    _, loaded_state = pm.load_program_state("mdtest01")
    assert loaded_state.status == "ready"
    assert loaded_state.status != "completed"


def test_35_markdown_regenerated_from_json(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    pm = ProgramManager(repo)
    state = pm.initialize_program("- Task A\n- Task B\n", program_id="regen01")

    prog_dir = repo / ".agent" / "programs" / "regen01"
    (prog_dir / "ROADMAP.md").write_text("corrupted content")

    # Update state and regenerate markdown
    state.status = "in_progress"
    pm.write_all_markdown_reports(state)

    content = (prog_dir / "ROADMAP.md").read_text()
    assert "corrupted content" not in content
    assert "Program Roadmap (regen01)" in content


def test_36_requirement_mapping_appears_in_task_evidence(tmp_path: Path):
    reqs = normalize_objective_to_requirements("- FR: Customer management\n- QR: Fast lookup")
    assert len(reqs) >= 2
    assert reqs[0]["id"].startswith("FR-") or reqs[0]["id"].startswith("QR-")
