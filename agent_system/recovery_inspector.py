import json
import subprocess
from pathlib import Path
from typing import Any

from agent_system.atomic_state import read_json_with_backup_recovery
from agent_system.schemas import ProgramState, ProgramTask
from agent_system.worktree_manager import get_default_worktree_root


class RecoveryAction:
    NOOP = "NOOP"
    SKIP_EXECUTION_PROCEED_TO_REVIEW = "SKIP_EXECUTION_PROCEED_TO_REVIEW"
    SKIP_INTEGRATION_MARK_COMPLETED = "SKIP_INTEGRATION_MARK_COMPLETED"
    FINALIZE_INTEGRATION_IDEMPOTENT = "FINALIZE_INTEGRATION_IDEMPOTENT"
    RECOVER_CHERRY_PICK_CONFLICT = "RECOVER_CHERRY_PICK_CONFLICT"
    RECONSTRUCT_WORKTREE = "RECONSTRUCT_WORKTREE"
    RESTORE_BACKUP_STATE = "RESTORE_BACKUP_STATE"
    ESCALATE_CORRUPTION = "ESCALATE_CORRUPTION"


def inspect_recovery_conditions(
    project_root: Path,
    state: ProgramState,
    target_branch: str = "main",
) -> dict[str, Any]:
    """Inspect repository, worktrees, locks, artifacts, and cherry-pick state for durable recovery.

    Guarantees:
    - Never duplicates completed work.
    - Never reruns completed OpenHands executions.
    - Never re-cherry-picks already integrated commits.
    - Recovers cherry-pick conflict or crashes safely and idempotently.
    """
    project_root = Path(project_root).expanduser().resolve()
    active_t: ProgramTask | None = None
    if state.active_task:
        active_t = next((t for t in state.tasks if t.task_id == state.active_task), None)

    # 1. Inspect Cherry-Pick state on target repo
    cherry_pick_head_file = project_root / ".git" / "CHERRY_PICK_HEAD"
    is_cherry_pick_active = cherry_pick_head_file.is_file()

    # 2. Inspect Target Branch HEAD
    res_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    current_target_head = res_head.stdout.strip() if res_head.returncode == 0 else "HEAD"

    # 3. Inspect Active Task and Run Artifacts
    execution_completed = False
    integration_completed = False
    commit_already_on_target = False
    worktree_exists = False
    worktree_path_actual = None

    if active_t:
        run_id = active_t.linked_run_id or state.active_run_id
        if run_id:
            clean_run = run_id[:8].lower()
            exec_file = project_root / ".agent" / "artifacts" / "executions" / f"execution_{clean_run}.json"
            integ_file = project_root / ".agent" / "artifacts" / "executions" / f"integration_{clean_run}.json"

            if exec_file.is_file():
                try:
                    ed, _ = read_json_with_backup_recovery(exec_file)
                    st = ed.get("status") or ed.get("result", {}).get("status")
                    if st in ("execution_review_ready", "approved_for_integration", "completed"):
                        execution_completed = True
                except Exception:
                    pass

            if integ_file.is_file():
                try:
                    id_data, _ = read_json_with_backup_recovery(integ_file)
                    ist = id_data.get("status") or id_data.get("result", {}).get("status")
                    if ist == "completed":
                        integration_completed = True
                except Exception:
                    pass

        # Check if commit is already present in target git log
        if active_t.completion_evidence:
            res_ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", str(active_t.completion_evidence), target_branch],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if res_ancestor.returncode == 0:
                commit_already_on_target = True

        # Check worktree
        wt_ref = state.worktree_reference or (f"{project_root.name}-openhands-{active_t.linked_run_id[:8]}" if active_t.linked_run_id else None)
        if wt_ref:
            wt_root = get_default_worktree_root(project_root)
            cand_wt = wt_root / wt_ref
            if cand_wt.is_dir():
                worktree_exists = True
                worktree_path_actual = cand_wt

    # Determine recovery decision
    recommended_action = RecoveryAction.NOOP
    reason = "Normal state."

    if is_cherry_pick_active:
        recommended_action = RecoveryAction.RECOVER_CHERRY_PICK_CONFLICT
        reason = "Cherry-pick conflict atau interupsi terdeteksi pada target repository."
    elif integration_completed or commit_already_on_target:
        recommended_action = RecoveryAction.SKIP_INTEGRATION_MARK_COMPLETED
        reason = "Integrasi sudah selesai sebelumnya. Menandai task completed secara idempotent."
    elif execution_completed:
        recommended_action = RecoveryAction.SKIP_EXECUTION_PROCEED_TO_REVIEW
        reason = "Eksekusi OpenHands sudah selesai dan valid. Melanjutkan ke approval/integrasi tanpa rerun."
    elif worktree_exists:
        recommended_action = RecoveryAction.RECONSTRUCT_WORKTREE
        reason = "Worktree terisolasi sudah ada di disk. Menggunakan kembali worktree tanpa duplikasi."

    return {
        "is_cherry_pick_active": is_cherry_pick_active,
        "current_target_head": current_target_head,
        "execution_completed": execution_completed,
        "integration_completed": integration_completed,
        "commit_already_on_target": commit_already_on_target,
        "worktree_exists": worktree_exists,
        "worktree_path": str(worktree_path_actual) if worktree_path_actual else None,
        "recommended_action": recommended_action,
        "reason": reason,
    }


def reconstruct_misclassified_approval_states(
    project_root: Path,
    state: ProgramState,
    event_logger: Any = None,
) -> bool:
    """General recovery path for legacy or broken program states where human_required was misclassified as failure.

    Detects tasks that failed solely due to human_required or retry exhaustion of human_required,
    reconstructs them to awaiting_input at the appropriate stage without silently resetting attempts,
    cleans up only derived blockers, and allows --resume --approve.
    """
    project_root = Path(project_root).expanduser().resolve()
    reconstructed = False

    for t in state.tasks:
        if t.status in ("failed", "blocked", "retrying"):
            is_human_req_failure = False
            linked_run_err = ""

            # Check blockers
            for b in state.blockers:
                if t.task_id in b and "human_required" in b.lower():
                    is_human_req_failure = True
                    linked_run_err = b
                    break

            # Check linked run artifact
            if not is_human_req_failure and t.linked_run_id:
                clean_run = t.linked_run_id[:8].lower()
                run_file = project_root / ".agent" / "artifacts" / "runs" / f"run_{clean_run}.json"
                if run_file.is_file():
                    try:
                        rd, _ = read_json_with_backup_recovery(run_file)
                        res_obj = rd.get("result", rd)
                        err_m = res_obj.get("error_message") or ""
                        if "human_required" in err_m.lower():
                            is_human_req_failure = True
                            linked_run_err = err_m
                    except Exception:
                        pass

            # Check retry_state
            if not is_human_req_failure and state.retry_state:
                if state.retry_state.get("task_id") == t.task_id and any("human_required" in b.lower() for b in state.blockers):
                    is_human_req_failure = True

            if is_human_req_failure:
                # 1. Determine approval stage (proposal vs execution)
                exec_file = None
                if t.linked_run_id:
                    clean_run = t.linked_run_id[:8].lower()
                    exec_file = project_root / ".agent" / "artifacts" / "executions" / f"execution_{clean_run}.json"
                stage = "execution" if (exec_file and exec_file.is_file()) else "proposal"

                # 2. Reconstruct task status
                t.status = "awaiting_input"
                t.approval_stage = stage

                # 3. Reconstruct program state
                state.status = "awaiting_ceo_input"
                state.current_phase = f"{stage}_approval"
                state.active_task = t.task_id
                state.active_run_id = t.linked_run_id
                state.next_action = f"python main.py ceo --project {project_root} --program {state.program_id} --resume --approve --autonomous"

                prop_art_id = f"proposal_{t.linked_run_id[:8]}" if t.linked_run_id else None
                state.approval_checkpoint = {
                    "program_id": state.program_id,
                    "task_id": t.task_id,
                    "active_run_id": t.linked_run_id,
                    "approval_stage": stage,
                    "proposal_artifact_id": prop_art_id,
                    "reason": linked_run_err or f"Reconstructed pending {stage} approval from misclassified human_required state.",
                    "risk": t.risk_level,
                    "next_action": state.next_action,
                }

                # 4. Clean up only derived blockers
                cleaned_blockers = []
                for b in state.blockers:
                    is_derived = (
                        (t.task_id in b and "human_required" in b.lower())
                        or (t.task_id in b and "failed after" in b.lower())
                        or b == "No ready tasks available and not all tasks completed."
                    )
                    if is_derived:
                        if event_logger:
                            event_logger.log(
                                event_type="recovery_completed",
                                safe_summary=f"Resolved derived blocker for {t.task_id}: {b}",
                                program_id=state.program_id,
                                task_id=t.task_id,
                            )
                        continue
                    cleaned_blockers.append(b)
                state.blockers = cleaned_blockers

                # 5. Clear misclassified retry state
                state.retry_state = {}

                # 6. Log recovery event
                if event_logger:
                    event_logger.log(
                        event_type="recovery_completed",
                        safe_summary=f"Reconstructed misclassified approval gate for task {t.task_id} at {stage} stage.",
                        program_id=state.program_id,
                        task_id=t.task_id,
                    )

                reconstructed = True

    return reconstructed

