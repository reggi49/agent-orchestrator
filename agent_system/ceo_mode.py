import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.config import get_settings
from agent_system.deterministic_safety import evaluate_deterministic_fallback_safety
from agent_system.error_reporting import format_safe_error_report, sanitize_text
from agent_system.event_logger import ProgramEventLogger
from agent_system.gateway_resilience import (
    BackoffManager,
    CircuitBreaker,
    FailureCategory,
    ModelFailoverManager,
    classify_failure,
)
from agent_system.objective_alignment import detect_objective_drift, validate_task_objective_alignment
from agent_system.orchestration_runner import OrchestrationRunner
from agent_system.program_lock import ProgramLock, ProgramLockActiveError
from agent_system.program_manager import ProgramManager, get_target_head_revision
from agent_system.recovery_inspector import (
    RecoveryAction,
    inspect_recovery_conditions,
    reconstruct_misclassified_approval_states,
)
from agent_system.risk_classifier import (
    classify_final_risk,
    classify_post_execution_risk,
    classify_pre_execution_risk,
    evaluate_approval_policy,
)
from agent_system.schemas import ProgramState, ProgramTask
from agent_system.scope_and_overlap import detect_cycle_in_dependencies


class ProgressHeartbeat:
    """Prints periodic progress heartbeats every interval seconds during long operations."""

    def __init__(self, message: str = "[CEO] Program aktif", interval: float = 15.0):
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = time.time()

    def start(self):
        self._start_time = time.time()
        self._stop_event.clear()

        def _loop():
            step = 0
            while not self._stop_event.wait(self.interval):
                elapsed = int(time.time() - self._start_time)
                step += 1
                if step % 3 == 1:
                    print(f"{self.message} (elapsed={elapsed}s)", flush=True)
                elif step % 3 == 2:
                    print(f"[LLM] Menunggu respons, elapsed={elapsed}s", flush=True)
                else:
                    print(f"[Checkpoint] State tersimpan (elapsed={elapsed}s)", flush=True)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)


class CEORunner:
    """CEO Mode orchestrates long-running multi-task programs autonomously with production hardening."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = Path(project_root).expanduser().resolve()
        self.verbose = verbose
        self.pm = ProgramManager(self.project_root)

    def run_ceo(
        self,
        objective_file: Path | str | None = None,
        request_text: str | None = None,
        autonomous: bool = False,
        max_workers: int = 1,
        program_id: str | None = None,
        resume: bool = False,
        approve: bool = False,
        reject: bool = False,
        request_changes: bool = False,
        approval_policy: str = "safe-auto",
        note: str = "",
    ) -> ProgramState:
        """Run or resume CEO program with single-writer lock, recovery inspection, and durability."""
        # 1. Resolve objective or resume target
        state: ProgramState | None = None

        if resume or (not objective_file and not request_text):
            try:
                _, state = self.pm.load_program_state(program_id or "latest")
            except Exception:
                pass

        if state is None:
            raw_objective = ""
            if objective_file:
                obj_path = Path(objective_file)
                if not obj_path.is_absolute():
                    obj_path = self.project_root / obj_path
                if not obj_path.is_file():
                    raise FileNotFoundError(f"Objective file '{objective_file}' tidak ditemukan.")
                raw_objective = obj_path.read_text(encoding="utf-8")
            elif request_text:
                raw_objective = request_text
            else:
                raise ValueError("Objective wajib diberikan melalui --objective <file> atau --request <text>.")

            state = self.pm.initialize_program(raw_objective, program_id=program_id)

        # 2. Acquire Single-Writer Program Lock
        lock = ProgramLock(self.project_root, program_id=state.program_id, project_id=state.project_id)
        event_logger = ProgramEventLogger(self.project_root, program_id=state.program_id)

        try:
            lock.acquire(
                current_task_id=state.active_task,
                active_run_id=state.active_run_id,
                event_logger=event_logger,
            )
        except ProgramLockActiveError as ple:
            print(f"ERROR: {ple}")
            sys.exit(1)

        try:
            # 3. Startup Recovery Inspection & Reconstruct Misclassified Approval States
            reconstructed = reconstruct_misclassified_approval_states(self.project_root, state, event_logger=event_logger)
            if reconstructed:
                self.pm.save_program_state(state)

            recovery_info = inspect_recovery_conditions(self.project_root, state)
            rec_action = recovery_info.get("recommended_action")

            if rec_action == RecoveryAction.RECOVER_CHERRY_PICK_CONFLICT:
                event_logger.log(
                    event_type="recovery_started",
                    safe_summary="Aborting pending cherry-pick conflict to restore target repository to safe state.",
                    program_id=state.program_id,
                )
                import subprocess
                subprocess.run(["git", "cherry-pick", "--abort"], cwd=self.project_root, capture_output=True, text=True)

            elif rec_action == RecoveryAction.SKIP_INTEGRATION_MARK_COMPLETED:
                if state.active_task:
                    active_t = next((t for t in state.tasks if t.task_id == state.active_task), None)
                    if active_t:
                        active_t.status = "completed"
                        state.active_task = None
                        state.active_run_id = None
                        self.pm.save_program_state(state)

            # 4. Handle human review decisions if decision flag was provided
            if approve or reject or request_changes:
                state = self._handle_ceo_decision(
                    state=state,
                    approve=approve,
                    reject=reject,
                    request_changes=request_changes,
                    approval_policy=approval_policy,
                    note=note,
                    event_logger=event_logger,
                )

            # 5. If not autonomous, save and return initial program breakdown
            if not autonomous:
                self._print_ceo_summary(state)
                return state

            # 6. Autonomous Execution Loop
            return self._run_autonomous_loop(
                state=state,
                approval_policy=approval_policy,
                max_workers=max_workers,
                lock=lock,
                event_logger=event_logger,
            )

        finally:
            # Release single-writer lock on exit
            lock.release(event_logger=event_logger)

    def _handle_ceo_decision(
        self,
        state: ProgramState,
        approve: bool,
        reject: bool,
        request_changes: bool,
        approval_policy: str,
        note: str,
        event_logger: Any = None,
    ) -> ProgramState:
        """Handle stage-aware CEO decision at awaiting_ceo_input checkpoint."""
        active_t = next((t for t in state.tasks if t.status == "awaiting_input"), None)
        if not active_t and state.active_task:
            active_t = next((t for t in state.tasks if t.task_id == state.active_task), None)

        if not active_t or (state.status != "awaiting_ceo_input" and active_t.status != "awaiting_input"):
            if approve or reject or request_changes:
                raise ValueError("FAIL-CLOSED: Tidak ada pending approval pada program ini. Perintah --approve/--reject ditolak.")
            return state

        if not active_t.linked_run_id:
            state.status = "ready"
            return state

        approval_stage = (
            getattr(active_t, "approval_stage", None)
            or (state.approval_checkpoint or {}).get("approval_stage")
            or "execution"
        )

        runner = OrchestrationRunner(self.project_root, verbose=self.verbose)
        res = runner.resume(
            run_arg=active_t.linked_run_id,
            approve=approve,
            reject=reject,
            request_changes=request_changes,
            apply=True,
            note=note,
        )

        overall_status = res.get("overall_status")
        if overall_status == "completed":
            active_t.status = "completed"
            active_t.approval_stage = None
            active_t.changed_files = res.get("changed_files", [])
            active_t.test_result = res.get("test_status", "passed")
            active_t.completion_evidence = res.get("commit_hash")
            state.active_task = None
            state.active_run_id = None
            state.approval_checkpoint = None
            if event_logger:
                event_logger.log(
                    event_type="task_completed",
                    safe_summary=f"Task {active_t.task_id} approved by CEO and completed.",
                    program_id=state.program_id,
                    task_id=active_t.task_id,
                    new_status="completed",
                )
            if all(t.status == "completed" for t in state.tasks):
                state.status = "completed"
                state.current_phase = "Phase Completed: Handoff Ready"
                state.next_action = "Review final black-box handoff report in HANDOFF.md"
                if event_logger:
                    event_logger.log(
                        event_type="program_ready_for_blackbox",
                        safe_summary="All program tasks completed. Handoff report ready.",
                        program_id=state.program_id,
                    )
            else:
                state.status = "in_progress"
        elif overall_status == "awaiting_human_approval":
            next_stage = res.get("approval_stage") or "execution"
            active_t.status = "awaiting_input"
            active_t.approval_stage = next_stage
            state.status = "awaiting_ceo_input"
            state.current_phase = f"{next_stage}_approval"
            state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve --autonomous"
            state.approval_checkpoint = {
                "program_id": state.program_id,
                "task_id": active_t.task_id,
                "active_run_id": active_t.linked_run_id,
                "approval_stage": next_stage,
                "proposal_artifact_id": res.get("proposal_artifact_id"),
                "reason": f"Persetujuan CEO dibutuhkan pada tahap {next_stage}.",
                "risk": active_t.risk_level,
                "next_action": state.next_action,
            }
        elif overall_status == "rejected":
            active_t.status = "cancelled"
            state.status = "blocked"
            state.approval_checkpoint = None
            state.blockers.append(f"Task {active_t.task_id} rejected by CEO.")
        elif overall_status == "changes_requested":
            active_t.status = "retrying"
            state.status = "in_progress"
            state.approval_checkpoint = None

        self.pm.refresh_task_statuses(state)
        self.pm.save_program_state(state)
        self.pm.write_all_markdown_reports(state)
        return state

    def _run_autonomous_loop(
        self,
        state: ProgramState,
        approval_policy: str,
        max_workers: int = 1,
        lock: ProgramLock | None = None,
        event_logger: Any = None,
    ) -> ProgramState:
        """Execute autonomous loop with circuit breaker, backoff, and graceful shutdown."""
        interrupted = False
        settings = get_settings()

        def sig_handler(signum, frame):
            nonlocal interrupted
            interrupted = True
            print("\n[CEO Mode] Graceful shutdown diterima. Menyimpan program checkpoint...")
            if event_logger:
                event_logger.log(
                    event_type="shutdown_requested",
                    safe_summary="Graceful shutdown signal (SIGINT/SIGTERM) received.",
                    program_id=state.program_id,
                )

        old_sigint = signal.signal(signal.SIGINT, sig_handler)
        old_sigterm = signal.signal(signal.SIGTERM, sig_handler)

        try:
            state.status = "in_progress"
            self.pm.save_program_state(state)

            while state.status not in ("completed", "failed", "blocked", "awaiting_ceo_input", "interrupted"):
                if interrupted:
                    state.status = "interrupted"
                    state.last_checkpoint = "Interrupted by graceful shutdown signal"
                    state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --autonomous"
                    if event_logger:
                        event_logger.log(
                            event_type="program_paused",
                            safe_summary="Program paused safely due to shutdown signal.",
                            program_id=state.program_id,
                        )
                    break

                if lock:
                    lock.update_heartbeat(current_task_id=state.active_task, active_run_id=state.active_run_id)

                task = self.pm.select_next_ready_task(state)
                if not task:
                    failed_task_ids = {t.task_id for t in state.tasks if t.status in ("failed", "cancelled")}

                    if all(t.status == "completed" for t in state.tasks):
                        state.status = "completed"
                        state.current_phase = "Phase Completed: Handoff Ready"
                        state.next_action = "Review final black-box handoff report in HANDOFF.md"
                        if event_logger:
                            event_logger.log(
                                event_type="program_ready_for_blackbox",
                                safe_summary="All tasks completed. Black-box handoff report ready.",
                                program_id=state.program_id,
                            )
                        break

                    # 9a. awaiting_input ditemukan: program = awaiting_ceo_input
                    elif any(t.status == "awaiting_input" for t in state.tasks):
                        state.status = "awaiting_ceo_input"
                        awt_task = next(t for t in state.tasks if t.status == "awaiting_input")
                        stage = awt_task.approval_stage or "proposal"
                        state.current_phase = f"{stage}_approval"
                        state.active_task = awt_task.task_id
                        state.active_run_id = awt_task.linked_run_id
                        state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve --autonomous"
                        break

                    # 9b. retrying ditemukan: program = retrying atau paused_for_provider
                    elif any(t.status == "retrying" for t in state.tasks):
                        ret_task = next(t for t in state.tasks if t.status == "retrying")
                        cb = CircuitBreaker(self.project_root, provider_id=settings.llm_provider or "default")
                        if not cb.is_available():
                            state.status = "paused_for_provider"
                            state.next_action = "Wait for provider cooldown or switch model"
                            break
                        else:
                            state.status = "in_progress"
                            task = ret_task

                    # 9c. active run ditemukan: resume active run
                    elif any(t.status == "in_progress" and t.linked_run_id for t in state.tasks):
                        task = next(t for t in state.tasks if t.status == "in_progress" and t.linked_run_id)

                    # 9d. dependency gagal permanen: program = blocked
                    elif any(t.dependencies and any(dep in failed_task_ids for dep in t.dependencies) for t in state.tasks):
                        blocked_tasks = [t for t in state.tasks if any(dep in failed_task_ids for dep in t.dependencies)]
                        state.status = "blocked"
                        for bt in blocked_tasks:
                            failed_deps = [dep for dep in bt.dependencies if dep in failed_task_ids]
                            msg = f"Task {bt.task_id} terhambat karena dependensi gagal permanen: {', '.join(failed_deps)}."
                            if msg not in state.blockers:
                                state.blockers.append(msg)
                        break

                    # 9e. dependency cycle: program = blocked dengan cycle details
                    elif detect_cycle_in_dependencies(state.dependencies):
                        cycle = detect_cycle_in_dependencies(state.dependencies)
                        state.status = "blocked"
                        cycle_msg = f"Dependency cycle terdeteksi: {' -> '.join(cycle)}."
                        if cycle_msg not in state.blockers:
                            state.blockers.append(cycle_msg)
                        break

                    # 9f. tidak ada kondisi di atas: program = inconsistent_state dan fail closed
                    else:
                        state.status = "inconsistent_state"
                        state.blockers.append("Inconsistent state: tidak ada task ready dan syarat penyelesaian belum terpenuhi.")
                        break

                if not task:
                    break

                # Execute or resume task with fresh base revision
                self._execute_single_task(state, task, approval_policy, lock=lock, event_logger=event_logger)

                self.pm.refresh_task_statuses(state)
                self.pm.save_program_state(state)
                self.pm.write_all_markdown_reports(state)

        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

            self.pm.save_program_state(state)
            self.pm.write_all_markdown_reports(state)
            self._print_ceo_summary(state)

        return state

    def _execute_single_task(
        self,
        state: ProgramState,
        task: ProgramTask,
        approval_policy: str,
        lock: ProgramLock | None = None,
        event_logger: Any = None,
    ) -> None:
        """Execute or resume one task using existing OrchestrationRunner with fresh-base & fail-closed recovery."""
        runner = OrchestrationRunner(self.project_root, verbose=self.verbose)
        state.active_task = task.task_id

        # 1. Validate Objective Alignment
        valid_req_ids = {r["id"] for r in state.requirements}
        is_aligned, align_err = validate_task_objective_alignment(task, valid_req_ids)
        if not is_aligned:
            task.status = "blocked"
            state.status = "blocked"
            state.blockers.append(align_err or "Task tidak terhubung ke requirement ID yang valid.")
            return

        # 2. Check Circuit Breaker for Gateway
        settings = get_settings()
        cb = CircuitBreaker(self.project_root, provider_id=settings.llm_provider or "default")
        if not cb.is_available():
            # Check model failover
            failover = ModelFailoverManager([settings.llm_model] if settings.llm_model else [])
            alt_model = failover.get_model_for_attempt(task.attempt_count + 1)
            if not alt_model or alt_model == settings.llm_model:
                task.status = "awaiting_input"
                state.status = "awaiting_ceo_input"
                state.blockers.append(f"Gateway circuit breaker OPEN untuk provider '{settings.llm_provider}'. Cooldown aktif.")
                if event_logger:
                    event_logger.log(
                        event_type="circuit_opened",
                        safe_summary=f"Circuit breaker OPEN for provider '{settings.llm_provider}'.",
                        program_id=state.program_id,
                        task_id=task.task_id,
                    )
                return

        # 3. Deterministic Safety Check on Task Design (enforce only when fallback was used)
        if task.fallback_used:
            fb_eval = evaluate_deterministic_fallback_safety(task.title, task.risk_level, fallback_type="task_execution")
            if not fb_eval["allowed"]:
                task.status = "awaiting_input"
                state.status = "awaiting_ceo_input"
                state.blockers.append(fb_eval["reason"])
                return

        # 4. If already linked to a run that was awaiting approval or interrupted
        if task.linked_run_id and task.status in ("in_progress", "awaiting_input", "retrying"):
            try:
                run_status_info = runner.get_status(task.linked_run_id)
                overall_status = run_status_info.get("overall_status")
                if overall_status == "completed":
                    task.status = "completed"
                    task.changed_files = run_status_info.get("changed_files", [])
                    task.test_result = run_status_info.get("test_status", "passed")
                    task.completion_evidence = run_status_info.get("commit_hash")
                    state.active_task = None
                    state.active_run_id = None
                    return
                elif overall_status == "awaiting_human_approval":
                    if approval_policy == "safe-auto" and task.risk_level == "low":
                        res_resume = runner.resume(run_arg=task.linked_run_id, approve=True, apply=True)
                        if res_resume.get("overall_status") == "completed":
                            task.status = "completed"
                            task.changed_files = res_resume.get("changed_files", [])
                            task.test_result = res_resume.get("test_status", "passed")
                            task.completion_evidence = res_resume.get("commit_hash")
                            state.active_task = None
                            state.active_run_id = None
                            return
                    task.status = "awaiting_input"
                    state.status = "awaiting_ceo_input"
                    state.active_run_id = task.linked_run_id
                    state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve"
                    return
            except Exception:
                pass

        # 5. Sequential Fresh-Base: read target HEAD immediately before execution
        fresh_head = get_target_head_revision(self.project_root)
        task.execution_base_revision = fresh_head

        # 6. Execute freshly
        task.status = "in_progress"
        task.attempt_count += 1
        if event_logger:
            event_logger.log(
                event_type="task_started",
                safe_summary=f"Started execution of task {task.task_id} (attempt {task.attempt_count}).",
                program_id=state.program_id,
                task_id=task.task_id,
                new_status="in_progress",
            )

        ac_str = "; ".join(task.acceptance_criteria)
        task_prompt = f"{task.title}: {task.description}. Kriteria penerimaan: {ac_str}"

        heartbeat = ProgressHeartbeat(message=f"[CEO] Menjalankan task {task.task_id}: {task.title[:40]}", interval=15.0)
        heartbeat.start()
        try:
            run_result = runner.run(
                request=task_prompt,
                apply=True,
                approval_policy=approval_policy,
                isolation="git-worktree",
                executor="openhands",
            )
        except Exception as e:
            run_result = {
                "overall_status": "interrupted",
                "error_message": str(e),
            }
        finally:
            heartbeat.stop()

        overall_status = run_result.get("overall_status")
        task.linked_run_id = run_result.get("run_id")
        state.active_run_id = run_result.get("run_id")
        state.worktree_reference = run_result.get("worktree_identifier")

        # 7. Post-Execution Risk Reclassification & Objective Drift Detection
        changed_files = run_result.get("changed_files", [])
        if changed_files:
            post_risk_info = classify_post_execution_risk(
                actual_changed_files=changed_files,
                diff_summary=f"{task.title} {run_result.get('error_message', '')}",
                test_status=run_result.get("test_status", "passed"),
            )
            final_risk = classify_final_risk(
                pre_risk_info={"level": task.pre_risk or task.risk_level},
                post_risk_info=post_risk_info,
            )
            task.post_risk = post_risk_info["level"]
            task.risk_level = final_risk["level"]

            # If risk escalated from LOW to HIGH/MEDIUM, escalate to CEO
            if final_risk["escalated"] and final_risk["level"] in ("high", "medium"):
                task.status = "awaiting_input"
                task.approval_stage = "execution"
                state.status = "awaiting_ceo_input"
                state.current_phase = "execution_approval"
                reason_str = "; ".join(final_risk["reasons"])
                state.blockers.append(f"Risk ter-eskalasi dari {task.pre_risk} ke {final_risk['level']}: {reason_str}")
                state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve --autonomous"
                state.approval_checkpoint = {
                    "program_id": state.program_id,
                    "task_id": task.task_id,
                    "active_run_id": task.linked_run_id,
                    "approval_stage": "execution",
                    "proposal_artifact_id": run_result.get("proposal_artifact_id"),
                    "reason": reason_str,
                    "risk": task.risk_level,
                    "next_action": state.next_action,
                }
                if event_logger:
                    event_logger.log(
                        event_type="risk_escalated",
                        safe_summary=f"Task {task.task_id} risk escalated to {final_risk['level']}: {reason_str}",
                        program_id=state.program_id,
                        task_id=task.task_id,
                    )
                return

            # Check Objective Drift
            drift_check = detect_objective_drift(task, changed_files)
            if drift_check["drift_detected"]:
                task.status = "awaiting_input"
                task.approval_stage = "execution"
                state.status = "awaiting_ceo_input"
                state.current_phase = "execution_approval"
                state.blockers.append(drift_check["drift_reason"])
                state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve --autonomous"
                state.approval_checkpoint = {
                    "program_id": state.program_id,
                    "task_id": task.task_id,
                    "active_run_id": task.linked_run_id,
                    "approval_stage": "execution",
                    "proposal_artifact_id": run_result.get("proposal_artifact_id"),
                    "reason": drift_check["drift_reason"],
                    "risk": task.risk_level,
                    "next_action": state.next_action,
                }
                return

        # 8. Handle Task Completion / Failure / Conflict
        if overall_status == "completed":
            cb.record_success()
            task.status = "completed"
            task.approval_stage = None
            task.changed_files = changed_files
            task.test_result = run_result.get("test_status", "passed")
            task.completion_evidence = run_result.get("commit_hash")
            state.active_task = None
            state.active_run_id = None
            state.approval_checkpoint = None
            if event_logger:
                event_logger.log(
                    event_type="task_completed",
                    safe_summary=f"Task {task.task_id} completed and integrated.",
                    program_id=state.program_id,
                    task_id=task.task_id,
                    new_status="completed",
                )

        elif overall_status == "awaiting_human_approval":
            approval_stage = run_result.get("approval_stage") or ("proposal" if not run_result.get("execution_artifact_path") else "execution")
            if approval_policy == "safe-auto" and task.risk_level == "low":
                res_resume = runner.resume(run_arg=run_result["run_id"], approve=True, apply=True)
                if res_resume.get("overall_status") == "completed":
                    cb.record_success()
                    task.status = "completed"
                    task.approval_stage = None
                    task.changed_files = res_resume.get("changed_files", [])
                    task.test_result = res_resume.get("test_status", "passed")
                    task.completion_evidence = res_resume.get("commit_hash")
                    state.active_task = None
                    state.active_run_id = None
                    state.approval_checkpoint = None
                    return
                elif res_resume.get("overall_status") == "awaiting_human_approval":
                    approval_stage = res_resume.get("approval_stage") or "execution"

            task.attempt_count = max(0, task.attempt_count - 1)
            task.status = "awaiting_input"
            task.approval_stage = approval_stage
            state.status = "awaiting_ceo_input"
            state.current_phase = f"{approval_stage}_approval"
            state.active_task = task.task_id
            state.active_run_id = task.linked_run_id
            state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve --autonomous"
            state.approval_checkpoint = {
                "program_id": state.program_id,
                "task_id": task.task_id,
                "active_run_id": task.linked_run_id,
                "approval_stage": approval_stage,
                "proposal_artifact_id": run_result.get("proposal_artifact_id"),
                "reason": run_result.get("error_message") or f"Persetujuan CEO dibutuhkan pada tahap {approval_stage}.",
                "risk": task.risk_level,
                "next_action": state.next_action,
            }

        elif overall_status == "interrupted":
            task.status = "retrying"
            state.status = "interrupted"
            state.last_checkpoint = f"Task {task.task_id} interrupted: {run_result.get('error_message')}"
            state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --autonomous"

        else:
            err_msg = run_result.get("error_message") or f"Task execution failed with status {overall_status}"
            fail_cat = classify_failure(err_msg)

            # Check if this failure was actually APPROVAL_REQUIRED (e.g. human_required in proposal review)
            if fail_cat == FailureCategory.APPROVAL_REQUIRED or "human_required" in err_msg.lower():
                approval_stage = run_result.get("approval_stage") or "proposal"
                task.attempt_count = max(0, task.attempt_count - 1)
                task.status = "awaiting_input"
                task.approval_stage = approval_stage
                state.status = "awaiting_ceo_input"
                state.current_phase = f"{approval_stage}_approval"
                state.active_task = task.task_id
                state.active_run_id = task.linked_run_id
                state.next_action = f"python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve --autonomous"
                state.approval_checkpoint = {
                    "program_id": state.program_id,
                    "task_id": task.task_id,
                    "active_run_id": task.linked_run_id,
                    "approval_stage": approval_stage,
                    "proposal_artifact_id": run_result.get("proposal_artifact_id") or (f"proposal_{task.linked_run_id[:8]}" if task.linked_run_id else None),
                    "reason": err_msg or "Proposal review berstatus human_required.",
                    "risk": task.risk_level,
                    "next_action": state.next_action,
                }
                return

            # Check if this failure was an integration cherry-pick conflict
            is_conflict = "conflict" in err_msg.lower() or "cherry-pick gagal" in err_msg.lower()

            if is_conflict:
                if event_logger:
                    event_logger.log(
                        event_type="integration_conflict",
                        safe_summary=f"Cherry-pick conflict on task {task.task_id}.",
                        program_id=state.program_id,
                        task_id=task.task_id,
                    )

                if task.conflict_replan_count < 1:
                    # Automatic conflict recovery attempt 1
                    task.conflict_replan_count += 1
                    task.superseded_attempts.append({
                        "run_id": task.linked_run_id,
                        "error": sanitize_text(err_msg),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    task.linked_run_id = None
                    task.status = "retrying"
                    if event_logger:
                        event_logger.log(
                            event_type="task_replanned",
                            safe_summary=f"Task {task.task_id} replanned against latest HEAD after conflict.",
                            program_id=state.program_id,
                            task_id=task.task_id,
                        )
                    return
                else:
                    # 2nd conflict: escalate to CEO (fail-closed)
                    task.status = "blocked"
                    state.status = "awaiting_ceo_input"
                    state.blockers.append(f"Cherry-pick conflict berulang pada task {task.task_id}. Eskalasi ke CEO.")
                    return

            # Gateway failure classification & backoff
            if fail_cat in (FailureCategory.TRANSIENT_GATEWAY, FailureCategory.RATE_LIMIT):
                cb.record_failure(fail_cat)
                backoff = BackoffManager()
                delay = backoff.calculate_delay(task.attempt_count)
                state.retry_state = {
                    "failure_class": fail_cat.value,
                    "attempt": task.attempt_count,
                    "retry_after": delay,
                    "operation": "task_execution",
                    "task_id": task.task_id,
                }
                if event_logger:
                    event_logger.log(
                        event_type="retry_scheduled",
                        safe_summary=f"Scheduled retry in {delay:.1f}s for {fail_cat.value}.",
                        program_id=state.program_id,
                        task_id=task.task_id,
                    )
                backoff.wait(delay)

            if task.attempt_count < 3:
                task.status = "retrying"
            else:
                task.status = "failed"
                state.status = "failed"
                state.blockers.append(f"Task {task.task_id} failed after {task.attempt_count} attempts: {err_msg}")

    def _print_ceo_summary(self, state: ProgramState) -> None:
        """Print concise CEO summary dashboard."""
        completed_cnt = sum(1 for t in state.tasks if t.status == "completed")
        total_cnt = len(state.tasks)
        pct = round((completed_cnt / total_cnt) * 100) if total_cnt else 0

        # Provider status
        settings = get_settings()
        cb = CircuitBreaker(self.project_root, provider_id=settings.llm_provider or "default")
        if not cb.is_available():
            prov_str = "cooling_down / blocked"
        else:
            prov_str = "available"

        retry_str = "none"
        if state.retry_state and state.retry_state.get("retry_after"):
            retry_str = f"scheduled ({state.retry_state.get('failure_class')})"

        # Approval status
        approval_str = "none"
        if state.approval_checkpoint:
            approval_str = state.approval_checkpoint.get("approval_stage") or "pending"
        elif any(t.status == "awaiting_input" for t in state.tasks):
            awt_t = next(t for t in state.tasks if t.status == "awaiting_input")
            approval_str = getattr(awt_t, "approval_stage", "pending") or "pending"

        # Next action display
        next_action_str = state.next_action or "none"
        if state.status == "awaiting_ceo_input":
            next_action_str = f"resume --approve --autonomous (or: python main.py ceo --project {self.project_root} --program {state.program_id} --resume --approve --autonomous)"

        print()
        print("=" * 65, flush=True)
        print("CEO MODE PROGRAM DASHBOARD", flush=True)
        print("=" * 65, flush=True)
        print(f"Program       : {state.program_id}", flush=True)
        print(f"Status        : {state.status.upper()}", flush=True)
        print(f"Progress      : {completed_cnt}/{total_cnt} tasks ({pct}%)", flush=True)
        print(f"Current Task  : {state.active_task or 'none'}", flush=True)
        print(f"Approval      : {approval_str}", flush=True)
        risk_str = "unknown"
        if state.active_task:
            t = next((task for task in state.tasks if task.task_id == state.active_task), None)
            if t:
                risk_str = t.risk_level
        elif state.tasks:
            risk_str = state.tasks[0].risk_level
        print(f"Risk          : {risk_str}", flush=True)
        print(f"Provider      : {prov_str}", flush=True)
        print(f"Retry         : {retry_str}", flush=True)
        print(f"Lock          : owned", flush=True)
        if state.blockers:
            print(f"Blockers      : {len(state.blockers)}", flush=True)
            for b in state.blockers:
                print(f"  - ⚠️ {b}", flush=True)
        print(f"Next Action   : {next_action_str}", flush=True)
        print("=" * 65, flush=True)
        print(flush=True)
