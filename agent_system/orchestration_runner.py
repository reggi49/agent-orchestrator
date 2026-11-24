import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.config import get_settings
from agent_system.error_reporting import format_safe_error_report, sanitize_text
from agent_system.execution_package import (
    extract_approved_files_from_proposal,
    extract_validation_commands_from_proposal,
)
from agent_system.execution_reviewer import (
    locate_execution_artifact,
    review_execution,
)
from agent_system.executors.openhands_executor import OpenHandsExecutor
from agent_system.graph import (
    build_execution_graph,
    build_graph,
    build_proposal_graph,
)
from agent_system.integrator import (
    check_source_repository_state,
    integrate_execution,
)
from agent_system.project_loader import load_project
from agent_system.risk_classifier import classify_risk, evaluate_approval_policy
from agent_system.worktree_manager import get_default_worktree_root


@contextmanager
def suppress_stdout(enabled: bool = True):
    """Context manager to suppress stdout when not in verbose mode."""
    if not enabled:
        yield
        return
    old_stdout = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def locate_run_checkpoint(project_root: Path, run_arg: str = "latest") -> tuple[Path, dict[str, Any]]:
    """Locate orchestration run checkpoint artifact."""
    runs_dir = project_root / ".agent" / "artifacts" / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Direktori runs '{runs_dir}' tidak ditemukan.")

    run_files = sorted(runs_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        raise FileNotFoundError(f"Tidak ditemukan checkpoint run di '{runs_dir}'.")

    if run_arg in ("latest", "", None):
        target_file = run_files[0]
    else:
        clean_name = run_arg if run_arg.endswith(".json") else f"{run_arg}.json"
        target_file = runs_dir / clean_name
        if not target_file.is_file() and not run_arg.startswith("run_"):
            target_file = runs_dir / f"run_{clean_name}"

    if not target_file.is_file():
        raise FileNotFoundError(f"Checkpoint run '{run_arg}' tidak ditemukan di '{runs_dir}'.")

    try:
        raw_data = json.loads(target_file.read_text(encoding="utf-8"))
        data = raw_data.get("result", raw_data) if isinstance(raw_data, dict) else raw_data
        if not isinstance(data, dict):
            raise ValueError(f"Checkpoint run '{target_file.name}' bukan objek JSON valid.")
        if "overall_status" not in data and isinstance(raw_data, dict) and "status" in raw_data:
            data["overall_status"] = raw_data["status"]
    except Exception as e:
        raise ValueError(f"Checkpoint run '{target_file.name}' rusak atau bukan JSON valid: {e}")

    return target_file, data


class OrchestrationRunner:
    """User-facing orchestration facade for multi-agent lifecycle."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = Path(project_root).expanduser().resolve()
        self.verbose = verbose
        self.validated_project = load_project(self.project_root)
        self.artifact_store = ProjectArtifactStore(self.project_root, self.validated_project.manifest)

    def _save_checkpoint(self, checkpoint: dict[str, Any]) -> Path:
        """Persist checkpoint to .agent/artifacts/runs/run_{run_id}.json."""
        run_id = checkpoint["run_id"]
        short_id = run_id[:8].lower()
        checkpoint["status"] = checkpoint.get("overall_status", "running")
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self.artifact_store.write_run(f"run_{short_id}", checkpoint)

    def run(
        self,
        request: str,
        apply: bool = False,
        approval_policy: str = "safe-auto",
        isolation: str = "git-worktree",
        executor: str = "openhands",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Orchestrate entire lifecycle from a single user request."""
        start_time = time.time()
        run_uuid = run_id or uuid.uuid4().hex
        short_id = run_uuid[:8].lower()
        now_iso = datetime.now(timezone.utc).isoformat()

        checkpoint = {
            "run_id": short_id,
            "project_id": self.validated_project.manifest.project.id,
            "request": request,
            "apply": apply,
            "dry_run": not apply,
            "approval_policy": approval_policy,
            "isolation": isolation,
            "executor": executor,
            "current_stage": "inspect",
            "overall_status": "running",
            "risk_level": "unknown",
            "risk_reasons": [],
            "approval_requirement": "unknown",
            "plan_artifact_id": None,
            "plan_artifact_path": None,
            "proposal_artifact_id": None,
            "proposal_artifact_path": None,
            "execution_artifact_id": None,
            "execution_artifact_path": None,
            "review_artifact_id": None,
            "review_artifact_path": None,
            "integration_artifact_id": None,
            "integration_artifact_path": None,
            "worktree_identifier": None,
            "worktree_path": None,
            "branch_name": None,
            "changed_files": [],
            "test_status": "none",
            "commit_hash": None,
            "next_action": None,
            "created_at": now_iso,
            "updated_at": now_iso,
            "duration_seconds": 0.0,
            "error_message": None,
        }
        self._save_checkpoint(checkpoint)

        try:
            return self._execute_lifecycle(checkpoint, start_time)
        except Exception as e:
            checkpoint["overall_status"] = "interrupted"
            checkpoint["error_message"] = sanitize_text(str(e))
            checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
            checkpoint["next_action"] = f"python main.py resume --project {self.project_root} --run {short_id}"
            self._save_checkpoint(checkpoint)
            if self.verbose:
                print(format_safe_error_report(e, context="Orchestration run"))
            return checkpoint

    def _execute_lifecycle(self, checkpoint: dict[str, Any], start_time: float) -> dict[str, Any]:
        """Execute or resume lifecycle stages sequentially."""
        short_id = checkpoint["run_id"]
        request = checkpoint["request"]
        apply = checkpoint["apply"]
        approval_policy = checkpoint["approval_policy"]

        # =====================================================================
        # STAGE 1: PLAN
        # =====================================================================
        plan_path_str = checkpoint.get("plan_artifact_path")
        plan_file = (self.project_root / plan_path_str) if plan_path_str else None
        plan_data = None

        if plan_file and plan_file.is_file():
            try:
                plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
            except Exception:
                plan_data = None

        if not plan_data:
            checkpoint["current_stage"] = "plan"
            self._save_checkpoint(checkpoint)

            initial_plan_state = {
                "run_id": short_id,
                "user_request": request,
                "project_root": str(self.project_root),
                "current_stage": "project_reader",
                "status": "running",
                "errors": [],
                "replan_count": 0,
                "max_replans": 2,
                "rejected_tasks": [],
            }

            plan_graph = build_graph()
            with suppress_stdout(not self.verbose):
                final_plan_state = plan_graph.invoke(initial_plan_state, config={"recursion_limit": 10})

            if final_plan_state.get("status") not in ("plan_ready", "valid"):
                checkpoint["overall_status"] = "failed"
                checkpoint["error_message"] = f"Tahap planning gagal dengan status '{final_plan_state.get('status')}'."
                checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
                self._save_checkpoint(checkpoint)
                return checkpoint

            analysis = final_plan_state.get("analysis", {})
            task_verification = final_plan_state.get("task_verification", {})
            rejected_tasks = final_plan_state.get("rejected_tasks", [])
            recommended_task = analysis.get("recommended_task", {}) if analysis else {}
            evidence_results = final_plan_state.get("repository_evidence", {}).get("results", [])
            evidence_paths = [r["path"] for r in evidence_results if isinstance(r, dict) and "path" in r]

            initial_candidate = rejected_tasks[0].get("title", "") if rejected_tasks else recommended_task.get("title", "")
            initial_status = rejected_tasks[0].get("status", "") if rejected_tasks else task_verification.get("status", "")
            missing_work = task_verification.get("missing_work", []) if task_verification.get("status") == "partially_implemented" else []

            plan_payload = {
                "initial_candidate": initial_candidate,
                "initial_verification_status": initial_status,
                "rejected_tasks": rejected_tasks,
                "replan_count": final_plan_state.get("replan_count", 0),
                "final_task": recommended_task.get("title", ""),
                "final_owner": recommended_task.get("owner", ""),
                "final_acceptance_criteria": recommended_task.get("acceptance_criteria", []),
                "evidence_paths": evidence_paths,
                "verification_status": task_verification.get("status", ""),
                "missing_work": missing_work,
                "status": final_plan_state.get("status"),
            }

            plan_file = self.artifact_store.write_plan(f"plan_{short_id}", plan_payload)
            plan_data = json.loads(plan_file.read_text(encoding="utf-8"))

            checkpoint["plan_artifact_id"] = f"plan_{short_id}"
            checkpoint["plan_artifact_path"] = str(plan_file.relative_to(self.project_root))
            checkpoint["current_stage"] = "propose"
            self._save_checkpoint(checkpoint)

        # =====================================================================
        # STAGE 2: PROPOSE
        # =====================================================================
        prop_path_str = checkpoint.get("proposal_artifact_path")
        prop_file = (self.project_root / prop_path_str) if prop_path_str else None
        prop_data = None

        if prop_file and prop_file.is_file():
            try:
                prop_data = json.loads(prop_file.read_text(encoding="utf-8"))
            except Exception:
                prop_data = None

        if not prop_data:
            checkpoint["current_stage"] = "propose"
            self._save_checkpoint(checkpoint)

            source_plan_rel = checkpoint["plan_artifact_path"]
            res = plan_data.get("result", {})
            verified_task = res.get("recommended_task") or res.get("final_task") or {}
            if isinstance(verified_task, str):
                task_title = verified_task
                task_owner = res.get("final_owner", "backend")
                acceptance_criteria = res.get("final_acceptance_criteria", [])
            else:
                task_title = verified_task.get("title", "")
                task_owner = verified_task.get("owner", "backend")
                acceptance_criteria = verified_task.get("acceptance_criteria", [])

            initial_prop_state = {
                "run_id": short_id,
                "project_root": str(self.project_root),
                "source_plan_path": source_plan_rel,
                "plan_data": plan_data,
                "task_verification": {
                    "status": res.get("verification_status", "verified_pending"),
                    "task_title": task_title,
                    "evidence_for": plan_data.get("evidence_paths", []),
                    "missing_work": res.get("missing_work", []),
                },
                "repository_evidence": {
                    "results": [{"path": p} for p in plan_data.get("evidence_paths", [])],
                },
                "current_stage": "implementation_proposal",
                "status": "running",
                "errors": [],
                "proposal_revision_count": 0,
                "max_proposal_revisions": 2,
                "rejected_proposal_reasons": [],
                "proposal_hashes": [],
            }

            prop_graph = build_proposal_graph()
            with suppress_stdout(not self.verbose):
                final_prop_state = prop_graph.invoke(initial_prop_state, config={"recursion_limit": 10})

            final_prop_status = final_prop_state.get("status")
            review = final_prop_state.get("proposal_review", {})
            proposal_payload = final_prop_state.get("implementation_proposal", {})

            if proposal_payload:
                rev_meta = {
                    "source_plan_path": source_plan_rel,
                    "proposal": proposal_payload,
                    "review_status": review.get("status"),
                    "proposal_revision_count": final_prop_state.get("proposal_revision_count", 0),
                    "status": final_prop_status,
                    "evidence_paths": proposal_payload.get("evidence_paths", []),
                }
                prop_file = self.artifact_store.write_proposal(f"proposal_{short_id}", rev_meta)
                prop_data = json.loads(prop_file.read_text(encoding="utf-8"))

                review_meta = {
                    "source_plan_path": source_plan_rel,
                    "proposal_id": proposal_payload.get("proposal_id"),
                    "review": review,
                    "status": final_prop_status,
                    "evidence_paths": proposal_payload.get("evidence_paths", []),
                }
                self.artifact_store.write_report(f"proposal_review_{short_id}", review_meta)

                checkpoint["proposal_artifact_id"] = f"proposal_{short_id}"
                checkpoint["proposal_artifact_path"] = str(prop_file.relative_to(self.project_root))

            if final_prop_status == "human_required":
                checkpoint["overall_status"] = "awaiting_human_approval"
                checkpoint["approval_stage"] = "proposal"
                checkpoint["current_stage"] = "proposal_approval"
                checkpoint["error_message"] = None
                checkpoint["next_action"] = f"python main.py resume --project {self.project_root} --run {short_id} --approve"
                checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
                self._save_checkpoint(checkpoint)
                return checkpoint
            elif final_prop_status != "approved_for_implementation":
                checkpoint["overall_status"] = "failed"
                checkpoint["error_message"] = f"Proposal review tidak disetujui (status: {final_prop_status})."
                checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
                self._save_checkpoint(checkpoint)
                return checkpoint

            checkpoint["current_stage"] = "execute"
            self._save_checkpoint(checkpoint)

        # =====================================================================
        # STAGE 3: EXECUTE
        # =====================================================================
        exec_path_str = checkpoint.get("execution_artifact_path")
        exec_file = (self.project_root / exec_path_str) if exec_path_str else None
        exec_data = None

        if exec_file and exec_file.is_file():
            try:
                exec_data = json.loads(exec_file.read_text(encoding="utf-8"))
            except Exception:
                exec_data = None

        if not exec_data:
            checkpoint["current_stage"] = "execute"
            self._save_checkpoint(checkpoint)

            prop_inner = prop_data.get("result", {}).get("proposal", {})
            review_inner = prop_data.get("result", {}).get("review", {})
            task_title = prop_inner.get("task_title", "Execution Task")
            task_words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in task_title.split()]
            task_tail = "-".join([w for w in task_words if w][-2:]) if task_words else "task"
            branch_name = f"agent/openhands-{task_tail}-{short_id}"
            wt_name = f"{self.project_root.name}-openhands-{short_id}"
            wt_root = get_default_worktree_root(self.project_root)
            planned_wt_path = wt_root / wt_name

            approved_files = extract_approved_files_from_proposal(prop_inner, review_inner)

            initial_exec_state = {
                "run_id": short_id,
                "project_root": str(self.project_root),
                "main_project_root": str(self.project_root),
                "source_plan_path": checkpoint["plan_artifact_path"],
                "source_proposal_path": checkpoint["proposal_artifact_path"],
                "plan_data": plan_data,
                "proposal_data": prop_data,
                "proposal_review": review_inner,
                "current_stage": "execution",
                "status": "running",
                "errors": [],
                "dry_run": not apply,
                "apply": apply,
                "executor": checkpoint["executor"],
                "isolation": checkpoint["isolation"],
                "approved_files": approved_files,
                "branch_name": branch_name,
                "worktree_descriptor": wt_name,
                "executor_retry_count": 0,
                "max_executor_retries": 1,
            }

            exec_graph = build_execution_graph()
            with suppress_stdout(not self.verbose):
                final_exec_state = exec_graph.invoke(initial_exec_state, config={"recursion_limit": 15})

            exec_status = final_exec_state.get("status", "unknown")
            preflight = final_exec_state.get("preflight_result", {})
            coding_res = final_exec_state.get("coding_execution_result", {})
            diff_rev = final_exec_state.get("diff_review", {})

            # In dry-run mode
            if not apply:
                checkpoint["worktree_identifier"] = wt_name
                checkpoint["branch_name"] = branch_name
                checkpoint["approved_files"] = approved_files
                checkpoint["overall_status"] = "dry_run_ready"
                checkpoint["next_action"] = f"python main.py run --project {self.project_root} --request '{request}' --apply"
                checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
                self._save_checkpoint(checkpoint)
                return checkpoint

            # Apply mode checks
            changed_files_actual = diff_rev.get("changed_files", coding_res.get("changed_files", []))
            cmd_results = coding_res.get("validation_results", [])
            test_exit_code = cmd_results[0].get("exit_code") if cmd_results else 0
            test_status = "passed" if test_exit_code == 0 and cmd_results else "failed"

            if exec_status not in ("execution_review_ready", "completed", "approved_for_integration"):
                checkpoint["overall_status"] = exec_status
                checkpoint["error_message"] = final_exec_state.get("final_summary") or f"Eksekusi gagal dengan status {exec_status}."
                checkpoint["changed_files"] = changed_files_actual
                checkpoint["test_status"] = test_status
                checkpoint["worktree_identifier"] = wt_name
                checkpoint["branch_name"] = branch_name
                checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
                self._save_checkpoint(checkpoint)
                return checkpoint

            # Save execution artifact
            val_summary = [
                {
                    "executable": c.get("executable"),
                    "arguments": c.get("arguments"),
                    "working_directory": c.get("working_directory"),
                    "exit_code": c.get("exit_code"),
                    "status": c.get("status"),
                }
                for c in cmd_results
            ]
            diff_check_passed = diff_rev.get("status") == "approved"

            exec_payload = {
                "executor": checkpoint["executor"],
                "source_proposal_path": checkpoint["proposal_artifact_path"],
                "source_revision": preflight.get("source_revision", {}),
                "branch_name": branch_name,
                "worktree_identifier": wt_name,
                "approved_files": approved_files,
                "actual_changed_files": changed_files_actual,
                "iterations": coding_res.get("iterations", 0),
                "validation_command_summary": val_summary,
                "validation_exit_code": test_exit_code,
                "diff_check_status": "passed" if diff_check_passed else "failed",
                "independent_review_status": diff_rev.get("status", "unknown"),
                "scope_violations": coding_res.get("scope_violations", []),
                "requested_additional_files": coding_res.get("requested_additional_files", []),
                "final_status": "execution_review_ready",
                "status": "execution_review_ready",
            }
            exec_file = self.artifact_store.write_execution(f"execution_{short_id}", exec_payload)
            self.artifact_store.write_report(f"execution_review_{short_id}", exec_payload)
            exec_data = json.loads(exec_file.read_text(encoding="utf-8"))

            checkpoint["execution_artifact_id"] = f"execution_{short_id}"
            checkpoint["execution_artifact_path"] = str(exec_file.relative_to(self.project_root))
            checkpoint["worktree_identifier"] = wt_name
            checkpoint["worktree_path"] = str(planned_wt_path)
            checkpoint["branch_name"] = branch_name
            checkpoint["changed_files"] = changed_files_actual
            checkpoint["test_status"] = test_status
            checkpoint["overall_status"] = "execution_review_ready"
            checkpoint["current_stage"] = "review"
            self._save_checkpoint(checkpoint)

        # =====================================================================
        # STAGE 4: RISK CLASSIFICATION & APPROVAL POLICY
        # =====================================================================
        res_exec = exec_data.get("result", {})
        changed_files = res_exec.get("actual_changed_files", []) or res_exec.get("changed_files", []) or checkpoint.get("changed_files", [])
        approved_files = res_exec.get("approved_files", [])
        val_cmds = res_exec.get("validation_command_summary", [])
        test_passed = all(c.get("exit_code") == 0 and c.get("status") == "passed" for c in val_cmds) if val_cmds else False
        test_status = "passed" if test_passed else "failed"
        diff_status = res_exec.get("independent_review_status", "approved")
        scope_violations = res_exec.get("scope_violations", [])

        risk_info = classify_risk(
            request=request,
            task_title=plan_data.get("result", {}).get("final_task", ""),
            task_desc="",
            changed_files=changed_files,
            approved_files=approved_files,
            scope_violations=scope_violations,
            test_status=test_status,
            diff_review_status=diff_status,
        )

        approval_eval = evaluate_approval_policy(
            risk_level=risk_info["level"],
            approval_policy=approval_policy,
        )

        checkpoint["risk_level"] = risk_info["level"]
        checkpoint["risk_reasons"] = risk_info["reasons"]
        checkpoint["approval_requirement"] = "none" if approval_eval["can_auto_approve"] else "human_approval_required"

        # Check if human approval is required
        if not approval_eval["can_auto_approve"]:
            checkpoint["overall_status"] = "awaiting_human_approval"
            checkpoint["next_action"] = f"python main.py resume --project {self.project_root} --run {short_id} --approve --apply"
            checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
            self._save_checkpoint(checkpoint)
            return checkpoint

        # Automated approval allowed
        if exec_data.get("status") != "approved_for_integration":
            review_res = review_execution(
                project_root=self.project_root,
                execution_file=exec_file,
                execution_data=exec_data,
                decision="approve",
                reviewer="safe_auto_reviewer",
                note=f"Auto-approved under safe-auto policy for {risk_info['level']} risk changes.",
            )
            checkpoint["review_artifact_id"] = f"human_execution_review_{short_id}"
            checkpoint["review_artifact_path"] = str(review_res["report_file"].relative_to(self.project_root))
            checkpoint["overall_status"] = "approved_for_integration"
            checkpoint["current_stage"] = "integrate"
            self._save_checkpoint(checkpoint)

        # =====================================================================
        # STAGE 5: INTEGRATE
        # =====================================================================
        if not apply:
            # Dry-run integration check
            int_res = integrate_execution(
                project_root=self.project_root,
                execution_file=exec_file,
                execution_data=exec_data,
                strategy="cherry-pick",
                apply=False,
            )
            checkpoint["overall_status"] = "dry_run_ready"
            checkpoint["next_action"] = f"python main.py run --project {self.project_root} --request '{request}' --apply"
            checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
            self._save_checkpoint(checkpoint)
            return checkpoint

        int_res = integrate_execution(
            project_root=self.project_root,
            execution_file=exec_file,
            execution_data=exec_data,
            strategy="cherry-pick",
            apply=True,
        )

        checkpoint["integration_artifact_id"] = f"integration_{short_id}"
        checkpoint["integration_artifact_path"] = str(int_res["integration_artifact_file"].relative_to(self.project_root))
        checkpoint["commit_hash"] = int_res["commit_hash"]
        checkpoint["overall_status"] = "completed"
        checkpoint["current_stage"] = "completed"
        checkpoint["next_action"] = "none"
        checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
        self._save_checkpoint(checkpoint)

        return checkpoint

    def resume(
        self,
        run_arg: str = "latest",
        approve: bool = False,
        reject: bool = False,
        request_changes: bool = False,
        apply: bool = False,
        note: str = "",
    ) -> dict[str, Any]:
        """Resume an orchestration run from its last checkpoint."""
        start_time = time.time()
        checkpoint_file, checkpoint = locate_run_checkpoint(self.project_root, run_arg)
        short_id = checkpoint["run_id"]
        status = checkpoint.get("overall_status")

        # Handle decisions at awaiting_human_approval stage
        if status == "awaiting_human_approval":
            approval_stage = checkpoint.get("approval_stage") or ("proposal" if not checkpoint.get("execution_artifact_path") else "execution")

            if approval_stage == "proposal":
                prop_path_str = checkpoint.get("proposal_artifact_path")
                if not prop_path_str:
                    raise ValueError("Checkpoint tidak memiliki proposal_artifact_path yang valid.")
                prop_file = self.project_root / prop_path_str
                prop_data = json.loads(prop_file.read_text(encoding="utf-8"))

                if reject:
                    checkpoint["overall_status"] = "rejected"
                    checkpoint["next_action"] = "none"
                    self._save_checkpoint(checkpoint)
                    return checkpoint

                if request_changes:
                    checkpoint["overall_status"] = "changes_requested"
                    checkpoint["next_action"] = "none"
                    self._save_checkpoint(checkpoint)
                    return checkpoint

                if approve:
                    # Update proposal artifact status to approved_for_implementation
                    prop_data["status"] = "approved_for_implementation"
                    if "result" in prop_data and isinstance(prop_data["result"], dict):
                        prop_data["result"]["status"] = "approved_for_implementation"
                        prop_data["result"]["review_status"] = "approved"
                    atomic_write_json(prop_file, prop_data, backup=True)

                    # Update review report if exists
                    rev_report_path = self.project_root / ".agent" / "artifacts" / "reports" / f"proposal_review_{short_id}.json"
                    if rev_report_path.is_file():
                        try:
                            rev_data = json.loads(rev_report_path.read_text(encoding="utf-8"))
                            rev_data["status"] = "approved"
                            if "result" in rev_data and isinstance(rev_data["result"], dict):
                                rev_data["result"]["status"] = "approved"
                                if "review" in rev_data["result"] and isinstance(rev_data["result"]["review"], dict):
                                    rev_data["result"]["review"]["status"] = "approved"
                            atomic_write_json(rev_report_path, rev_data, backup=False)
                        except Exception:
                            pass

                    checkpoint["approval_stage"] = "execution"
                    checkpoint["overall_status"] = "running"
                    checkpoint["current_stage"] = "execute"
                    if apply:
                        checkpoint["apply"] = True
                        checkpoint["dry_run"] = False
                    self._save_checkpoint(checkpoint)
                    return self._execute_lifecycle(checkpoint, start_time)

            else:  # execution stage
                exec_path_str = checkpoint.get("execution_artifact_path")
                if not exec_path_str:
                    raise ValueError("Checkpoint tidak memiliki execution_artifact_path yang valid.")
                exec_file = self.project_root / exec_path_str
                exec_data = json.loads(exec_file.read_text(encoding="utf-8"))

                if reject:
                    rev_res = review_execution(
                        project_root=self.project_root,
                        execution_file=exec_file,
                        execution_data=exec_data,
                        decision="reject",
                        reviewer=os.getenv("USER") or "human_operator",
                        note=note,
                    )
                    checkpoint["overall_status"] = "rejected"
                    checkpoint["review_artifact_path"] = str(rev_res["report_file"].relative_to(self.project_root))
                    checkpoint["next_action"] = "none"
                    self._save_checkpoint(checkpoint)
                    return checkpoint

                if request_changes:
                    rev_res = review_execution(
                        project_root=self.project_root,
                        execution_file=exec_file,
                        execution_data=exec_data,
                        decision="request-changes",
                        reviewer=os.getenv("USER") or "human_operator",
                        note=note,
                    )
                    checkpoint["overall_status"] = "changes_requested"
                    checkpoint["review_artifact_path"] = str(rev_res["report_file"].relative_to(self.project_root))
                    checkpoint["next_action"] = "none"
                    self._save_checkpoint(checkpoint)
                    return checkpoint

                if approve:
                    rev_res = review_execution(
                        project_root=self.project_root,
                        execution_file=exec_file,
                        execution_data=exec_data,
                        decision="approve",
                        reviewer=os.getenv("USER") or "human_operator",
                        note=note,
                    )
                    checkpoint["review_artifact_id"] = f"human_execution_review_{short_id}"
                    checkpoint["review_artifact_path"] = str(rev_res["report_file"].relative_to(self.project_root))
                    checkpoint["overall_status"] = "approved_for_integration"
                    checkpoint["current_stage"] = "integrate"

                    # Check if we should integrate immediately
                    should_apply = apply or checkpoint.get("apply", False)
                    if should_apply:
                        int_res = integrate_execution(
                            project_root=self.project_root,
                            execution_file=exec_file,
                            execution_data=exec_data,
                            strategy="cherry-pick",
                            apply=True,
                        )
                        checkpoint["integration_artifact_id"] = f"integration_{short_id}"
                        checkpoint["integration_artifact_path"] = str(int_res["integration_artifact_file"].relative_to(self.project_root))
                        checkpoint["commit_hash"] = int_res["commit_hash"]
                        checkpoint["overall_status"] = "completed"
                        checkpoint["current_stage"] = "completed"
                        checkpoint["next_action"] = "none"
                    else:
                        checkpoint["next_action"] = f"python main.py resume --project {self.project_root} --run {short_id} --apply"

                    checkpoint["duration_seconds"] = round(time.time() - start_time, 1)
                    self._save_checkpoint(checkpoint)
                    return checkpoint

            raise ValueError("Run berstatus 'awaiting_human_approval'. Berikan salah satu opsi: --approve, --reject, atau --request-changes.")

        # If interrupted/failed at earlier stages, continue lifecycle
        if apply:
            checkpoint["apply"] = True
            checkpoint["dry_run"] = False

        checkpoint["overall_status"] = "running"
        self._save_checkpoint(checkpoint)
        return self._execute_lifecycle(checkpoint, start_time)

    def get_status(self, run_arg: str = "latest") -> dict[str, Any]:
        """Inspect and return current status of a run."""
        _, checkpoint = locate_run_checkpoint(self.project_root, run_arg)
        return checkpoint
