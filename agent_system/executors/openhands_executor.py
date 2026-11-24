import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from agent_system.config import get_settings
from agent_system.executors.base import (
    CodingAgentExecutor,
    CodingExecutionRequest,
    CodingExecutionResult,
)

OPENHANDS_EXECUTION_TIMEOUT_SECONDS = 900
OPENHANDS_MAX_ITERATIONS = 30

# Ensure banner does not pollute console
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"


class OpenHandsExecutor(CodingAgentExecutor):
    """OpenHands Software Agent SDK Executor using LocalWorkspace (No Docker)."""

    def __init__(self, settings: Any = None):
        self.settings = settings or get_settings()
        self._active_conversations: dict[str, Any] = {}
        self._cancel_flags: dict[str, bool] = {}

    def validate_configuration(self) -> dict[str, Any]:
        """Validate OpenHands configuration, environment, and dependencies."""
        try:
            import openhands.sdk as sdk
            from openhands.sdk import LocalWorkspace

            sdk_version = getattr(sdk, "__version__", "1.53.0")
        except ImportError as e:
            return {
                "valid": False,
                "error": f"OpenHands SDK tidak terpasang: {e}",
                "workspace_type": "none",
                "docker_used": False,
            }

        if not self.settings.has_api_key:
            return {
                "valid": False,
                "error": "API key belum dikonfigurasi di environment.",
                "workspace_type": "LocalWorkspace",
                "docker_used": False,
            }

        if not self.settings.llm_model:
            return {
                "valid": False,
                "error": "LLM_MODEL belum dikonfigurasi di environment.",
                "workspace_type": "LocalWorkspace",
                "docker_used": False,
            }

        return {
            "valid": True,
            "executor": "openhands",
            "sdk_version": sdk_version,
            "workspace_type": "LocalWorkspace",
            "docker_used": False,
            "model": self.settings.llm_model,
            "has_api_key": True,
            "base_url": self.settings.llm_base_url,
        }

    def prepare_execution(self, request: CodingExecutionRequest) -> dict[str, Any]:
        """Prepare task instructions, policies, and workspace context from proposal."""
        # Ensure all approved paths are relative
        relative_approved_files = []
        for p in request.approved_files:
            clean_p = str(p).replace("\\", "/").strip()
            if clean_p.startswith("/"):
                clean_p = clean_p.lstrip("/")
            relative_approved_files.append(clean_p)

        # Build validation commands description
        val_cmds_str = ""
        for i, cmd in enumerate(request.validation_commands, 1):
            exe = cmd.get("executable", "")
            args = " ".join(cmd.get("arguments", []))
            cwd = cmd.get("working_directory", ".")
            val_cmds_str += f"\n  {i}. {exe} {args} (working_dir: {cwd})"

        if not val_cmds_str:
            val_cmds_str = "\n  (None specified)"

        ac_str = ""
        for i, ac in enumerate(request.acceptance_criteria, 1):
            ac_str += f"\n  {i}. {ac}"

        approved_files_str = "\n".join(f"  - {f}" for f in relative_approved_files)

        # Build task instruction without API key
        existing_code_blocks = []
        wt_dir = Path(request.worktree_identifier).expanduser().resolve()
        if wt_dir.is_dir():
            for f in relative_approved_files:
                f_path = wt_dir / f
                if f_path.is_file():
                    try:
                        content = f_path.read_text(encoding="utf-8")
                        if len(content) <= 30000:
                            existing_code_blocks.append(f"--- File: {f} ---\n{content}")
                    except Exception:
                        pass

        existing_code_str = ""
        if existing_code_blocks:
            existing_code_str = "\n\nEXISTING CODE OF APPROVED FILES:\n" + "\n\n".join(existing_code_blocks) + "\n"

        instruction = (
            f"TASK TITLE: {request.task_title}\n\n"
            f"TASK DESCRIPTION:\n{request.task_description}\n\n"
            f"APPROVED WRITABLE FILES (STRICT SCOPE):\n{approved_files_str}\n\n"
            f"ACCEPTANCE CRITERIA:{ac_str}\n\n"
            f"APPROVED VALIDATION COMMANDS:{val_cmds_str}\n\n"
            "WORKFLOW INSTRUCTIONS:\n"
            "1. Read the existing code provided below and modify the approved files directly.\n"
            f"2. Modify ONLY the approved files listed above to implement changes for {request.task_title}.\n"
            "3. Run the validation command: `go test ./...` in the `backend` directory using terminal.\n"
            "4. If tests fail, read the error output and fix the implementation strictly within the approved files.\n"
            "5. Stop immediately once the tests pass or the maximum iteration limit is reached.\n\n"
            "STRICT RULES & FORBIDDEN ACTIONS:\n"
            "1. ONLY read and modify files inside the current worktree workspace.\n"
            "2. NEVER access files outside the worktree or the main project repository.\n"
            "3. NEVER read, print, or modify .env, .env.*, secrets, or credentials.\n"
            "4. NEVER run git commit, git push, git merge, git reset, or git clean.\n"
            "5. NEVER modify database migrations (e.g. backend/migrations/*).\n"
            "6. NEVER modify frontend code or dependencies (e.g. package.json, go.mod, go.sum).\n"
            "7. NEVER install new dependencies (no go get, npm install, pip install, etc.).\n"
            "8. NEVER download from the network (no curl, wget, etc.).\n"
            "9. NEVER use sudo or system administrative commands.\n"
            "10. You are ONLY authorized to edit the APPROVED WRITABLE FILES listed above.\n"
            "11. If you determine that the task CANNOT be completed without modifying additional files outside the approved list:\n"
            "   DO NOT modify any unapproved files!\n"
            "   Instead, explicitly declare: REQUEST_ADDITIONAL_FILES: <file1>, <file2>\n"
            "   and stop immediately so a proposal revision can be requested.\n\n"
            f"STOP CONDITIONS & ITERATION LIMIT:\n"
            f"- Maximum iterations: {request.max_iterations}\n"
            "- Stop immediately when all acceptance criteria are met and validation commands pass.\n"
            f"- Stop immediately if unapproved files would be required.\n"
            f"{existing_code_str}"
        )

        # Sanity check: Ensure API key is NEVER leaked in prompt
        if self.settings.llm_api_key and self.settings.llm_api_key in instruction:
            raise ValueError("SECURITY VIOLATION: API key detected in task instruction prompt.")

        return {
            "instruction": instruction,
            "approved_files": relative_approved_files,
            "max_iterations": request.max_iterations,
            "timeout_seconds": request.timeout_seconds,
            "worktree_identifier": request.worktree_identifier,
        }

    def execute(self, request: CodingExecutionRequest) -> CodingExecutionResult:
        """Execute coding task within isolated workspace using OpenHands LocalWorkspace."""
        self._cancel_flags[request.run_id] = False
        worktree_path = Path(request.worktree_identifier).expanduser().resolve()

        if not worktree_path.is_dir():
            return CodingExecutionResult(
                status="failed",
                executor="openhands",
                worktree_identifier=request.worktree_identifier,
                summary=f"Worktree directory '{worktree_path}' tidak ditemukan.",
                error_category="workspace_error",
            )

        # Safety check: Verify worktree is NOT the main project
        if self.settings.default_project_path:
            main_resolved = self.settings.default_project_path.resolve()
            if worktree_path == main_resolved:
                return CodingExecutionResult(
                    status="scope_violation",
                    executor="openhands",
                    worktree_identifier=request.worktree_identifier,
                    scope_violations=[str(worktree_path)],
                    summary="SECURITY ERROR: Workspace tidak boleh sama dengan project utama.",
                    error_category="security_violation",
                )

        prep = self.prepare_execution(request)
        instruction = prep["instruction"]

        # Run OpenHands Conversation
        iterations_run = 0
        timed_out = False
        execution_error: str | None = None
        error_category: str | None = None
        requested_files: list[str] = []

        start_time = time.time()
        timeout_seconds = min(request.timeout_seconds, OPENHANDS_EXECUTION_TIMEOUT_SECONDS)
        max_iterations = min(request.max_iterations, OPENHANDS_MAX_ITERATIONS)

        # Log initial progress
        self._log_progress(
            executor="openhands",
            iteration=0,
            elapsed_time=0.0,
            action_category="initialization",
            status="starting",
        )

        conv = None
        try:
            try:
                import openhands.tools as oh_tools
                oh_tools.register_default_tools()
            except Exception:
                pass

            from openhands.sdk import Agent, Conversation, LLM, LocalWorkspace, Tool

            # Setup LocalWorkspace pointing strictly to worktree (NO DOCKER)
            workspace = LocalWorkspace(working_dir=worktree_path)

            # Setup LLM configuration from environment (API key passed to LLM config, NEVER prompt)
            model_name = self.settings.llm_model
            if not any(model_name.startswith(p) for p in ["openai/", "anthropic/", "azure/", "ollama/"]):
                model_name = f"openai/{model_name}"

            llm_kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": self.settings.llm_api_key,
                "temperature": 0.0,
            }
            if self.settings.llm_base_url:
                llm_kwargs["base_url"] = self.settings.llm_base_url

            llm = LLM(**llm_kwargs)
            try:
                tools_list = [Tool(name="terminal"), Tool(name="file_editor")]
                agent = Agent(llm=llm, tools=tools_list)
            except Exception:
                agent = Agent(llm=llm)

            conv = Conversation(agent=agent, workspace=workspace, visualizer=None)
            if hasattr(conv, "max_iteration_per_run"):
                conv.max_iteration_per_run = max_iterations

            self._active_conversations[request.run_id] = conv

            # Send task instruction
            conv.send_message(instruction)

            # Run in worker thread bounded by timeout
            def _runner():
                try:
                    conv.run()
                except Exception as ex:
                    nonlocal execution_error, error_category
                    execution_error = str(ex)
                    error_category = type(ex).__name__

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()

            # Monitor progress with safe timeout loop
            last_logged_iter = -1
            while thread.is_alive():
                elapsed = time.time() - start_time
                if self._cancel_flags.get(request.run_id, False):
                    self._safely_stop_conversation(conv)
                    break

                if elapsed > timeout_seconds:
                    timed_out = True
                    self._safely_stop_conversation(conv)
                    break

                # Count iterations from conversation events if available
                action_category = "execution"
                if hasattr(conv, "state") and hasattr(conv.state, "events"):
                    agent_events = [ev for ev in conv.state.events if getattr(ev, "source", "") == "agent"]
                    iterations_run = len(agent_events)
                    if agent_events:
                        last_ev = agent_events[-1]
                        if hasattr(last_ev, "tool_name") and getattr(last_ev, "tool_name"):
                            action_category = getattr(last_ev, "tool_name")
                        elif hasattr(last_ev, "action"):
                            act = getattr(last_ev, "action")
                            action_category = getattr(act, "kind", getattr(act, "tool_name", "action"))
                        elif hasattr(last_ev, "thought") and getattr(last_ev, "thought"):
                            action_category = "thinking"
                        else:
                            action_category = type(last_ev).__name__

                if iterations_run != last_logged_iter:
                    self._log_progress(
                        executor="openhands",
                        iteration=iterations_run,
                        elapsed_time=round(elapsed, 1),
                        action_category=action_category,
                        status="running",
                    )
                    last_logged_iter = iterations_run

                time.sleep(2.0)

            thread.join(timeout=5.0)

            if hasattr(conv, "state") and hasattr(conv.state, "events"):
                iterations_run = sum(
                    1 for ev in conv.state.events if getattr(ev, "source", "") == "agent"
                )
                # Inspect messages for REQUEST_ADDITIONAL_FILES (from agent events only)
                for ev in conv.state.events:
                    if getattr(ev, "source", "") != "agent":
                        continue
                    txt = getattr(ev, "message", "") or ""
                    if not txt and hasattr(ev, "llm_message"):
                        llm_msg = getattr(ev, "llm_message")
                        if hasattr(llm_msg, "content"):
                            txt = str(llm_msg.content)
                    match = re.search(r"REQUEST_ADDITIONAL_FILES\s*:\s*([^\n\r]+)", txt)
                    if match:
                        raw_files = match.group(1).split(",")
                        for rf in raw_files:
                            clean_rf = rf.strip().strip("'\"`<> ")
                            if clean_rf and not clean_rf.startswith("<file") and clean_rf not in requested_files:
                                requested_files.append(clean_rf)

        except Exception as e:
            execution_error = str(e)
            error_category = type(e).__name__
        finally:
            self._active_conversations.pop(request.run_id, None)

        total_elapsed = round(time.time() - start_time, 1)

        # Collect result and check scope violations
        return self.collect_result(
            request=request,
            iterations=iterations_run,
            timed_out=timed_out,
            execution_error=execution_error,
            error_category=error_category,
            requested_additional_files=requested_files,
            elapsed_seconds=total_elapsed,
        )

    def collect_result(
        self,
        request: CodingExecutionRequest,
        iterations: int = 0,
        timed_out: bool = False,
        execution_error: str | None = None,
        error_category: str | None = None,
        requested_additional_files: list[str] | None = None,
        elapsed_seconds: float = 0.0,
    ) -> CodingExecutionResult:
        """Collect and verify result after execution against Git diff and approved scope."""
        worktree_path = Path(request.worktree_identifier).expanduser().resolve()
        requested_files = requested_additional_files or []

        # 1. Inspect actual Git diff
        changed_files: list[str] = []
        if worktree_path.is_dir():
            try:
                res_names = subprocess.run(
                    ["git", "diff", "--name-only"],
                    cwd=worktree_path,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                )
                if res_names.returncode == 0:
                    changed_files = [
                        line.strip() for line in res_names.stdout.splitlines() if line.strip()
                    ]

                # Also inspect untracked files
                res_untracked = subprocess.run(
                    ["git", "status", "--porcelain", "-uall"],
                    cwd=worktree_path,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                )
                if res_untracked.returncode == 0:
                    for line in res_untracked.stdout.splitlines():
                        line = line.strip()
                        if line.startswith("??"):
                            parts = line.split(maxsplit=1)
                            if len(parts) == 2:
                                untracked_path = parts[1].strip()
                                if untracked_path not in changed_files:
                                    changed_files.append(untracked_path)
            except Exception as e:
                execution_error = f"Gagal membaca status git worktree: {e}"

        # 2. Check changed files against approved scope
        approved_set = {str(f).replace("\\", "/").strip().lstrip("/") for f in request.approved_files}
        scope_violations: list[str] = []
        for cf in changed_files:
            clean_cf = cf.replace("\\", "/").strip().lstrip("/")
            if clean_cf not in approved_set:
                scope_violations.append(clean_cf)

        # 3. Run approved validation commands
        validation_results: list[dict[str, Any]] = []
        all_passed = True
        if not timed_out and not scope_violations and not requested_files and worktree_path.is_dir():
            for cmd in request.validation_commands:
                exe = cmd.get("executable", "")
                args = cmd.get("arguments", [])
                rel_cwd = cmd.get("working_directory", ".")
                target_cwd = (worktree_path / rel_cwd).resolve()

                if not exe:
                    continue

                try:
                    res_cmd = subprocess.run(
                        [exe] + args,
                        cwd=target_cwd,
                        shell=False,
                        capture_output=True,
                        text=True,
                        timeout=120.0,
                    )
                    passed = res_cmd.returncode == 0
                    if not passed:
                        all_passed = False
                    validation_results.append({
                        "executable": exe,
                        "arguments": args,
                        "working_directory": rel_cwd,
                        "exit_code": res_cmd.returncode,
                        "status": "passed" if passed else "failed",
                    })
                except Exception as ex:
                    all_passed = False
                    validation_results.append({
                        "executable": exe,
                        "arguments": args,
                        "working_directory": rel_cwd,
                        "exit_code": -1,
                        "status": "failed",
                        "error": str(ex),
                    })

        # 4. Determine final status
        if timed_out:
            final_status = "timed_out"
            summary = f"OpenHands eksekusi melebihi batas waktu ({OPENHANDS_EXECUTION_TIMEOUT_SECONDS} detik)."
        elif scope_violations:
            final_status = "scope_violation"
            summary = (
                f"OpenHands memodifikasi file tak terduga di luar approved scope ({len(scope_violations)} file): "
                f"{', '.join(scope_violations)}."
            )
        elif requested_files:
            final_status = "proposal_revision_required"
            summary = (
                f"OpenHands memerlukan file tambahan ({len(requested_files)} file): "
                f"{', '.join(requested_files)}. Diperlukan revisi proposal."
            )
        elif execution_error or not all_passed:
            final_status = "failed"
            summary = execution_error or "Validation command gagal setelah eksekusi OpenHands."
        elif len(changed_files) == 0:
            final_status = "execution_failed"
            summary = "Tidak ada file yang diubah (zero changed files). Eksekusi gagal."
        else:
            final_status = "completed"
            summary = (
                f"OpenHands berhasil menyelesaikan implementasi pada {len(changed_files)} file "
                f"dalam {iterations} iterasi ({elapsed_seconds}s)."
            )

        self._log_progress(
            executor="openhands",
            iteration=iterations,
            elapsed_time=elapsed_seconds,
            action_category="completion",
            status=final_status,
        )

        return CodingExecutionResult(
            status=final_status,
            executor="openhands",
            worktree_identifier=request.worktree_identifier,
            changed_files=changed_files,
            validation_results=validation_results,
            iterations=iterations,
            summary=summary,
            requested_additional_files=requested_files,
            scope_violations=scope_violations,
            timed_out=timed_out,
            error_category=error_category,
        )

    def cancel(self, run_id: str) -> None:
        """Cancel running execution safely."""
        self._cancel_flags[run_id] = True
        conv = self._active_conversations.get(run_id)
        if conv:
            self._safely_stop_conversation(conv)

    def cleanup(self, request: CodingExecutionRequest) -> None:
        """Clean up executor resources safely without deleting unreviewed worktree changes."""
        self._active_conversations.pop(request.run_id, None)
        self._cancel_flags.pop(request.run_id, None)

    def _safely_stop_conversation(self, conv: Any) -> None:
        """Stop OpenHands conversation safely without hanging."""
        try:
            if hasattr(conv, "pause"):
                conv.pause()
            if hasattr(conv, "_cancel_token") and hasattr(conv._cancel_token, "cancel"):
                conv._cancel_token.cancel()
        except Exception:
            pass

    def _log_progress(
        self,
        executor: str,
        iteration: int,
        elapsed_time: float,
        action_category: str,
        status: str,
    ) -> None:
        """Log progress conforming strictly to privacy and safety constraints.

        Never prints raw prompt, raw response, API keys, or secrets.
        """
        print(
            f"[{executor.upper()}] iter={iteration} "
            f"elapsed={elapsed_time}s "
            f"action={action_category} "
            f"status={status}"
        )
