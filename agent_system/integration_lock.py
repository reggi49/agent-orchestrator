import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.atomic_state import atomic_write_json
from agent_system.program_lock import get_process_start_identifier, is_pid_alive


class IntegrationLockError(Exception):
    """Base exception for integration lock errors."""
    pass


class IntegrationLockActiveError(IntegrationLockError):
    """Raised when integration lock is held by another active process."""
    pass


class TargetHeadChangedError(IntegrationLockError):
    """Raised when target branch HEAD changes after preflight or while acquiring lock."""
    pass


INTEGRATION_PHASES = [
    "acquired",
    "validating_worktree",
    "creating_source_commit",
    "preparing_target",
    "cherry_picking",
    "post_validation",
    "completed",
    "failed",
]


class IntegrationLock:
    """Lock manager for exclusive integration operations per target branch."""

    def __init__(
        self,
        project_root: Path,
        project_id: str,
        target_branch: str,
        program_id: str = "default",
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.project_id = project_id
        self.target_branch = target_branch
        self.program_id = program_id
        self.safe_branch = target_branch.replace("/", "_").replace("\\", "_")
        self.locks_dir = self.project_root / ".agent" / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.locks_dir / f"integration_{project_id}_{self.safe_branch}.lock"
        self.owned = False
        self._current_pid = os.getpid()
        self._hostname = platform.node() or "localhost"

    def acquire(
        self,
        task_id: str | None = None,
        run_id: str | None = None,
        execution_id: str | None = None,
        expected_target_head: str | None = None,
        event_logger: Any = None,
    ) -> dict[str, Any]:
        """Atomically acquire integration lock and verify target HEAD."""
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = {
            "program_id": self.program_id,
            "project_id": self.project_id,
            "task_id": task_id,
            "run_id": run_id,
            "execution_id": execution_id,
            "target_branch": self.target_branch,
            "expected_target_head": expected_target_head,
            "process_id": self._current_pid,
            "hostname": self._hostname,
            "created_at": now_iso,
            "heartbeat_at": now_iso,
            "integration_phase": "acquired",
        }

        # Check existing lock file
        if self.lock_file.is_file():
            insp = self.inspect()
            if insp.get("is_active"):
                owner_pid = insp.get("process_id")
                raise IntegrationLockActiveError(
                    f"Integrasi ke branch '{self.target_branch}' sedang berjalan "
                    f"oleh proses PID {owner_pid} pada fase '{insp.get('integration_phase')}'."
                )
            # Stale lock recovery
            if event_logger and hasattr(event_logger, "log"):
                event_logger.log(
                    event_type="recovery_started",
                    safe_summary=f"Recovering stale integration lock for branch '{self.target_branch}'.",
                    program_id=self.program_id,
                    task_id=task_id,
                )
            self.lock_file.unlink(missing_ok=True)

        # Atomic create
        atomic_write_json(self.lock_file, metadata, backup=False, monotonic_revision=False)
        self.owned = True

        # Re-verify target HEAD after acquiring lock
        if expected_target_head:
            res_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            actual_head = res_head.stdout.strip()
            if res_head.returncode == 0 and actual_head != expected_target_head:
                self.update_phase("failed")
                self.release()
                raise TargetHeadChangedError(
                    f"Target branch HEAD berubah setelah preflight! "
                    f"Expected: {expected_target_head[:8]}, Actual: {actual_head[:8]}. Re-validasi diperlukan."
                )

        if event_logger and hasattr(event_logger, "log"):
            event_logger.log(
                event_type="checkpoint_written",
                safe_summary=f"Acquired integration lock on branch '{self.target_branch}'.",
                program_id=self.program_id,
                task_id=task_id,
            )

        return metadata

    def update_phase(self, phase: str) -> None:
        """Update integration phase in lock file."""
        if not self.owned or not self.lock_file.is_file():
            return
        if phase not in INTEGRATION_PHASES:
            raise ValueError(f"Fase integrasi '{phase}' tidak valid.")

        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            data = json.loads(self.lock_file.read_text(encoding="utf-8"))
            if data.get("process_id") == self._current_pid:
                data["integration_phase"] = phase
                data["heartbeat_at"] = now_iso
                atomic_write_json(self.lock_file, data, backup=False, monotonic_revision=False)
        except Exception:
            pass

    def release(self) -> None:
        """Release integration lock."""
        if not self.lock_file.is_file():
            self.owned = False
            return
        try:
            data = json.loads(self.lock_file.read_text(encoding="utf-8"))
            if data.get("process_id") == self._current_pid:
                self.lock_file.unlink(missing_ok=True)
                self.owned = False
        except Exception:
            pass

    def inspect(self) -> dict[str, Any]:
        """Inspect integration lock state."""
        if not self.lock_file.is_file():
            return {"exists": False, "is_active": False, "is_stale": False}

        try:
            data = json.loads(self.lock_file.read_text(encoding="utf-8"))
        except Exception:
            return {"exists": True, "is_active": False, "is_stale": True}

        pid = data.get("process_id", -1)
        alive = is_pid_alive(pid) if data.get("hostname") == self._hostname else False
        is_active = alive
        is_stale = not alive

        return {
            "exists": True,
            "is_active": is_active,
            "is_stale": is_stale,
            "process_id": pid,
            "hostname": data.get("hostname"),
            "target_branch": data.get("target_branch"),
            "expected_target_head": data.get("expected_target_head"),
            "integration_phase": data.get("integration_phase"),
            "created_at": data.get("created_at"),
            "heartbeat_at": data.get("heartbeat_at"),
            "program_id": data.get("program_id"),
            "task_id": data.get("task_id"),
        }
