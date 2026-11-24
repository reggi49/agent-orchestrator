import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.atomic_state import atomic_write_json


class ProgramLockError(Exception):
    """Base exception for program lock errors."""
    pass


class ProgramLockActiveError(ProgramLockError):
    """Raised when program lock is currently held by an active running process."""
    pass


def get_process_start_identifier(pid: int) -> str | None:
    """Get process start time / identifier using ps -p <pid> -o lstart= for PID reuse detection."""
    if pid <= 0:
        return None
    try:
        res = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID exists and can receive signals."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but belongs to different user
        return True
    except OSError:
        return False


class ProgramLock:
    """Single-writer lock manager per program with heartbeat and stale lock recovery."""

    def __init__(self, project_root: Path, program_id: str, project_id: str = "project"):
        self.project_root = Path(project_root).expanduser().resolve()
        self.program_id = program_id
        self.project_id = project_id
        self.locks_dir = self.project_root / ".agent" / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.locks_dir / f"program_{program_id}.lock"
        self.owned = False
        self._current_pid = os.getpid()
        self._hostname = platform.node() or "localhost"
        self._process_start_id = get_process_start_identifier(self._current_pid) or str(time.time())

    def acquire(
        self,
        current_task_id: str | None = None,
        active_run_id: str | None = None,
        event_logger: Any = None,
    ) -> dict[str, Any]:
        """Atomically acquire single-writer program lock or recover stale lock safely."""
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = {
            "program_id": self.program_id,
            "project_id": self.project_id,
            "process_id": self._current_pid,
            "hostname": self._hostname,
            "created_at": now_iso,
            "heartbeat_at": now_iso,
            "current_task_id": current_task_id,
            "active_run_id": active_run_id,
            "process_start_identifier": self._process_start_id,
        }

        # Try atomic creation first
        try:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            fd = os.open(str(self.lock_file), flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            self.owned = True
            if event_logger and hasattr(event_logger, "log"):
                event_logger.log(
                    event_type="checkpoint_written",
                    safe_summary=f"Acquired program lock for PID {self._current_pid}.",
                    program_id=self.program_id,
                )
            return metadata
        except FileExistsError:
            pass

        # Lock file exists: check if owner is active or stale
        existing_meta = self.inspect()
        if existing_meta.get("is_active"):
            owner_pid = existing_meta.get("process_id")
            owner_host = existing_meta.get("hostname")
            raise ProgramLockActiveError(
                f"Program '{self.program_id}' sedang dijalankan oleh proses aktif "
                f"PID {owner_pid} pada host '{owner_host}'."
            )

        # Stale lock recovery
        # 1. Log stale lock recovery event
        if event_logger and hasattr(event_logger, "log"):
            event_logger.log(
                event_type="recovery_started",
                safe_summary=(
                    f"Recovering stale lock from dead PID {existing_meta.get('process_id')}. "
                    f"New owner PID {self._current_pid} taking lock."
                ),
                program_id=self.program_id,
            )

        # 2. Archive old stale lock
        try:
            archive_path = self.locks_dir / f"stale_program_{self.program_id}_{int(time.time())}.bak"
            if self.lock_file.is_file():
                self.lock_file.replace(archive_path)
        except Exception:
            pass

        # 3. Write new lock metadata atomically
        atomic_write_json(self.lock_file, metadata, backup=False, monotonic_revision=False)
        self.owned = True

        if event_logger and hasattr(event_logger, "log"):
            event_logger.log(
                event_type="recovery_completed",
                safe_summary=f"Stale lock successfully replaced by PID {self._current_pid}.",
                program_id=self.program_id,
            )

        return metadata

    def update_heartbeat(
        self,
        current_task_id: str | None = None,
        active_run_id: str | None = None,
    ) -> None:
        """Update heartbeat timestamp and active task on owned lock."""
        if not self.owned or not self.lock_file.is_file():
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            data = json.loads(self.lock_file.read_text(encoding="utf-8"))
            if data.get("process_id") == self._current_pid:
                data["heartbeat_at"] = now_iso
                if current_task_id is not None:
                    data["current_task_id"] = current_task_id
                if active_run_id is not None:
                    data["active_run_id"] = active_run_id
                atomic_write_json(self.lock_file, data, backup=False, monotonic_revision=False)
        except Exception:
            pass

    def release(self, event_logger: Any = None) -> None:
        """Release program lock if owned by this process."""
        if not self.lock_file.is_file():
            self.owned = False
            return
        try:
            data = json.loads(self.lock_file.read_text(encoding="utf-8"))
            if data.get("process_id") == self._current_pid:
                self.lock_file.unlink(missing_ok=True)
                self.owned = False
                if event_logger and hasattr(event_logger, "log"):
                    event_logger.log(
                        event_type="checkpoint_written",
                        safe_summary=f"Released program lock for PID {self._current_pid}.",
                        program_id=self.program_id,
                    )
        except Exception:
            pass

    def inspect(self) -> dict[str, Any]:
        """Inspect lock file without modifying it."""
        if not self.lock_file.is_file():
            return {"exists": False, "is_active": False, "is_stale": False}

        try:
            data = json.loads(self.lock_file.read_text(encoding="utf-8"))
        except Exception:
            return {"exists": True, "is_active": False, "is_stale": True, "error": "Unreadable lock file"}

        pid = data.get("process_id", -1)
        host = data.get("hostname", "")
        recorded_start_id = data.get("process_start_identifier")

        is_same_host = (host == self._hostname)
        alive = is_pid_alive(pid) if is_same_host else False

        # PID reuse verification if start identifier is available
        is_same_process = False
        if alive and is_same_host and recorded_start_id:
            current_start_id = get_process_start_identifier(pid)
            if current_start_id and current_start_id == recorded_start_id:
                is_same_process = True
            elif not current_start_id:
                # If ps couldn't fetch start id, assume alive if PID matches
                is_same_process = True
        elif alive and is_same_host:
            is_same_process = True

        is_active = alive and is_same_process
        is_stale = not is_active

        return {
            "exists": True,
            "is_active": is_active,
            "is_stale": is_stale,
            "process_id": pid,
            "hostname": host,
            "created_at": data.get("created_at"),
            "heartbeat_at": data.get("heartbeat_at"),
            "current_task_id": data.get("current_task_id"),
            "active_run_id": data.get("active_run_id"),
            "process_start_identifier": recorded_start_id,
        }
