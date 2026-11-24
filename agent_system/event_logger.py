import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.error_reporting import sanitize_text


class ProgramEventLogger:
    """Append-only sanitized event logger for program observability and audit."""

    def __init__(self, project_root: Path, program_id: str):
        self.project_root = Path(project_root).expanduser().resolve()
        self.program_id = program_id
        self.events_dir = self.project_root / ".agent" / "artifacts" / "programs"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.events_dir / f"events_{program_id}.jsonl"
        self._sequence = self._init_sequence()

    def _init_sequence(self) -> int:
        if not self.events_file.is_file():
            return 0
        count = 0
        try:
            with open(self.events_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception:
            pass
        return count

    def log(
        self,
        event_type: str,
        safe_summary: str,
        task_id: str | None = None,
        run_id: str | None = None,
        previous_status: str | None = None,
        new_status: str | None = None,
        state_revision: int = 1,
        program_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a sanitized event to events_<program_id>.jsonl with fsync."""
        self._sequence += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        actual_prog_id = program_id or self.program_id

        clean_summary = sanitize_text(safe_summary)
        clean_extra = {}
        if extra:
            for k, v in extra.items():
                if isinstance(v, str):
                    clean_extra[k] = sanitize_text(v)
                elif isinstance(v, (int, float, bool)):
                    clean_extra[k] = v
                elif isinstance(v, list):
                    clean_extra[k] = [sanitize_text(str(x)) for x in v]

        event = {
            "sequence": self._sequence,
            "event_id": event_id,
            "event_type": event_type,
            "program_id": actual_prog_id,
            "task_id": task_id,
            "run_id": run_id,
            "timestamp": now_iso,
            "previous_status": previous_status,
            "new_status": new_status,
            "safe_summary": clean_summary,
            "state_revision": state_revision,
        }
        if clean_extra:
            event["metadata"] = clean_extra

        line = json.dumps(event, ensure_ascii=False) + "\n"

        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        return event

    def get_events(self) -> list[dict[str, Any]]:
        """Read all events from file."""
        if not self.events_file.is_file():
            return []
        events = []
        with open(self.events_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        return events
