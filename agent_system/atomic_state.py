import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.error_reporting import sanitize_text


class StateCorruptionError(Exception):
    """Raised when both primary state and its backup are corrupt or invalid (fail-closed)."""
    pass


def compute_checksum(data: Any) -> str:
    """Compute deterministic SHA-256 checksum for a state payload."""
    if isinstance(data, dict):
        clean_copy = {k: v for k, v in data.items() if k not in ("checksum",)}
    else:
        clean_copy = data
    canonical_json = json.dumps(clean_copy, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _fsync_parent_dir(parent_dir: Path) -> None:
    """Fsync parent directory descriptor on POSIX platforms."""
    try:
        dir_fd = os.open(str(parent_dir), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def atomic_write_json(
    file_path: Path,
    data: dict[str, Any],
    backup: bool = True,
    monotonic_revision: bool = True,
) -> Path:
    """Atomically write JSON state with fsync, validation, monotonic revision, and backup.

    Flow:
    1. Serialize data to JSON and validate by deserializing back.
    2. Write to temporary file in the same directory.
    3. Flush and fsync file descriptor.
    4. Maintain last valid backup (.bak).
    5. Atomic replace (rename) into target file.
    6. Fsync parent directory.
    """
    file_path = file_path.resolve()
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Monotonic revision tracking
    if monotonic_revision and isinstance(data, dict):
        prev_rev = 0
        bak_file = parent_dir / f"{file_path.name}.bak"
        # Check current file
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as cur_f:
                    cur_d = json.load(cur_f)
                    if isinstance(cur_d, dict):
                        prev_rev = max(
                            prev_rev,
                            cur_d.get("state_revision", 0),
                            cur_d.get("result", {}).get("state_revision", 0),
                        )
            except Exception:
                pass
        # Check backup file
        if bak_file.is_file():
            try:
                with open(bak_file, "r", encoding="utf-8") as bak_f:
                    bak_d = json.load(bak_f)
                    if isinstance(bak_d, dict):
                        prev_rev = max(
                            prev_rev,
                            bak_d.get("state_revision", 0),
                            bak_d.get("result", {}).get("state_revision", 0),
                        )
            except Exception:
                pass

        current_rev = data.get("state_revision", 0)
        data["state_revision"] = max(prev_rev + 1, current_rev + 1)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "schema_version" not in data:
            data["schema_version"] = "1.0"
        data["checksum"] = compute_checksum(data)

    # 1. Serialize and validate deserialize
    serialized = json.dumps(data, indent=2, ensure_ascii=False)
    try:
        re_read = json.loads(serialized)
        if not isinstance(re_read, (dict, list)):
            raise ValueError("Serialized payload is not a valid JSON object or list.")
    except Exception as e:
        raise ValueError(f"State serialization validation failed before write: {e}")

    # 2. Write to temporary file in same directory
    tmp_file = parent_dir / f".tmp_{file_path.name}_{uuid.uuid4().hex}.json"

    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            # 3. Fsync file descriptor
            os.fsync(f.fileno())

        # 4. Backup last valid state if file_path exists and is valid
        if backup and file_path.is_file():
            bak_path = parent_dir / f"{file_path.name}.bak"
            try:
                with open(file_path, "r", encoding="utf-8") as existing_f:
                    existing_data = json.load(existing_f)
                if isinstance(existing_data, (dict, list)):
                    shutil.copyfile(file_path, bak_path)
                    with open(bak_path, "r", encoding="utf-8") as bf:
                        os.fsync(bf.fileno())
            except Exception:
                pass

        # 5. Atomic replace
        os.replace(tmp_file, file_path)

        # 6. Fsync parent directory
        _fsync_parent_dir(parent_dir)

    finally:
        # Clean up tmp file if still exists (in case of exception before replace)
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass

    return file_path


def read_json_with_backup_recovery(
    file_path: Path,
    event_logger: Any = None,
    program_id: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Read JSON state with automatic backup recovery if primary file is corrupt.

    Returns (data, recovered_from_backup).
    Raises StateCorruptionError if both primary and backup are corrupt or invalid (fail-closed).
    """
    file_path = file_path.resolve()
    parent_dir = file_path.parent
    bak_path = parent_dir / f"{file_path.name}.bak"

    primary_err: Exception | None = None
    data: Any = None

    if file_path.is_file():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    raise ValueError("File is empty (0 bytes).")
                data = json.loads(content)
                if isinstance(data, dict):
                    return data, False
                raise ValueError("JSON content is not a dict.")
        except Exception as e:
            primary_err = e
    else:
        primary_err = FileNotFoundError(f"State file '{file_path}' does not exist.")

    # Primary failed or was corrupt. Try backup.
    if bak_path.is_file():
        try:
            with open(bak_path, "r", encoding="utf-8") as bf:
                bak_content = bf.read()
                if not bak_content.strip():
                    raise ValueError("Backup file is empty (0 bytes).")
                bak_data = json.loads(bak_content)
                if isinstance(bak_data, dict):
                    # Restore primary file from backup
                    atomic_write_json(file_path, bak_data, backup=False, monotonic_revision=False)
                    if event_logger and hasattr(event_logger, "log"):
                        event_logger.log(
                            event_type="recovery_completed",
                            safe_summary=f"Recovered state '{file_path.name}' from valid backup.",
                            program_id=program_id or bak_data.get("program_id", "unknown"),
                        )
                    return bak_data, True
        except Exception as bak_e:
            raise StateCorruptionError(
                f"FAIL-CLOSED: Both primary state '{file_path.name}' (error: {primary_err}) "
                f"and backup '{bak_path.name}' (error: {bak_e}) are corrupt or invalid."
            )

    raise StateCorruptionError(
        f"FAIL-CLOSED: State file '{file_path.name}' is corrupt or missing ({primary_err}) and no valid backup exists."
    )
