import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any

from agent_system.path_safety import resolve_project_path, validate_relative_pattern

SECRET_FILE_PATTERNS = [
    re.compile(r"(^|/)\.env(\..+)?$", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"(^|/)credentials\.[a-z0-9]+$", re.IGNORECASE),
    re.compile(r"(^|/)auth\.json$", re.IGNORECASE),
    re.compile(r"(^|/)id_rsa(\..+)?$", re.IGNORECASE),
]

PROTOCOL_FILES = {
    ".agent/manifest.yaml",
    ".agent/policy.yaml",
    ".agent/workflow.yaml",
}


def is_secret_path(rel_path: str) -> bool:
    clean = rel_path.replace("\\", "/").strip("/")
    return any(p.search(clean) for p in SECRET_FILE_PATTERNS)


def compute_file_hash(path: Path) -> str | None:
    """Compute sha256 hash of a file if it exists."""
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class UnapprovedFileError(ValueError):
    """Raised when an attempt is made to write an unapproved file."""
    pass


class ControlledFileWriter:
    """Safe, controlled file writer that only writes approved files atomically with backups."""

    def __init__(
        self,
        project_root: Path,
        approved_files: list[str],
        run_id: str,
    ):
        self.project_root = project_root.expanduser().resolve()
        self.approved_files = set(approved_files)
        self.run_id = run_id
        self.backup_dir = self.project_root / ".agent" / "runtime" / "backups" / run_id
        self.tmp_dir = self.project_root / ".agent" / "tmp"

        self.file_hashes_before: dict[str, str] = {}
        self.file_hashes_after: dict[str, str] = {}
        self.written_files: list[str] = []

    def write_file(self, relative_path: str, content: str) -> Path:
        """Write content to an approved file safely and atomically."""
        # 1 & 2. Relative pattern & traversal validation
        clean_path = validate_relative_pattern(relative_path)

        # 3. Protocol files rejection
        if clean_path in PROTOCOL_FILES or clean_path.startswith(".agent/manifest") or clean_path.startswith(".agent/policy") or clean_path.startswith(".agent/workflow"):
            raise ValueError(f"Dilarang mengubah file protokol: '{clean_path}'")

        # 4. Secret files rejection
        if is_secret_path(clean_path):
            raise ValueError(f"Dilarang mengubah file credential/secret: '{clean_path}'")

        # 5. Approved files check
        if clean_path not in self.approved_files:
            raise UnapprovedFileError(
                f"Penulisan ditolak: file '{clean_path}' tidak tercantum dalam approved_files {list(self.approved_files)}."
            )

        # 6. Resolve path inside project root
        target_path = resolve_project_path(self.project_root, clean_path)
        try:
            target_path.relative_to(self.project_root)
        except ValueError:
            raise ValueError(f"Target path keluar dari project root: '{clean_path}'")

        # 7. Symlink check
        if target_path.is_symlink():
            resolved = target_path.resolve()
            try:
                resolved.relative_to(self.project_root)
            except ValueError:
                raise ValueError(f"Target file adalah symlink yang keluar dari project root: '{clean_path}'")

        # 8. Record hash before and create backup
        existing_mode = None
        has_crlf = False
        if target_path.is_file():
            hash_before = compute_file_hash(target_path)
            if hash_before:
                self.file_hashes_before[clean_path] = hash_before

            existing_mode = target_path.stat().st_mode

            # Detect line endings
            with open(target_path, "rb") as f:
                raw_sample = f.read(4096)
                if b"\r\n" in raw_sample:
                    has_crlf = True

            # Backup
            backup_file = self.backup_dir / clean_path
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, backup_file)
        else:
            self.file_hashes_before[clean_path] = ""

        # Normalize or preserve line endings
        normalized_content = content
        if has_crlf and "\r\n" not in normalized_content:
            normalized_content = normalized_content.replace("\n", "\r\n")

        # 9. Atomic write via .agent/tmp
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_file = self.tmp_dir / f"tmp_write_{clean_path.replace('/', '_')}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(normalized_content)
            f.flush()
            os.fsync(f.fileno())

        if existing_mode is not None:
            try:
                os.chmod(tmp_file, existing_mode)
            except Exception:
                pass

        os.replace(tmp_file, target_path)

        # 10. Record hash after
        hash_after = compute_file_hash(target_path)
        if hash_after:
            self.file_hashes_after[clean_path] = hash_after
        self.written_files.append(clean_path)

        return target_path
