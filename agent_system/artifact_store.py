import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.atomic_state import atomic_write_json, read_json_with_backup_recovery
from agent_system.error_reporting import sanitize_text
from agent_system.path_safety import resolve_project_path, validate_relative_pattern
from agent_system.project_protocol import ProjectManifest
from agent_system.source_revision import get_source_revision


SENSITIVE_KEY_PATTERNS = [
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
]


def sanitize_artifact_payload(obj: Any) -> Any:
    """Sanitize data structure before writing to artifact.
    
    - Redacts sensitive keys
    - Removes full prompt dumps
    - Converts Path objects to relative strings
    """
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            key_str = str(k)
            # Check sensitive key
            if any(p.search(key_str) for p in SENSITIVE_KEY_PATTERNS):
                sanitized[key_str] = "[REDACTED]"
            elif key_str in {"worktree_path", "target_worktree_path", "absolute_worktree_path"}:
                sanitized["worktree_descriptor"] = Path(str(v)).name
            elif key_str.lower() in {
                "prompt",
                "full_prompt",
                "system_prompt",
                "raw_prompt",
                "full_diff",
                "raw_diff",
                "diff",
                "patch",
                "source_code",
                "full_source",
                "raw_response",
                "chain_of_thought",
            }:
                sanitized[key_str] = "[OMITTED_FOR_SAFETY]"
            else:
                sanitized[key_str] = sanitize_artifact_payload(v)
        return sanitized
    elif isinstance(obj, list):
        return [sanitize_artifact_payload(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, str):
        if re.match(r"^(?:/Users/|/home/|[A-Za-z]:[\\/])", obj):
            return Path(obj).name
        return sanitize_text(obj)
    return obj

class ProjectArtifactStore:
    """Project-owned artifact store.
    
    Writes artifacts exclusively into the target project's artifact directory,
    enforcing path safety, atomic writes, and metadata tracking.
    """

    def __init__(self, project_root: Path, manifest: ProjectManifest):
        self.project_root = project_root.expanduser().resolve()
        self.manifest = manifest
        
        self.artifact_root = resolve_project_path(self.project_root, manifest.artifact_storage.root)
        cats = manifest.artifact_storage.categories
        self.plans_dir = resolve_project_path(
            self.project_root,
            f"{manifest.artifact_storage.root}/{cats.plans}",
        )
        self.reports_dir = resolve_project_path(
            self.project_root,
            f"{manifest.artifact_storage.root}/{cats.reports}",
        )
        self.decisions_dir = resolve_project_path(
            self.project_root,
            f"{manifest.artifact_storage.root}/{cats.decisions}",
        )
        self.proposals_dir = resolve_project_path(
            self.project_root,
            f"{manifest.artifact_storage.root}/{cats.proposals}",
        )
        self.implementations_dir = resolve_project_path(
            self.project_root,
            f"{manifest.artifact_storage.root}/{cats.implementations}",
        )
        self.tmp_dir = resolve_project_path(self.project_root, ".agent/tmp")

    def _prepare_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Wrap payload with protocol metadata."""
        validate_relative_pattern(artifact_id)
        sanitized_data = sanitize_artifact_payload(data)

        rev = get_source_revision(self.project_root)
        source_rev_dict = {
            "commit_head": rev.commit_head,
            "is_dirty": rev.is_dirty,
        }
        if rev.warning:
            source_rev_dict["warning"] = rev.warning

        evidence_paths = sanitized_data.pop("evidence_paths", [])
        status = sanitized_data.pop("status", "completed")

        return {
            "schema_version": "1.0",
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "project_id": self.manifest.project.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "agent-orchestrator",
            "source_revision": source_rev_dict,
            "status": status,
            "evidence_paths": evidence_paths,
            "result": sanitized_data,
        }

    def _atomic_write(self, target_dir: Path, filename: str, content_dict: dict[str, Any]) -> Path:
        """Atomic write using temporary file, fsync, monotonic revision, and backup."""
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        try:
            target_file.relative_to(self.artifact_root)
        except ValueError:
            raise ValueError(f"Target file escapes artifact root: {target_file}")

        return atomic_write_json(target_file, content_dict, backup=True, monotonic_revision=True)

    def write_plan(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write a plan artifact."""
        artifact = self._prepare_artifact(artifact_id, "plan", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(self.plans_dir, filename, artifact)

    def write_report(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write a report artifact."""
        artifact = self._prepare_artifact(artifact_id, "report", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(self.reports_dir, filename, artifact)

    def write_decision(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write a decision artifact."""
        artifact = self._prepare_artifact(artifact_id, "decision", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(self.decisions_dir, filename, artifact)

    def write_proposal(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write a proposal artifact."""
        artifact = self._prepare_artifact(artifact_id, "proposal", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(self.proposals_dir, filename, artifact)

    def write_implementation(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write an implementation artifact."""
        artifact = self._prepare_artifact(artifact_id, "implementation", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(self.implementations_dir, filename, artifact)

    def write_execution(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write an execution artifact."""
        executions_dir = resolve_project_path(
            self.project_root,
            f"{self.manifest.artifact_storage.root}/executions",
        )
        artifact = self._prepare_artifact(artifact_id, "execution", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(executions_dir, filename, artifact)

    def write_run(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write an orchestration run / checkpoint artifact."""
        runs_dir = resolve_project_path(
            self.project_root,
            f"{self.manifest.artifact_storage.root}/runs",
        )
        artifact = self._prepare_artifact(artifact_id, "run", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(runs_dir, filename, artifact)

    def write_program(self, artifact_id: str, data: dict[str, Any]) -> Path:
        """Write a program artifact."""
        programs_dir = resolve_project_path(
            self.project_root,
            f"{self.manifest.artifact_storage.root}/programs",
        )
        artifact = self._prepare_artifact(artifact_id, "program", data)
        filename = f"{artifact_id}.json" if not artifact_id.endswith(".json") else artifact_id
        return self._atomic_write(programs_dir, filename, artifact)

    def read_artifact(self, relative_path: str) -> dict[str, Any]:
        """Read artifact from artifact root with automatic backup recovery."""
        file_path = resolve_project_path(self.project_root, relative_path)
        try:
            file_path.relative_to(self.artifact_root)
        except ValueError:
            raise ValueError(f"Path is outside artifact root: {relative_path}")

        data, _ = read_json_with_backup_recovery(file_path)
        return data

