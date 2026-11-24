import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_system.path_safety import resolve_project_path
from agent_system.policy_generator import generate_default_policy
from agent_system.project_detector import detect_project
from agent_system.project_protocol import (
    ProjectManifest,
    ProjectPolicy,
    ProjectWorkflow,
)
from agent_system.workflow_generator import generate_default_workflow


RECOMMENDED_GITIGNORE = """# Recommended gitignore entries for .agent runtime artifacts
# Note: Do NOT ignore the entire .agent/ directory because versioned configuration files
# (.agent/manifest.yaml, .agent/policy.yaml, .agent/workflow.yaml) must remain tracked.
artifacts/
runtime/
cache/
tmp/
logs/
"""

GITIGNORE_USER_NOTICE = (
    "Recommended .gitignore entries for .agent runtime artifacts:\n"
    "  .agent/artifacts/\n"
    "  .agent/runtime/\n"
    "  .agent/cache/\n"
    "  .agent/tmp/\n"
    "  .agent/logs/\n"
    "Note: Do NOT ignore the entire .agent/ directory because versioned configuration files "
    "(.agent/manifest.yaml, .agent/policy.yaml, .agent/workflow.yaml) must remain tracked in git.\n"
    "Root .gitignore was NOT modified automatically. You may manually add these rules to your root .gitignore if desired."
)


@dataclass
class InitializationPlan:
    project_root: Path
    directories_to_create: list[str]
    files_to_create: list[str]
    existing_files: list[str]
    warnings: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    gitignore_recommendation: str = GITIGNORE_USER_NOTICE
    manifest: ProjectManifest | None = None
    policy: ProjectPolicy | None = None
    workflow: ProjectWorkflow | None = None


def build_initialization_plan(project_root: Path) -> InitializationPlan:
    """Build the initialization plan without touching disk."""
    root = project_root.expanduser().resolve()

    # Detect project draft
    manifest = detect_project(root)
    policy = generate_default_policy(manifest)
    workflow = generate_default_workflow()

    dirs = [
        ".agent",
        ".agent/artifacts",
        ".agent/artifacts/plans",
        ".agent/artifacts/reports",
        ".agent/artifacts/decisions",
        ".agent/runtime",
        ".agent/cache",
        ".agent/tmp",
    ]

    files = [
        ".agent/manifest.yaml",
        ".agent/policy.yaml",
        ".agent/workflow.yaml",
        ".agent/.gitignore.recommended",
    ]

    existing: list[str] = []
    conflicts: list[str] = []
    warnings: list[str] = []

    for f_rel in files:
        target = root / f_rel
        if target.exists():
            existing.append(f_rel)
            conflicts.append(f"File already exists: {f_rel}")

    if existing:
        warnings.append(
            f"{len(existing)} configuration file(s) already exist. "
            "Use force=True to overwrite with backup."
        )

    return InitializationPlan(
        project_root=root,
        directories_to_create=dirs,
        files_to_create=files,
        existing_files=existing,
        warnings=warnings,
        conflicts=conflicts,
        gitignore_recommendation=GITIGNORE_USER_NOTICE,
        manifest=manifest,
        policy=policy,
        workflow=workflow,
    )



def _dump_yaml(data: dict[str, Any]) -> str:
    """Dump dictionary to deterministic, safe YAML."""
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    )


def initialize_project(
    project_root: Path,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Initialize portable project protocol in project_root.
    
    If apply=False: Dry run only, does not modify any files.
    If apply=True: Writes only to .agent/**, never modifies files outside .agent/.
    """
    root = project_root.expanduser().resolve()
    plan = build_initialization_plan(root)

    if not apply:
        return {
            "status": "dry_run",
            "applied": False,
            "project_root": root,
            "directories_to_create": plan.directories_to_create,
            "files_to_create": plan.files_to_create,
            "existing_files": plan.existing_files,
            "warnings": plan.warnings,
            "conflicts": plan.conflicts,
            "gitignore_recommendation": plan.gitignore_recommendation,
            "manifest": plan.manifest,
            "policy": plan.policy,
            "workflow": plan.workflow,
        }

    # apply=True: Execute initialization
    agent_dir = root / ".agent"
    if agent_dir.is_symlink():
        raise ValueError("Security violation: .agent cannot be a symlink.")

    # Create directories strictly within root
    created_dirs: list[str] = []
    for d_rel in plan.directories_to_create:
        target_dir = resolve_project_path(root, d_rel)
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(d_rel)

    # Prepare file contents
    manifest_data = plan.manifest.model_dump(mode="python", exclude_none=True)
    policy_data = plan.policy.model_dump(mode="python", exclude_none=True)
    # Ensure enum values are serialized as str
    if "mode" in policy_data and hasattr(policy_data["mode"], "value"):
        policy_data["mode"] = policy_data["mode"].value

    workflow_data = plan.workflow.model_dump(mode="python", exclude_none=True)

    files_content = {
        ".agent/manifest.yaml": _dump_yaml(manifest_data),
        ".agent/policy.yaml": _dump_yaml(policy_data),
        ".agent/workflow.yaml": _dump_yaml(workflow_data),
        ".agent/.gitignore.recommended": RECOMMENDED_GITIGNORE,
    }

    created_files: list[str] = []
    backed_up_files: list[str] = []
    skipped_files: list[str] = []

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tmp_dir = resolve_project_path(root, ".agent/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in files_content.items():
        file_path = resolve_project_path(root, rel_path)

        if file_path.exists():
            if not force:
                skipped_files.append(rel_path)
                continue
            else:
                # Backup existing file before overwrite
                backup_name = f"backup_{file_path.name}_{timestamp}"
                backup_dest = tmp_dir / backup_name
                shutil.copy2(file_path, backup_dest)
                backed_up_files.append(f"{rel_path} -> .agent/tmp/{backup_name}")

        file_path.write_text(content, encoding="utf-8")
        created_files.append(rel_path)

    return {
        "status": "applied",
        "applied": True,
        "project_root": root,
        "created_directories": created_dirs,
        "created_files": created_files,
        "backed_up_files": backed_up_files,
        "skipped_files": skipped_files,
        "gitignore_recommendation": plan.gitignore_recommendation,
        "manifest": plan.manifest,
        "policy": plan.policy,
        "workflow": plan.workflow,
    }

    for f_rel in files:
        target = root / f_rel
        if target.exists():
            existing.append(f_rel)
            conflicts.append(f"File already exists: {f_rel}")

    if existing:
        warnings.append(
            f"{len(existing)} configuration file(s) already exist. "
            "Use force=True to overwrite with backup."
        )

    return InitializationPlan(
        project_root=root,
        directories_to_create=dirs,
        files_to_create=files,
        existing_files=existing,
        warnings=warnings,
        conflicts=conflicts,
        gitignore_recommendation=GITIGNORE_USER_NOTICE,
        manifest=manifest,
        policy=policy,
        workflow=workflow,
    )
