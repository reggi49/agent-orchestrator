from pathlib import Path

import yaml

from agent_system.path_safety import resolve_project_path
from agent_system.project_protocol import (
    ProjectManifest,
    ProjectPolicy,
    ProjectWorkflow,
    ValidatedProject,
)


class ProjectNotFoundError(Exception):
    """Raised when .agent/manifest.yaml cannot be found."""
    pass


def find_project_root(start_path: Path) -> Path:
    """Find project root by searching upward for .agent/manifest.yaml."""
    current = start_path.expanduser().resolve()
    
    # If start_path is a file, start from its parent
    if current.is_file():
        current = current.parent

    # Search upward for .agent/manifest.yaml
    while True:
        candidate = current / ".agent" / "manifest.yaml"
        if candidate.is_file():
            return current

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            raise ProjectNotFoundError(
                f"Could not find .agent/manifest.yaml starting from {start_path}. "
                "Run 'init' command first to initialize the project."
            )
        current = parent


def load_project(project_root: Path) -> ValidatedProject:
    """Load and validate project protocol configuration.
    
    Reads manifest.yaml, policy.yaml, workflow.yaml and validates:
    - All paths are relative
    - No path traversal
    - Component roots exist within project
    - Artifact storage is within .agent/
    - No secrets in configuration
    """
    root = project_root.expanduser().resolve()

    manifest_path = root / ".agent" / "manifest.yaml"
    policy_path = root / ".agent" / "policy.yaml"
    workflow_path = root / ".agent" / "workflow.yaml"

    # Check all required files exist
    if not manifest_path.is_file():
        raise ProjectNotFoundError(
            f"Missing manifest.yaml at {manifest_path}. "
            "Run 'init --apply' to initialize the project."
        )
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Missing policy.yaml at {policy_path}. "
            "Run 'init --apply' to initialize the project."
        )
    if not workflow_path.is_file():
        raise FileNotFoundError(
            f"Missing workflow.yaml at {workflow_path}. "
            "Run 'init --apply' to initialize the project."
        )

    # Load YAML safely
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = yaml.safe_load(f)
    
    with open(policy_path, "r", encoding="utf-8") as f:
        policy_data = yaml.safe_load(f)
    
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_data = yaml.safe_load(f)

    # Validate with Pydantic (includes path validation)
    manifest = ProjectManifest.model_validate(manifest_data)
    policy = ProjectPolicy.model_validate(policy_data)
    workflow = ProjectWorkflow.model_validate(workflow_data)

    # Additional validation: component roots must exist within project
    for comp_name, component in manifest.components.items():
        comp_root_path = resolve_project_path(root, component.root)
        if not comp_root_path.exists():
            raise ValueError(
                f"Component '{comp_name}' root does not exist: {component.root}"
            )

        # Verify manifest file exists if specified
        if component.manifest:
            manifest_file = resolve_project_path(root, component.manifest)
            if not manifest_file.is_file():
                raise ValueError(
                    f"Component '{comp_name}' manifest file not found: {component.manifest}"
                )

    # Validate artifact storage root
    artifact_root = resolve_project_path(root, manifest.artifact_storage.root)
    # Already validated to be within .agent/ by Pydantic validator

    return ValidatedProject(
        root=root,
        manifest=manifest,
        policy=policy,
        workflow=workflow,
    )
