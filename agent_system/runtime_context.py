from dataclasses import dataclass
from pathlib import Path

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.project_protocol import ValidatedProject


@dataclass
class ProjectRuntimeContext:
    """Runtime context for portable project execution."""
    project_root: Path
    project: ValidatedProject
    artifact_store: ProjectArtifactStore
    run_id: str
