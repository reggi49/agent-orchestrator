from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_system.path_safety import validate_relative_pattern


class PolicyMode(str, Enum):
    READ_ONLY = "read_only"
    PROPOSAL = "proposal"
    ARTIFACT_WRITE = "artifact_write"
    BRANCH_WRITE = "branch_write"
    RELEASE = "release"


class ProjectIdentity(BaseModel):
    id: str
    name: str
    type: str

    def __repr__(self) -> str:
        return f"ProjectIdentity(id={self.id!r}, name={self.name!r}, type={self.type!r})"


class ContextSources(BaseModel):
    rules: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    handoffs: list[str] = Field(default_factory=list)
    reports: list[str] = Field(default_factory=list)

    @field_validator("rules", "requirements", "tasks", "handoffs", "reports", mode="after")
    @classmethod
    def validate_patterns(cls, patterns: list[str]) -> list[str]:
        return [validate_relative_pattern(p) for p in patterns]


class ProjectComponent(BaseModel):
    root: str
    kind: str
    language: str
    framework: str | None = None
    manifest: str | None = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("root", mode="after")
    @classmethod
    def validate_component_root(cls, v: str) -> str:
        return validate_relative_pattern(v)

    @field_validator("manifest", mode="after")
    @classmethod
    def validate_manifest_path(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_relative_pattern(v)
        return v


class ArtifactCategories(BaseModel):
    plans: str = "plans"
    reports: str = "reports"
    decisions: str = "decisions"
    proposals: str = "proposals"
    implementations: str = "implementations"

    @field_validator("plans", "reports", "decisions", "proposals", "implementations", mode="after")
    @classmethod
    def validate_category(cls, v: str) -> str:
        return validate_relative_pattern(v)


class ArtifactStorage(BaseModel):
    root: str = ".agent/artifacts"
    categories: ArtifactCategories = Field(default_factory=ArtifactCategories)

    @field_validator("root", mode="after")
    @classmethod
    def validate_storage_root(cls, v: str) -> str:
        clean = validate_relative_pattern(v)
        # Must be located within .agent/
        if not (clean == ".agent" or clean.startswith(".agent/")):
            raise ValueError(f"Artifact storage root must be inside .agent/: '{clean}'")
        return clean


class ProjectManifest(BaseModel):
    schema_version: str = "1.0"
    project: ProjectIdentity
    sources: ContextSources
    components: dict[str, ProjectComponent] = Field(default_factory=dict)
    artifact_storage: ArtifactStorage = Field(default_factory=ArtifactStorage)
    generated: dict = Field(default_factory=dict)


class AgentAccess(BaseModel):
    read: list[str] = Field(default_factory=list)
    write: list[str] = Field(default_factory=list)
    commands: list[list[str]] = Field(default_factory=list)

    @field_validator("read", "write", mode="after")
    @classmethod
    def validate_paths(cls, patterns: list[str]) -> list[str]:
        return [validate_relative_pattern(p) for p in patterns]


class SecurityPolicy(BaseModel):
    deny_hidden_files_by_default: bool = True
    denied_paths: list[str] = Field(default_factory=list)

    @field_validator("denied_paths", mode="after")
    @classmethod
    def validate_denied_paths(cls, patterns: list[str]) -> list[str]:
        return [validate_relative_pattern(p) for p in patterns]


class ProjectPolicy(BaseModel):
    schema_version: str = "1.0"
    mode: Literal["read_only", "proposal", "artifact_write", "branch_write", "release"]
    security: SecurityPolicy
    agents: dict[str, AgentAccess] = Field(default_factory=dict)


class WorkflowLimits(BaseModel):
    max_replans: int = 2
    max_repairs: int = 3
    max_files_per_search: int = 50
    max_context_characters: int = 100000


class WorkflowVerification(BaseModel):
    repository_evidence_required: bool = True
    minimum_evidence_files: int = 1
    allow_handoff_as_final_evidence: bool = False


class WorkflowApprovals(BaseModel):
    source_write: str = "human"
    command_execution: str = "human"
    production_release: str = "human"


class ProjectWorkflow(BaseModel):
    schema_version: str = "1.0"
    profile: str = "software_delivery"
    stages: list[str] = Field(default_factory=lambda: ["discover", "analyze", "verify", "plan"])
    verification: WorkflowVerification = Field(default_factory=WorkflowVerification)
    limits: WorkflowLimits = Field(default_factory=WorkflowLimits)
    approvals: WorkflowApprovals = Field(default_factory=WorkflowApprovals)


class ValidatedProject(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    manifest: ProjectManifest
    policy: ProjectPolicy
    workflow: ProjectWorkflow


    def __repr__(self) -> str:
        return (
            f"ValidatedProject(root={str(self.root)!r}, "
            f"project_id={self.manifest.project.id!r}, "
            f"mode={self.policy.mode!r})"
        )
