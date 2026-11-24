from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel, Field

CodingExecutionStatus = Literal[
    "completed",
    "failed",
    "execution_failed",
    "timed_out",
    "scope_violation",
    "proposal_revision_required",
    "human_required",
    "execution_review_ready",
    "approved_for_integration",
    "changes_requested",
    "rejected",
    "integration_in_progress",
    "integration_failed",
]


class CodingExecutionRequest(BaseModel):
    run_id: str
    project_id: str
    task_title: str
    task_description: str
    worktree_identifier: str
    approved_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    validation_commands: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    max_iterations: int = 30
    timeout_seconds: int = 900


class CodingExecutionResult(BaseModel):
    status: CodingExecutionStatus
    executor: str
    worktree_identifier: str
    changed_files: list[str] = Field(default_factory=list)
    validation_results: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    summary: str = ""
    requested_additional_files: list[str] = Field(default_factory=list)
    scope_violations: list[str] = Field(default_factory=list)
    timed_out: bool = False
    error_category: str | None = None


class CodingAgentExecutor(ABC):
    """Interface standard for coding executor agents."""

    @abstractmethod
    def validate_configuration(self) -> dict[str, Any]:
        """Validate executor configuration, environment, and dependencies."""
        pass

    @abstractmethod
    def prepare_execution(self, request: CodingExecutionRequest) -> dict[str, Any]:
        """Prepare task instructions, policies, and workspace context."""
        pass

    @abstractmethod
    def execute(self, request: CodingExecutionRequest) -> CodingExecutionResult:
        """Execute coding task within isolated workspace."""
        pass

    @abstractmethod
    def collect_result(self, request: CodingExecutionRequest) -> CodingExecutionResult:
        """Collect and verify result after execution."""
        pass

    @abstractmethod
    def cancel(self, run_id: str) -> None:
        """Cancel running execution safely."""
        pass

    @abstractmethod
    def cleanup(self, request: CodingExecutionRequest) -> None:
        """Clean up executor resources."""
        pass
