import uuid
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class ProjectFile(BaseModel):
    path: str
    content: str


class StackComponent(BaseModel):
    name: str
    root: str
    language: str
    framework: str | None = None
    manifest: str | None = None
    available_scripts: list[str] = Field(default_factory=list)


class RepositoryContext(BaseModel):
    project_name: str
    project_root: str
    repository_type: Literal["monorepo", "single-project", "unknown"]
    stack: list[StackComponent] = Field(default_factory=list)
    important_files: list[ProjectFile] = Field(default_factory=list)
    repository_summary: str = ""


class RecommendedTask(BaseModel):
    title: str
    owner: Literal[
        "backend",
        "frontend",
        "fullstack",
        "analyst",
        "qa",
        "unknown",
    ]
    description: str
    reason: str
    dependencies: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class ProjectAnalysis(BaseModel):
    project_summary: str
    current_status: str

    rules: list[str] = Field(default_factory=list)
    completed_work: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    recommended_task: RecommendedTask

    next_action: Literal[
        "backend",
        "frontend",
        "fullstack",
        "qa",
        "human_required",
    ]


class RepositorySearchRequest(BaseModel):
    query: str
    target_paths: list[str] = Field(default_factory=list)
    file_extensions: list[str] = Field(default_factory=list)
    reason: str = ""
    max_results: int = 20


class RepositorySearchResult(BaseModel):
    path: str
    line_start: int
    line_end: int
    excerpt: str
    match_reason: str = ""


class RepositoryEvidence(BaseModel):
    candidate_task: str
    inspected_files: list[str] = Field(default_factory=list)
    results: list[RepositorySearchResult] = Field(default_factory=list)
    summary: str = ""


VerificationStatus = Literal[
    "verified_pending",
    "already_implemented",
    "partially_implemented",
    "insufficient_evidence",
    "invalid_task",
]


class TaskVerification(BaseModel):
    status: VerificationStatus
    task_title: str
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    missing_work: list[str] = Field(default_factory=list)
    recommendation: str = ""
    should_replan: bool = False


# ======================================================================
# Implementation Proposal & Review Schemas
# ======================================================================

ProposedFileAction = Literal["create", "modify", "delete"]
EstimatedComplexity = Literal["low", "medium", "high"]
ProposalRiskSeverity = Literal["low", "medium", "high", "critical"]
ProposalStatus = Literal[
    "draft",
    "under_review",
    "revision_required",
    "approved_for_implementation",
    "rejected",
    "human_required",
]
ProposalReviewStatus = Literal[
    "approved",
    "revision_required",
    "rejected",
    "human_required",
]
FindingCategory = Literal[
    "grounding",
    "scope",
    "architecture",
    "testing",
    "security",
    "command_safety",
    "acceptance_criteria",
    "rollback",
    "dependency",
    "file_ownership",
]


class ProposedFileChange(BaseModel):
    path: str
    action: ProposedFileAction
    purpose: str
    changes: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    estimated_complexity: EstimatedComplexity = "medium"


class ProposedTest(BaseModel):
    path: str
    test_type: str
    scenarios: list[str] = Field(default_factory=list)
    expected_results: list[str] = Field(default_factory=list)
    related_acceptance_criteria: list[str] = Field(default_factory=list)


class ProposedCommand(BaseModel):
    executable: str
    arguments: list[str] = Field(default_factory=list)
    working_directory: str
    purpose: str
    requires_approval: bool = True


class ImplementationStep(BaseModel):
    order: int
    title: str
    description: str
    affected_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[int] = Field(default_factory=list)


class ProposalRisk(BaseModel):
    severity: ProposalRiskSeverity
    description: str
    mitigation: str
    evidence_paths: list[str] = Field(default_factory=list)


class ImplementationProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: f"prop_{uuid.uuid4().hex[:8]}")
    task_title: str = ""
    owner: str = "backend"
    objective: str = ""
    verification_status: str = "verified_pending"
    source_plan_path: str = ""
    affected_components: list[str] = Field(default_factory=list)
    files_to_create: list[ProposedFileChange] = Field(default_factory=list)
    files_to_modify: list[ProposedFileChange] = Field(default_factory=list)
    files_to_delete: list[ProposedFileChange] = Field(default_factory=list)
    tests_to_create: list[ProposedTest] = Field(default_factory=list)
    tests_to_modify: list[ProposedTest] = Field(default_factory=list)
    approved_writable_files: list[str] = Field(default_factory=list)
    original_proposal_path: str = ""
    dry_run_report_path: str = ""
    revision_reason: str = ""
    file_scope_before: list[str] = Field(default_factory=list)
    file_scope_after: list[str] = Field(default_factory=list)
    validation_commands: list[ProposedCommand] = Field(default_factory=list)
    implementation_steps: list[ImplementationStep] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    risks: list[ProposalRisk] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    rollback_plan: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    status: ProposalStatus = "draft"

    @model_validator(mode="before")
    @classmethod
    def _coerce_proposal_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "proposal_id" not in d or not d["proposal_id"]:
            d["proposal_id"] = f"prop_{uuid.uuid4().hex[:8]}"
        if "task_title" not in d or not d["task_title"]:
            d["task_title"] = d.get("title", "") or d.get("summary", "Task")
        if "owner" not in d or not d["owner"]:
            d["owner"] = d.get("component", "backend")
        if "objective" not in d or not d["objective"]:
            d["objective"] = d.get("summary", "") or d.get("description", "")
        if "verification_status" not in d or not d["verification_status"]:
            d["verification_status"] = "verified_pending"
        if "source_plan_path" not in d or not d["source_plan_path"]:
            d["source_plan_path"] = ""
        if isinstance(d.get("rollback_plan"), str):
            d["rollback_plan"] = [d["rollback_plan"]]

        # Map commands -> validation_commands if missing
        if "commands" in d and ("validation_commands" not in d or not d["validation_commands"]):
            d["validation_commands"] = d["commands"]

        # Map steps -> implementation_steps if missing
        if "steps" in d and ("implementation_steps" not in d or not d["implementation_steps"]):
            mapped_steps = []
            for s in d["steps"]:
                if isinstance(s, dict):
                    order = s.get("order") or s.get("step_number") or len(mapped_steps) + 1
                    mapped_steps.append({
                        "order": order,
                        "title": s.get("title", f"Step {order}"),
                        "description": s.get("description", ""),
                        "affected_files": s.get("affected_files", []),
                        "acceptance_criteria": s.get("acceptance_criteria", []),
                    })
                else:
                    mapped_steps.append(s)
            d["implementation_steps"] = mapped_steps

        # Map affected_files -> files_to_modify / files_to_create if missing
        if "affected_files" in d and ("files_to_modify" not in d or not d["files_to_modify"]):
            mods = []
            crts = []
            for af in d["affected_files"]:
                if isinstance(af, dict):
                    act = af.get("action", "modify").lower()
                    entry = {
                        "path": af.get("path", ""),
                        "action": act,
                        "purpose": af.get("purpose") or af.get("description", ""),
                        "changes": af.get("changes") or ([af.get("description")] if af.get("description") else []),
                    }
                    if act == "create":
                        crts.append(entry)
                    else:
                        mods.append(entry)
            if not d.get("files_to_modify"):
                d["files_to_modify"] = mods
            if not d.get("files_to_create"):
                d["files_to_create"] = crts

        # Map proposed_tests -> tests_to_create / tests_to_modify if missing
        if "proposed_tests" in d and ("tests_to_create" not in d or not d["tests_to_create"]):
            t_crts = []
            for pt in d["proposed_tests"]:
                if isinstance(pt, dict):
                    t_crts.append({
                        "path": pt.get("file_path") or pt.get("path", ""),
                        "test_type": pt.get("test_type", "unit"),
                        "scenarios": [pt.get("description", "")] if pt.get("description") else [],
                        "expected_results": ["Tests pass"],
                        "related_acceptance_criteria": [pt.get("mapped_criteria")] if pt.get("mapped_criteria") else [],
                    })
            if not d.get("tests_to_create"):
                d["tests_to_create"] = t_crts

        return d


class ProposalFinding(BaseModel):
    severity: str
    category: str
    message: str
    evidence_paths: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)


class ProposalReview(BaseModel):
    status: ProposalReviewStatus
    proposal_id: str
    findings: list[ProposalFinding] = Field(default_factory=list)
    missing_acceptance_criteria: list[str] = Field(default_factory=list)
    unsupported_changes: list[str] = Field(default_factory=list)
    scope_violations: list[str] = Field(default_factory=list)
    unsafe_commands: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    approved_files: list[str] = Field(default_factory=list)
    approved_writable_files: list[str] = Field(default_factory=list)
    rejected_files: list[str] = Field(default_factory=list)
    summary: str = ""


# ======================================================================
# Implementation Agent, Command Runner, & Diff Reviewer Schemas
# ======================================================================

ImplementationAction = Literal["create", "modify"]
ImplementationStatus = Literal[
    "implementation_generated",
    "proposal_revision_required",
    "blocked",
    "human_required",
    "implementation_generation_failed",
]
CommandExecutionStatus = Literal["passed", "failed", "timed_out", "rejected"]
DiffReviewStatus = Literal[
    "approved",
    "repair_required",
    "proposal_revision_required",
    "human_required",
]
CriterionStatus = Literal["satisfied", "verified", "missing", "unverifiable"]


class ImplementationChange(BaseModel):
    path: str
    action: ImplementationAction
    content: str
    reason: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)


class ImplementationResult(BaseModel):
    status: ImplementationStatus
    task_title: str
    changes: list[ImplementationChange] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    files_requested_outside_scope: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    validation_commands: list[ProposedCommand] = Field(default_factory=list)
    summary: str


class CommandExecutionResult(BaseModel):
    executable: str
    arguments: list[str] = Field(default_factory=list)
    working_directory: str
    exit_code: int
    stdout_summary: str
    stderr_summary: str
    timed_out: bool = False
    duration_seconds: float = 0.0
    status: CommandExecutionStatus


class DiffFinding(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    category: str
    message: str
    file_path: str
    line_number: int | None = None


class AcceptanceCriterionEvidence(BaseModel):
    criterion: str
    status: CriterionStatus
    evidence_detail: str = ""


# ======================================================================
# Post-Execution Review & Integration Schemas
# ======================================================================

HumanReviewDecision = Literal["approve", "request-changes", "reject"]
ExecutionReviewStatus = Literal[
    "execution_review_ready",
    "approved_for_integration",
    "changes_requested",
    "rejected",
    "integration_in_progress",
    "completed",
    "integration_failed",
]


class HumanExecutionReview(BaseModel):
    execution_id: str
    proposal_id: str
    decision: HumanReviewDecision
    reviewer: str
    timestamp: str
    changed_files: list[str] = Field(default_factory=list)
    test_summary: list[dict[str, Any]] = Field(default_factory=list)
    source_revision: dict[str, Any] = Field(default_factory=dict)
    worktree_branch: str
    worktree_identifier: str
    note: str = ""
    status: ExecutionReviewStatus
    review_type: Literal["human_review"] = "human_review"
    summary: str = ""


class IntegrationResult(BaseModel):
    execution_id: str
    proposal_id: str
    status: Literal["completed", "integration_failed"]
    strategy: Literal["cherry-pick"] = "cherry-pick"
    target_branch: str
    source_agent_branch: str
    agent_branch: str = ""
    source_commit: str | None = None
    integrated_commit: str | None = None
    commit_hash: str | None = None
    cherry_pick_occurred: bool = False
    changed_files: list[str] = Field(default_factory=list)
    validation_results: list[dict[str, Any]] = Field(default_factory=list)
    post_integration_validation: dict[str, Any] = Field(default_factory=dict)
    post_validation_results: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str
    recovery_instructions: str | None = None
    error_message: str | None = None
    summary: str = ""


# ======================================================================
# CEO Mode & Program Manager Schemas
# ======================================================================

TaskStatus = Literal[
    "pending",
    "ready",
    "in_progress",
    "awaiting_input",
    "retrying",
    "blocked",
    "completed",
    "failed",
    "cancelled",
]

ProgramStatus = Literal[
    "planning",
    "ready",
    "in_progress",
    "awaiting_ceo_input",
    "blocked",
    "failed",
    "completed",
    "cancelled",
    "interrupted",
]


class ProgramTask(BaseModel):
    task_id: str
    epic_id: str
    title: str
    description: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    owner_role: str = "backend"
    risk_level: str = "low"
    pre_risk: str | None = None
    post_risk: str | None = None
    status: TaskStatus = "pending"
    attempt_count: int = 0
    linked_run_id: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    test_result: Any = "none"
    completion_evidence: Any = None
    mapped_requirement_ids: list[str] = Field(default_factory=list)
    explicit_out_of_scope: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    readable_files: list[str] = Field(default_factory=list)
    expected_writable_files: list[str] = Field(default_factory=list)
    conditionally_writable_files: list[str] = Field(default_factory=list)
    planned_base_revision: str | None = None
    execution_base_revision: str | None = None
    integration_target_revision: str | None = None
    integrated_revision: str | None = None
    fallback_used: bool = False
    fallback_metadata: dict[str, Any] | None = None
    objective_alignment_status: str = "aligned"
    unmapped_changes: list[str] = Field(default_factory=list)
    drift_reason: str | None = None
    conflict_replan_count: int = 0
    superseded_attempts: list[dict[str, Any]] = Field(default_factory=list)
    approval_stage: str | None = None


class ProgramEpic(BaseModel):
    epic_id: str
    title: str
    description: str = ""
    status: str = "pending"
    tasks: list[str] = Field(default_factory=list)
    mapped_requirement_ids: list[str] = Field(default_factory=list)


class ProgramState(BaseModel):
    program_id: str
    project_id: str
    objective: str
    status: ProgramStatus = "planning"
    current_phase: str = "Phase 1: Foundation"
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    epics: list[ProgramEpic] = Field(default_factory=list)
    tasks: list[ProgramTask] = Field(default_factory=list)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    task_statuses: dict[str, str] = Field(default_factory=dict)
    attempts: dict[str, int] = Field(default_factory=dict)
    active_task: str | None = None
    active_run_id: str | None = None
    worktree_reference: str | None = None
    last_checkpoint: str = ""
    next_action: str | None = None
    approval_checkpoint: dict[str, Any] | None = None
    blockers: list[str] = Field(default_factory=list)
    overlap_analysis: dict[str, Any] = Field(default_factory=dict)
    circuit_breaker_status: dict[str, Any] = Field(default_factory=dict)
    retry_state: dict[str, Any] = Field(default_factory=dict)
    state_revision: int = 1
    checksum: str = ""
    schema_version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""



class DiffReview(BaseModel):
    status: DiffReviewStatus
    changed_files: list[str] = Field(default_factory=list)
    approved_files: list[str] = Field(default_factory=list)
    unexpected_files: list[str] = Field(default_factory=list)
    findings: list[DiffFinding] = Field(default_factory=list)
    acceptance_criteria_status: list[AcceptanceCriterionEvidence] = Field(default_factory=list)
    test_status: str
    summary: str


class AgentState(TypedDict, total=False):
    run_id: str
    user_request: str
    project_root: str

    repository_context: dict
    analysis: dict

    current_stage: str
    status: str
    final_summary: str
    errors: list[str]

    # Replan Loop fields
    repository_evidence: dict
    task_verification: dict
    replan_count: int
    max_replans: int
    rejected_tasks: list[dict]

    # Proposal & Review Loop fields
    source_plan_path: str
    plan_data: dict
    implementation_proposal: dict
    proposal_review: dict
    proposal_revision_count: int
    max_proposal_revisions: int
    rejected_proposal_reasons: list[str]
    proposal_hashes: list[str]

    # Implementation & Repair Loop fields
    source_proposal_path: str
    proposal_data: dict
    approved_files: list[str]
    branch_name: str
    preflight_result: dict
    implementation_result: dict
    command_results: list[dict]
    diff_review: dict
    implementation_repair_count: int
    max_implementation_repairs: int
    implementation_failures: list[dict]
    dry_run: bool
    apply: bool
    file_hashes_before: dict[str, str]
    file_hashes_after: dict[str, str]

    # OpenHands Coding Executor fields
    executor: str
    worktree_path: str
    worktree_descriptor: str
    main_project_root: str
    coding_execution_request: dict
    coding_execution_result: dict
    executor_retry_count: int
    max_executor_retries: int
