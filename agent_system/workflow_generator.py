from agent_system.project_protocol import (
    ProjectWorkflow,
    WorkflowApprovals,
    WorkflowLimits,
    WorkflowVerification,
)


def generate_default_workflow() -> ProjectWorkflow:
    """Generate default workflow configuration."""
    return ProjectWorkflow(
        schema_version="1.0",
        profile="software_delivery",
        stages=["discover", "analyze", "verify", "plan"],
        verification=WorkflowVerification(
            repository_evidence_required=True,
            minimum_evidence_files=1,
            allow_handoff_as_final_evidence=False,
        ),
        limits=WorkflowLimits(
            max_replans=2,
            max_repairs=3,
            max_files_per_search=50,
            max_context_characters=100000,
        ),
        approvals=WorkflowApprovals(
            source_write="human",
            command_execution="human",
            production_release="human",
        ),
    )
