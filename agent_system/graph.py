from langgraph.graph import END, START, StateGraph

from agent_system.nodes import (
    analyst_node,
    branch_guard_node,
    coding_executor_node,
    diff_reviewer_node,
    implementation_agent_node,
    implementation_preflight_node,
    implementation_proposal_node,
    independent_diff_review_node,
    project_reader_node,
    proposal_reviewer_node,
    repair_agent_node,
    repository_explorer_node,
    supervisor_node,
    task_verifier_node,
    test_runner_node,
)
from agent_system.schemas import AgentState


def route_after_verification(state: AgentState) -> str:
    """Route from task_verifier to supervisor or analyst replan."""
    verification = state.get("task_verification", {})
    status = verification.get("status")
    replan_count = state.get("replan_count", 0)
    max_replans = state.get("max_replans", 2)

    # 1. verified_pending -> supervisor
    if status == "verified_pending":
        return "supervisor"

    # 2. partially_implemented with supported missing_work -> supervisor
    if status == "partially_implemented" and verification.get("missing_work"):
        return "supervisor"

    # 6. max_replans reached -> supervisor (will mark human_required)
    if replan_count >= max_replans:
        return "supervisor"

    # 3, 4, 5. already_implemented, invalid_task, insufficient_evidence -> analyst
    return "analyst"


def route_after_proposal_review(state: AgentState) -> str:
    """Route from proposal_reviewer to supervisor or proposal revision."""
    review = state.get("proposal_review", {})
    status = review.get("status")
    revision_count = state.get("proposal_revision_count", 0)
    max_revisions = state.get("max_proposal_revisions", 2)

    # 1. approved -> supervisor
    if status == "approved":
        return "supervisor"

    # 2. revision_required and revision available -> implementation_proposal
    if status == "revision_required" and revision_count <= max_revisions:
        return "implementation_proposal"

    # 3. rejected, human_required, or max revisions exhausted -> supervisor
    return "supervisor"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("project_reader", project_reader_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("repository_explorer", repository_explorer_node)
    builder.add_node("task_verifier", task_verifier_node)
    builder.add_node("supervisor", supervisor_node)

    builder.add_edge(START, "project_reader")
    builder.add_edge("project_reader", "analyst")
    builder.add_edge("analyst", "repository_explorer")
    builder.add_edge("repository_explorer", "task_verifier")

    builder.add_conditional_edges(
        "task_verifier",
        route_after_verification,
        {
            "analyst": "analyst",
            "supervisor": "supervisor",
        },
    )

    builder.add_edge("supervisor", END)

    return builder.compile()


def build_proposal_graph():
    builder = StateGraph(AgentState)

    builder.add_node("implementation_proposal", implementation_proposal_node)
    builder.add_node("proposal_reviewer", proposal_reviewer_node)
    builder.add_node("supervisor", supervisor_node)

    builder.add_edge(START, "implementation_proposal")
    builder.add_edge("implementation_proposal", "proposal_reviewer")

    builder.add_conditional_edges(
        "proposal_reviewer",
        route_after_proposal_review,
        {
            "implementation_proposal": "implementation_proposal",
            "supervisor": "supervisor",
        },
    )

    builder.add_edge("supervisor", END)

    return builder.compile()
def route_after_preflight(state: AgentState) -> str:
    """Route from implementation_preflight to branch_guard or supervisor."""
    status = state.get("status")
    if status in {"proposal_revision_required", "stale_proposal", "human_required", "rejected"}:
        return "supervisor"
    return "branch_guard"


def route_after_branch_guard(state: AgentState) -> str:
    """Route from branch_guard to implementation_agent, coding_executor, or supervisor."""
    status = state.get("status")
    if status in {"human_required", "rejected"}:
        return "supervisor"
    if state.get("executor") == "openhands":
        return "coding_executor"
    return "implementation_agent"


def route_after_coding_executor(state: AgentState) -> str:
    """Route from coding_executor to independent_diff_review or supervisor."""
    status = state.get("status")
    retry_count = state.get("executor_retry_count", 0)
    max_retries = state.get("max_executor_retries", 1)

    if status == "completed":
        return "independent_diff_review"
    if status == "failed" and retry_count < max_retries:
        return "coding_executor"
    return "supervisor"


def route_after_implementation_agent(state: AgentState) -> str:
    """Route from implementation_agent to test_runner or supervisor."""
    status = state.get("status")
    if status in {"proposal_revision_required", "blocked", "human_required"}:
        return "supervisor"
    if state.get("dry_run", True):
        return "supervisor"
    return "test_runner"


def route_after_test_runner(state: AgentState) -> str:
    """Route from test_runner to repair_agent, diff_reviewer, or supervisor."""
    cmd_results = state.get("command_results", [])
    any_failed = any(c.get("status") != "passed" for c in cmd_results)
    repairs = state.get("implementation_repair_count", 0)
    max_repairs = state.get("max_implementation_repairs", 2)

    if any_failed:
        if repairs < max_repairs:
            return "repair_agent"
        return "supervisor"
    return "diff_reviewer"


def route_after_diff_reviewer(state: AgentState) -> str:
    """Route from diff_reviewer to supervisor or repair_agent."""
    review = state.get("diff_review", {})
    status = review.get("status")
    unexpected_files = review.get("unexpected_files", [])
    repairs = state.get("implementation_repair_count", 0)
    max_repairs = state.get("max_implementation_repairs", 2)

    if unexpected_files or status == "human_required":
        return "supervisor"
    if status == "approved":
        return "supervisor"
    if status == "repair_required":
        if repairs < max_repairs:
            return "repair_agent"
        return "supervisor"
    return "supervisor"


def build_implementation_graph():
    """Build controlled implementation workflow graph."""
    builder = StateGraph(AgentState)

    builder.add_node("preflight", implementation_preflight_node)
    builder.add_node("branch_guard", branch_guard_node)
    builder.add_node("implementation_agent", implementation_agent_node)
    builder.add_node("test_runner", test_runner_node)
    builder.add_node("diff_reviewer", diff_reviewer_node)
    builder.add_node("repair_agent", repair_agent_node)
    builder.add_node("supervisor", supervisor_node)

    builder.add_edge(START, "preflight")

    builder.add_conditional_edges(
        "preflight",
        route_after_preflight,
        {
            "branch_guard": "branch_guard",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "branch_guard",
        route_after_branch_guard,
        {
            "implementation_agent": "implementation_agent",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "implementation_agent",
        route_after_implementation_agent,
        {
            "test_runner": "test_runner",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "test_runner",
        route_after_test_runner,
        {
            "repair_agent": "repair_agent",
            "diff_reviewer": "diff_reviewer",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "diff_reviewer",
        route_after_diff_reviewer,
        {
            "repair_agent": "repair_agent",
            "supervisor": "supervisor",
        },
    )

    builder.add_edge("repair_agent", "test_runner")
    builder.add_edge("supervisor", END)

    return builder.compile()


def build_execution_graph():
    """Build controlled execution workflow graph using OpenHands executor."""
    builder = StateGraph(AgentState)

    builder.add_node("preflight", implementation_preflight_node)
    builder.add_node("branch_guard", branch_guard_node)
    builder.add_node("coding_executor", coding_executor_node)
    builder.add_node("independent_diff_review", independent_diff_review_node)
    builder.add_node("supervisor", supervisor_node)

    builder.add_edge(START, "preflight")

    builder.add_conditional_edges(
        "preflight",
        route_after_preflight,
        {
            "branch_guard": "branch_guard",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "branch_guard",
        route_after_branch_guard,
        {
            "coding_executor": "coding_executor",
            "implementation_agent": "coding_executor",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "coding_executor",
        route_after_coding_executor,
        {
            "independent_diff_review": "independent_diff_review",
            "coding_executor": "coding_executor",
            "supervisor": "supervisor",
        },
    )

    builder.add_edge("independent_diff_review", "supervisor")
    builder.add_edge("supervisor", END)

    return builder.compile()



