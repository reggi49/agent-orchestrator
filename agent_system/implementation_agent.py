import json
import warnings
from pathlib import Path
from typing import Any

warnings.warn(
    "run_implementation_agent and LLMImplementationExecutor are deprecated. "
    "Use agent_system.executors.OpenHandsExecutor instead.",
    DeprecationWarning,
    stacklevel=2,
)

from agent_system.implementation_executor import (
    ImplementationExecutor,
    LLMImplementationExecutor,
)
from agent_system.implementation_preflight import inspect_scope_requirements
from agent_system.schemas import (
    ImplementationChange,
    ImplementationResult,
    ProposedCommand,
)


def run_implementation_agent(
    project_root: Path,
    proposal_data: dict[str, Any],
    plan_data: dict[str, Any] | None = None,
    proposal_review: dict[str, Any] | None = None,
    approved_files: list[str] | None = None,
    repair_feedback: str | None = None,
    executor: ImplementationExecutor | None = None,
) -> ImplementationResult:
    """Implementation Agent: generates code changes strictly within approved_files using generic executor."""
    project_root = project_root.expanduser().resolve()
    prop_inner = proposal_data.get("result", {}).get("proposal", {})
    if not prop_inner and "proposal_id" in proposal_data:
        prop_inner = proposal_data

    task_title = prop_inner.get("task_title", "")
    if not approved_files:
        if proposal_review and "approved_writable_files" in proposal_review and proposal_review.get("approved_writable_files"):
            approved_files = proposal_review.get("approved_writable_files", [])
        elif proposal_review and "approved_files" in proposal_review:
            approved_files = proposal_review.get("approved_files", [])
        elif prop_inner.get("approved_writable_files"):
            approved_files = prop_inner.get("approved_writable_files", [])
        else:
            files_to_mod = prop_inner.get("files_to_modify", [])
            files_to_crt = prop_inner.get("files_to_create", [])
            tests_to_mod = prop_inner.get("tests_to_modify", [])
            tests_to_crt = prop_inner.get("tests_to_create", [])
            approved_files = [
                f["path"] if isinstance(f, dict) else f.path
                for f in (files_to_mod + files_to_crt + tests_to_mod + tests_to_crt)
            ]

    # Scope inspection
    outside_files = inspect_scope_requirements(
        project_root,
        approved_files,
        task_title,
        prop_inner,
    )

    if outside_files:
        outside_scope_paths = [f[0] for f in outside_files]
        reasons = [f"- {f[0]}: {f[1]}" for f in outside_files]
        summary = (
            f"Implementasi memerlukan modifikasi pada file di luar proposal ({len(outside_scope_paths)} file):\n"
            + "\n".join(reasons)
            + "\nImplementasi dihentikan. Diperlukan revisi proposal untuk memperluas cakupan."
        )
        return ImplementationResult(
            status="proposal_revision_required",
            task_title=task_title,
            changes=[],
            files_changed=[],
            files_requested_outside_scope=outside_scope_paths,
            assumptions=["Implementasi tidak boleh mengubah file di luar approved_files."],
            risks=["Perubahan tanpa persetujuan scope dapat merusak dependency wiring atau API kontrak."],
            validation_commands=[],
            summary=summary,
        )

    # Use generic implementation executor
    if executor is None:
        executor = LLMImplementationExecutor()

    ctx = executor.prepare_context(
        project_root=project_root,
        proposal_data=proposal_data,
        approved_files=approved_files,
        plan_data=plan_data,
        proposal_review=proposal_review,
    )

    return executor.generate_changes(ctx)

