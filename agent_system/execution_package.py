import json
from pathlib import Path
from typing import Any

from agent_system.executors.base import CodingExecutionRequest
from agent_system.project_protocol import ProjectPolicy


def extract_acceptance_criteria_from_proposal(prop_inner: dict[str, Any]) -> list[str]:
    """Extract all acceptance criteria and test expectations from proposal."""
    criteria: list[str] = []

    # 1. From implementation steps
    for step in prop_inner.get("implementation_steps", []):
        ac_list = step.get("acceptance_criteria", []) if isinstance(step, dict) else getattr(step, "acceptance_criteria", [])
        for ac in ac_list:
            if ac and ac not in criteria:
                criteria.append(ac)

    # 2. From tests to modify / create
    for test in prop_inner.get("tests_to_modify", []) + prop_inner.get("tests_to_create", []):
        scenarios = test.get("scenarios", []) if isinstance(test, dict) else getattr(test, "scenarios", [])
        for sc in scenarios:
            if sc and sc not in criteria:
                criteria.append(sc)

        rel_ac = test.get("related_acceptance_criteria", []) if isinstance(test, dict) else getattr(test, "related_acceptance_criteria", [])
        for ac in rel_ac:
            if ac and ac not in criteria:
                criteria.append(ac)

    return criteria


def extract_approved_files_from_proposal(
    prop_inner: dict[str, Any],
    proposal_review: dict[str, Any] | None = None,
) -> list[str]:
    """Extract approved writable files list strictly from proposal artifact."""
    if proposal_review and proposal_review.get("approved_writable_files"):
        raw_list = proposal_review.get("approved_writable_files", [])
    elif proposal_review and proposal_review.get("approved_files"):
        raw_list = proposal_review.get("approved_files", [])
    elif prop_inner.get("approved_writable_files"):
        raw_list = prop_inner.get("approved_writable_files", [])
    else:
        files_to_mod = prop_inner.get("files_to_modify", [])
        files_to_crt = prop_inner.get("files_to_create", [])
        tests_to_mod = prop_inner.get("tests_to_modify", [])
        tests_to_crt = prop_inner.get("tests_to_create", [])
        raw_list = [
            f["path"] if isinstance(f, dict) else f.path
            for f in (files_to_mod + files_to_crt + tests_to_mod + tests_to_crt)
        ]

    # Normalize: strip leading slashes to guarantee relative paths
    approved = []
    for f in raw_list:
        clean = str(f).replace("\\", "/").strip().lstrip("/")
        if clean and clean not in approved:
            approved.append(clean)
    return approved


def extract_validation_commands_from_proposal(prop_inner: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract validation commands from proposal."""
    cmds: list[dict[str, Any]] = []
    for cmd in prop_inner.get("validation_commands", []):
        if isinstance(cmd, dict):
            exe = cmd.get("executable", "")
            args = cmd.get("arguments", [])
            cwd = cmd.get("working_directory", ".")
            purpose = cmd.get("purpose", "")
            requires_approval = cmd.get("requires_approval", True)
        else:
            exe = getattr(cmd, "executable", "")
            args = getattr(cmd, "arguments", [])
            cwd = getattr(cmd, "working_directory", ".")
            purpose = getattr(cmd, "purpose", "")
            requires_approval = getattr(cmd, "requires_approval", True)

        # Ensure cwd is relative
        clean_cwd = str(cwd).replace("\\", "/").strip().lstrip("/") or "."
        cmds.append({
            "executable": exe,
            "arguments": args,
            "working_directory": clean_cwd,
            "purpose": purpose,
            "requires_approval": requires_approval,
        })
    return cmds


def build_execution_request_from_proposal(
    proposal_data: dict[str, Any],
    worktree_path: str | Path,
    run_id: str,
    project_id: str = "project",
    proposal_review: dict[str, Any] | None = None,
    policy: ProjectPolicy | None = None,
    max_iterations: int = 30,
    timeout_seconds: int = 900,
) -> CodingExecutionRequest:
    """Build CodingExecutionRequest from approved proposal artifact.

    All file paths are guaranteed to be relative.
    """
    prop_inner = proposal_data.get("result", {}).get("proposal", {})
    if not prop_inner and "proposal_id" in proposal_data:
        prop_inner = proposal_data

    task_title = prop_inner.get("task_title", "Coding Task")
    task_desc = prop_inner.get("objective", "") or prop_inner.get("task_description", "")
    if not task_desc:
        task_desc = f"Implement changes for '{task_title}' according to proposal specification."

    planned_details = []
    for f in prop_inner.get("files_to_modify", []):
        f_path = f.get("path", "") if isinstance(f, dict) else getattr(f, "path", "")
        f_purpose = f.get("purpose", "") if isinstance(f, dict) else getattr(f, "purpose", "")
        f_changes = f.get("changes", []) if isinstance(f, dict) else getattr(f, "changes", [])
        if f_path:
            detail = f"- File: {f_path}\n  Purpose: {f_purpose}"
            if f_changes:
                detail += "\n  Planned modifications:\n" + "\n".join(f"    * {ch}" for ch in f_changes)
            planned_details.append(detail)

    if planned_details:
        task_desc = f"{task_desc}\n\nPLANNED SPECIFICATIONS (FROM APPROVED PROPOSAL):\n" + "\n\n".join(planned_details)

    approved_files = extract_approved_files_from_proposal(prop_inner, proposal_review)
    acceptance_criteria = extract_acceptance_criteria_from_proposal(prop_inner)
    validation_commands = extract_validation_commands_from_proposal(prop_inner)

    # Clean evidence paths (must be relative)
    raw_ev = prop_inner.get("evidence_paths", []) or proposal_data.get("evidence_paths", [])
    evidence_paths = []
    for ep in raw_ev:
        clean_ep = str(ep).split(":")[0].replace("\\", "/").strip().lstrip("/")
        if clean_ep and clean_ep not in evidence_paths:
            evidence_paths.append(clean_ep)

    # Constraints from policy
    constraints = [
        "Do not modify files outside approved_files",
        "Do not modify or inspect .env or secrets",
        "Do not execute git write operations (no commit, push, merge, reset)",
        "Do not install dependencies or make network requests",
    ]
    if policy:
        if policy.security.deny_hidden_files_by_default:
            constraints.append("Deny hidden files by default")
        for dp in policy.security.denied_paths:
            constraints.append(f"Denied path: {dp}")

    resolved_project_id = (
        proposal_data.get("project_id")
        or prop_inner.get("project_id")
        or project_id
    )

    return CodingExecutionRequest(
        run_id=run_id,
        project_id=resolved_project_id,
        task_title=task_title,
        task_description=task_desc,
        worktree_identifier=str(worktree_path),
        approved_files=approved_files,
        acceptance_criteria=acceptance_criteria,
        validation_commands=validation_commands,
        constraints=constraints,
        evidence_paths=evidence_paths,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
    )
