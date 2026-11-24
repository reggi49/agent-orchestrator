import json
import re
from pathlib import Path
from typing import Any

from agent_system.project_protocol import ProjectManifest, ProjectPolicy
from agent_system.proposal_agent import (
    compute_approved_writable_files,
    compute_proposal_hash,
    detect_code_or_patch_in_text,
    validate_all_paths_relative,
    validate_approved_writable_files,
)
from agent_system.schemas import (
    ImplementationProposal,
    ProposalFinding,
    ProposalReview,
)

ALLOWED_EXECUTABLES = {"go", "npm", "npx", "git"}

FORBIDDEN_EXECUTABLES = {
    "sh",
    "bash",
    "zsh",
    "fish",
    "sudo",
    "rm",
    "mv",
    "curl",
    "wget",
    "ssh",
    "scp",
    "git commit",
    "git push",
    "git reset",
    "git clean",
    "git checkout",
    "npm publish",
    "go install",
}

FORBIDDEN_SHELL_OPERATORS = [
    re.compile(r"&&"),
    re.compile(r"\|\|"),
    re.compile(r";"),
    re.compile(r"\|"),
    re.compile(r">>"),
    re.compile(r">"),
    re.compile(r"<"),
    re.compile(r"`"),
    re.compile(r"\$\("),
]


def validate_commands(
    commands: list[dict[str, Any]],
) -> tuple[list[ProposalFinding], list[str]]:
    """Validate proposal validation commands against security and format policy."""
    findings: list[ProposalFinding] = []
    unsafe_commands: list[str] = []

    for idx, cmd in enumerate(commands):
        executable = cmd.get("executable", "").strip()
        args = cmd.get("arguments", [])
        requires_approval = cmd.get("requires_approval", False)
        cmd_str = f"{executable} {' '.join(str(a) for a in args)}".strip()

        # 1. Shell command string rejection (arguments must be a list)
        if not isinstance(args, list):
            msg = f"Command #{idx + 1} '{cmd_str}' arguments must be a list, not string."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="command_safety",
                    message=msg,
                    affected_files=[],
                )
            )
            unsafe_commands.append(cmd_str)
            continue

        # 2. Check forbidden executables
        exe_lower = executable.lower()
        if exe_lower in FORBIDDEN_EXECUTABLES or any(
            fe in cmd_str.lower() for fe in FORBIDDEN_EXECUTABLES
        ):
            msg = f"Forbidden executable or destructive command in #{idx + 1}: '{cmd_str}'."
            findings.append(
                ProposalFinding(
                    severity="critical",
                    category="command_safety",
                    message=msg,
                    affected_files=[],
                )
            )
            unsafe_commands.append(cmd_str)
            continue

        # 3. Check allowed executables
        if exe_lower not in ALLOWED_EXECUTABLES:
            msg = f"Executable '{executable}' is not in allowlist (go, npm, npx, git)."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="command_safety",
                    message=msg,
                    affected_files=[],
                )
            )
            unsafe_commands.append(cmd_str)
            continue

        # 4. Check git read-only restriction
        if exe_lower == "git":
            is_allowed_git = False
            for allowed in ALLOWED_GIT_ARGS:
                if args == allowed:
                    is_allowed_git = True
                    break
            if not is_allowed_git:
                msg = f"Git command '{cmd_str}' is not an allowed read-only git command."
                findings.append(
                    ProposalFinding(
                        severity="high",
                        category="command_safety",
                        message=msg,
                        affected_files=[],
                    )
                )
                unsafe_commands.append(cmd_str)
                continue

        # 5. Check shell operators in arguments or executable
        combined_text = f"{executable} " + " ".join(str(a) for a in args)
        has_operator = False
        for op in FORBIDDEN_SHELL_OPERATORS:
            if op.search(combined_text):
                msg = f"Command #{idx + 1} contains forbidden shell operator '{op.pattern}': '{cmd_str}'."
                findings.append(
                    ProposalFinding(
                        severity="critical",
                        category="command_safety",
                        message=msg,
                        affected_files=[],
                    )
                )
                unsafe_commands.append(cmd_str)
                has_operator = True
                break
        if has_operator:
            continue

        # 6. requires_approval must be True
        if requires_approval is not True:
            msg = f"Command #{idx + 1} '{cmd_str}' must set requires_approval: true."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="command_safety",
                    message=msg,
                    affected_files=[],
                )
            )
            unsafe_commands.append(cmd_str)

    return findings, unsafe_commands


ALLOWED_GIT_ARGS = [
    ["status", "--short"],
    ["diff", "--check"],
    ["diff", "--stat"],
]


def review_proposal_deterministically(
    proposal: ImplementationProposal | dict[str, Any],
    plan_data: dict[str, Any],
    manifest: ProjectManifest | None = None,
    policy: ProjectPolicy | None = None,
    previous_hashes: list[str] | None = None,
) -> ProposalReview:
    """Perform deterministic review checks on implementation proposal."""
    data = (
        proposal.model_dump(mode="json")
        if isinstance(proposal, ImplementationProposal)
        else dict(proposal)
    )
    proposal_id = data.get("proposal_id", "prop_unknown")

    findings: list[ProposalFinding] = []
    missing_criteria: list[str] = []
    unsupported_changes: list[str] = []
    scope_violations: list[str] = []
    unsafe_commands: list[str] = []
    approved_files: list[str] = []
    rejected_files: list[str] = []
    revision_instructions: list[str] = []

    # 1. Proposal must be derived from verified plan
    verified_task = (
        plan_data.get("result", {}).get("recommended_task")
        or plan_data.get("result", {}).get("final_task")
        or {}
    )
    if isinstance(verified_task, str):
        verified_task_title = verified_task
    else:
        verified_task_title = verified_task.get("title", "")

    prop_task_title = data.get("task_title", "")
    if not verified_task_title or not prop_task_title:
        findings.append(
            ProposalFinding(
                severity="critical",
                category="grounding",
                message="Proposal tidak memiliki referensi task dari verified plan yang valid.",
                affected_files=[],
            )
        )

    # 2. Check for identical proposal across revisions
    current_hash = compute_proposal_hash(data)
    if previous_hashes and current_hash in previous_hashes:
        findings.append(
            ProposalFinding(
                severity="critical",
                category="architecture",
                message="Proposal identik setelah menerima revision_required.",
                affected_files=[],
            )
        )

    # 3. Path safety: All paths relative, no absolute, no traversal
    path_errors = validate_all_paths_relative(data)
    for pe in path_errors:
        findings.append(
            ProposalFinding(
                severity="critical",
                category="security",
                message=pe,
                affected_files=[],
            )
        )

    # 4. Check for forbidden source code, diffs, patches
    for f in data.get("files_to_create", []) + data.get("files_to_modify", []):
        f_path = f.get("path", "")
        for change_desc in f.get("changes", []):
            code_issues = detect_code_or_patch_in_text(change_desc)
            for ci in code_issues:
                findings.append(
                    ProposalFinding(
                        severity="high",
                        category="architecture",
                        message=f"File {f_path}: {ci}",
                        affected_files=[f_path],
                    )
                )

    # 5. Component ownership & Scope checks
    task_owner = (data.get("owner") or "").lower().strip()

    for f in (
        data.get("files_to_create", [])
        + data.get("files_to_modify", [])
        + data.get("files_to_delete", [])
    ):
        f_path = f.get("path", "")
        f_action = f.get("action", "")
        f_evidence = f.get("evidence_paths", [])

        # Backend modifying frontend
        if task_owner == "backend" and f_path.startswith("frontend/"):
            msg = f"Backend task proposes modifying frontend file '{f_path}' without cross-component contract evidence."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="file_ownership",
                    message=msg,
                    affected_files=[f_path],
                )
            )
            scope_violations.append(f_path)
            rejected_files.append(f_path)
            continue

        # Frontend modifying backend
        if task_owner == "frontend" and f_path.startswith("backend/"):
            msg = f"Frontend task proposes modifying backend file '{f_path}' without cross-component contract evidence."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="file_ownership",
                    message=msg,
                    affected_files=[f_path],
                )
            )
            scope_violations.append(f_path)
            rejected_files.append(f_path)
            continue

        # Deleting files requires strong evidence
        if f_action == "delete" and not f_evidence:
            msg = f"Deleting file '{f_path}' without supporting evidence is not allowed."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="scope",
                    message=msg,
                    affected_files=[f_path],
                )
            )
            rejected_files.append(f_path)
            continue

        # Check if file has purpose and changes
        if not f.get("purpose") or not f.get("changes"):
            msg = f"File change '{f_path}' lacks clear purpose or concrete change descriptions."
            findings.append(
                ProposalFinding(
                    severity="medium",
                    category="grounding",
                    message=msg,
                    affected_files=[f_path],
                )
            )
            unsupported_changes.append(f_path)
            rejected_files.append(f_path)
            continue

        # Check forbidden migration modification
        if f_action in {"modify", "delete"} and ("migrations/" in f_path or "/migrations/" in f_path):
            msg = f"Modifying or deleting existing migration '{f_path}' is strictly forbidden."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="database_safety",
                    message=msg,
                    affected_files=[f_path],
                )
            )
            rejected_files.append(f_path)
            continue

        # Check forbidden dependency manifest modification
        manifest_names = {
            "go.mod", "go.sum", "package.json", "package-lock.json",
            "yarn.lock", "pnpm-lock.yaml", "requirements.txt",
            "poetry.lock", "pyproject.toml",
        }
        if f_action in {"modify", "delete"} and Path(f_path).name in manifest_names:
            msg = f"Modifying or deleting dependency manifest '{f_path}' is forbidden without explicit authorization."
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="dependency_safety",
                    message=msg,
                    affected_files=[f_path],
                )
            )
            rejected_files.append(f_path)
            continue

        approved_files.append(f_path)

    # 5b. Tests must be included in approved files
    for t in data.get("tests_to_create", []) + data.get("tests_to_modify", []):
        t_path = t.get("path", "")
        if not t_path:
            continue
        if not t.get("scenarios") or not t.get("expected_results"):
            msg = f"Proposed test '{t_path}' lacks clear scenarios or expected results."
            findings.append(
                ProposalFinding(
                    severity="medium",
                    category="testing",
                    message=msg,
                    affected_files=[t_path],
                )
            )
            rejected_files.append(t_path)
            continue
        if t_path not in approved_files:
            approved_files.append(t_path)

    # 5c. Validate approved writable files deterministically
    awf_errors = validate_approved_writable_files(data)
    for err in awf_errors:
        if "absolute" in err.lower() or "traversal" in err.lower():
            findings.append(
                ProposalFinding(
                    severity="critical",
                    category="security",
                    message=err,
                    affected_files=[],
                )
            )
        elif "test" in err.lower():
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="testing",
                    message=err,
                    affected_files=[],
                )
            )
        else:
            findings.append(
                ProposalFinding(
                    severity="high",
                    category="scope",
                    message=err,
                    affected_files=[],
                )
            )


    # 6. Acceptance Criteria to Test Mapping
    expected_criteria = (
        verified_task.get("acceptance_criteria", [])
        if isinstance(verified_task, dict)
        else []
    )
    if not expected_criteria:
        expected_criteria = plan_data.get("result", {}).get(
            "final_acceptance_criteria", []
        )

    all_tests = data.get("tests_to_create", []) + data.get("tests_to_modify", [])
    if not all_tests:
        findings.append(
            ProposalFinding(
                severity="high",
                category="testing",
                message="Proposal must include at least one test in tests_to_create or tests_to_modify.",
                affected_files=[],
            )
        )

    mapped_criteria_set = set()
    for t in all_tests:
        for rc in t.get("related_acceptance_criteria", []):
            mapped_criteria_set.add(rc.strip().lower())

    for ec in expected_criteria:
        ec_clean = ec.strip().lower()
        matched = any(
            ec_clean in mc or mc in ec_clean for mc in mapped_criteria_set
        )
        if not matched:
            missing_criteria.append(ec)
            findings.append(
                ProposalFinding(
                    severity="medium",
                    category="acceptance_criteria",
                    message=f"Acceptance criterion has no corresponding test mapping: '{ec}'.",
                    affected_files=[],
                )
            )

    # 7. Rollback Plan & Out-of-scope check
    rollback = data.get("rollback_plan", [])
    if not rollback:
        findings.append(
            ProposalFinding(
                severity="high",
                category="rollback",
                message="Rollback plan is mandatory and cannot be empty.",
                affected_files=[],
            )
        )
    else:
        combined_rollback = " ".join(rollback).lower()
        for f in data.get("files_to_modify", []):
            f_path = f.get("path", "") if isinstance(f, dict) else getattr(f, "path", "")
            f_name = Path(f_path).name.lower()
            if f_path and f_path.lower() not in combined_rollback and f_name not in combined_rollback:
                findings.append(
                    ProposalFinding(
                        severity="medium",
                        category="rollback",
                        message=f"Rollback plan does not cover modified file '{f_path}'.",
                        affected_files=[f_path],
                    )
                )

    out_of_scope = data.get("out_of_scope", [])
    if not out_of_scope:
        findings.append(
            ProposalFinding(
                severity="medium",
                category="scope",
                message="Out-of-scope definition is mandatory and cannot be empty.",
                affected_files=[],
            )
        )

    # 8. Command Policy Checks
    cmd_findings, bad_cmds = validate_commands(data.get("validation_commands", []))
    findings.extend(cmd_findings)
    unsafe_commands.extend(bad_cmds)

    # 9. Determine Status
    has_critical = any(f.severity == "critical" for f in findings)
    has_rejected = any(
        "identik" in f.message.lower() or (f.category == "grounding" and f.severity == "critical")
        for f in findings
    )
    has_high = any(f.severity == "high" for f in findings)
    has_medium = any(f.severity == "medium" for f in findings)

    if has_rejected:
        status = "rejected"
        summary = "Proposal ditolak karena pelanggaran fundamental atau proposal identik setelah revisi."
    elif has_critical:
        status = "human_required"
        summary = "Proposal mengandung temuan kritis terkait keamanan atau perintah destruktif."
    elif has_high or has_medium or missing_criteria or scope_violations or unsafe_commands:
        status = "revision_required"
        summary = f"Proposal memerlukan revisi ({len(findings)} temuan ditemukan)."
        for f in findings:
            if f.severity in {"high", "medium"}:
                revision_instructions.append(f"[{f.category}] {f.message}")
    else:
        status = "approved"
        summary = "Proposal lolos verifikasi review deterministik tanpa temuan blocker."

    return ProposalReview(
        status=status,
        proposal_id=proposal_id,
        findings=findings,
        missing_acceptance_criteria=missing_criteria,
        unsupported_changes=unsupported_changes,
        scope_violations=scope_violations,
        unsafe_commands=unsafe_commands,
        revision_instructions=revision_instructions,
        approved_files=sorted(list(dict.fromkeys(approved_files))),
        approved_writable_files=compute_approved_writable_files(data),
        rejected_files=rejected_files,
        summary=summary,
    )

