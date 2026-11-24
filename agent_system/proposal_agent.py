import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agent_system.schemas import (
    ImplementationProposal,
    ProposedCommand,
    ProposedFileChange,
    ProposedTest,
)


FORBIDDEN_DIFF_PATTERNS = [
    re.compile(r"diff\s+--git", re.IGNORECASE),
    re.compile(r"^---\s+[ab]/", re.MULTILINE),
    re.compile(r"^\+\+\+\s+[ab]/", re.MULTILINE),
    re.compile(r"^@@\s+-[0-9]+,[0-9]+\s+\+[0-9]+,[0-9]+\s+@@", re.MULTILINE),
]

FORBIDDEN_CODE_PATTERNS = [
    re.compile(r"^package\s+[a-zA-Z0-9_]+", re.MULTILINE),
    re.compile(r"^func\s+\([a-zA-Z0-9_\*]+\s+[a-zA-Z0-9_]+\)\s+[a-zA-Z0-9_]+", re.MULTILINE),
    re.compile(r"^func\s+[a-zA-Z0-9_]+\s*\(.*?\)\s*\{", re.MULTILINE),
    re.compile(r"^import\s+\(", re.MULTILINE),
    re.compile(r"^export\s+(default\s+)?function", re.MULTILINE),
    re.compile(r"^class\s+[a-zA-Z0-9_]+.*?:", re.MULTILINE),
    re.compile(r"^def\s+[a-zA-Z0-9_]+\s*\(.*?\):", re.MULTILINE),
]


def normalize_proposal(proposal: ImplementationProposal | dict[str, Any]) -> str:
    """Produce deterministic canonical representation of a proposal for hashing."""
    data = (
        proposal.model_dump(mode="json")
        if isinstance(proposal, ImplementationProposal)
        else dict(proposal)
    )

    canonical = {
        "task_title": (data.get("task_title") or "").strip().lower(),
        "owner": (data.get("owner") or "").strip().lower(),
        "files_to_create": sorted(
            [f.get("path", "").strip() for f in data.get("files_to_create", [])]
        ),
        "files_to_modify": sorted(
            [f.get("path", "").strip() for f in data.get("files_to_modify", [])]
        ),
        "files_to_delete": sorted(
            [f.get("path", "").strip() for f in data.get("files_to_delete", [])]
        ),
        "tests_to_create": sorted(
            [t.get("path", "").strip() for t in data.get("tests_to_create", [])]
        ),
        "tests_to_modify": sorted(
            [t.get("path", "").strip() for t in data.get("tests_to_modify", [])]
        ),
        "validation_commands": [
            {
                "executable": c.get("executable"),
                "arguments": c.get("arguments", []),
                "working_directory": c.get("working_directory"),
            }
            for c in data.get("validation_commands", [])
        ],
        "implementation_steps": [
            s.get("title", "").strip() for s in data.get("implementation_steps", [])
        ],
        "rollback_plan": [r.strip() for r in data.get("rollback_plan", [])],
        "out_of_scope": [o.strip() for o in data.get("out_of_scope", [])],
        "approved_writable_files": compute_approved_writable_files(data),
    }
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False)


def compute_approved_writable_files(proposal: ImplementationProposal | dict[str, Any]) -> list[str]:
    """Compute normalized (deduplicated, sorted) list of approved writable files."""
    data = (
        proposal.model_dump(mode="json")
        if isinstance(proposal, ImplementationProposal)
        else dict(proposal)
    )
    raw: list[str] = []
    for f in data.get("files_to_create", []):
        p = f.get("path") if isinstance(f, dict) else getattr(f, "path", "")
        if p and isinstance(p, str):
            raw.append(p.strip())
    for f in data.get("files_to_modify", []):
        p = f.get("path") if isinstance(f, dict) else getattr(f, "path", "")
        if p and isinstance(p, str):
            raw.append(p.strip())
    for t in data.get("tests_to_create", []):
        p = t.get("path") if isinstance(t, dict) else getattr(t, "path", "")
        if p and isinstance(p, str):
            raw.append(p.strip())
    for t in data.get("tests_to_modify", []):
        p = t.get("path") if isinstance(t, dict) else getattr(t, "path", "")
        if p and isinstance(p, str):
            raw.append(p.strip())

    # Return normalized: unique and sorted
    return sorted(list(dict.fromkeys(raw)))


def validate_approved_writable_files(proposal: ImplementationProposal | dict[str, Any]) -> list[str]:
    """Deterministic validation for approved writable files according to specification:
    1. Every ProposedTest to create must be in files_to_create or approved_writable_files.
    2. Every ProposedTest to modify must be in files_to_modify or approved_writable_files.
    3. Preflight/review does not consider test file out of scope if in approved_writable_files.
    4. No approved writable file without ProposedFileChange or ProposedTest.
    5. All approved writable files must be relative to project root.
    """
    data = (
        proposal.model_dump(mode="json")
        if isinstance(proposal, ImplementationProposal)
        else dict(proposal)
    )
    errors: list[str] = []

    files_to_crt = [
        f.get("path") if isinstance(f, dict) else getattr(f, "path", "")
        for f in data.get("files_to_create", [])
    ]
    files_to_mod = [
        f.get("path") if isinstance(f, dict) else getattr(f, "path", "")
        for f in data.get("files_to_modify", [])
    ]
    tests_to_crt = [
        t.get("path") if isinstance(t, dict) else getattr(t, "path", "")
        for t in data.get("tests_to_create", [])
    ]
    tests_to_mod = [
        t.get("path") if isinstance(t, dict) else getattr(t, "path", "")
        for t in data.get("tests_to_modify", [])
    ]

    all_proposed = set(files_to_crt + files_to_mod + tests_to_crt + tests_to_mod)

    approved = data.get("approved_writable_files")
    if approved is None or len(approved) == 0:
        approved = compute_approved_writable_files(data)

    # 4. No approved writable file without ProposedFileChange or ProposedTest
    for p in approved:
        if p not in all_proposed:
            errors.append(
                f"Approved writable file '{p}' does not correspond to any ProposedFileChange or ProposedTest."
            )

    # 1. Every ProposedTest to create must be in files_to_create or approved_writable_files
    for p in tests_to_crt:
        if p and p not in files_to_crt and p not in approved:
            errors.append(
                f"Proposed test to create '{p}' is not in files_to_create or approved_writable_files."
            )

    # 2. Every ProposedTest to modify must be in files_to_modify or approved_writable_files
    for p in tests_to_mod:
        if p and p not in files_to_mod and p not in approved:
            errors.append(
                f"Proposed test to modify '{p}' is not in files_to_modify or approved_writable_files."
            )

    # 5. All approved writable files must be relative to project root
    for p in approved:
        if not p:
            continue
        if p.startswith("/") or Path(p).is_absolute():
            errors.append(
                f"Approved writable file '{p}' must be relative to project root, absolute path found."
            )
        elif ".." in Path(p).parts:
            errors.append(
                f"Approved writable file '{p}' must not contain path traversal ('..')."
            )

    return errors


def compute_proposal_hash(proposal: ImplementationProposal | dict[str, Any]) -> str:
    """Compute sha256 hash of normalized proposal."""
    normalized = normalize_proposal(proposal)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def detect_code_or_patch_in_text(text: str) -> list[str]:
    """Check text for forbidden diff markers or full source code implementations."""
    issues = []
    for pattern in FORBIDDEN_DIFF_PATTERNS:
        if pattern.search(text):
            issues.append(f"Contains git diff/patch pattern: {pattern.pattern}")
            break

    for pattern in FORBIDDEN_CODE_PATTERNS:
        if pattern.search(text):
            issues.append(f"Contains source code implementation: {pattern.pattern}")
            break

    return issues


def validate_all_paths_relative(proposal: ImplementationProposal | dict[str, Any]) -> list[str]:
    """Check that all paths in proposal are strictly relative and do not contain traversal."""
    data = (
        proposal.model_dump(mode="json")
        if isinstance(proposal, ImplementationProposal)
        else dict(proposal)
    )

    errors = []
    paths_to_check: list[tuple[str, str]] = []

    for f in data.get("files_to_create", []):
        paths_to_check.append(("files_to_create", f.get("path", "")))
        for ep in f.get("evidence_paths", []):
            paths_to_check.append(("files_to_create.evidence_paths", ep))

    for f in data.get("files_to_modify", []):
        paths_to_check.append(("files_to_modify", f.get("path", "")))
        for ep in f.get("evidence_paths", []):
            paths_to_check.append(("files_to_modify.evidence_paths", ep))

    for f in data.get("files_to_delete", []):
        paths_to_check.append(("files_to_delete", f.get("path", "")))
        for ep in f.get("evidence_paths", []):
            paths_to_check.append(("files_to_delete.evidence_paths", ep))

    for t in data.get("tests_to_create", []):
        paths_to_check.append(("tests_to_create", t.get("path", "")))

    for t in data.get("tests_to_modify", []):
        paths_to_check.append(("tests_to_modify", t.get("path", "")))

    for c in data.get("validation_commands", []):
        wd = c.get("working_directory", "")
        if wd:
            paths_to_check.append(("validation_commands.working_directory", wd))

    for ep in data.get("evidence_paths", []):
        paths_to_check.append(("evidence_paths", ep))

    for p in data.get("approved_writable_files", []):
        paths_to_check.append(("approved_writable_files", p))

    if data.get("original_proposal_path"):
        paths_to_check.append(("original_proposal_path", data["original_proposal_path"]))

    if data.get("dry_run_report_path"):
        paths_to_check.append(("dry_run_report_path", data["dry_run_report_path"]))

    if data.get("source_plan_path"):
        paths_to_check.append(("source_plan_path", data["source_plan_path"]))

    for p in data.get("file_scope_before", []):
        paths_to_check.append(("file_scope_before", p))

    for p in data.get("file_scope_after", []):
        paths_to_check.append(("file_scope_after", p))

    for loc, p in paths_to_check:
        if not p:
            continue
        if p.startswith("/") or Path(p).is_absolute():
            errors.append(f"Absolute path found in {loc}: '{p}'")
        elif ".." in Path(p).parts:
            errors.append(f"Path traversal ('..') found in {loc}: '{p}'")

    return errors
