import re
import subprocess
from pathlib import Path
from typing import Any

from agent_system.schemas import (
    AcceptanceCriterionEvidence,
    DiffFinding,
    DiffReview,
)

DEBUG_PRINT_PATTERNS = [
    re.compile(r"fmt\.Print(ln|f)?\s*\(\s*\"debug", re.IGNORECASE),
    re.compile(r"console\.log\s*\(\s*['\"]debug", re.IGNORECASE),
    re.compile(r"print\s*\(\s*['\"]DEBUG", re.IGNORECASE),
]

TODO_PATTERNS = [
    re.compile(r"//\s*TODO:?\s*implement", re.IGNORECASE),
    re.compile(r"#\s*TODO:?\s*implement", re.IGNORECASE),
    re.compile(r"//\s*FIXME", re.IGNORECASE),
    re.compile(r"#\s*FIXME", re.IGNORECASE),
]

DEPENDENCY_MANIFESTS = {
    "backend/go.mod",
    "backend/go.sum",
    "frontend/package.json",
    "frontend/package-lock.json",
    "package.json",
    "package-lock.json",
    "requirements.txt",
    "pyproject.toml",
}

SECRET_PATTERNS = [
    re.compile(r"(^|/)\.env(\..+)?$", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"(^|/)credentials\.[a-z0-9]+$", re.IGNORECASE),
    re.compile(r"(^|/)auth\.json$", re.IGNORECASE),
]

PROTOCOL_PATTERNS = [
    re.compile(r"\.agent/manifest\.yaml$"),
    re.compile(r"\.agent/policy\.yaml$"),
    re.compile(r"\.agent/workflow\.yaml$"),
]


def is_secret_file(p: str) -> bool:
    clean = p.replace("\\", "/").strip("/")
    return any(pat.search(clean) for pat in SECRET_PATTERNS)


def is_protocol_file(p: str) -> bool:
    clean = p.replace("\\", "/").strip("/")
    return any(pat.search(clean) for pat in PROTOCOL_PATTERNS)


def review_diff(
    project_root: Path,
    approved_files: list[str],
    acceptance_criteria: list[str],
    test_status: str = "passed",
    timeout: float = 10.0,
) -> DiffReview:
    """Inspect Git diff safely and deterministically without storing raw full diff."""
    project_root = project_root.expanduser().resolve()
    findings: list[DiffFinding] = []
    approved_set = set(approved_files)

    # 1. git diff --name-only
    res_names = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=project_root,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    changed_files = [line.strip() for line in res_names.stdout.splitlines() if line.strip()]

    # Also check untracked files
    res_untracked = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=project_root,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in res_untracked.stdout.splitlines():
        line = line.strip()
        if line.startswith("??"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                untracked_path = parts[1].strip()
                if is_protocol_file(untracked_path) or not untracked_path.startswith(".agent/"):
                    if untracked_path not in changed_files:
                        changed_files.append(untracked_path)

    unexpected_files = [f for f in changed_files if f not in approved_set]

    # Check for secret, protocol, frontend, old migrations, manifests in changed files
    for f in changed_files:
        if is_secret_file(f):
            findings.append(
                DiffFinding(
                    severity="critical",
                    category="security",
                    message=f"File rahasia diubah: '{f}'",
                    file_path=f,
                )
            )
        if is_protocol_file(f):
            findings.append(
                DiffFinding(
                    severity="critical",
                    category="protocol",
                    message=f"File protokol diubah tanpa otorisasi: '{f}'",
                    file_path=f,
                )
            )
        if f.startswith("frontend/"):
            findings.append(
                DiffFinding(
                    severity="critical",
                    category="scope",
                    message=f"File frontend diubah pada task backend: '{f}'",
                    file_path=f,
                )
            )
        if f.startswith("backend/migrations/"):
            findings.append(
                DiffFinding(
                    severity="critical",
                    category="database",
                    message=f"File migrasi lama diubah: '{f}'",
                    file_path=f,
                )
            )
        if f in DEPENDENCY_MANIFESTS:
            findings.append(
                DiffFinding(
                    severity="critical",
                    category="dependency",
                    message=f"Dependency manifest diubah tanpa persetujuan: '{f}'",
                    file_path=f,
                )
            )

    # 2. git diff --check (whitespace errors)
    res_check = subprocess.run(
        ["git", "diff", "--check"],
        cwd=project_root,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res_check.returncode != 0:
        findings.append(
            DiffFinding(
                severity="medium",
                category="code_style",
                message=f"Ditemukan whitespace error oleh git diff --check: {res_check.stdout.strip()[:200]}",
                file_path=changed_files[0] if changed_files else "",
            )
        )

    # 3. Read diff for approved files to detect debug prints, TODO placeholders
    if approved_files:
        res_diff = subprocess.run(
            ["git", "diff", "--"] + [f for f in approved_files if f in changed_files],
            cwd=project_root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        current_file = ""
        current_line_no = 0
        for line in res_diff.stdout.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
            elif line.startswith("@@"):
                m = re.search(r"\+(\d+)", line)
                if m:
                    current_line_no = int(m.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                added_code = line[1:].strip()
                for pat in DEBUG_PRINT_PATTERNS:
                    if pat.search(added_code):
                        findings.append(
                            DiffFinding(
                                severity="medium",
                                category="debug_code",
                                message=f"Ditemukan debug print: '{added_code}'",
                                file_path=current_file,
                                line_number=current_line_no,
                            )
                        )
                for pat in TODO_PATTERNS:
                    if pat.search(added_code):
                        findings.append(
                            DiffFinding(
                                severity="high",
                                category="incomplete_code",
                                message=f"Ditemukan TODO/FIXME placeholder: '{added_code}'",
                                file_path=current_file,
                                line_number=current_line_no,
                            )
                        )
                current_line_no += 1
            elif not line.startswith("-"):
                current_line_no += 1

    # 4. Acceptance criteria status
    ac_status_list: list[AcceptanceCriterionEvidence] = []
    for ac in acceptance_criteria:
        if test_status == "passed" and not findings:
            status = "satisfied"
            detail = "Diverifikasi oleh test runner yang lolos dan diff review bersih."
        elif test_status == "passed":
            status = "verified"
            detail = "Test lolos tetapi terdapat temuan style/code review."
        else:
            status = "missing"
            detail = f"Test status: {test_status}"
        ac_status_list.append(
            AcceptanceCriterionEvidence(
                criterion=ac,
                status=status,
                evidence_detail=detail,
            )
        )

    # 5. Determine overall review status
    if unexpected_files:
        status = "human_required"
        summary = f"Ditemukan {len(unexpected_files)} file tak terduga di luar approved files: {unexpected_files}"
    elif any(f.severity == "critical" for f in findings):
        status = "human_required"
        summary = f"Ditemukan temuan kritis pada diff: {[f.message for f in findings if f.severity == 'critical']}"
    elif findings:
        status = "repair_required"
        summary = f"Ditemukan {len(findings)} temuan yang memerlukan perbaikan (repair_required)."
    else:
        status = "approved"
        summary = f"Diff review disetujui: {len(changed_files)} file berubah dalam batas approved scope."

    return DiffReview(
        status=status,
        changed_files=changed_files,
        approved_files=approved_files,
        unexpected_files=unexpected_files,
        findings=findings,
        acceptance_criteria_status=ac_status_list,
        test_status=test_status,
        summary=summary,
    )

