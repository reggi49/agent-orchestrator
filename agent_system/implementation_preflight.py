import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_system.path_safety import resolve_project_path, validate_relative_pattern
from agent_system.project_protocol import ProjectManifest, ProjectPolicy
from agent_system.source_revision import get_source_revision


# Allowed executables for validation commands
ALLOWED_VALIDATION_EXECUTABLES = {
    "go",
    "pytest",
    "python",
    "python3",
    "npm",
    "pnpm",
    "yarn",
    "cargo",
    "mvn",
    "gradle",
    "make",
}

FORBIDDEN_EXECUTABLES = {
    "sh",
    "bash",
    "zsh",
    "sudo",
    "rm",
    "curl",
    "wget",
    "nc",
    "git",
    "dd",
    "chmod",
    "chown",
}

SHELL_INJECTION_PATTERN = re.compile(r"[|;&><`$\n\r]")

SECRET_FILE_PATTERNS = [
    re.compile(r"(^|/)\.env(\..+)?$", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"(^|/)credentials\.[a-z0-9]+$", re.IGNORECASE),
    re.compile(r"(^|/)auth\.json$", re.IGNORECASE),
    re.compile(r"(^|/)id_rsa(\..+)?$", re.IGNORECASE),
]


def is_secret_file_path(path: str) -> bool:
    clean = path.replace("\\", "/").strip("/")
    return any(p.search(clean) for p in SECRET_FILE_PATTERNS)


class PreflightResult(BaseModel):
    is_valid: bool
    status: str  # "valid", "stale_proposal", "human_required", "proposal_revision_required", "rejected"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    approved_files: list[str] = Field(default_factory=list)
    files_to_modify: list[str] = Field(default_factory=list)
    files_to_create: list[str] = Field(default_factory=list)
    files_requested_outside_scope: list[str] = Field(default_factory=list)
    validation_commands: list[dict] = Field(default_factory=list)
    source_revision: dict = Field(default_factory=dict)
    summary: str = ""


def check_git_dirty_for_files(project_root: Path, target_files: list[str]) -> list[str]:
    """Check if any of the target files have uncommitted local changes."""
    if not target_files:
        return []
    try:
        cmd = ["git", "status", "--porcelain", "--"] + target_files
        res = subprocess.run(
            cmd,
            cwd=project_root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if res.returncode != 0:
            return []
        dirty_files = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                dirty_files.append(parts[1].strip())
        return dirty_files
    except Exception:
        return []


def is_symlink_escaping(project_root: Path, rel_path: str) -> bool:
    """Check if target path or any parent is a symlink pointing outside project."""
    try:
        full_path = project_root / rel_path
        cur = full_path
        while True:
            if cur.is_symlink():
                target = cur.resolve()
                try:
                    target.relative_to(project_root)
                except ValueError:
                    return True
            if cur == project_root or cur.parent == cur:
                break
            cur = cur.parent
        return False
    except Exception:
        return True


def run_implementation_preflight(
    project_root: Path,
    manifest: ProjectManifest,
    policy: ProjectPolicy,
    source_plan_path: str,
    source_proposal_path: str,
    plan_data: dict[str, Any] | None = None,
    proposal_data: dict[str, Any] | None = None,
    proposal_review: dict[str, Any] | None = None,
    apply: bool = False,
    target_branch_name: str | None = None,
    target_worktree_path: Path | None = None,
    dry_run_report_data: dict[str, Any] | None = None,
) -> PreflightResult:
    """Validate all preflight invariants before controlled implementation."""
    project_root = project_root.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    # 1 & 2. Verified plan check
    if not source_plan_path:
        errors.append("source_plan_path tidak disediakan.")
    else:
        try:
            validate_relative_pattern(source_plan_path)
            plan_file = resolve_project_path(project_root, source_plan_path)
            if not plan_file.is_file():
                errors.append(f"Verified plan file tidak ditemukan: '{source_plan_path}'")
            elif plan_data is None:
                plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"Gagal memuat verified plan '{source_plan_path}': {e}")

    if plan_data:
        plan_status = plan_data.get("status")
        if plan_status != "plan_ready":
            errors.append(
                f"Verified plan status adalah '{plan_status}', harus 'plan_ready'."
            )

    # 3 & 4. Proposal check
    if not source_proposal_path:
        errors.append("source_proposal_path tidak disediakan.")
    else:
        try:
            validate_relative_pattern(source_proposal_path)
            prop_file = resolve_project_path(project_root, source_proposal_path)
            if not prop_file.is_file():
                errors.append(f"Proposal file tidak ditemukan: '{source_proposal_path}'")
            elif proposal_data is None:
                proposal_data = json.loads(prop_file.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"Gagal memuat proposal '{source_proposal_path}': {e}")

    prop_inner = {}
    if proposal_data:
        prop_status = proposal_data.get("status")
        if prop_status != "approved_for_implementation":
            errors.append(
                f"Proposal status adalah '{prop_status}', harus 'approved_for_implementation'."
            )
        prop_inner = proposal_data.get("result", {}).get("proposal", {})
        if not prop_inner and "proposal_id" in proposal_data:
            prop_inner = proposal_data

    # 5. Proposal review check
    review_status = None
    if proposal_review:
        review_status = proposal_review.get("status")
    elif proposal_data:
        review_status = proposal_data.get("result", {}).get("review_status")
        if not review_status:
            prop_id = proposal_data.get("artifact_id", "").replace("proposal_", "")
            if prop_id:
                report_file = project_root / ".agent" / "artifacts" / "reports" / f"proposal_review_{prop_id}.json"
                if report_file.is_file():
                    try:
                        rep_data = json.loads(report_file.read_text(encoding="utf-8"))
                        review_status = rep_data.get("result", {}).get("review", {}).get("status")
                        if proposal_review is None:
                            proposal_review = rep_data.get("result", {}).get("review", {})
                    except Exception:
                        pass

    if review_status != "approved":
        errors.append(f"Proposal review status adalah '{review_status}', harus 'approved'.")

    # 5b. Dry-run preflight check in apply mode
    if apply:
        dr_status = None
        if dry_run_report_data:
            dr_status = (
                dry_run_report_data.get("result", {}).get("preflight_status")
                or dry_run_report_data.get("preflight_status")
                or dry_run_report_data.get("status")
            )
        else:
            rep_dir = project_root / ".agent" / "artifacts" / "reports"
            if rep_dir.is_dir():
                dr_files = sorted(
                    list(rep_dir.glob("implementation_review_*.json"))
                    + list(rep_dir.glob("execution_review_*.json")),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for df in dr_files:
                    try:
                        ddata = json.loads(df.read_text(encoding="utf-8"))
                        res_obj = ddata.get("result", {})
                        if res_obj.get("is_dry_run") or ddata.get("is_dry_run"):
                            dr_status = (
                                res_obj.get("preflight_status")
                                or ddata.get("preflight_status")
                                or ddata.get("status")
                            )
                            break
                    except Exception:
                        pass
        if dr_status and dr_status not in {"preflight_passed", "valid", "dry_run_ready"}:
            errors.append(f"Dry-run report status adalah '{dr_status}', harus 'preflight_passed'.")

    # 5c. Target branch & worktree collision check in apply mode
    if apply and target_branch_name:
        try:
            res_br = subprocess.run(
                ["git", "branch", "--list", target_branch_name],
                cwd=project_root,
                shell=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if res_br.stdout.strip():
                errors.append(f"Branch tujuan '{target_branch_name}' sudah ada di repository.")
        except Exception:
            pass

    if apply and target_worktree_path:
        wt_resolved = target_worktree_path.expanduser().resolve()
        if wt_resolved.exists() and wt_resolved.is_dir() and any(wt_resolved.iterdir()):
            errors.append(f"Worktree tujuan '{target_worktree_path}' sudah ada dan tidak kosong.")

    # 6. Source revision check
    cur_rev = get_source_revision(project_root)
    source_rev_dict = {
        "commit_head": cur_rev.commit_head,
        "is_dirty": cur_rev.is_dirty,
    }
    prop_rev = proposal_data.get("source_revision", {}) if proposal_data else {}
    expected_head = prop_rev.get("commit_head")
    if expected_head and cur_rev.commit_head and cur_rev.commit_head != expected_head:
        return PreflightResult(
            is_valid=False,
            status="stale_proposal",
            errors=[
                f"Source revision mismatch: proposal commit_head '{expected_head}' "
                f"!= current HEAD '{cur_rev.commit_head}' (stale_proposal)."
            ],
            source_revision=source_rev_dict,
            summary="Implementasi dibatalkan: proposal menggunakan source revision yang sudah usang (stale_proposal).",
        )

    # Extract proposed files & commands
    files_to_modify_objs = prop_inner.get("files_to_modify", [])
    files_to_create_objs = prop_inner.get("files_to_create", [])
    files_to_delete_objs = prop_inner.get("files_to_delete", [])
    tests_to_modify_objs = prop_inner.get("tests_to_modify", [])
    tests_to_create_objs = prop_inner.get("tests_to_create", [])

    files_to_modify = [
        f["path"] if isinstance(f, dict) else f.path for f in files_to_modify_objs
    ]
    files_to_create = [
        f["path"] if isinstance(f, dict) else f.path for f in files_to_create_objs
    ]
    files_to_delete = [
        f["path"] if isinstance(f, dict) else f.path for f in files_to_delete_objs
    ]
    tests_to_modify = [
        t["path"] if isinstance(t, dict) else t.path for t in tests_to_modify_objs
    ]
    tests_to_create = [
        t["path"] if isinstance(t, dict) else t.path for t in tests_to_create_objs
    ]

    approved_files = []
    if proposal_review and "approved_writable_files" in proposal_review and proposal_review.get("approved_writable_files"):
        approved_files = list(proposal_review.get("approved_writable_files"))
    elif proposal_review and "approved_files" in proposal_review and proposal_review.get("approved_files"):
        approved_files = list(proposal_review.get("approved_files"))
    elif prop_inner.get("approved_writable_files"):
        approved_files = list(prop_inner.get("approved_writable_files"))
    else:
        approved_files = sorted(list(set(files_to_modify + files_to_create + tests_to_modify + tests_to_create)))

    all_target_files = sorted(
        set(files_to_modify + files_to_create + files_to_delete + tests_to_modify + tests_to_create)
    )

    # 7. Project policy check
    denied_patterns = policy.security.denied_paths
    for p in all_target_files:
        for den in denied_patterns:
            den_clean = den.replace("**/", "").replace("/**", "").strip("/")
            if den_clean and den_clean in p:
                errors.append(f"Target file '{p}' melanggar security policy denied_paths: '{den}'")

        if policy.security.deny_hidden_files_by_default:
            parts = Path(p).parts
            if any(part.startswith(".") for part in parts):
                errors.append(f"Target file '{p}' adalah hidden file yang ditolak policy.")

    # 8 & 9. Path validation
    for p in all_target_files:
        try:
            validate_relative_pattern(p)
            resolved = resolve_project_path(project_root, p)
            resolved.relative_to(project_root)
        except Exception as e:
            errors.append(f"Target file '{p}' tidak valid atau keluar dari project root: {e}")

    # 10. Component check
    comp_roots = [c.root.rstrip("/") for c in manifest.components.values()]
    for p in all_target_files:
        matches_component = any(
            p == root or p.startswith(f"{root}/") for root in comp_roots
        )
        if not matches_component and comp_roots:
            errors.append(
                f"Target file '{p}' tidak berada di dalam komponen terdaftar ({comp_roots})."
            )

    # 11 & 12. Validation commands check
    val_cmds = prop_inner.get("validation_commands", [])
    for cmd in val_cmds:
        exe = cmd.get("executable") if isinstance(cmd, dict) else cmd.executable
        args = cmd.get("arguments", []) if isinstance(cmd, dict) else cmd.arguments
        cwd = cmd.get("working_directory", "") if isinstance(cmd, dict) else cmd.working_directory

        if not exe or exe.lower() not in ALLOWED_VALIDATION_EXECUTABLES:
            errors.append(
                f"Executable '{exe}' tidak diizinkan. Allowlist: {sorted(ALLOWED_VALIDATION_EXECUTABLES)}"
            )
        if exe and exe.lower() in FORBIDDEN_EXECUTABLES:
            errors.append(f"Executable '{exe}' secara eksplisit dilarang.")

        for arg in args:
            if SHELL_INJECTION_PATTERN.search(str(arg)):
                errors.append(f"Command argument '{arg}' mengandung karakter shell terlarang.")

        try:
            validate_relative_pattern(cwd)
            cwd_path = resolve_project_path(project_root, cwd)
            if not cwd_path.is_dir():
                errors.append(f"Working directory command '{cwd}' tidak ditemukan di disk.")
        except Exception as e:
            errors.append(f"Working directory command '{cwd}' tidak valid: {e}")

    # 14. File to modify existence
    for p in files_to_modify:
        target_f = project_root / p
        if not target_f.is_file():
            errors.append(f"File yang akan diubah tidak ditemukan: '{p}'")

    # 15. Symlink check
    for p in all_target_files:
        if is_symlink_escaping(project_root, p):
            errors.append(f"Target file atau jalurnya adalah symlink yang keluar dari project: '{p}'")

    # 16. Secret file check
    for p in all_target_files:
        if is_secret_file_path(p):
            errors.append(f"Target file '{p}' terdeteksi sebagai secret/credential file.")

    if errors:
        return PreflightResult(
            is_valid=False,
            status="rejected",
            errors=errors,
            warnings=warnings,
            approved_files=approved_files,
            files_to_modify=files_to_modify,
            files_to_create=files_to_create,
            validation_commands=val_cmds,
            source_revision=source_rev_dict,
            summary=f"Preflight gagal dengan {len(errors)} kesalahan.",
        )

    # 13. Source repository clean for target files
    dirty_targets = check_git_dirty_for_files(project_root, all_target_files)
    if dirty_targets:
        return PreflightResult(
            is_valid=False,
            status="human_required",
            errors=[
                f"Target file memiliki perubahan lokal yang belum di-commit: {dirty_targets}"
            ],
            approved_files=approved_files,
            files_to_modify=files_to_modify,
            files_to_create=files_to_create,
            source_revision=source_rev_dict,
            summary=(
                f"Implementasi dihentikan: target file {dirty_targets} sedang memiliki "
                f"perubahan lokal. Diperlukan tindakan manual (human_required)."
            ),
        )

    # Scope inspection
    task_title = prop_inner.get("task_title", "")
    outside_files = inspect_scope_requirements(
        project_root,
        approved_files,
        task_title,
        prop_inner,
    )
    if outside_files:
        outside_scope_paths = [f[0] for f in outside_files]
        reasons = [f"{f[0]}: {f[1]}" for f in outside_files]
        return PreflightResult(
            is_valid=True,
            status="proposal_revision_required",
            errors=[],
            warnings=reasons,
            approved_files=approved_files,
            files_to_modify=files_to_modify,
            files_to_create=files_to_create,
            files_requested_outside_scope=outside_scope_paths,
            validation_commands=val_cmds,
            source_revision=source_rev_dict,
            summary=(
                f"Preflight mendeteksi implementasi membutuhkan {len(outside_scope_paths)} file "
                f"di luar proposal yang disetujui ({outside_scope_paths}). Diperlukan proposal_revision_required."
            ),
        )

    return PreflightResult(
        is_valid=True,
        status="preflight_passed",
        errors=[],
        warnings=warnings,
        approved_files=approved_files,
        files_to_modify=files_to_modify,
        files_to_create=files_to_create,
        validation_commands=val_cmds,
        source_revision=source_rev_dict,
        summary="Preflight validasi berhasil: semua kondisi siap untuk implementasi terisolasi.",
    )


def inspect_scope_requirements(
    project_root: Path,
    approved_files: list[str],
    task_title: str,
    proposal_inner: dict[str, Any],
) -> list[tuple[str, str]]:
    """Inspect whether implementing the proposal requires files outside approved_files."""
    outside_files: list[tuple[str, str]] = []
    approved_set = set(approved_files)

    # 1. Any file/test declared in proposal that is missing from approved_files
    for key, label in [
        ("files_to_modify", "File to modify"),
        ("files_to_create", "File to create"),
        ("tests_to_modify", "Test to modify"),
        ("tests_to_create", "Test to create"),
    ]:
        for item in proposal_inner.get(key, []):
            f_path = item.get("path") if isinstance(item, dict) else getattr(item, "path", None)
            if f_path and f_path not in approved_set:
                if not any(f[0] == f_path for f in outside_files):
                    outside_files.append((
                        f_path,
                        f"{label} '{f_path}' tercantum dalam proposal namun tidak termasuk dalam approved_files.",
                    ))

    # 2. Check implementation steps affected_files
    for step in proposal_inner.get("implementation_steps", []):
        aff_files = step.get("affected_files", []) if isinstance(step, dict) else getattr(step, "affected_files", [])
        for f_path in aff_files:
            if f_path and f_path not in approved_set:
                if not any(f[0] == f_path for f in outside_files):
                    outside_files.append((
                        f_path,
                        f"File '{f_path}' tercantum dalam affected_files namun tidak termasuk dalam approved_files.",
                    ))

    # 3. Check explicitly requested outside scope files
    for f_path in proposal_inner.get("files_requested_outside_scope", []):
        if f_path and f_path not in approved_set:
            if not any(f[0] == f_path for f in outside_files):
                outside_files.append((
                    f_path,
                    f"File '{f_path}' diminta untuk diubah namun berada di luar approved_files.",
                ))

    return outside_files

