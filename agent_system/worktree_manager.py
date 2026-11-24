import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent_system.branch_guard import (
    BRANCH_NAME_SAFE_PATTERN,
    TASK_SLUG_CLEAN_PATTERN,
    validate_branch_name,
)

SHELL_OPERATORS = ["|", ";", "&", ">", "<", "`", "$", "\n", "\r", "*", "?", "~", "!", "\t"]


def generate_task_slug(task_title: str, max_words: int = 3) -> str:
    """Generate a clean kebab-case task slug from task title."""
    clean = TASK_SLUG_CLEAN_PATTERN.sub("-", task_title.lower()).strip("-")
    words = [w for w in clean.split("-") if w][:max_words]
    slug = "-".join(words) if words else "task"
    return slug[:35].strip("-")


def generate_branch_name(task_title: str, run_id: str, suffix: str | None = None) -> str:
    """Generate safe branch name according to protocol."""
    slug = generate_task_slug(task_title)
    short_id = run_id[:8].lower()
    if suffix:
        short_suffix = suffix[:4].lower()
        branch_name = f"agent/{slug}-{short_id}-{short_suffix}"
    else:
        branch_name = f"agent/{slug}-{short_id}"
    return branch_name


def generate_worktree_name(project_root: Path, run_id: str) -> str:
    """Generate unique worktree directory name <project_name>-<short-run-id>."""
    short_id = run_id[:8].lower()
    proj_name = project_root.name.strip()
    return f"{proj_name}-{short_id}"


def get_default_worktree_root(project_root: Path, custom_root: Path | str | None = None) -> Path:
    """Determine base directory for git worktrees with strict precedence."""
    if custom_root:
        return Path(custom_root).expanduser().resolve()
    env_root = os.environ.get("AGENT_WORKTREE_ROOT")
    if env_root and env_root.strip():
        return Path(env_root).expanduser().resolve()
    return (project_root.expanduser().resolve().parent / ".agent-worktrees").resolve()


def validate_repository(project_root: Path, timeout: float = 5.0) -> str:
    """Validate that project_root is a valid git repository and return commit HEAD."""
    root_resolved = project_root.expanduser().resolve()
    if not root_resolved.is_dir():
        raise ValueError(f"Project root '{project_root}' bukan direktori yang valid.")

    res_top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root_resolved,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res_top.returncode != 0:
        raise ValueError(f"Direktori '{project_root}' bukan Git repository: {res_top.stderr.strip()[:200]}")

    git_top = Path(res_top.stdout.strip()).resolve()
    if git_top != root_resolved:
        raise ValueError(f"Git toplevel '{git_top}' tidak cocok dengan project root '{root_resolved}'.")

    res_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root_resolved,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res_head.returncode != 0:
        raise RuntimeError(f"Gagal membaca commit HEAD repository: {res_head.stderr.strip()[:200]}")

    return res_head.stdout.strip()

def validate_worktree_path(worktree_path: Path, project_root: Path) -> Path:
    """Validate that worktree path satisfies all isolation and safety constraints."""
    resolved_wt = worktree_path.expanduser().resolve()
    resolved_pr = project_root.expanduser().resolve()

    raw_str = str(worktree_path)
    if any(op in raw_str for op in SHELL_OPERATORS):
        raise ValueError(f"Worktree path '{worktree_path}' mengandung karakter berbahaya atau operator command.")
    if ".." in worktree_path.parts or "/../" in raw_str or raw_str.endswith("/.."):
        raise ValueError(f"Worktree path '{worktree_path}' mengandung path traversal '..'.")

    if resolved_wt == resolved_pr:
        raise ValueError("Worktree path tidak boleh sama dengan folder project utama.")
    try:
        resolved_wt.relative_to(resolved_pr)
        raise ValueError(f"Worktree path '{worktree_path}' berada di dalam folder project utama '{project_root}'.")
    except ValueError as e:
        if "berada di dalam" in str(e):
            raise

    if resolved_wt.exists() and resolved_wt.is_dir():
        if any(resolved_wt.iterdir()):
            raise ValueError(f"Worktree path '{worktree_path}' sudah ada dan tidak kosong.")
    elif resolved_wt.exists() and not resolved_wt.is_dir():
        raise ValueError(f"Worktree path '{worktree_path}' sudah ada dan bukan direktori.")

    return resolved_wt


def get_worktree_status(worktree_path: Path, timeout: float = 10.0) -> dict[str, Any]:
    """Check uncommitted changes in worktree safely."""
    resolved_wt = worktree_path.expanduser().resolve()
    if not resolved_wt.is_dir():
        return {"clean": True, "has_changes": False, "lines": []}

    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=resolved_wt,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Gagal memeriksa status worktree: {res.stderr.strip()[:200]}")

    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    return {
        "clean": len(lines) == 0,
        "has_changes": len(lines) > 0,
        "lines": lines,
    }

def inspect_worktree(
    worktree_path: Path,
    approved_files: list[str] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Inspect status, diff stat, diff check, and name-only in worktree safely."""
    resolved_wt = worktree_path.expanduser().resolve()
    if not resolved_wt.is_dir():
        raise ValueError(f"Worktree path '{worktree_path}' tidak ditemukan.")

    res_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved_wt,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    head_rev = res_head.stdout.strip() if res_head.returncode == 0 else ""

    res_stat = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=resolved_wt,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    status_output = res_stat.stdout.strip()

    res_check = subprocess.run(
        ["git", "diff", "--check"],
        cwd=resolved_wt,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    res_diff_stat = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=resolved_wt,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    res_name_only = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=resolved_wt,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    changed_approved = ""
    if approved_files:
        cmd_app = ["git", "diff", "--"] + approved_files
        res_app = subprocess.run(
            cmd_app,
            cwd=resolved_wt,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        changed_approved = res_app.stdout.strip()

    return {
        "head": head_rev,
        "status_porcelain": status_output,
        "diff_check_passed": res_check.returncode == 0,
        "diff_check_output": res_check.stdout.strip() or res_check.stderr.strip(),
        "diff_stat": res_diff_stat.stdout.strip(),
        "changed_files": [l.strip() for l in res_name_only.stdout.splitlines() if l.strip()],
        "approved_diff": changed_approved,
    }


def create_worktree(
    project_root: Path,
    branch_name: str,
    worktree_path: Path,
    base_revision: str | None = None,
    timeout: float = 30.0,
) -> Path:
    """Safely create isolated Git worktree without modifying branch of main project."""
    head_rev = validate_repository(project_root, timeout=timeout)
    if not validate_branch_name(branch_name):
        raise ValueError(f"Branch name '{branch_name}' tidak valid menurut aturan keamanan.")

    resolved_pr = project_root.expanduser().resolve()
    resolved_wt = validate_worktree_path(worktree_path, resolved_pr)

    res_br = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=resolved_pr,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res_br.stdout.strip():
        raise ValueError(f"Branch '{branch_name}' sudah ada di repository.")

    res_wt_list = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=resolved_pr,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in res_wt_list.stdout.splitlines():
        if line.startswith("worktree "):
            wt_entry = line.split(maxsplit=1)[1].strip()
            if Path(wt_entry).resolve() == resolved_wt:
                raise ValueError(f"Worktree path '{resolved_wt}' sudah terdaftar dalam Git worktree.")

    target_base = base_revision or head_rev
    resolved_wt.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "worktree",
        "add",
        "-b",
        branch_name,
        str(resolved_wt),
        target_base,
    ]
    res_add = subprocess.run(
        cmd,
        cwd=resolved_pr,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res_add.returncode != 0:
        err_msg = res_add.stderr.strip() or res_add.stdout.strip()
        raise RuntimeError(f"Gagal membuat git worktree: {err_msg[:500]}")

    return resolved_wt


def remove_worktree(
    project_root: Path,
    worktree_path: Path,
    force: bool = False,
    timeout: float = 15.0,
) -> None:
    """Safely clean up worktree directory only if it has no changes (or force=True)."""
    resolved_wt = worktree_path.expanduser().resolve()
    if not resolved_wt.exists():
        return

    status = get_worktree_status(resolved_wt, timeout=timeout)
    if status.get("has_changes") and not force:
        raise ValueError(
            f"Worktree '{worktree_path}' memiliki perubahan yang belum disimpan. Penghapusan ditolak."
        )

    shutil.rmtree(resolved_wt, ignore_errors=True)
