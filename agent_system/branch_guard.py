import re
import subprocess
import uuid
from pathlib import Path

BRANCH_NAME_SAFE_PATTERN = re.compile(r"^agent/[a-z0-9]+(-[a-z0-9]+)*-[a-f0-9]{8}(-[a-f0-9]{4})?$")
TASK_SLUG_CLEAN_PATTERN = re.compile(r"[^a-z0-9]+")


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


def validate_branch_name(branch_name: str) -> bool:
    """Validate branch name against strict safety regex."""
    if not branch_name or not isinstance(branch_name, str):
        return False
    # Reject spaces, shell metacharacters, path traversal
    if any(c in branch_name for c in [" ", ";", "&", "|", "`", "$", "..", "\\", "*", "?", "~", "^", ":"]):
        return False
    return bool(BRANCH_NAME_SAFE_PATTERN.match(branch_name))


class BranchGuard:
    """Safely manages isolated Git branch creation without commit, push, or merge."""

    def __init__(self, project_root: Path, base_branches: list[str] | None = None):
        self.project_root = project_root.expanduser().resolve()
        self.base_branches = base_branches or ["main", "master"]

    def get_current_branch(self, timeout: float = 5.0) -> str:
        """Get current active Git branch safely."""
        res = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.project_root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Gagal membaca branch saat ini: {res.stderr.strip()}")
        branch = res.stdout.strip()
        if not branch:
            # Check rev-parse for detached HEAD
            res_rev = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            branch = res_rev.stdout.strip() if res_rev.returncode == 0 else ""
        return branch

    def branch_exists(self, branch_name: str, timeout: float = 5.0) -> bool:
        """Check if branch already exists in local repository."""
        res = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            cwd=self.project_root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return res.returncode == 0

    def create_isolated_branch(
        self,
        task_title: str,
        run_id: str,
        timeout: float = 10.0,
    ) -> str:
        """Validate base branch, generate unique branch name, and switch to it."""
        current_branch = self.get_current_branch(timeout=timeout)
        if current_branch not in self.base_branches:
            raise ValueError(
                f"Implementasi ditolak: branch saat ini adalah '{current_branch}', "
                f"harus salah satu dari base branch {self.base_branches}."
            )

        branch_name = generate_branch_name(task_title, run_id)
        if not validate_branch_name(branch_name):
            raise ValueError(f"Branch name '{branch_name}' tidak valid menurut aturan keamanan.")

        # If branch already exists, append unique suffix
        if self.branch_exists(branch_name, timeout=timeout):
            suffix = uuid.uuid4().hex[:4]
            branch_name = generate_branch_name(task_title, run_id, suffix=suffix)
            if not validate_branch_name(branch_name):
                raise ValueError(f"Suffixed branch name '{branch_name}' tidak valid.")
            if self.branch_exists(branch_name, timeout=timeout):
                raise ValueError(f"Branch '{branch_name}' sudah ada di repository.")

        # Execute fixed command: git switch -c <branch-name>
        res = subprocess.run(
            ["git", "switch", "-c", branch_name],
            cwd=self.project_root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Gagal membuat branch '{branch_name}': {res.stderr.strip()}")

        return branch_name
