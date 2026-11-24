import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceRevision:
    commit_head: str | None
    is_dirty: bool | None
    warning: str | None = None


def get_source_revision(project_root: Path, timeout: float = 5.0) -> SourceRevision:
    """Get Git commit hash and dirty status for the project safely.
    
    Only runs:
    - git rev-parse HEAD
    - git status --porcelain
    Does not run any other git commands.
    """
    root = project_root.expanduser().resolve()

    try:
        # Check commit HEAD
        res_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res_head.returncode != 0:
            return SourceRevision(
                commit_head=None,
                is_dirty=None,
                warning=f"Git revision unavailable (not a git repository or no commits): {res_head.stderr.strip()}",
            )
        commit = res_head.stdout.strip()

        # Check dirty status
        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        is_dirty = bool(res_status.stdout.strip()) if res_status.returncode == 0 else False

        return SourceRevision(
            commit_head=commit,
            is_dirty=is_dirty,
            warning=None,
        )
    except FileNotFoundError:
        return SourceRevision(
            commit_head=None,
            is_dirty=None,
            warning="git binary not found on system PATH",
        )
    except subprocess.TimeoutExpired:
        return SourceRevision(
            commit_head=None,
            is_dirty=None,
            warning="git revision query timed out",
        )
    except Exception as e:
        return SourceRevision(
            commit_head=None,
            is_dirty=None,
            warning=f"Error checking git revision: {e}",
        )
