import subprocess
from pathlib import Path

import pytest

from agent_system.branch_guard import (
    BranchGuard,
    generate_branch_name,
    validate_branch_name,
)


def init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=path, check=True, capture_output=True)
    return path


def test_11_safe_branch_name_generated():
    """11. Branch name aman dibuat."""
    name = generate_branch_name("Implementasi Persistensi Riwayat Transaksi", "a1b2c3d4e5f6")
    assert name.startswith("agent/implementasi-persistensi-")
    assert name.endswith("-a1b2c3d4")
    assert validate_branch_name(name) is True


def test_12_dangerous_branch_name_rejected():
    """12. Branch name berbahaya ditolak."""
    assert validate_branch_name("agent/branch;rm -rf /") is False
    assert validate_branch_name("agent/branch && rm -rf /") is False
    assert validate_branch_name("agent/../escape") is False
    assert validate_branch_name("agent/branch with space") is False
    assert validate_branch_name("agent/branch$name") is False
    assert validate_branch_name("agent/branch`id`") is False


def test_13_existing_branch_not_overwritten(tmp_path):
    """13. Existing branch tidak dioverwrite."""
    repo = init_git_repo(tmp_path / "repo")
    guard = BranchGuard(repo)

    # First creation
    b1 = guard.create_isolated_branch("My Task", "12345678")
    assert b1.startswith("agent/my-task-12345678")

    # Switch back to main
    subprocess.run(["git", "switch", "main"], cwd=repo, check=True, capture_output=True)

    # Second creation with same parameters gets unique suffix
    b2 = guard.create_isolated_branch("My Task", "12345678")
    assert b2 != b1
    assert b2.startswith("agent/my-task-12345678-")


def test_14_main_not_modified_directly(tmp_path):
    """14. Main tidak dimodifikasi langsung (harus di branch terisolasi)."""
    repo = init_git_repo(tmp_path / "repo")
    guard = BranchGuard(repo)

    current_before = guard.get_current_branch()
    assert current_before == "main"

    new_branch = guard.create_isolated_branch("New Task", "87654321")
    current_after = guard.get_current_branch()
    assert current_after == new_branch
    assert current_after != "main"


def test_15_no_commits_made(tmp_path):
    """15. Tidak ada commit dibuat oleh branch guard."""
    repo = init_git_repo(tmp_path / "repo")
    head_before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    guard = BranchGuard(repo)
    guard.create_isolated_branch("New Task", "abcdef12")

    head_after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert head_before == head_after


def test_16_no_push_executed(tmp_path):
    """16. Tidak ada push ke remote."""
    repo = init_git_repo(tmp_path / "repo")
    guard = BranchGuard(repo)
    guard.create_isolated_branch("New Task", "abcdef12")

    # Remotes check
    remotes = subprocess.run(["git", "remote"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert remotes == ""


def test_17_no_merge_executed(tmp_path):
    """17. Tidak ada merge dijalankan."""
    repo = init_git_repo(tmp_path / "repo")
    guard = BranchGuard(repo)
    guard.create_isolated_branch("Task 1", "11112222")

    log_count = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert log_count == "1"
