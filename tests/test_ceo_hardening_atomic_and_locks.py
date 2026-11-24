import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.atomic_state import (
    StateCorruptionError,
    atomic_write_json,
    compute_checksum,
    read_json_with_backup_recovery,
)
from agent_system.integration_lock import (
    IntegrationLock,
    IntegrationLockActiveError,
    TargetHeadChangedError,
)
from agent_system.program_lock import (
    ProgramLock,
    ProgramLockActiveError,
)
from agent_system.project_initializer import initialize_project


def setup_test_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test Repo\n")
    (path / "backend").mkdir(parents=True, exist_ok=True)
    (path / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"ok\" }\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    initialize_project(path, apply=True)
    return path


# ==============================================================================
# Category A: Atomic State Tests (1 - 6)
# ==============================================================================

def test_01_atomic_write_produces_valid_json(tmp_path: Path):
    target = tmp_path / "state.json"
    data = {"program_id": "test01", "status": "ready"}
    written = atomic_write_json(target, data, backup=True, monotonic_revision=True)
    assert written.is_file()

    with open(written, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["program_id"] == "test01"
    assert loaded["state_revision"] == 1
    assert "checksum" in loaded


def test_02_interrupted_temporary_write_does_not_corrupt_final_state(tmp_path: Path):
    target = tmp_path / "program_valid.json"
    initial_data = {"program_id": "prog_ok", "status": "completed", "completed_tasks": ["t1"]}
    atomic_write_json(target, initial_data, backup=True, monotonic_revision=True)

    # Simulate crash/interrupted write leaving a tmp file
    tmp_file = tmp_path / f".tmp_{target.name}_crash.json"
    tmp_file.write_text("{ incomplete json...")

    # Final file must remain 100% valid and untainted
    data, recovered = read_json_with_backup_recovery(target)
    assert data["program_id"] == "prog_ok"
    assert data["status"] == "completed"
    assert not recovered


def test_03_corrupt_current_state_recovered_from_valid_backup(tmp_path: Path):
    target = tmp_path / "program_recover.json"
    valid_data = {"program_id": "prog_rec", "status": "in_progress", "tasks": ["t1", "t2"]}
    atomic_write_json(target, valid_data, backup=True, monotonic_revision=True)

    # Next update creates valid backup
    updated_data = {"program_id": "prog_rec", "status": "in_progress", "tasks": ["t1", "t2", "t3"]}
    atomic_write_json(target, updated_data, backup=True, monotonic_revision=True)

    # Corrupt target file directly (e.g. 0 bytes or garbage)
    target.write_text("{ corrupt broken json !!")

    # Recovery must restore from .bak
    data, recovered = read_json_with_backup_recovery(target)
    assert recovered is True
    assert data["program_id"] == "prog_rec"
    # Target file should now also be restored and readable
    assert json.loads(target.read_text())["program_id"] == "prog_rec"


def test_04_current_and_backup_corrupt_produces_fail_closed(tmp_path: Path):
    target = tmp_path / "program_fail.json"
    target.write_text("{ corrupt 1")
    bak = tmp_path / f"{target.name}.bak"
    bak.write_text("{ corrupt 2")

    with pytest.raises(StateCorruptionError, match="FAIL-CLOSED"):
        read_json_with_backup_recovery(target)


def test_05_state_revision_increments_monotonically(tmp_path: Path):
    target = tmp_path / "program_mono.json"
    data = {"program_id": "p1"}
    atomic_write_json(target, data, backup=True, monotonic_revision=True)
    rev1 = json.loads(target.read_text())["state_revision"]
    assert rev1 == 1

    atomic_write_json(target, data, backup=True, monotonic_revision=True)
    rev2 = json.loads(target.read_text())["state_revision"]
    assert rev2 == 2

    atomic_write_json(target, data, backup=True, monotonic_revision=True)
    rev3 = json.loads(target.read_text())["state_revision"]
    assert rev3 == 3


def test_06_secret_not_stored_in_backup(tmp_path: Path):
    from agent_system.artifact_store import sanitize_artifact_payload

    target = tmp_path / "program_sec.json"
    payload = {
        "program_id": "sec_test",
        "api_key": "sk-1234567890abcdefsecret",
        "token": "Bearer supersecret123",
        "raw_diff": "diff --git secret line",
    }
    sanitized = sanitize_artifact_payload(payload)
    atomic_write_json(target, sanitized, backup=True, monotonic_revision=True)
    # Trigger backup creation on second write
    atomic_write_json(target, sanitized, backup=True, monotonic_revision=True)

    bak = tmp_path / f"{target.name}.bak"
    assert bak.is_file()
    bak_content = bak.read_text()
    assert "sk-1234567890abcdefsecret" not in bak_content
    assert "supersecret123" not in bak_content
    assert "diff --git" not in bak_content


# ==============================================================================
# Category B: Program Lock Tests (7 - 12)
# ==============================================================================

def test_07_two_runners_cannot_hold_same_program_lock(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    lock1 = ProgramLock(repo, "prog_dual")
    lock1.acquire()

    lock2 = ProgramLock(repo, "prog_dual")
    with pytest.raises(ProgramLockActiveError, match="sedang dijalankan oleh proses aktif"):
        lock2.acquire()

    lock1.release()


def test_08_active_lock_cannot_be_usurped(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    lock = ProgramLock(repo, "prog_active")
    lock.acquire()

    # Second process on same host
    other_runner = ProgramLock(repo, "prog_active")
    insp = other_runner.inspect()
    assert insp["is_active"] is True
    assert insp["is_stale"] is False

    with pytest.raises(ProgramLockActiveError):
        other_runner.acquire()

    lock.release()


def test_09_stale_lock_detected_via_process_identity(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    lock_file = repo / ".agent" / "locks" / "program_prog_stale.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    # Write lock with a dead PID (e.g. 99999999)
    metadata = {
        "program_id": "prog_stale",
        "project_id": "repo",
        "process_id": 99999999,
        "hostname": lock_file.name,
        "created_at": "2026-01-01T00:00:00Z",
        "heartbeat_at": "2026-01-01T00:00:00Z",
    }
    lock_file.write_text(json.dumps(metadata))

    checker = ProgramLock(repo, "prog_stale")
    insp = checker.inspect()
    assert insp["is_stale"] is True
    assert insp["is_active"] is False


def test_10_pid_reuse_not_considered_same_owner(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    lock_file = repo / ".agent" / "locks" / "program_prog_pid_reuse.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    checker = ProgramLock(repo, "prog_pid_reuse")
    current_pid = os.getpid()
    metadata = {
        "program_id": "prog_pid_reuse",
        "project_id": "repo",
        "process_id": current_pid,
        "hostname": checker._hostname,
        "created_at": "2026-01-01T00:00:00Z",
        "heartbeat_at": "2026-01-01T00:00:00Z",
        "process_start_identifier": "Wed Jan 01 00:00:00 1970",
    }
    lock_file.write_text(json.dumps(metadata))

    insp = checker.inspect()
    # Different start identifier means PID was reused -> lock is stale!
    assert insp["is_stale"] is True


def test_11_graceful_shutdown_releases_lock_after_checkpoint(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    lock = ProgramLock(repo, "prog_shutdown")
    lock.acquire()
    assert lock.lock_file.is_file()

    # Release
    lock.release()
    assert not lock.lock_file.is_file()
    assert lock.owned is False


def test_12_crash_leaves_lock_safely_recoverable(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    dead_pid = 99999998
    lock_file = repo / ".agent" / "locks" / "program_prog_crash.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "program_id": "prog_crash",
        "project_id": "repo",
        "process_id": dead_pid,
        "hostname": (new_runner := ProgramLock(repo, "prog_crash"))._hostname,
        "created_at": "2026-01-01T00:00:00Z",
        "heartbeat_at": "2026-01-01T00:00:00Z",
    }
    lock_file.write_text(json.dumps(metadata))

    # New runner should safely recover and take the lock
    acquired_meta = new_runner.acquire()
    assert acquired_meta["process_id"] == os.getpid()
    assert new_runner.owned is True
    new_runner.release()


# ==============================================================================
# Category C: Integration Lock Tests (13 - 17)
# ==============================================================================

def test_13_only_one_integration_apply_runs_per_target_branch(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    lock1 = IntegrationLock(repo, "repo", target_branch="main")
    lock1.acquire(task_id="t1")

    lock2 = IntegrationLock(repo, "repo", target_branch="main")
    with pytest.raises(IntegrationLockActiveError, match="sedang berjalan"):
        lock2.acquire(task_id="t2")

    lock1.release()


def test_14_target_head_change_triggers_revalidation(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    initial_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    # Move target HEAD by committing
    (repo / "file_new.txt").write_text("concurrent edit\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "concurrent commit"], cwd=repo, check=True)

    # Attempt to acquire with old expected head
    lock = IntegrationLock(repo, "repo", target_branch="main")
    with pytest.raises(TargetHeadChangedError, match="Target branch HEAD berubah"):
        lock.acquire(expected_target_head=initial_head)


def test_15_integration_lock_persists_during_post_validation(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "repo")
    lock = IntegrationLock(repo, "repo", target_branch="main")
    lock.acquire(task_id="t15")

    lock.update_phase("post_validation")
    insp = lock.inspect()
    assert insp["integration_phase"] == "post_validation"
    assert insp["is_active"] is True

    lock.release()


def test_16_completed_integration_not_repeated_after_crash(tmp_path: Path):
    from agent_system.recovery_inspector import inspect_recovery_conditions, RecoveryAction
    from agent_system.schemas import ProgramState, ProgramTask

    repo = setup_test_git_repo(tmp_path / "repo")
    state = ProgramState(
        program_id="p16",
        project_id="repo",
        objective="Obj",
        active_task="t1",
        active_run_id="run_16",
        tasks=[
            ProgramTask(
                task_id="t1",
                epic_id="e1",
                title="T1",
                status="in_progress",
                linked_run_id="run_16",
            )
        ],
    )

    # Write completed integration artifact
    integ_file = repo / ".agent" / "artifacts" / "executions" / "integration_run_16.json"
    integ_file.parent.mkdir(parents=True, exist_ok=True)
    integ_file.write_text(json.dumps({"status": "completed", "result": {"status": "completed"}}))

    res = inspect_recovery_conditions(repo, state)
    assert res["integration_completed"] is True
    assert res["recommended_action"] == RecoveryAction.SKIP_INTEGRATION_MARK_COMPLETED


def test_17_lock_recovery_inspects_cherry_pick_state(tmp_path: Path):
    from agent_system.recovery_inspector import inspect_recovery_conditions, RecoveryAction
    from agent_system.schemas import ProgramState

    repo = setup_test_git_repo(tmp_path / "repo")
    state = ProgramState(program_id="p17", project_id="repo", objective="Obj")

    # Simulate active cherry-pick state file
    cp_head = repo / ".git" / "CHERRY_PICK_HEAD"
    cp_head.write_text("commit_hash_123\n")

    res = inspect_recovery_conditions(repo, state)
    assert res["is_cherry_pick_active"] is True
    assert res["recommended_action"] == RecoveryAction.RECOVER_CHERRY_PICK_CONFLICT
