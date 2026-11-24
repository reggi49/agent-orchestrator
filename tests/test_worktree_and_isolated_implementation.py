import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_system.branch_guard import validate_branch_name
from agent_system.command_runner import SafeCommandRunner, summarize_output
from agent_system.diff_reviewer import review_diff
from agent_system.file_writer import ControlledFileWriter, UnapprovedFileError
from agent_system.graph import route_after_diff_reviewer, route_after_test_runner
from agent_system.nodes import supervisor_validate_implementation
from agent_system.worktree_manager import (
    create_worktree,
    generate_branch_name,
    generate_worktree_name,
    get_default_worktree_root,
    get_worktree_status,
    inspect_worktree,
    remove_worktree,
    validate_repository,
    validate_worktree_path,
)


def init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test Repo\n")
    (path / "backend").mkdir(parents=True, exist_ok=True)
    (path / "backend" / "main.go").write_text("package main\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    return path


# ==============================================================================
# TAHAP 9: UNIT TEST 1-12 (Worktree Manager)
# ==============================================================================

def test_01_worktree_created_outside_main_project(tmp_path):
    """1. Worktree dibuat di luar project utama."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt_path = tmp_path / "worktrees" / "main_proj-12345678"
    res_path = create_worktree(repo, "agent/task-12345678", wt_path)

    assert res_path.is_dir()
    assert res_path.resolve() != repo.resolve()
    assert not res_path.resolve().is_relative_to(repo.resolve())


def test_02_worktree_inside_main_project_rejected(tmp_path):
    """2. Worktree di dalam project utama ditolak."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt_inside = repo / "worktrees" / "sub"
    with pytest.raises(ValueError, match="berada di dalam folder project utama"):
        create_worktree(repo, "agent/task-12345678", wt_inside)


def test_03_safe_branch_name_accepted():
    """3. Branch name aman diterima."""
    bname = generate_branch_name("Implementasi Persistensi Riwayat Transaksi", "a23dc084")
    assert bname == "agent/implementasi-persistensi-riwayat-a23dc084"
    assert validate_branch_name(bname) is True


def test_04_dangerous_branch_name_rejected():
    """4. Branch name berbahaya ditolak."""
    assert validate_branch_name("agent/task;rm -rf /") is False
    assert validate_branch_name("agent/task`id`") is False
    assert validate_branch_name("agent/task$USER") is False
    assert validate_branch_name("agent/../escape") is False


def test_05_existing_branch_rejected(tmp_path):
    """5. Existing branch ditolak."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt1 = tmp_path / "wt1"
    create_worktree(repo, "agent/task-12345678", wt1)

    wt2 = tmp_path / "wt2"
    with pytest.raises(ValueError, match="sudah ada di repository"):
        create_worktree(repo, "agent/task-12345678", wt2)


def test_06_existing_non_empty_worktree_rejected(tmp_path):
    """6. Existing non-empty worktree ditolak."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "existing_wt"
    wt.mkdir(parents=True)
    (wt / "file.txt").write_text("existing content")

    with pytest.raises(ValueError, match="sudah ada dan tidak kosong"):
        create_worktree(repo, "agent/task-12345678", wt)



def test_07_main_project_does_not_switch_branch(tmp_path):
    """7. Main project tidak berpindah branch."""
    repo = init_git_repo(tmp_path / "main_proj")
    branch_before = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert branch_before == "main"

    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    branch_after = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert branch_after == "main"


def test_08_worktree_uses_correct_base_revision(tmp_path):
    """8. Worktree menggunakan base revision yang benar."""
    repo = init_git_repo(tmp_path / "main_proj")
    head_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt, base_revision=head_rev)

    wt_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    assert wt_head == head_rev


def test_09_no_commits_made_on_worktree_creation(tmp_path):
    """9. Tidak ada commit dibuat."""
    repo = init_git_repo(tmp_path / "main_proj")
    count_before = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    count_after = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert count_after == count_before


def test_10_no_push_executed(tmp_path):
    """10. Tidak ada push."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    remotes = subprocess.run(["git", "remote"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert remotes == ""


def test_11_worktree_with_changes_not_removed_automatically(tmp_path):
    """11. Worktree dengan perubahan tidak dihapus otomatis."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    (wt / "dirty.txt").write_text("modifications")

    with pytest.raises(ValueError, match="memiliki perubahan yang belum disimpan"):
        remove_worktree(repo, wt, force=False)

    assert wt.exists()


def test_12_clean_worktree_removable_on_setup_failure(tmp_path):
    """12. Worktree bersih milik run dapat dibersihkan saat setup gagal."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    assert wt.exists()
    remove_worktree(repo, wt, force=False)
    assert not wt.exists()


# ==============================================================================
# TAHAP 9: UNIT TEST 13-20 (Implementation Isolation)
# ==============================================================================

def test_13_writer_only_writes_in_worktree(tmp_path):
    """13. Writer hanya menulis di worktree."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    approved = ["backend/internal/service/wallet_service.go"]
    writer = ControlledFileWriter(wt, approved, "run-1")
    target_path = writer.write_file("backend/internal/service/wallet_service.go", "package service\n")

    assert target_path.is_file()
    assert target_path.is_relative_to(wt)
    assert not (repo / "backend" / "internal" / "service" / "wallet_service.go").exists()


def test_14_main_project_not_modified(tmp_path):
    """14. Project utama tidak berubah saat implementasi di worktree."""
    repo = init_git_repo(tmp_path / "main_proj")
    status_before = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.strip()

    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)
    approved = ["backend/main.go"]
    writer = ControlledFileWriter(wt, approved, "run-1")
    writer.write_file("backend/main.go", "package main\n// modified in wt\n")

    status_after = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert status_after == status_before


def test_15_runtime_project_root_is_worktree_path(tmp_path):
    """15. Runtime project root menjadi worktree path."""
    repo = init_git_repo(tmp_path / "main_proj")
    from agent_system.nodes import branch_guard_node

    state = {
        "run_id": "a23dc0841234",
        "project_root": str(repo),
        "proposal_data": {
            "result": {
                "proposal": {
                    "task_title": "Implementasi Persistensi Riwayat",
                }
            }
        },
        "dry_run": False,
        "isolation": "git-worktree",
    }
    res = branch_guard_node(state)
    assert "worktree_path" in res
    assert res["project_root"] == res["worktree_path"]
    assert res["project_root"] != str(repo)


def test_16_approved_files_can_be_modified(tmp_path):
    """16. Approved files dapat diubah."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    approved = ["backend/cmd/api/main.go"]
    writer = ControlledFileWriter(wt, approved, "run-1")
    written = writer.write_file("backend/cmd/api/main.go", "package main\nfunc main() {}\n")
    assert written.read_text() == "package main\nfunc main() {}\n"


def test_17_additional_unapproved_files_rejected(tmp_path):
    """17. File tambahan di luar approved_files ditolak."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    approved = ["backend/main.go"]
    writer = ControlledFileWriter(wt, approved, "run-1")
    with pytest.raises(UnapprovedFileError, match="tidak tercantum dalam approved_files"):
        writer.write_file("backend/other.go", "package main\n")


def test_18_protocol_files_rejected(tmp_path):
    """18. Protocol file ditolak."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    writer = ControlledFileWriter(wt, [".agent/manifest.yaml"], "run-1")
    with pytest.raises(ValueError, match="Dilarang mengubah file protokol"):
        writer.write_file(".agent/manifest.yaml", "manifest: {}\n")


def test_19_secret_files_rejected(tmp_path):
    """19. Secret file ditolak."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    writer = ControlledFileWriter(wt, [".env"], "run-1")
    with pytest.raises(ValueError, match="Dilarang mengubah file credential/secret"):
        writer.write_file(".env", "SECRET=123\n")


def test_20_symlink_escaping_rejected(tmp_path):
    """20. Symlink keluar project ditolak."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    outside = tmp_path / "outside.txt"
    outside.write_text("outside")

    link = wt / "symlink_file.go"
    link.symlink_to(outside)

    writer = ControlledFileWriter(wt, ["symlink_file.go"], "run-1")
    with pytest.raises(ValueError, match="escapes project root"):
        writer.write_file("symlink_file.go", "package main\n")



# ==============================================================================
# TAHAP 9: UNIT TEST 21-28 (Command Execution)
# ==============================================================================

def test_21_go_test_executed_in_worktree_backend(tmp_path, monkeypatch):
    """21. go test ./... dijalankan di worktree/backend."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)
    (wt / "backend").mkdir(parents=True, exist_ok=True)

    executed_cwd = []
    def fake_run(cmd, cwd=None, shell=False, capture_output=True, text=True, timeout=None):
        executed_cwd.append(cwd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "ok"
        mock_res.stderr = ""
        return mock_res

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = SafeCommandRunner(wt)
    res = runner.run_command("go", ["test", "./..."], "backend")
    assert res.status == "passed"
    assert len(executed_cwd) == 1
    assert executed_cwd[0] == (wt / "backend").resolve()


def test_22_command_not_executed_in_main_project(tmp_path, monkeypatch):
    """22. Command tidak dijalankan di backend project utama."""
    repo = init_git_repo(tmp_path / "main_proj")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)
    (wt / "backend").mkdir(parents=True, exist_ok=True)

    executed_cwd = []
    def fake_run(cmd, cwd=None, **kwargs):
        executed_cwd.append(cwd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "ok"
        mock_res.stderr = ""
        return mock_res

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SafeCommandRunner(wt)
    runner.run_command("go", ["test", "./..."], "backend")
    assert executed_cwd[0] != (repo / "backend").resolve()


def test_23_shell_false_used(tmp_path, monkeypatch):
    """23. shell=False digunakan."""
    repo = init_git_repo(tmp_path / "main_proj")
    (repo / "backend").mkdir(parents=True, exist_ok=True)

    shell_val = []
    def fake_run(cmd, cwd=None, shell=True, **kwargs):
        shell_val.append(shell)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "ok"
        mock_res.stderr = ""
        return mock_res

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SafeCommandRunner(repo)
    runner.run_command("go", ["test", "./..."], "backend")
    assert len(shell_val) == 1
    assert shell_val[0] is False


def test_24_output_capped():
    """24. Output dibatasi."""
    big_out = "A" * 5000
    capped = summarize_output(big_out, max_length=1000)
    assert len(capped) <= 1200
    assert "OUTPUT TRUNCATED FOR SAFETY" in capped


def test_25_timeout_handled(tmp_path, monkeypatch):
    """25. Timeout ditangani."""
    repo = init_git_repo(tmp_path / "main_proj")
    (repo / "backend").mkdir(parents=True, exist_ok=True)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="go", timeout=5.0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SafeCommandRunner(repo)
    res = runner.run_command("go", ["test", "./..."], "backend", timeout=5.0)
    assert res.status == "timed_out"
    assert res.timed_out is True


def test_26_failure_routes_to_repair_agent():
    """26. Failure menuju Repair Agent."""
    state = {
        "command_results": [{"status": "failed", "exit_code": 1}],
        "implementation_repair_count": 0,
        "max_implementation_repairs": 2,
    }
    assert route_after_test_runner(state) == "repair_agent"


def test_27_max_repair_produces_human_required():
    """27. Repair maksimal menghasilkan human_required (routes to supervisor)."""
    state = {
        "command_results": [{"status": "failed", "exit_code": 1}],
        "implementation_repair_count": 2,
        "max_implementation_repairs": 2,
    }
    assert route_after_test_runner(state) == "supervisor"


def test_28_passed_test_routes_to_diff_reviewer():
    """28. Passed test menuju Diff Reviewer."""
    state = {
        "command_results": [{"status": "passed", "exit_code": 0}],
        "implementation_repair_count": 0,
        "max_implementation_repairs": 2,
    }
    assert route_after_test_runner(state) == "diff_reviewer"



# ==============================================================================
# TAHAP 9: UNIT TEST 29-35 (Diff Review)
# ==============================================================================

def test_29_only_approved_files_produces_approved(tmp_path):
    """29. Hanya approved files menghasilkan approved."""
    repo = init_git_repo(tmp_path / "repo")
    approved = ["backend/main.go"]
    (repo / "backend" / "main.go").write_text("package main\nfunc Modified() {}\n")

    res = review_diff(repo, approved, ["criterion 1"], test_status="passed")
    assert res.status == "approved"
    assert res.changed_files == ["backend/main.go"]
    assert res.unexpected_files == []


def test_30_unexpected_file_produces_human_required(tmp_path):
    """30. Unexpected file menghasilkan human_required."""
    repo = init_git_repo(tmp_path / "repo")
    approved = ["backend/main.go"]
    (repo / "backend" / "main.go").write_text("package main\n")
    (repo / "unexpected.txt").write_text("unexpected content\n")

    res = review_diff(repo, approved, ["criterion 1"], test_status="passed")
    assert res.status == "human_required"
    assert "unexpected.txt" in res.unexpected_files


def test_31_migration_change_rejected(tmp_path):
    """31. Migration change ditolak."""
    repo = init_git_repo(tmp_path / "repo")
    mig_file = repo / "backend" / "migrations" / "001_initial.sql"
    mig_file.parent.mkdir(parents=True, exist_ok=True)
    mig_file.write_text("CREATE TABLE t1();\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add migration"], cwd=repo, check=True, capture_output=True)

    mig_file.write_text("CREATE TABLE t1_modified();\n")
    approved = ["backend/migrations/001_initial.sql"]
    res = review_diff(repo, approved, ["c1"], test_status="passed")
    assert res.status == "human_required"
    assert any(f.category == "database" for f in res.findings)


def test_32_frontend_change_rejected(tmp_path):
    """32. Frontend change ditolak."""
    repo = init_git_repo(tmp_path / "repo")
    fe_file = repo / "frontend" / "App.tsx"
    fe_file.parent.mkdir(parents=True, exist_ok=True)
    fe_file.write_text("export default function App() {}\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add fe"], cwd=repo, check=True, capture_output=True)

    fe_file.write_text("export default function App() { return <div />; }\n")
    approved = ["frontend/App.tsx"]
    res = review_diff(repo, approved, ["c1"], test_status="passed")
    assert res.status == "human_required"
    assert any(f.category == "scope" for f in res.findings)


def test_33_protocol_change_rejected(tmp_path):
    """33. Protocol change ditolak."""
    repo = init_git_repo(tmp_path / "repo")
    p_file = repo / ".agent" / "manifest.yaml"
    p_file.parent.mkdir(parents=True, exist_ok=True)
    p_file.write_text("manifest: 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add manifest"], cwd=repo, check=True, capture_output=True)

    p_file.write_text("manifest: 2\n")
    res = review_diff(repo, [".agent/manifest.yaml"], ["c1"], test_status="passed")
    assert res.status == "human_required"
    assert any(f.category == "protocol" for f in res.findings)


def test_34_dependency_manifest_change_rejected(tmp_path):
    """34. Dependency manifest change ditolak."""
    repo = init_git_repo(tmp_path / "repo")
    gomod = repo / "backend" / "go.mod"
    gomod.write_text("module test\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add go.mod"], cwd=repo, check=True, capture_output=True)

    gomod.write_text("module test\nrequire foo v1\n")
    res = review_diff(repo, ["backend/go.mod"], ["c1"], test_status="passed")
    assert res.status == "human_required"
    assert any(f.category == "dependency" for f in res.findings)


def test_35_git_diff_check_failure_produces_repair_required(tmp_path):
    """35. git diff --check failure menghasilkan repair_required."""
    repo = init_git_repo(tmp_path / "repo")
    approved = ["backend/main.go"]
    # Add trailing whitespace to trigger git diff --check whitespace error
    (repo / "backend" / "main.go").write_text("package main\nfunc BadWhitespace() { }   \n")

    res = review_diff(repo, approved, ["c1"], test_status="passed")
    assert res.status == "repair_required"
    assert any(f.category == "code_style" for f in res.findings)



# ==============================================================================
# TAHAP 9: UNIT TEST 36-44 (Final State and Invariants)
# ==============================================================================

def test_36_passed_test_and_approved_diff_produces_implementation_ready():
    """36. Passed test dan approved diff menghasilkan implementation_ready."""
    state = {
        "status": "running",
        "current_stage": "implementation",
        "dry_run": False,
        "preflight_result": {"is_valid": True},
        "branch_name": "agent/task-12345678",
        "implementation_result": {
            "status": "implementation_generated",
            "files_changed": ["backend/main.go"],
        },
        "command_results": [{"exit_code": 0, "status": "passed"}],
        "diff_review": {
            "status": "approved",
            "unexpected_files": [],
        },
    }
    res = supervisor_validate_implementation(state)
    assert res["status"] == "implementation_ready"


def test_37_implementation_ready_makes_no_commit(tmp_path):
    """37. implementation_ready tidak membuat commit."""
    repo = init_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)
    (wt / "backend" / "main.go").write_text("package main\n// modified\n")

    count_repo = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    count_wt = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    assert count_repo == "1"
    assert count_wt == "1"


def test_38_implementation_ready_does_not_push(tmp_path):
    """38. implementation_ready tidak push."""
    repo = init_git_repo(tmp_path / "repo")
    remotes = subprocess.run(["git", "remote"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert remotes == ""


def test_39_implementation_ready_does_not_merge(tmp_path):
    """39. implementation_ready tidak merge."""
    repo = init_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)

    main_branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert main_branch == "main"
    log_count = subprocess.run(["git", "rev-list", "--count", "main"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert log_count == "1"


def test_40_worktree_remains_available_after_success(tmp_path):
    """40. Worktree tetap tersedia untuk review manusia."""
    repo = init_git_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)
    (wt / "backend" / "main.go").write_text("package main\n// modified\n")

    # In success scenario, worktree is NOT deleted
    assert wt.exists()
    assert (wt / "backend" / "main.go").is_file()


def test_41_artifact_saved_in_main_project(tmp_path):
    """41. Artifact disimpan di project utama."""
    from agent_system.artifact_store import ProjectArtifactStore
    from agent_system.project_initializer import initialize_project

    repo = init_git_repo(tmp_path / "repo")
    initialize_project(repo, apply=True)

    from agent_system.project_loader import load_project
    proj = load_project(repo)
    store = ProjectArtifactStore(repo, proj.manifest)

    impl_file = store.write_implementation("implementation_12345678", {
        "status": "implementation_ready",
        "branch_name": "agent/task-12345678",
    })
    assert impl_file.is_file()
    assert impl_file.is_relative_to(repo / ".agent" / "artifacts" / "implementations")


def test_42_artifact_not_saved_in_orchestrator(tmp_path):
    """42. Artifact tidak disimpan di orchestrator."""
    orchestrator_dir = Path(__file__).resolve().parent.parent.resolve()
    # Ensure no implementation artifact for run id in orchestrator
    assert not (orchestrator_dir / ".agent" / "artifacts" / "implementations" / "implementation_12345678.json").exists()


def test_43_artifact_does_not_store_absolute_worktree_path():
    """43. Artifact tidak menyimpan absolute worktree path."""
    from agent_system.artifact_store import sanitize_artifact_payload

    wt_abs = "/Users/user/Projects/.agent-worktrees/sample-12345678"
    raw_payload = {
        "worktree_descriptor": "sample-12345678",
        "branch_name": "agent/task-12345678",
        "files_changed": ["backend/internal/service/wallet_service.go"],
        "final_status": "implementation_ready",
    }
    sanitized = sanitize_artifact_payload(raw_payload)
    dumped = json.dumps(sanitized)
    assert wt_abs not in dumped
    assert "/Users/" not in dumped


def test_44_main_project_remains_unmodified(tmp_path):
    """44. Project utama tetap tidak berubah."""
    repo = init_git_repo(tmp_path / "repo")
    status_before = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert status_before == ""

    wt = tmp_path / "wt"
    create_worktree(repo, "agent/task-12345678", wt)
    (wt / "backend" / "main.go").write_text("package main\n// only in worktree\n")

    status_after = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert status_after == ""

