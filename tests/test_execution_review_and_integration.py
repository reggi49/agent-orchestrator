import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.execution_reviewer import (
    locate_execution_artifact,
    review_execution,
    validate_execution_for_human_approval,
)
from agent_system.integrator import (
    check_source_repository_state,
    integrate_execution,
)
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project


def setup_test_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    (path / "backend").mkdir(parents=True, exist_ok=True)
    (path / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"ok\" }\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    initialize_project(path, apply=True)
    return path


def setup_test_worktree_and_execution(repo: Path, tmp_path: Path, branch_name: str = "agent/test-branch"):
    wt_id = "test-wt-1"
    wt_path = tmp_path / ".agent-worktrees" / wt_id
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "worktree", "add", "-b", branch_name, str(wt_path), "HEAD"], cwd=repo, check=True, capture_output=True)

    # Make change in worktree
    changed_file = wt_path / "backend" / "app_test.go"
    changed_file.write_text("package main\n\nimport \"testing\"\n\nfunc TestRun(t *testing.T) {}\n")

    val_proj = load_project(repo)
    store = ProjectArtifactStore(repo, val_proj.manifest)

    exec_data = {
        "artifact_id": "execution_test_01",
        "schema_version": "1.0",
        "status": "execution_review_ready",
        "source_revision": {"commit_head": "HEAD", "is_dirty": False},
        "result": {
            "source_proposal_path": ".agent/artifacts/proposals/proposal_01.json",
            "worktree_identifier": wt_id,
            "branch_name": branch_name,
            "approved_files": ["backend/app.go", "backend/app_test.go"],
            "actual_changed_files": ["backend/app_test.go"],
            "validation_exit_code": 0,
            "validation_command_summary": [
                {
                    "executable": "go",
                    "arguments": ["test", "./..."],
                    "working_directory": "backend",
                    "exit_code": 0,
                    "status": "passed",
                }
            ],
            "diff_check_status": "passed",
            "independent_review_status": "approved",
            "scope_violations": [],
        },
    }
    exec_file = store.write_execution("execution_test_01", exec_data)

    return wt_path, exec_file, exec_data


# ==============================================================================
# 1. Test approve execution
# ==============================================================================
def test_01_approve_execution(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)

        res = review_execution(
            project_root=repo,
            execution_file=exec_file,
            execution_data=exec_data,
            decision="approve",
            reviewer="operator_alice",
            note="Lulus validasi formal",
        )

        assert res["decision"] == "approve"
        assert res["status"] == "approved_for_integration"
        assert res["reviewer"] == "operator_alice"
        assert res["report_file"].is_file()

        # Verify saved report
        report_data = json.loads(res["report_file"].read_text())
        assert report_data["status"] == "approved_for_integration"
        assert report_data["result"]["decision"] == "approve"
        assert report_data["result"]["reviewer"] == "operator_alice"
        assert report_data["result"]["review_type"] == "human_review"

        # Verify updated execution file
        updated_exec = json.loads(exec_file.read_text())
        assert updated_exec["status"] == "approved_for_integration"
        assert updated_exec["result"]["human_review"]["decision"] == "approve"


# ==============================================================================
# 2. Test request changes
# ==============================================================================
def test_02_request_changes(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)

        res = review_execution(
            project_root=repo,
            execution_file=exec_file,
            execution_data=exec_data,
            decision="request-changes",
            reviewer="operator_bob",
            note="Mohon tambahkan skenario error",
        )

        assert res["decision"] == "request-changes"
        assert res["status"] == "changes_requested"
        assert res["report_file"].is_file()

        updated_exec = json.loads(exec_file.read_text())
        assert updated_exec["status"] == "changes_requested"


# ==============================================================================
# 3. Test reject execution
# ==============================================================================
def test_03_reject_execution(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)

        res = review_execution(
            project_root=repo,
            execution_file=exec_file,
            execution_data=exec_data,
            decision="reject",
            reviewer="operator_charlie",
            note="Pendekatan tidak sesuai arsitektur",
        )

        assert res["decision"] == "reject"
        assert res["status"] == "rejected"
        assert res["report_file"].is_file()

        updated_exec = json.loads(exec_file.read_text())
        assert updated_exec["status"] == "rejected"


# ==============================================================================
# 4. Test menolak execution dengan status tidak valid
# ==============================================================================
def test_04_reject_execution_with_invalid_status(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "running"

        with pytest.raises(ValueError, match="Hanya execution dengan status 'execution_review_ready'"):
            review_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                decision="approve",
            )


# ==============================================================================
# 5. Test menolak changed files kosong
# ==============================================================================
def test_05_reject_empty_changed_files(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["result"]["actual_changed_files"] = []
        exec_data["result"]["changed_files"] = []

        with pytest.raises(ValueError, match="zero changed files"):
            review_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                decision="approve",
            )


# ==============================================================================
# 6. Test menolak test result gagal
# ==============================================================================
def test_06_reject_failed_test_result(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["result"]["validation_exit_code"] = 1
        exec_data["result"]["validation_command_summary"][0]["status"] = "failed"
        exec_data["result"]["validation_command_summary"][0]["exit_code"] = 1

        with pytest.raises(ValueError, match="Validasi test gagal"):
            review_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                decision="approve",
            )


# ==============================================================================
# 7. Test dry-run tidak membuat commit
# ==============================================================================
def test_07_dry_run_does_not_create_commit(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"

        wt_commits_before = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=wt_path, capture_output=True, text=True).stdout.strip()

        res = integrate_execution(
            project_root=repo,
            execution_file=exec_file,
            execution_data=exec_data,
            strategy="cherry-pick",
            apply=False,
        )

        assert res["mode"] == "dry-run"
        assert res["status"] == "dry_run_ready"

        wt_commits_after = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=wt_path, capture_output=True, text=True).stdout.strip()
        assert wt_commits_after == wt_commits_before


# ==============================================================================
# 8. Test dry-run tidak melakukan cherry-pick
# ==============================================================================
def test_08_dry_run_does_not_cherry_pick(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"

        main_commits_before = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

        res = integrate_execution(
            project_root=repo,
            execution_file=exec_file,
            execution_data=exec_data,
            strategy="cherry-pick",
            apply=False,
        )

        assert res["mode"] == "dry-run"
        main_commits_after = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        assert main_commits_after == main_commits_before


# ==============================================================================
# 9. Test integrate menolak unexpected changed file
# ==============================================================================
def test_09_integrate_rejects_unexpected_changed_file(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"

        # Create unexpected file in worktree
        (wt_path / "unexpected.txt").write_text("malicious content")

        with pytest.raises(ValueError, match="scope violation"):
            integrate_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                strategy="cherry-pick",
                apply=True,
            )


# ==============================================================================
# 10. Test integrate menolak dirty source repository
# ==============================================================================
def test_10_integrate_rejects_dirty_source_repository(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"

        # Dirty the main source repository
        (repo / "backend" / "app.go").write_text("package main\n\n// dirty edit\n")

        with pytest.raises(ValueError, match="perubahan source code yang belum di-commit"):
            integrate_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                strategy="cherry-pick",
                apply=True,
            )


# ==============================================================================
# 11. Test .agent/artifacts dapat dibedakan dari source changes
# ==============================================================================
def test_11_agent_artifacts_distinguished_from_source_changes(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    # Write untracked artifact under .agent
    art_file = repo / ".agent" / "artifacts" / "test_dummy.json"
    art_file.parent.mkdir(parents=True, exist_ok=True)
    art_file.write_text("{}")

    state = check_source_repository_state(repo)

    assert state["has_uncommitted_source_changes"] is False
    assert state["has_runtime_artifacts"] is True
    assert state["is_completely_clean"] is False
    assert len(state["source_changes"]) == 0
    assert len(state["runtime_artifacts"]) > 0


# ==============================================================================
# 12. Test successful cherry-pick flow menggunakan temporary Git
# ==============================================================================
def test_12_successful_cherry_pick_flow(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"
        # Mock validation command execution so test passes cleanly in temp dir
        exec_data["result"]["validation_command_summary"] = [
            {
                "executable": "git",
                "arguments": ["status"],
                "working_directory": ".",
                "exit_code": 0,
                "status": "passed",
            }
        ]

        main_commits_before = int(subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip())

        res = integrate_execution(
            project_root=repo,
            execution_file=exec_file,
            execution_data=exec_data,
            strategy="cherry-pick",
            apply=True,
        )

        assert res["mode"] == "apply"
        assert res["status"] == "completed"
        assert res["commit_hash"] is not None
        assert res["source_commit"] is not None
        assert res["integrated_commit"] is not None
        assert len(res["source_commit"]) == 40
        assert len(res["integrated_commit"]) == 40
        assert res["target_branch"] == "main"
        assert res["source_agent_branch"] == "agent/test-branch"
        assert res["post_integration_validation"]["status"] == "passed"
        assert (repo / "backend" / "app_test.go").is_file()

        main_commits_after = int(subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip())
        assert main_commits_after == main_commits_before + 1

        # Verify integration artifact
        assert res["integration_artifact_file"].is_file()
        assert res["integration_report_file"].is_file()

        art_data = json.loads(res["integration_artifact_file"].read_text())
        art_res = art_data.get("result", art_data)
        assert art_res["source_commit"] == res["source_commit"]
        assert art_res["integrated_commit"] == res["integrated_commit"]
        assert art_res["target_branch"] == "main"
        assert art_res["source_agent_branch"] == "agent/test-branch"
        assert art_res["post_integration_validation"]["status"] == "passed"

        # Verify updated execution file status is completed
        updated_exec = json.loads(exec_file.read_text())
        assert updated_exec["status"] == "completed"
        assert updated_exec["result"]["integration"]["source_commit"] == res["source_commit"]
        assert updated_exec["result"]["integration"]["integrated_commit"] == res["integrated_commit"]


# ==============================================================================
# 13. Test commit message generator classification
# ==============================================================================
def test_13_commit_message_generator_classification():
    from agent_system.integrator import (
        classify_commit_type,
        generate_commit_message,
    )

    # 1. Only test files -> test(scope)
    msg_test = generate_commit_message(
        changed_files=["backend/internal/handler/http/health_handler_test.go"],
        task_title="Implementasi endpoint GET /health dan test verifikasi",
        branch_name="agent/test-branch",
        task_owner="backend",
    )
    assert msg_test.startswith("test(backend):")
    assert not msg_test.startswith("feat(")

    # 2. Only docs files -> docs(scope)
    msg_docs = generate_commit_message(
        changed_files=["docs/API.md", "README.md"],
        task_title="Update petunjuk penggunaan API",
        branch_name="agent/docs-branch",
        task_owner="backend",
    )
    assert msg_docs.startswith("docs(")

    # 3. Bug fix -> fix(scope)
    msg_fix = generate_commit_message(
        changed_files=["backend/cmd/api/main.go"],
        task_title="Perbaiki bug routing timeout pada gateway",
        branch_name="agent/fix-branch",
        task_owner="backend",
    )
    assert msg_fix.startswith("fix(backend):")

    # 4. Feature -> feat(scope)
    msg_feat = generate_commit_message(
        changed_files=["backend/cmd/api/main.go", "backend/tests/api_test.go"],
        task_title="Implementasi fitur topup wallet",
        branch_name="agent/feat-branch",
        task_owner="backend",
    )
    assert msg_feat.startswith("feat(backend):")

    # 5. Refactor -> refactor(scope)
    msg_refactor = generate_commit_message(
        changed_files=["backend/internal/service/wallet.go"],
        task_title="Refactor struktur service layer wallet",
        branch_name="agent/refactor-branch",
        task_owner="backend",
    )
    assert msg_refactor.startswith("refactor(backend):")


# ==============================================================================
# 14. Test post-integration validation failure produces integration_failed
# ==============================================================================
def test_14_post_integration_validation_failure_produces_integration_failed(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"

        # Mock worktree test passes, but main project post-validation fails
        orig_run = subprocess.run

        def mock_subproc(cmd, *a, **kw):
            # Check if running mock test in main repo vs worktree
            cwd = kw.get("cwd") or (a[1] if len(a) > 1 else None)
            if isinstance(cmd, list) and cmd and cmd[0] == "mock_test_exe":
                if str(cwd) == str(repo):
                    return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="test failed in main")
                return subprocess.CompletedProcess(cmd, returncode=0, stdout="test passed in wt", stderr="")
            return orig_run(cmd, *a, **kw)

        exec_data["result"]["validation_command_summary"] = [
            {
                "executable": "mock_test_exe",
                "arguments": ["run"],
                "working_directory": ".",
                "exit_code": 0,
                "status": "passed",
            }
        ]

        with patch("subprocess.run", side_effect=mock_subproc):
            with pytest.raises(RuntimeError, match="Post-integration validation pada repository utama gagal"):
                integrate_execution(
                    project_root=repo,
                    execution_file=exec_file,
                    execution_data=exec_data,
                    strategy="cherry-pick",
                    apply=True,
                )

        # Check execution file status is integration_failed (NOT completed)
        updated_exec = json.loads(exec_file.read_text())
        assert updated_exec["status"] == "integration_failed"
        assert updated_exec["result"]["final_status"] == "integration_failed"
        assert updated_exec["result"]["integration"]["cherry_pick_occurred"] is True
        assert updated_exec["result"]["integration"]["integrated_commit"] is not None
        assert updated_exec["result"]["integration"]["source_commit"] is not None
        assert "git revert" in updated_exec["result"]["integration"]["recovery_instructions"]


# ==============================================================================
# 15. Test integrator does not execute forbidden git commands
# ==============================================================================
def test_15_integrator_does_not_run_forbidden_git_commands(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"
        exec_data["result"]["validation_command_summary"] = [
            {
                "executable": "git",
                "arguments": ["status"],
                "working_directory": ".",
                "exit_code": 0,
                "status": "passed",
            }
        ]

        executed_git_commands: list[list[str]] = []
        orig_run = subprocess.run

        def intercept_run(cmd, *a, **kw):
            if isinstance(cmd, list) and cmd and cmd[0] == "git":
                executed_git_commands.append(cmd)
            return orig_run(cmd, *a, **kw)

        with patch("subprocess.run", side_effect=intercept_run):
            integrate_execution(
                project_root=repo,
                execution_file=exec_file,
                execution_data=exec_data,
                strategy="cherry-pick",
                apply=True,
            )

        forbidden_verbs = {"push", "merge", "rebase", "reset", "tag"}
        for cmd in executed_git_commands:
            cmd_verbs = set(cmd[1:])
            assert not (cmd_verbs & forbidden_verbs), f"Forbidden git command executed: {cmd}"
            assert "--force" not in cmd and "-f" not in cmd, f"Force flag executed: {cmd}"


# ==============================================================================
# 16. Test integration artifact sanitization
# ==============================================================================
def test_16_integration_artifact_sanitization():
    from agent_system.artifact_store import sanitize_artifact_payload

    dirty_payload = {
        "worktree_path": "/Users/user/Projects/.agent-worktrees/sample-secret-1234",
        "raw_diff": "diff --git a/file b/file\n+secret line",
        "full_diff": "--- a/secret\n+++ b/secret",
        "patch": "patch content with secrets",
        "raw_prompt": "System: do not leak key sk-1234567890abcdef",
        "system_prompt": "Instruction prompt",
        "api_key": "sk-realapikey1234567890",
        "headers": "Authorization: Bearer secret_token_xyz98765",
        "custom_secret": "Bearer abc123def456token",
        "env_secret": "password='supersecretpass'",
    }

    sanitized = sanitize_artifact_payload(dirty_payload)
    dumped = json.dumps(sanitized)

    assert "/Users/user/Projects/.agent-worktrees" not in dumped
    assert "sk-1234567890abcdef" not in dumped
    assert "sk-realapikey1234567890" not in dumped
    assert "secret_token_xyz98765" not in dumped
    assert "abc123def456token" not in dumped
    assert "supersecretpass" not in dumped
    assert "diff --git" not in dumped
    assert sanitized.get("raw_diff") == "[OMITTED_FOR_SAFETY]"
    assert sanitized.get("patch") == "[OMITTED_FOR_SAFETY]"
    assert sanitized.get("raw_prompt") == "[OMITTED_FOR_SAFETY]"
    assert sanitized.get("worktree_descriptor") == "sample-secret-1234"


# ==============================================================================
# 17. Test runtime artifacts policy and gitignore recommendation
# ==============================================================================
def test_17_runtime_artifacts_policy_and_gitignore_recommendation(tmp_path: Path):
    from agent_system.project_initializer import RECOMMENDED_GITIGNORE, GITIGNORE_USER_NOTICE

    assert "artifacts/" in RECOMMENDED_GITIGNORE
    assert ".agent/artifacts/" in GITIGNORE_USER_NOTICE
    # Policy ensures entire .agent/ is NOT ignored
    assert "Do NOT ignore the entire .agent/ directory" in GITIGNORE_USER_NOTICE
    assert "manifest.yaml" in GITIGNORE_USER_NOTICE


# ==============================================================================
# 18. Test source_commit and integrated_commit are not assumed equal
# ==============================================================================
def test_18_source_and_integrated_commit_differ_when_target_branch_progresses(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    with patch("agent_system.worktree_manager.get_default_worktree_root", return_value=tmp_path / ".agent-worktrees"):
        wt_path, exec_file, exec_data = setup_test_worktree_and_execution(repo, tmp_path)
        exec_data["status"] = "approved_for_integration"
        exec_data["result"]["validation_command_summary"] = [
            {
                "executable": "git",
                "arguments": ["status"],
                "working_directory": ".",
                "exit_code": 0,
                "status": "passed",
            }
        ]

        # Target branch progresses with another commit
        (repo / "docs_update.txt").write_text("documentation update on main\n")
        subprocess.run(["git", "add", "docs_update.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "docs: update on main"], cwd=repo, check=True)

        res = integrate_execution(
            project_root=repo,
            execution_file=exec_file,
            execution_data=exec_data,
            strategy="cherry-pick",
            apply=True,
        )

        assert res["status"] == "completed"
        assert res["source_commit"] is not None
        assert res["integrated_commit"] is not None
        # Must be different commits!
        assert res["source_commit"] != res["integrated_commit"]
        assert len(res["source_commit"]) == 40
        assert len(res["integrated_commit"]) == 40


