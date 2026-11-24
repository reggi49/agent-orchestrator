import subprocess
from pathlib import Path

import pytest

from agent_system.diff_reviewer import review_diff


def init_diff_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "backend").mkdir(parents=True, exist_ok=True)
    (path / "backend" / "test.go").write_text("package backend\nfunc Foo() {}\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=path, check=True, capture_output=True)
    return path


def test_45_only_approved_files_changed_yields_approved(tmp_path):
    """45. Hanya approved file berubah menghasilkan approved."""
    p = init_diff_repo(tmp_path / "repo")
    (p / "backend" / "test.go").write_text("package backend\nfunc Foo() { /* updated */ }\n")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go"],
        acceptance_criteria=["ac1"],
        test_status="passed",
    )
    assert res.status == "approved"
    assert res.changed_files == ["backend/test.go"]
    assert res.unexpected_files == []


def test_46_unexpected_file_yields_human_required(tmp_path):
    """46. Unexpected file menghasilkan human_required."""
    p = init_diff_repo(tmp_path / "repo")
    (p / "backend" / "other.go").write_text("package other\n")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go"],
        acceptance_criteria=["ac1"],
    )
    assert res.status == "human_required"
    assert "backend/other.go" in res.unexpected_files


def test_47_protocol_change_rejected(tmp_path):
    """47. Protocol change ditolak."""
    p = init_diff_repo(tmp_path / "repo")
    (p / ".agent").mkdir(parents=True, exist_ok=True)
    (p / ".agent" / "policy.yaml").write_text("modified")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go", ".agent/policy.yaml"],
        acceptance_criteria=[],
    )
    assert res.status == "human_required"
    assert any(f.category == "protocol" for f in res.findings)


def test_48_secret_change_rejected(tmp_path):
    """48. Secret change ditolak."""
    p = init_diff_repo(tmp_path / "repo")
    (p / "backend" / ".env").write_text("SECRET=1")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go", "backend/.env"],
        acceptance_criteria=[],
    )
    assert res.status == "human_required"
    assert any(f.category == "security" for f in res.findings)


def test_49_frontend_change_rejected(tmp_path):
    """49. Frontend change ditolak."""
    p = init_diff_repo(tmp_path / "repo")
    (p / "frontend").mkdir(parents=True, exist_ok=True)
    (p / "frontend" / "app.tsx").write_text("const a = 1;")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go", "frontend/app.tsx"],
        acceptance_criteria=[],
    )
    assert res.status == "human_required"
def test_50_whitespace_error_detected(tmp_path):
    """50. Whitespace error ditemukan."""
    p = init_diff_repo(tmp_path / "repo")
    # Trailing whitespace triggers git diff --check
    (p / "backend" / "test.go").write_text("package backend\nfunc Foo() { }   \n")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go"],
        acceptance_criteria=[],
    )
    assert any(f.category == "code_style" for f in res.findings)


def test_51_todo_placeholder_detected(tmp_path):
    """51. TODO placeholder ditemukan."""
    p = init_diff_repo(tmp_path / "repo")
    (p / "backend" / "test.go").write_text("package backend\n// TODO: implement logic\nfunc Foo() {}\n")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go"],
        acceptance_criteria=[],
    )
    assert res.status == "repair_required"
    assert any(f.category == "incomplete_code" for f in res.findings)


def test_52_dependency_manifest_change_rejected(tmp_path):
    """52. Dependency manifest change ditolak."""
    p = init_diff_repo(tmp_path / "repo")
    (p / "backend" / "go.mod").write_text("module backend\nrequire new/pkg v1\n")

    res = review_diff(
        project_root=p,
        approved_files=["backend/test.go"],
        acceptance_criteria=[],
    )
    assert res.status == "human_required"
    assert any(f.category == "dependency" for f in res.findings)


def test_53_full_diff_not_stored_in_artifact():
    """53. Full diff tidak disimpan di artifact."""
    from agent_system.diff_reviewer import DiffReview
    review = DiffReview(
        status="approved",
        changed_files=["backend/test.go"],
        approved_files=["backend/test.go"],
        test_status="passed",
        summary="ok",
    )
    dumped = review.model_dump()
    assert "diff" not in dumped
    assert "full_diff" not in dumped
    assert "patch" not in dumped
