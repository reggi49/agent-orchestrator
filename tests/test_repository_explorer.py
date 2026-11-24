import os
from pathlib import Path

import pytest

from agent_system.project_protocol import (
    AgentAccess,
    ProjectPolicy,
    SecurityPolicy,
)
from agent_system.repository_explorer import (
    RepositoryExplorer,
    is_denied_by_policy,
    is_secret_path,
    is_text_file,
)


def create_dummy_policy(denied_paths=None) -> ProjectPolicy:
    return ProjectPolicy(
        schema_version="1.0",
        mode="read_only",
        security=SecurityPolicy(
            deny_hidden_files_by_default=True,
            denied_paths=denied_paths or [".agent/secrets/**", "backend/secret_config.yaml"],
        ),
        agents={
            "orchestrator": AgentAccess(
                read=["**/*"],
                write=[],
            )
        },
    )



@pytest.fixture
def repo_fixture(tmp_path: Path):
    root = tmp_path / "dummy_project"
    root.mkdir()

    # Secret files
    (root / ".env").write_text("SECRET=123")
    (root / ".env.local").write_text("LOCAL_SECRET=456")
    (root / "cert.pem").write_text("-----BEGIN CERT-----")
    (root / "private.key").write_text("-----BEGIN KEY-----")
    (root / "credentials.json").write_text('{"key": "val"}')
    (root / "auth.json").write_text('{"auth": "token"}')

    # Ignored directories
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config")

    nm_dir = root / "node_modules" / "some_pkg"
    nm_dir.mkdir(parents=True)
    (nm_dir / "index.js").write_text("console.log('nm');")

    # Policy denied file
    secret_dir = root / ".agent" / "secrets"
    secret_dir.mkdir(parents=True)
    (secret_dir / "token.txt").write_text("topsecret")

    # Regular files
    migrations_dir = root / "backend" / "migrations"
    migrations_dir.mkdir(parents=True)
    (migrations_dir / "001_create_users.sql").write_text(
        "CREATE TABLE users (\n    id UUID PRIMARY KEY,\n    email TEXT\n);\n"
    )
    (migrations_dir / "002_create_wallets.sql").write_text(
        "CREATE TABLE wallets (\n    id UUID PRIMARY KEY\n);\n"
    )

    src_dir = root / "backend" / "internal"
    src_dir.mkdir(parents=True)
    long_code = "\n".join([f"line_{i} = {i}" for i in range(1, 200)])
    (src_dir / "app.go").write_text(f"package main\n\nfunc main() {{\n{long_code}\n}}\n")

    # Binary file
    (root / "app.bin").write_bytes(b"\x00\x01\x02\xff\xfe\xfd")

    return root


def test_1_path_traversal_rejected(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    assert explorer._safe_resolve("../outside.txt") is None
    assert explorer.list_files("../") == []
    excerpt, lines = explorer.read_excerpt("../outside.txt")
    assert excerpt == "" and lines == 0


def test_2_absolute_path_rejected(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    assert explorer._safe_resolve("/etc/passwd") is None
    excerpt, lines = explorer.read_excerpt("/etc/passwd")
    assert excerpt == "" and lines == 0


def test_3_symlink_pointing_outside_project_rejected(repo_fixture: Path, tmp_path: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("outside data")

    symlink_file = repo_fixture / "link_outside.txt"
    try:
        symlink_file.symlink_to(outside_target)
        resolved = explorer._safe_resolve("link_outside.txt")
        assert resolved is None
    except OSError:
        pass


def test_4_5_secrets_ignored(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    all_files = explorer.list_files(".")
    assert ".env" not in all_files
    assert ".env.local" not in all_files


def test_6_7_8_9_cert_and_keys_ignored(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    all_files = explorer.list_files(".")
    assert "cert.pem" not in all_files
    assert "private.key" not in all_files
    assert "credentials.json" not in all_files
    assert "auth.json" not in all_files


def test_10_11_git_and_node_modules_ignored(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    all_files = explorer.list_files(".")
    for f in all_files:
        assert not f.startswith(".git")
        assert not f.startswith("node_modules")



def test_12_migration_sql_found(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    sql_files = explorer.find_files(["backend/migrations"], extensions=[".sql"])
    assert "backend/migrations/001_create_users.sql" in sql_files
    assert "backend/migrations/002_create_wallets.sql" in sql_files


def test_13_text_search_correct_line_number(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    results = explorer.search_text("CREATE TABLE users", ["backend/migrations"])
    assert len(results) >= 1
    assert results[0]["path"] == "backend/migrations/001_create_users.sql"
    assert results[0]["line"] == 1


def test_14_excerpt_limited(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    excerpt, total_lines = explorer.read_excerpt("backend/internal/app.go", max_lines=10, max_chars=100)
    assert total_lines > 100
    assert len(excerpt.splitlines()) <= 10
    assert len(excerpt) <= 100


def test_15_results_count_limited(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy, max_results=1)
    results = explorer.list_files("backend/migrations")
    assert len(results) == 1


def test_16_total_context_chars_limited(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy, max_total_chars=50)
    results = explorer.search_text("line_", ["backend/internal"])
    total_len = sum(len(r["excerpt"]) for r in results)
    assert total_len <= 150


def test_17_binary_file_skipped(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    excerpt, lines = explorer.read_excerpt("app.bin")
    assert excerpt == "" and lines == 0


def test_18_explorer_does_not_modify_repo(repo_fixture: Path):
    policy = create_dummy_policy()
    before_mtime = (repo_fixture / "backend/migrations/001_create_users.sql").stat().st_mtime
    explorer = RepositoryExplorer(repo_fixture, policy)
    _ = explorer.list_files(".")
    _ = explorer.search_text("users", ["backend"])
    _ = explorer.read_excerpt("backend/migrations/001_create_users.sql")
    after_mtime = (repo_fixture / "backend/migrations/001_create_users.sql").stat().st_mtime
    assert before_mtime == after_mtime


def test_19_denied_paths_respected(repo_fixture: Path):
    policy = create_dummy_policy(denied_paths=[".agent/secrets/**", "backend/migrations/002*"])
    explorer = RepositoryExplorer(repo_fixture, policy)
    all_files = explorer.list_files(".")
    assert not any(f.startswith(".agent/secrets") for f in all_files)
    assert "backend/migrations/002_create_wallets.sql" not in all_files


def test_20_all_results_use_relative_paths(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    files = explorer.list_files(".")
    for f in files:
        assert not f.startswith("/")
        assert not f.startswith("\\")
        assert not ".." in f

    searches = explorer.search_text("CREATE", ["backend/migrations"])
    for s in searches:
        assert not s["path"].startswith("/")


def test_21_source_file_not_entirely_copied(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    evidence = explorer.inspect_candidate_task("App implementation", ["line_"], ["backend/internal"])
    for r in evidence["search_results"]:
        assert len(r["excerpt"]) <= 500


def test_22_internal_symlink_handled_safely(repo_fixture: Path):
    policy = create_dummy_policy()
    explorer = RepositoryExplorer(repo_fixture, policy)
    internal_target = repo_fixture / "backend" / "migrations" / "001_create_users.sql"
    link = repo_fixture / "users_link.sql"
    try:
        link.symlink_to(internal_target)
        resolved = explorer._safe_resolve("users_link.sql")
        if resolved:
            assert resolved.relative_to(repo_fixture)
    except OSError:
        pass
