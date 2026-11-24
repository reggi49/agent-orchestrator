import pytest
from pathlib import Path
from pydantic import ValidationError

from agent_system.path_safety import PathSafetyError, resolve_project_path, validate_relative_pattern
from agent_system.project_loader import ProjectNotFoundError, load_project
from agent_system.project_protocol import (
    ArtifactCategories,
    ArtifactStorage,
    ContextSources,
    ProjectComponent,
    ProjectIdentity,
    ProjectManifest,
    ProjectPolicy,
    SecurityPolicy,
)


def test_1_valid_manifest(tmp_path: Path):
    manifest = ProjectManifest(
        schema_version="1.0",
        project=ProjectIdentity(id="test-proj", name="test-proj", type="monorepo"),
        sources=ContextSources(rules=["AGENT_RULES.md"]),
        components={
            "backend": ProjectComponent(
                root="backend",
                kind="backend",
                language="Go",
                manifest="backend/go.mod",
            )
        },
        artifact_storage=ArtifactStorage(root=".agent/artifacts"),
    )
    assert manifest.project.id == "test-proj"
    assert manifest.components["backend"].root == "backend"


def test_2_absolute_component_path_rejected():
    with pytest.raises((ValidationError, PathSafetyError)):
        ProjectComponent(
            root="/absolute/path",
            kind="backend",
            language="Go",
        )


def test_3_path_traversal_rejected():
    with pytest.raises((ValidationError, PathSafetyError)):
        ProjectComponent(
            root="../outside",
            kind="backend",
            language="Go",
        )


def test_4_absolute_artifact_root_rejected():
    with pytest.raises((ValidationError, PathSafetyError, ValueError)):
        ArtifactStorage(root="/tmp/artifacts")


def test_5_artifact_root_outside_agent_rejected():
    with pytest.raises((ValidationError, ValueError)):
        ArtifactStorage(root="artifacts")


def test_6_missing_manifest_raises(tmp_path: Path):
    with pytest.raises((ProjectNotFoundError, FileNotFoundError)):
        load_project(tmp_path)


def test_7_missing_policy_raises(tmp_path: Path):
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "manifest.yaml").write_text("schema_version: '1.0'\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path)


def test_8_missing_workflow_raises(tmp_path: Path):
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "manifest.yaml").write_text("schema_version: '1.0'\n", encoding="utf-8")
    (agent_dir / "policy.yaml").write_text("schema_version: '1.0'\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path)


def test_9_secret_not_in_config_repr():
    policy = ProjectPolicy(
        schema_version="1.0",
        mode="read_only",
        security=SecurityPolicy(
            deny_hidden_files_by_default=True,
            denied_paths=["**/.env", "**/*.key"],
        ),
    )
    repr_str = repr(policy)
    assert "sk-" not in repr_str
    assert "password" not in repr_str


def test_10_unknown_policy_mode_rejected():
    with pytest.raises(ValidationError):
        ProjectPolicy(
            schema_version="1.0",
            mode="super_admin_write",  # Invalid mode
            security=SecurityPolicy(
                deny_hidden_files_by_default=True,
                denied_paths=[],
            ),
        )
