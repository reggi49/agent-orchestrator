import json
from pathlib import Path

from agent_system.project_detector import detect_project
from agent_system.project_initializer import initialize_project


def setup_sample_project(project_dir: Path) -> None:
    # Setup sample monorepo structure
    (project_dir / "backend").mkdir(parents=True, exist_ok=True)
    (project_dir / "backend" / "go.mod").write_text("module example.com/app\n\ngo 1.22\n", encoding="utf-8")

    (project_dir / "frontend").mkdir(parents=True, exist_ok=True)
    pkg = {
        "name": "frontend-app",
        "scripts": {"dev": "next dev", "build": "next build"},
        "dependencies": {"next": "14.0.0", "react": "18.2.0"},
    }
    (project_dir / "frontend" / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
    (project_dir / "frontend" / "next.config.mjs").write_text("export default {};\n", encoding="utf-8")

    (project_dir / "README.md").write_text("# Project Readme\n", encoding="utf-8")
    (project_dir / "AGENT_RULES.md").write_text("# Agent Rules\n", encoding="utf-8")
    (project_dir / "TASKS.md").write_text("# Tasks\n", encoding="utf-8")
    (project_dir / ".env").write_text("SECRET_KEY=123456\n", encoding="utf-8")

    (project_dir / "frontend" / "node_modules").mkdir(parents=True, exist_ok=True)
    (project_dir / "frontend" / "node_modules" / "some-pkg.md").write_text("dummy", encoding="utf-8")


def test_11_dry_run_does_not_create_files(tmp_path: Path):
    setup_sample_project(tmp_path)
    result = initialize_project(tmp_path, apply=False)

    assert result["applied"] is False
    assert not (tmp_path / ".agent").exists()


def test_12_apply_creates_only_agent_dir(tmp_path: Path):
    setup_sample_project(tmp_path)
    before_files = sorted([str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if not p.name.startswith(".agent")])

    result = initialize_project(tmp_path, apply=True)
    assert result["applied"] is True
    assert (tmp_path / ".agent" / "manifest.yaml").is_file()
    assert (tmp_path / ".agent" / "policy.yaml").is_file()
    assert (tmp_path / ".agent" / "workflow.yaml").is_file()
    assert (tmp_path / ".agent" / "artifacts" / "plans").is_dir()

    # Check that non-.agent files remain unchanged
    after_files = sorted([str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if not str(p.relative_to(tmp_path)).startswith(".agent")])
    assert before_files == after_files


def test_13_existing_config_not_overwritten(tmp_path: Path):
    setup_sample_project(tmp_path)
    initialize_project(tmp_path, apply=True)

    manifest_file = tmp_path / ".agent" / "manifest.yaml"
    manifest_file.write_text("custom_content: true\n", encoding="utf-8")

    # Second init without force
    result2 = initialize_project(tmp_path, apply=True, force=False)
    assert ".agent/manifest.yaml" in result2["skipped_files"]
    assert manifest_file.read_text(encoding="utf-8") == "custom_content: true\n"


def test_14_force_only_modifies_agent_with_backup(tmp_path: Path):
    setup_sample_project(tmp_path)
    initialize_project(tmp_path, apply=True)

    manifest_file = tmp_path / ".agent" / "manifest.yaml"
    manifest_file.write_text("custom_content: true\n", encoding="utf-8")

    result_force = initialize_project(tmp_path, apply=True, force=True)
    assert ".agent/manifest.yaml" in result_force["created_files"]
    assert any("manifest.yaml" in b for b in result_force["backed_up_files"])
    # Backup file exists in .agent/tmp
    backups = list((tmp_path / ".agent" / "tmp").glob("backup_manifest.yaml_*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "custom_content: true\n"


def test_15_yaml_has_no_absolute_paths(tmp_path: Path):
    setup_sample_project(tmp_path)
    initialize_project(tmp_path, apply=True)

    for yml in ["manifest.yaml", "policy.yaml", "workflow.yaml"]:
        content = (tmp_path / ".agent" / yml).read_text(encoding="utf-8")
        assert str(tmp_path) not in content
        assert "/Users/" not in content
        # Check that there are no absolute paths starting with /
        for line in content.splitlines():
            stripped = line.strip()
            # If line is a list item or key-value with string
            if stripped.startswith("- /") or (": /" in stripped and not ": //" in stripped):
                assert False, f"Found absolute path in {yml}: {line}"



def test_16_go_backend_detected(tmp_path: Path):
    setup_sample_project(tmp_path)
    manifest = detect_project(tmp_path)
    assert "backend" in manifest.components
    assert manifest.components["backend"].language == "Go"
    assert manifest.components["backend"].manifest == "backend/go.mod"


def test_17_nextjs_frontend_detected(tmp_path: Path):
    setup_sample_project(tmp_path)
    manifest = detect_project(tmp_path)
    assert "frontend" in manifest.components
    assert manifest.components["frontend"].framework == "Next.js"
    assert manifest.components["frontend"].manifest == "frontend/package.json"


def test_18_context_sources_detected(tmp_path: Path):
    setup_sample_project(tmp_path)
    manifest = detect_project(tmp_path)
    assert "AGENT_RULES.md" in manifest.sources.rules
    assert "README.md" in manifest.sources.requirements
    assert "TASKS.md" in manifest.sources.tasks


def test_19_env_files_not_in_manifest(tmp_path: Path):
    setup_sample_project(tmp_path)
    manifest = detect_project(tmp_path)
    all_sources = (
        manifest.sources.rules
        + manifest.sources.requirements
        + manifest.sources.tasks
        + manifest.sources.handoffs
        + manifest.sources.reports
    )
    assert not any(".env" in s for s in all_sources)


def test_20_node_modules_not_in_manifest(tmp_path: Path):
    setup_sample_project(tmp_path)
    manifest = detect_project(tmp_path)
    all_sources = (
        manifest.sources.rules
        + manifest.sources.requirements
        + manifest.sources.tasks
        + manifest.sources.handoffs
        + manifest.sources.reports
    )
    assert not any("node_modules" in s for s in all_sources)
