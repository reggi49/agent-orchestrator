import json
import re
from pathlib import Path

from agent_system.project_protocol import (
    ArtifactCategories,
    ArtifactStorage,
    ContextSources,
    ProjectComponent,
    ProjectIdentity,
    ProjectManifest,
)


IGNORED_PATTERNS = {
    ".git",
    ".venv",
    "node_modules",
    ".next",
    ".agent",
}

SECRET_EXTENSIONS = {".pem", ".key"}


def slugify(name: str) -> str:
    """Create a stable slug from directory name."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return clean or "unnamed-project"




def detect_components(project_root: Path) -> dict[str, ProjectComponent]:
    """Detect software components without using LLM."""
    components: dict[str, ProjectComponent] = {}

    # 1. Backend Go (in backend/ or root)
    backend_candidates = [
        ("backend", project_root / "backend" / "go.mod", "backend/go.mod"),
        ("backend", project_root / "go.mod", "go.mod"),
    ]
    for comp_name, mod_path, rel_manifest in backend_candidates:
        if mod_path.is_file():
            comp_root = "backend" if "backend" in rel_manifest else "."
            components["backend"] = ProjectComponent(
                root=comp_root,
                kind="backend",
                language="Go",
                framework=None,
                manifest=rel_manifest,
                metadata={},
            )
            break

    # 2. Frontend Node/TypeScript/Next.js (in frontend/ or root)
    frontend_candidates = [
        ("frontend", project_root / "frontend" / "package.json", "frontend/package.json"),
        ("frontend", project_root / "package.json", "package.json"),
    ]
    for comp_name, pkg_path, rel_manifest in frontend_candidates:
        if pkg_path.is_file():
            comp_root_dir = pkg_path.parent
            comp_root = "frontend" if "frontend" in rel_manifest else "."

            # Read package.json safely
            scripts: list[str] = []
            has_next_dep = False
            try:
                data = json.loads(pkg_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    raw_scripts = data.get("scripts", {})
                    if isinstance(raw_scripts, dict):
                        scripts = sorted(raw_scripts.keys())
                    deps = data.get("dependencies", {})
                    dev_deps = data.get("devDependencies", {})
                    if isinstance(deps, dict) and "next" in deps:
                        has_next_dep = True
                    if isinstance(dev_deps, dict) and "next" in dev_deps:
                        has_next_dep = True
            except Exception:
                pass

            # Detect Next.js config files
            has_next_config = any(
                (comp_root_dir / cfg).is_file()
                for cfg in [
                    "next.config.js",
                    "next.config.mjs",
                    "next.config.ts",
                ]
            )

            is_next = has_next_dep or has_next_config

            # Language detection: TypeScript if tsconfig.json exists
            is_typescript = (comp_root_dir / "tsconfig.json").is_file()
            lang = "TypeScript" if is_typescript else "JavaScript"

            components["frontend"] = ProjectComponent(
                root=comp_root,
                kind="frontend",
                language=lang,
                framework="Next.js" if is_next else None,
                manifest=rel_manifest,
                metadata={"available_scripts": scripts} if scripts else {},
            )
            break

    return components



def is_secret_file(path_obj: Path) -> bool:
    """Check if file matches known secret filename patterns."""
    name = path_obj.name.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    if name in {"auth.json", "credentials.json", "credentials.yml", "credentials.yaml"}:
        return True
    if name.startswith("credentials."):
        return True
    if path_obj.suffix.lower() in SECRET_EXTENSIONS:
        return True
    return False


def detect_context_sources(project_root: Path) -> ContextSources:
    """Detect documentation and context sources that actually exist."""
    rules: list[str] = []
    requirements: list[str] = []
    tasks: list[str] = []
    handoffs: list[str] = []
    reports: list[str] = []

    # Potential candidate paths
    potential_rules = [
        "AGENT_RULES.md",
        "frontend/AGENTS.md",
        "frontend/CLAUDE.md",
    ]
    potential_requirements = [
        "docs/prd.md",
        "frontend/docs/frontend_prd.md",
        "README.md",
        "frontend/README.md",
    ]
    potential_tasks = [
        "TASKS.md",
    ]
    potential_reports = [
        "tests/TEST_REPORT.md",
        "tests/COVERAGE_NOTES.md",
    ]

    for rel in potential_rules:
        if (project_root / rel).is_file():
            rules.append(rel)

    for rel in potential_requirements:
        if (project_root / rel).is_file():
            requirements.append(rel)

    for rel in potential_tasks:
        if (project_root / rel).is_file():
            tasks.append(rel)

    for rel in potential_reports:
        if (project_root / rel).is_file():
            reports.append(rel)

    # Detect handoffs (e.g. handoffs/**/*.md or handoffs/*.md)
    handoffs_dir = project_root / "handoffs"
    if handoffs_dir.is_dir():
        for p in sorted(handoffs_dir.rglob("*.md")):
            if p.is_file() and not is_secret_file(p):
                rel = str(p.relative_to(project_root))
                handoffs.append(rel)

    return ContextSources(
        rules=rules,
        requirements=requirements,
        tasks=tasks,
        handoffs=handoffs,
        reports=reports,
    )


def detect_project(project_root: Path) -> ProjectManifest:
    """Detect project structure and create a draft ProjectManifest."""
    resolved_root = project_root.expanduser().resolve()
    dir_name = resolved_root.name
    proj_id = slugify(dir_name)

    components = detect_components(resolved_root)
    sources = detect_context_sources(resolved_root)

    repo_type = (
        "monorepo"
        if len(components) > 1
        else "single-project"
        if len(components) == 1
        else "unknown"
    )

    identity = ProjectIdentity(
        id=proj_id,
        name=dir_name,
        type=repo_type,
    )

    storage = ArtifactStorage(
        root=".agent/artifacts",
        categories=ArtifactCategories(
            plans="plans",
            reports="reports",
            decisions="decisions",
        ),
    )

    return ProjectManifest(
        schema_version="1.0",
        project=identity,
        sources=sources,
        components=components,
        artifact_storage=storage,
        generated={},
    )


