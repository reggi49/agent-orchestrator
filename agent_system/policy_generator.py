from agent_system.project_protocol import (
    AgentAccess,
    ProjectManifest,
    ProjectPolicy,
    SecurityPolicy,
)


DEFAULT_DENIED_PATHS = [
    ".git/**",
    ".agent/runtime/**",
    ".agent/cache/**",
    ".agent/tmp/**",
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/credentials.*",
    "**/auth.json",
    "**/node_modules/**",
    "**/.next/**",
    "**/.venv/**",
]


def collect_manifest_source_paths(manifest: ProjectManifest) -> list[str]:
    """Collect all source paths declared in manifest."""
    paths: list[str] = []
    sources = manifest.sources
    paths.extend(sources.rules)
    paths.extend(sources.requirements)
    paths.extend(sources.tasks)
    paths.extend(sources.handoffs)
    paths.extend(sources.reports)
    for comp in manifest.components.values():
        if comp.manifest:
            paths.append(comp.manifest)
    return sorted(list(set(paths)))


def generate_default_policy(manifest: ProjectManifest) -> ProjectPolicy:
    """Generate default safe policy for the project."""
    manifest_sources = collect_manifest_source_paths(manifest)

    security = SecurityPolicy(
        deny_hidden_files_by_default=True,
        denied_paths=DEFAULT_DENIED_PATHS,
    )

    # Agents setup
    agents: dict[str, AgentAccess] = {}

    # 1. Project Reader
    agents["project_reader"] = AgentAccess(
        read=manifest_sources,
        write=[],
        commands=[],
    )

    # 2. Analyst
    agents["analyst"] = AgentAccess(
        read=manifest_sources,
        write=[
            ".agent/artifacts/plans/**",
            ".agent/artifacts/reports/**",
        ],
        commands=[],
    )

    # 3. Supervisor
    agents["supervisor"] = AgentAccess(
        read=[
            ".agent/artifacts/**",
            *manifest_sources,
        ],
        write=[
            ".agent/artifacts/plans/**",
            ".agent/artifacts/reports/**",
        ],
        commands=[],
    )

    # 4. Backend
    backend_comp = manifest.components.get("backend")
    backend_read = list(manifest_sources)
    if backend_comp:
        root_pattern = f"{backend_comp.root}/**" if backend_comp.root != "." else "**"
        backend_read.insert(0, root_pattern)

    agents["backend"] = AgentAccess(
        read=backend_read,
        write=[],
        commands=[],
    )

    # 5. Frontend
    frontend_comp = manifest.components.get("frontend")
    frontend_read = list(manifest_sources)
    if frontend_comp:
        root_pattern = f"{frontend_comp.root}/**" if frontend_comp.root != "." else "**"
        frontend_read.insert(0, root_pattern)

    agents["frontend"] = AgentAccess(
        read=frontend_read,
        write=[],
        commands=[],
    )

    # 6. QA
    qa_read = list(manifest_sources)
    for comp in manifest.components.values():
        root_pattern = f"{comp.root}/**" if comp.root != "." else "**"
        if root_pattern not in qa_read:
            qa_read.append(root_pattern)

    agents["qa"] = AgentAccess(
        read=qa_read,
        write=[
            ".agent/artifacts/reports/**",
        ],
        commands=[],
    )

    return ProjectPolicy(
        schema_version="1.0",
        mode="read_only",
        security=security,
        agents=agents,
    )
