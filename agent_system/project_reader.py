import fnmatch
import json
from pathlib import Path

from agent_system.config import get_settings
from agent_system.path_safety import resolve_project_path
from agent_system.project_detector import detect_project, is_secret_file
from agent_system.project_loader import ProjectNotFoundError, load_project
from agent_system.project_protocol import ValidatedProject
from agent_system.schemas import (
    ProjectFile,
    RepositoryContext,
    StackComponent,
)


settings = get_settings()


def is_path_denied(rel_path: str, denied_patterns: list[str]) -> bool:
    """Check if a relative path matches any denied pattern."""
    for pattern in denied_patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        try:
            if Path(rel_path).match(pattern):
                return True
        except Exception:
            pass
    return False


def read_text_file(path: Path, max_chars: int) -> str:
    """Read text file safely with character limit."""
    if not path.is_file():
        return ""

    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""

    return content[:max_chars]


def gather_source_paths(project: ValidatedProject) -> list[str]:
    """Gather all source paths declared in project manifest."""
    paths: list[str] = []
    
    sources = project.manifest.sources
    paths.extend(sources.rules)
    paths.extend(sources.requirements)
    paths.extend(sources.tasks)
    paths.extend(sources.handoffs)
    paths.extend(sources.reports)
    
    for comp in project.manifest.components.values():
        if comp.manifest:
            paths.append(comp.manifest)
    
    return paths


def build_stack_components(project: ValidatedProject) -> list[StackComponent]:
    """Build StackComponent list from manifest components."""
    components: list[StackComponent] = []
    
    for name, comp in project.manifest.components.items():
        scripts = comp.metadata.get("available_scripts", [])
        components.append(
            StackComponent(
                name=name,
                root=comp.root,
                language=comp.language,
                framework=comp.framework,
                manifest=comp.manifest,
                available_scripts=scripts if isinstance(scripts, list) else [],
            )
        )
    
    return components


def read_repository(project_root: Path) -> RepositoryContext:
    """Read repository using portable project protocol.
    
    If .agent/manifest.yaml exists, uses validated project configuration.
    Otherwise, falls back to auto-detection (legacy mode).
    """
    project_root = project_root.expanduser().resolve()

    if not project_root.is_dir():
        raise ValueError(
            f"Project directory tidak ditemukan: {project_root}"
        )

    # Try to load validated project
    try:
        validated_project = load_project(project_root)
        using_protocol = True
    except (ProjectNotFoundError, FileNotFoundError):
        # Fallback: auto-detect (legacy mode for backward compatibility)
        manifest = detect_project(project_root)
        from agent_system.policy_generator import generate_default_policy
        from agent_system.workflow_generator import generate_default_workflow
        
        validated_project = ValidatedProject(
            root=project_root,
            manifest=manifest,
            policy=generate_default_policy(manifest),
            workflow=generate_default_workflow(),
        )
        using_protocol = False

    source_paths = gather_source_paths(validated_project)
    denied_patterns = validated_project.policy.security.denied_paths
    max_files = validated_project.workflow.limits.max_files_per_search
    max_total_chars = validated_project.workflow.limits.max_context_characters
    max_per_file_chars = settings.max_file_characters

    files: list[ProjectFile] = []
    total_chars = 0
    files_read = 0

    for rel_path in source_paths:
        if files_read >= max_files:
            break
        
        if is_path_denied(rel_path, denied_patterns):
            continue
        
        try:
            abs_path = resolve_project_path(project_root, rel_path)
        except Exception:
            continue
        
        if is_secret_file(abs_path):
            continue
        
        # Handle glob patterns
        if "*" in rel_path:
            try:
                for matched in project_root.glob(rel_path):
                    if files_read >= max_files or total_chars >= max_total_chars:
                        break
                    if matched.is_file() and not is_secret_file(matched):
                        rel_matched = str(matched.relative_to(project_root))
                        if not is_path_denied(rel_matched, denied_patterns):
                            content = read_text_file(matched, max_per_file_chars)
                            if content:
                                files.append(ProjectFile(path=rel_matched, content=content))
                                total_chars += len(content)
                                files_read += 1
            except Exception:
                pass
        else:
            if abs_path.is_file():
                content = read_text_file(abs_path, max_per_file_chars)
                if content:
                    files.append(ProjectFile(path=rel_path, content=content))
                    total_chars += len(content)
                    files_read += 1

    stack = build_stack_components(validated_project)
    repository_type = validated_project.manifest.project.type
    stack_names = ", ".join(
        f"{component.name}: {component.language}"
        for component in stack
    )

    protocol_note = " (using portable protocol)" if using_protocol else " (auto-detected)"
    summary = (
        f"Repository {validated_project.manifest.project.name} terdeteksi sebagai "
        f"{repository_type}{protocol_note}. Stack: {stack_names or 'unknown'}."
    )

    return RepositoryContext(
        project_name=validated_project.manifest.project.name,
        project_root=str(project_root),
        repository_type=repository_type,
        stack=stack,
        important_files=files,
        repository_summary=summary,
    )

        