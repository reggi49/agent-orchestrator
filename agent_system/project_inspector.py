from dataclasses import dataclass, field
from pathlib import Path

from agent_system.path_safety import validate_relative_pattern
from agent_system.project_loader import load_project
from agent_system.project_protocol import ValidatedProject


@dataclass
class InspectionReport:
    project_id: str
    project_name: str
    project_type: str
    project_root: Path
    schema_versions: dict[str, str]
    components: dict[str, dict]
    context_sources_found: list[str]
    context_sources_missing: list[str]
    policy_mode: str
    workflow_profile: str
    artifact_storage_root: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    portability_status: str = "unknown"
    portability_issues: list[str] = field(default_factory=list)


def inspect_project(project_root: Path) -> InspectionReport:
    """Generate inspection report without modifying files or calling LLM."""
    root = project_root.expanduser().resolve()
    
    # Load validated project
    project = load_project(root)
    
    # Collect basic info
    manifest = project.manifest
    policy = project.policy
    workflow = project.workflow
    
    project_id = manifest.project.id
    project_name = manifest.project.name
    project_type = manifest.project.type
    
    schema_versions = {
        "manifest": manifest.schema_version,
        "policy": policy.schema_version,
        "workflow": workflow.schema_version,
    }
    
    # Components summary
    components_summary = {}
    for name, comp in manifest.components.items():
        components_summary[name] = {
            "root": comp.root,
            "kind": comp.kind,
            "language": comp.language,
            "framework": comp.framework,
            "manifest": comp.manifest,
        }
    
    # Check which context sources exist
    context_found: list[str] = []
    context_missing: list[str] = []
    
    all_sources = (
        manifest.sources.rules +
        manifest.sources.requirements +
        manifest.sources.tasks +
        manifest.sources.handoffs +
        manifest.sources.reports
    )
    
    for src in all_sources:
        src_path = root / src
        if src_path.exists():
            context_found.append(src)
        else:
            context_missing.append(src)
    
    # Policy and workflow info
    policy_mode = policy.mode if isinstance(policy.mode, str) else policy.mode.value
    workflow_profile = workflow.profile
    artifact_storage_root = manifest.artifact_storage.root
    
    # Portability validation
    portability_issues: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    
    # Check for absolute paths in all config
    try:
        for comp_name, comp in manifest.components.items():
            if comp.root.startswith("/") or (len(comp.root) > 1 and comp.root[1] == ":"):
                portability_issues.append(f"Component {comp_name} has absolute root: {comp.root}")
        
        if manifest.artifact_storage.root.startswith("/"):
            portability_issues.append(f"Artifact storage has absolute root: {manifest.artifact_storage.root}")
        
        # Check artifact storage is within .agent/
        if not (manifest.artifact_storage.root == ".agent" or manifest.artifact_storage.root.startswith(".agent/")):
            portability_issues.append(
                f"Artifact storage root must be within .agent/: {manifest.artifact_storage.root}"
            )
    except Exception as e:
        errors.append(f"Path validation error: {e}")
    
    # Check for missing context sources
    if context_missing:
        warnings.append(f"{len(context_missing)} context source(s) declared but not found on disk")
    
    # Determine portability status
    if errors:
        portability_status = "invalid"
    elif portability_issues:
        portability_status = "not_portable"
    elif warnings:
        portability_status = "portable_with_warnings"
    else:
        portability_status = "portable"
    
    return InspectionReport(
        project_id=project_id,
        project_name=project_name,
        project_type=project_type,
        project_root=root,
        schema_versions=schema_versions,
        components=components_summary,
        context_sources_found=context_found,
        context_sources_missing=context_missing,
        policy_mode=policy_mode,
        workflow_profile=workflow_profile,
        artifact_storage_root=artifact_storage_root,
        warnings=warnings,
        errors=errors,
        portability_status=portability_status,
        portability_issues=portability_issues,
    )


def format_inspection_report(report: InspectionReport) -> str:
    """Format inspection report as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("PROJECT INSPECTION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Project ID:   {report.project_id}")
    lines.append(f"Project Name: {report.project_name}")
    lines.append(f"Project Type: {report.project_type}")
    lines.append(f"Project Root: {report.project_root}")
    lines.append("")
    lines.append("Schema Versions:")
    for key, version in report.schema_versions.items():
        lines.append(f"  {key}: {version}")
    lines.append("")
    lines.append(f"Policy Mode:      {report.policy_mode}")
    lines.append(f"Workflow Profile: {report.workflow_profile}")
    lines.append(f"Artifact Storage: {report.artifact_storage_root}")
    lines.append("")
    lines.append(f"Components ({len(report.components)}):")
    if report.components:
        for name, info in report.components.items():
            lines.append(f"  {name}:")
            lines.append(f"    root:      {info['root']}")
            lines.append(f"    kind:      {info['kind']}")
            lines.append(f"    language:  {info['language']}")
            if info.get('framework'):
                lines.append(f"    framework: {info['framework']}")
            if info.get('manifest'):
                lines.append(f"    manifest:  {info['manifest']}")
    else:
        lines.append("  (none detected)")
    lines.append("")
    lines.append(f"Context Sources Found: {len(report.context_sources_found)}")
    if report.context_sources_found:
        for src in report.context_sources_found[:10]:
            lines.append(f"  ✓ {src}")
        if len(report.context_sources_found) > 10:
            lines.append(f"  ... and {len(report.context_sources_found) - 10} more")
    lines.append("")
    if report.context_sources_missing:
        lines.append(f"Context Sources Missing: {len(report.context_sources_missing)}")
        for src in report.context_sources_missing[:5]:
            lines.append(f"  ✗ {src}")
        if len(report.context_sources_missing) > 5:
            lines.append(f"  ... and {len(report.context_sources_missing) - 5} more")
        lines.append("")
    if report.warnings:
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  ⚠ {w}")
        lines.append("")
    if report.errors:
        lines.append("Errors:")
        for e in report.errors:
            lines.append(f"  ✗ {e}")
        lines.append("")
    if report.portability_issues:
        lines.append("Portability Issues:")
        for issue in report.portability_issues:
            lines.append(f"  ✗ {issue}")
        lines.append("")
    lines.append(f"Portability Status: {report.portability_status.upper()}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)

