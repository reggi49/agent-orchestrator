import os
import re
from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when a path violates security constraints."""
    pass


def validate_relative_pattern(pattern: str) -> str:
    """Validate that a pattern or path is strictly relative and safe.
    
    Rejects:
    - empty string / whitespace
    - null bytes
    - absolute paths
    - path traversal (..)
    - home expansion (~)
    - file URI (file:)
    - URLs (e.g. http://, https://)
    - Windows drive paths (e.g. C:) or UNC paths (\\\\)
    """
    if pattern is None:
        raise PathSafetyError("Pattern cannot be None.")
    
    clean_pattern = pattern.strip()
    if not clean_pattern:
        raise PathSafetyError("Pattern cannot be empty.")

    if "\0" in clean_pattern:
        raise PathSafetyError("Pattern contains null byte.")

    # Check for URLs and URIs
    lower = clean_pattern.lower()
    if lower.startswith("file:") or "://" in lower or lower.startswith("http:") or lower.startswith("https:"):
        raise PathSafetyError(f"Pattern cannot be a URL or URI: {pattern}")

    # Check for Windows drive paths or UNC paths
    if re.match(r"^[a-zA-Z]:", clean_pattern) or clean_pattern.startswith(r"\\"):
        raise PathSafetyError(f"Windows drive or UNC paths not allowed: {pattern}")

    # Check for home expansion
    if clean_pattern.startswith("~"):
        raise PathSafetyError(f"Home expansion (~) not allowed: {pattern}")

    # Check for absolute path
    if clean_pattern.startswith("/") or clean_pattern.startswith("\\"):
        raise PathSafetyError(f"Absolute paths not allowed: {pattern}")

    # Check for path traversal (..)
    parts = re.split(r"[/\\]+", clean_pattern)
    if ".." in parts:
        raise PathSafetyError(f"Path traversal (..) not allowed: {pattern}")

    return clean_pattern


def resolve_project_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a relative path against project_root safely.
    
    Guarantees:
    - Path is relative and does not traverse outside project_root
    - No null bytes
    - No absolute paths
    - Result stays strictly within project_root
    """
    if project_root is None:
        raise PathSafetyError("project_root cannot be None.")
    
    resolved_root = project_root.expanduser().resolve()
    
    # Validate the relative path pattern first
    safe_rel = validate_relative_pattern(relative_path)

    # Join and resolve
    candidate = (resolved_root / safe_rel).resolve()

    # Verify that candidate is inside or equal to resolved_root
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise PathSafetyError(
            f"Resolved path escapes project root: '{relative_path}' -> '{candidate}' outside '{resolved_root}'"
        )

    return candidate
