"""Repository Explorer - Read-only filesystem operations for agent verification.

Provides secure file listing, searching, and content reading within project boundaries.
All operations enforce path safety, respect denied paths, and return relative paths only.
"""

import fnmatch
import re
from pathlib import Path
from typing import Any

from agent_system.path_safety import resolve_project_path, validate_relative_pattern
from agent_system.project_protocol import ProjectPolicy


ALLOWED_TEXT_EXTENSIONS = {
    ".py", ".go", ".sql", ".ts", ".tsx", ".js", ".jsx", ".json",
    ".md", ".yaml", ".yml", ".toml", ".mod", ".sum",
    ".txt", ".env.example", ".gitignore",
    ".rs", ".java", ".c", ".cpp", ".h", ".cs", ".rb", ".php", ".sh",
}

IGNORED_DIRS = {
    ".git", "node_modules", ".next", ".venv", "__pycache__",
    "vendor", "dist", "build", ".cache",
}

SECRET_FILES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "credentials.json", "auth.json", "service-account.json",
}

SECRET_EXTENSIONS = {".pem", ".key", ".p12", ".pfx"}


def is_secret_path(path: Path) -> bool:
    """Check if path points to a secret file."""
    if path.name in SECRET_FILES:
        return True
    if path.suffix in SECRET_EXTENSIONS:
        return True
    if any(p in path.name for p in ["credential", "secret", "private"]):
        return True
    return False


def is_text_file(path: Path) -> bool:
    """Check if file is a readable text file."""
    if not path.is_file():
        return False
    if path.suffix not in ALLOWED_TEXT_EXTENSIONS and path.suffix:
        return False
    return True


def is_denied_by_policy(rel_path: str, policy: ProjectPolicy) -> bool:
    """Check if path is denied by project policy."""
    denied = policy.security.denied_paths
    for pattern in denied:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        try:
            if Path(rel_path).match(pattern):
                return True
        except Exception:
            pass
    return False


class RepositoryExplorer:
    """Secure read-only repository exploration."""

    def __init__(
        self,
        project_root: Path,
        policy: ProjectPolicy,
        max_file_size: int = 500_000,
        max_results: int = 50,
        max_total_chars: int = 200_000,
    ):
        self.project_root = project_root.expanduser().resolve()
        self.policy = policy
        self.max_file_size = max_file_size
        self.max_results = max_results
        self.max_total_chars = max_total_chars

    def _safe_resolve(self, rel_path: str) -> Path | None:
        """Resolve relative path safely within project root."""
        try:
            validate_relative_pattern(rel_path)
            resolved = resolve_project_path(self.project_root, rel_path)
            
            # Ensure resolved path is inside project root
            try:
                resolved.relative_to(self.project_root)
            except ValueError:
                return None
            
            return resolved
        except (ValueError, OSError):
            return None

    def _should_skip_dir(self, dir_path: Path) -> bool:
        """Check if directory should be skipped."""
        if dir_path.name in IGNORED_DIRS:
            return True
        if dir_path.name.startswith(".") and dir_path.name not in {".agent"}:
            return True
        return False

    def list_files(
        self,
        target_path: str = ".",
        pattern: str = "*",
        recursive: bool = True,
    ) -> list[str]:
        """List files matching pattern in target path.
        
        Returns relative paths only.
        """
        resolved = self._safe_resolve(target_path)
        if not resolved or not resolved.is_dir():
            return []

        results: list[str] = []
        
        try:
            if recursive:
                iterator = resolved.rglob(pattern)
            else:
                iterator = resolved.glob(pattern)
            
            for path in iterator:
                if path.is_file():
                    try:
                        rel_path = str(path.relative_to(self.project_root))
                    except ValueError:
                        continue
                    
                    if any(self._should_skip_dir(p) for p in path.parents if p != self.project_root):
                        continue
                    
                    if is_secret_path(path):
                        continue
                    
                    if is_denied_by_policy(rel_path, self.policy):
                        continue
                    
                    results.append(rel_path)
                    
                    if len(results) >= self.max_results:
                        break
        except (PermissionError, OSError):
            pass
        
        return sorted(results)

    def find_files(
        self,
        target_paths: list[str],
        extensions: list[str] | None = None,
    ) -> list[str]:
        """Find files in multiple target paths with optional extension filter.
        
        Returns relative paths only.
        """
        results: list[str] = []
        seen = set()
        
        for target in target_paths:
            if len(results) >= self.max_results:
                break
            
            resolved = self._safe_resolve(target)
            if not resolved:
                continue
            
            if resolved.is_file():
                try:
                    rel_path = str(resolved.relative_to(self.project_root))
                except ValueError:
                    continue
                
                if rel_path in seen:
                    continue
                
                if is_secret_path(resolved):
                    continue
                
                if is_denied_by_policy(rel_path, self.policy):
                    continue
                
                if extensions and resolved.suffix not in extensions:
                    continue
                
                results.append(rel_path)
                seen.add(rel_path)
            
            elif resolved.is_dir():
                for path in resolved.rglob("*"):
                    if len(results) >= self.max_results:
                        break
                    
                    if not path.is_file():
                        continue
                    
                    try:
                        rel_path = str(path.relative_to(self.project_root))
                    except ValueError:
                        continue
                    
                    if rel_path in seen:
                        continue
                    
                    if any(self._should_skip_dir(p) for p in path.parents if p != self.project_root):
                        continue
                    
                    if is_secret_path(path):
                        continue
                    
                    if is_denied_by_policy(rel_path, self.policy):
                        continue
                    
                    if extensions and path.suffix not in extensions:
                        continue
                    
                    results.append(rel_path)
                    seen.add(rel_path)
        
        return sorted(results)

    def read_excerpt(
        self,
        rel_path: str,
        max_lines: int = 100,
        max_chars: int = 50_000,
    ) -> tuple[str, int]:
        """Read file excerpt safely.
        
        Returns (content, total_lines).
        """
        resolved = self._safe_resolve(rel_path)
        if not resolved or not resolved.is_file():
            return ("", 0)
        
        if is_secret_path(resolved):
            return ("", 0)
        
        if is_denied_by_policy(rel_path, self.policy):
            return ("", 0)
        
        if not is_text_file(resolved):
            return ("", 0)
        
        try:
            if resolved.stat().st_size > self.max_file_size:
                return ("", 0)
            
            content = resolved.read_text(encoding="utf-8")
            lines = content.splitlines()
            total_lines = len(lines)
            
            if max_lines and total_lines > max_lines:
                excerpt_lines = lines[:max_lines]
                excerpt = "\n".join(excerpt_lines)
            else:
                excerpt = content
            
            if len(excerpt) > max_chars:
                excerpt = excerpt[:max_chars]
            
            return (excerpt, total_lines)
        except (UnicodeDecodeError, OSError):
            return ("", 0)

    def search_text(
        self,
        query: str,
        target_paths: list[str],
        context_lines: int = 3,
    ) -> list[dict[str, Any]]:
        """Search for text pattern in files.
        
        Returns list of matches with file path, line numbers, and excerpt.
        All paths are relative to project root.
        """
        results: list[dict[str, Any]] = []
        total_chars = 0
        
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        files = self.find_files(target_paths)
        
        for rel_path in files:
            if len(results) >= self.max_results:
                break
            
            if total_chars >= self.max_total_chars:
                break
            
            resolved = self._safe_resolve(rel_path)
            if not resolved or not resolved.is_file():
                continue
            
            try:
                content = resolved.read_text(encoding="utf-8")
                lines = content.splitlines()
                
                for i, line in enumerate(lines):
                    if pattern.search(line):
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        excerpt = "\n".join(lines[start:end])
                        
                        result = {
                            "path": rel_path,
                            "line": i + 1,
                            "excerpt": excerpt,
                            "match": line.strip(),
                        }
                        results.append(result)
                        total_chars += len(excerpt)
                        
                        if len(results) >= self.max_results:
                            break
                        
                        if total_chars >= self.max_total_chars:
                            break
            except (UnicodeDecodeError, OSError):
                continue
        
        return results

    def inspect_candidate_task(
        self,
        task_title: str,
        search_terms: list[str],
        target_paths: list[str],
    ) -> dict[str, Any]:
        """Inspect repository for evidence related to a candidate task.
        
        Returns summary with files found and key excerpts.
        """
        evidence = {
            "task_title": task_title,
            "files_found": [],
            "search_results": [],
            "summary": "",
        }
        
        files = self.find_files(target_paths)
        evidence["files_found"] = files[:self.max_results]
        
        for term in search_terms[:10]:
            results = self.search_text(term, target_paths, context_lines=2)
            for result in results[:5]:
                evidence["search_results"].append({
                    "term": term,
                    "path": result["path"],
                    "line": result["line"],
                    "excerpt": result["excerpt"][:500],
                })
        
        if files:
            evidence["summary"] = (
                f"Found {len(files)} relevant files. "
                f"Search found {len(evidence['search_results'])} matches across key terms."
            )
        else:
            evidence["summary"] = "No relevant files found for this task."
        
        return evidence

