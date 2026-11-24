"""Task Verifier - Compare recommended tasks against actual repository state.

Verifies whether a recommended task has already been completed, is partially done,
or is genuinely pending, using repository evidence and source code inspection.
"""

import re
from typing import Any

from agent_system.schemas import TaskVerification


def normalize_task_title(title: str) -> str:
    """Normalize task title for comparison."""
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    words = cleaned.split()
    # Remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "to", "for", "in", "with", "of", "on"}
    words = [w for w in words if w not in stop_words]
    return " ".join(sorted(words))


def are_tasks_substantially_identical(title1: str, title2: str, threshold: float = 0.6) -> bool:
    """Check if two task titles are substantially the same.
    
    Uses Jaccard similarity on normalized word sets.
    """
    w1 = set(re.findall(r"\w+", title1.lower()))
    w2 = set(re.findall(r"\w+", title2.lower()))
    
    # Remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "to", "for", "in", "with", "of", "on", "create", "define", "implement"}
    w1 -= stop_words
    w2 -= stop_words
    
    if not w1 or not w2:
        return title1.strip().lower() == title2.strip().lower()
    
    intersection = w1 & w2
    union = w1 | w2
    
    if not union:
        return False
    
    similarity = len(intersection) / len(union)
    return similarity > threshold


def is_duplicate_task(candidate_title: str, rejected_tasks: list[dict[str, Any]]) -> bool:
    """Check if candidate task is a duplicate of any rejected task."""
    for rejected in rejected_tasks:
        rejected_title = rejected.get("title", "")
        if are_tasks_substantially_identical(candidate_title, rejected_title):
            return True
    return False


def extract_migration_keywords(task_title: str, task_description: str) -> list[str]:
    """Extract keywords to search for migration-related tasks."""
    text = f"{task_title} {task_description}".lower()
    
    keywords = []
    
    # Check for migration-related terms
    if any(term in text for term in ["migration", "schema", "database", "table", "sql"]):
        keywords.extend([
            "CREATE TABLE",
            "DROP TABLE",
            "CREATE INDEX",
            "ALTER TABLE",
            "REFERENCES",
            "PRIMARY KEY",
            "FOREIGN KEY",
        ])
        
        # Entity names
        if any(entity in text for entity in ["user", "users"]):
            keywords.append("users")
        if any(entity in text for entity in ["wallet", "wallets"]):
            keywords.append("wallets")
        if any(entity in text for entity in ["product", "products"]):
            keywords.append("products")
        if any(entity in text for entity in ["transaction", "transactions"]):
            keywords.append("transactions")
    
    return keywords


def build_search_targets(task_title: str, task_description: str, components: dict[str, Any]) -> list[str]:
    """Build list of directories/files to search based on task content."""
    text = f"{task_title} {task_description}".lower()
    targets = []
    
    # Migration/database tasks
    if any(term in text for term in ["migration", "schema", "database", "table", "sql"]):
        targets.extend([
            "backend/migrations",
            "migrations",
            "backend/internal",
            "tests",
        ])
    
    # Backend tasks
    if "backend" in text or any(term in text for term in ["go", "api", "server", "repository", "service"]):
        targets.extend([
            "backend",
            "backend/internal",
            "backend/cmd",
        ])
    
    # Frontend tasks
    if "frontend" in text or any(term in text for term in ["next.js", "react", "ui", "component", "page"]):
        targets.extend([
            "frontend",
            "frontend/src",
            "frontend/app",
            "frontend/components",
        ])
    
    # Test tasks
    if "test" in text:
        targets.extend([
            "tests",
            "backend/internal",
            "frontend/__tests__",
        ])
    
    # If no specific targets, search common areas
    if not targets:
        targets.extend([
            "backend",
            "frontend",
            "tests",
            "docs",
        ])
    
    return list(set(targets))  # Remove duplicates
