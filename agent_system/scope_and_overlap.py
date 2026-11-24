from pathlib import Path
from typing import Any

from agent_system.schemas import ProgramTask


class OverlapType:
    NONE = "NONE"
    POSSIBLE = "POSSIBLE"
    DIRECT = "DIRECT"


def extract_package_or_module(file_path: str) -> str:
    """Extract parent package or directory from a relative file path."""
    p = Path(file_path.replace("\\", "/"))
    return str(p.parent)


def detect_task_overlap(task_a: ProgramTask, task_b: ProgramTask) -> str:
    """Classify overlap between two tasks into DIRECT, POSSIBLE, or NONE."""
    writable_a = set(task_a.expected_writable_files + task_a.conditionally_writable_files)
    writable_b = set(task_b.expected_writable_files + task_b.conditionally_writable_files)

    # 1. DIRECT OVERLAP: same expected or conditionally writable file
    direct_overlap = writable_a.intersection(writable_b)
    if direct_overlap:
        return OverlapType.DIRECT

    # Also check changed_files if already populated
    if task_a.changed_files and task_b.changed_files:
        if set(task_a.changed_files).intersection(set(task_b.changed_files)):
            return OverlapType.DIRECT

    # 2. POSSIBLE OVERLAP: same package, module, component, or shared entry point
    packages_a = {extract_package_or_module(f) for f in writable_a}
    packages_b = {extract_package_or_module(f) for f in writable_b}

    if packages_a.intersection(packages_b):
        return OverlapType.POSSIBLE

    if task_a.owner_role == task_b.owner_role and task_a.owner_role in ("backend", "frontend"):
        return OverlapType.POSSIBLE

    return OverlapType.NONE


def detect_cycle_in_dependencies(dependencies: dict[str, list[str]]) -> list[str] | None:
    """Detect cycles in dependency DAG using DFS. Returns cycle path if found."""
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in dependencies.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                path.append(neighbor)
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    for node in dependencies:
        if node not in visited:
            if dfs(node):
                return path

    return None


def analyze_backlog_overlaps_and_enforce_dependencies(
    tasks: list[ProgramTask],
) -> tuple[dict[str, Any], bool, str | None]:
    """Analyze overlap across task backlog, enforce dependencies, and check for cycles.

    Returns:
    - analysis_dict: machine-readable overlap matrix
    - is_valid: True if DAG is valid, False if cycle detected (fail-closed)
    - cycle_error: description of cycle if invalid
    """
    matrix: list[dict[str, Any]] = []
    dep_map: dict[str, list[str]] = {t.task_id: list(t.dependencies) for t in tasks}

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t_a = tasks[i]
            t_b = tasks[j]
            overlap = detect_task_overlap(t_a, t_b)

            entry = {
                "task_a": t_a.task_id,
                "task_b": t_b.task_id,
                "overlap": overlap,
            }
            matrix.append(entry)

            # For DIRECT overlap: enforce explicit dependency (task_b depends on task_a)
            if overlap == OverlapType.DIRECT:
                if t_a.task_id not in dep_map[t_b.task_id]:
                    dep_map[t_b.task_id].append(t_a.task_id)
                if t_a.task_id not in t_b.dependencies:
                    t_b.dependencies.append(t_a.task_id)

    # Check for dependency cycle
    cycle_path = detect_cycle_in_dependencies(dep_map)
    if cycle_path:
        cycle_str = " -> ".join(cycle_path)
        return (
            {"matrix": matrix, "dependencies": dep_map, "has_cycle": True, "cycle_path": cycle_path},
            False,
            f"FAIL-CLOSED: Dependency cycle terdeteksi pada overlap resolution: {cycle_str}",
        )

    return (
        {"matrix": matrix, "dependencies": dep_map, "has_cycle": False},
        True,
        None,
    )
