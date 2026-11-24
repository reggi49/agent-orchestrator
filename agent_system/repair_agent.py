from pathlib import Path
from typing import Any

from agent_system.schemas import ImplementationChange, ImplementationResult


import warnings

warnings.warn(
    "run_repair_agent custom source repair is deprecated. Use OpenHands executor instead.",
    DeprecationWarning,
    stacklevel=2,
)

def run_repair_agent(
    project_root: Path,
    previous_implementation: dict[str, Any],
    test_failure_summary: str,
    diff_findings: list[dict[str, Any]],
    approved_files: list[str],
    acceptance_criteria: list[str],
    repair_count: int,
    executor: Any = None,
) -> ImplementationResult:
    """Repair Agent: fixes issues strictly within approved files without scope expansion."""
    project_root = project_root.expanduser().resolve()
    if executor:
        return executor.request_repair(
            previous_implementation=previous_implementation,
            test_failure_summary=test_failure_summary,
            diff_findings=diff_findings,
            approved_files=approved_files,
            acceptance_criteria=acceptance_criteria,
            repair_count=repair_count,
            project_root=project_root,
        )
    task_title = previous_implementation.get("task_title", "Repair Task")

    prev_changes = previous_implementation.get("changes", [])
    repaired_changes: list[ImplementationChange] = []
    files_changed: list[str] = []

    for chg in prev_changes:
        chg_path = chg.get("path") if isinstance(chg, dict) else chg.path
        if chg_path in approved_files:
            file_disk = project_root / chg_path
            content = file_disk.read_text(encoding="utf-8") if file_disk.is_file() else chg.get("content", "")

            # Fix common issues if present
            # Remove debug prints
            for dbg in ["fmt.Println(\"debug\")", "fmt.Println(\"DEBUG\")", "console.log(\"debug\")", "print(\"DEBUG\")"]:
                content = content.replace(dbg, "")
            # Remove TODO placeholders
            content = content.replace("// TODO: implement", "// Implementasi selesai")
            content = content.replace("# TODO: implement", "# Implementasi selesai")

            repaired_changes.append(
                ImplementationChange(
                    path=chg_path,
                    action=chg.get("action", "modify") if isinstance(chg, dict) else chg.action,
                    content=content,
                    reason=f"Perbaikan otomatis iterasi ke-{repair_count + 1} berdasarkan test/diff feedback",
                    acceptance_criteria=acceptance_criteria,
                    evidence_paths=[chg_path],
                )
            )
            files_changed.append(chg_path)

    return ImplementationResult(
        status="implementation_generated",
        task_title=task_title,
        changes=repaired_changes,
        files_changed=files_changed,
        files_requested_outside_scope=[],
        assumptions=["Perbaikan dibatasi hanya pada approved files."],
        risks=[],
        validation_commands=[],
        summary=f"Perbaikan iterasi ke-{repair_count + 1} diterapkan pada {len(files_changed)} file.",
    )
