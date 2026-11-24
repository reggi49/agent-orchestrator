import re
from pathlib import Path
from typing import Any

from agent_system.integrator import is_doc_file, is_test_file
from agent_system.schemas import ProgramTask


def normalize_objective_to_requirements(objective: str) -> list[dict[str, str]]:
    """Normalize CEO Objective into structured Requirement IDs (FR-*, QR-*, SEC-*, etc.)."""
    requirements: list[dict[str, str]] = []
    lines = [l.strip() for l in objective.strip().splitlines() if l.strip()]

    fr_count = 0
    qr_count = 0
    sec_count = 0
    const_count = 0

    for line in lines:
        clean = line.lstrip("#-*1234567890. ").strip()
        if not clean or len(clean) < 4:
            continue

        lower = clean.lower()
        if any(k in lower for k in ("security", "auth", "token", "password", "aman", "keamanan")):
            sec_count += 1
            req_id = f"SEC-{sec_count:03d}"
            cat = "security"
        elif any(k in lower for k in ("performance", "latency", "quality", "test", "pengujian", "black-box")):
            qr_count += 1
            req_id = f"QR-{qr_count:03d}"
            cat = "quality"
        elif any(k in lower for k in ("constraint", "jangan", "dilarang", "batasan", "limit")):
            const_count += 1
            req_id = f"CONSTRAINT-{const_count:03d}"
            cat = "constraint"
        else:
            fr_count += 1
            req_id = f"FR-{fr_count:03d}"
            cat = "functional"

        requirements.append({
            "id": req_id,
            "category": cat,
            "description": clean,
        })

    if not requirements:
        requirements = [
            {"id": "FR-001", "category": "functional", "description": objective.strip()[:100]},
            {"id": "DOD-001", "category": "definition_of_done", "description": "Seluruh test lulus dan diff review disetujui."},
        ]

    return requirements


def validate_task_objective_alignment(
    task: ProgramTask,
    valid_requirement_ids: set[str],
) -> tuple[bool, str | None]:
    """Validate task is mapped to at least one valid requirement ID before execution."""
    if not task.mapped_requirement_ids:
        return False, f"Task '{task.task_id}' ({task.title}) tidak memiliki mapped_requirement_ids."

    for req_id in task.mapped_requirement_ids:
        if req_id not in valid_requirement_ids:
            return False, f"Task '{task.task_id}' memetakan requirement ID '{req_id}' yang tidak terdaftar dalam program."

    return True, None


def detect_objective_drift(
    task: ProgramTask,
    actual_changed_files: list[str],
) -> dict[str, Any]:
    """Detect unmapped production code changes or architecture drift against mapped scope."""
    allowed_writable = set(task.expected_writable_files + task.conditionally_writable_files)
    unmapped_production_files = []

    for f in actual_changed_files:
        clean = f.replace("\\", "/").lstrip("/")
        # Allow test files and doc files
        if is_test_file(clean) or is_doc_file(clean):
            continue
        # Production files not in allowed writable scope are unmapped changes
        if allowed_writable:
            matched = any(
                clean == aw
                or clean.startswith(aw.rstrip("/"))
                or (aw.endswith("/") and clean.startswith(aw))
                or Path(clean).name == Path(aw).name
                for aw in allowed_writable
            )
            if not matched:
                unmapped_production_files.append(clean)

    if unmapped_production_files:
        return {
            "drift_detected": True,
            "status": "drift_detected",
            "unmapped_changes": unmapped_production_files,
            "drift_reason": (
                f"Ditemukan perubahan pada kode produksi di luar lingkup requirement yang dipetakan: "
                f"{', '.join(unmapped_production_files)}."
            ),
        }

    return {
        "drift_detected": False,
        "status": "aligned",
        "unmapped_changes": [],
        "drift_reason": None,
    }
