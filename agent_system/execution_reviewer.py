import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.project_loader import load_project
from agent_system.schemas import HumanExecutionReview
from agent_system.worktree_manager import get_default_worktree_root


def locate_execution_artifact(project_path: Path, execution_arg: str = "latest") -> tuple[Path, dict[str, Any]]:
    """Locate execution artifact from project directory.

    Supports:
    - full relative or absolute path
    - filename (e.g. execution_ea39ba24.json)
    - clean ID (e.g. ea39ba24 or execution_ea39ba24)
    - 'latest' keyword
    """
    execs_dir = project_path / ".agent" / "artifacts" / "executions"
    if not execs_dir.is_dir():
        raise FileNotFoundError(f"Direktori eksekusi '{execs_dir}' tidak ditemukan.")

    exec_files = sorted(execs_dir.glob("execution_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not exec_files:
        raise FileNotFoundError(f"Tidak ditemukan file execution di '{execs_dir}'.")

    if execution_arg in ("latest", "", None):
        # Prefer execution with status execution_review_ready, otherwise newest
        for ef in exec_files:
            try:
                data = json.loads(ef.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("status") == "execution_review_ready":
                    return ef, data
            except Exception:
                continue

        # Fallback to newest readable
        for ef in exec_files:
            try:
                data = json.loads(ef.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return ef, data
            except Exception:
                continue
        raise FileNotFoundError("Tidak ditemukan file execution yang valid.")

    # Explicit argument resolution
    cand = project_path / execution_arg
    if not cand.is_file():
        cand = Path(execution_arg)
    if not cand.is_file():
        clean_name = execution_arg if execution_arg.endswith(".json") else f"{execution_arg}.json"
        cand = execs_dir / clean_name
        if not cand.is_file() and not execution_arg.startswith("execution_"):
            cand = execs_dir / f"execution_{clean_name}"

    if not cand.is_file():
        raise FileNotFoundError(f"Execution artifact file '{execution_arg}' tidak ditemukan di '{execs_dir}'.")

    try:
        data = json.loads(cand.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Execution file '{cand.name}' rusak atau bukan JSON valid: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Execution file '{cand.name}' bukan objek JSON valid.")

    return cand, data


def validate_execution_for_human_approval(
    project_root: Path,
    execution_data: dict[str, Any],
) -> dict[str, Any]:
    """Validate that execution meets all strict prerequisites for human approval."""
    # 1. Artifact integrity
    if not execution_data.get("artifact_id") or not execution_data.get("result"):
        return {
            "valid": False,
            "error": "Integritas artifact eksekusi gagal: field wajib (artifact_id/result) tidak lengkap.",
        }

    res = execution_data.get("result", {})
    status = execution_data.get("status")

    # 2. Status must be execution_review_ready
    if status != "execution_review_ready":
        return {
            "valid": False,
            "error": f"Execution memiliki status '{status}', bukan 'execution_review_ready'. Hanya status 'execution_review_ready' yang dapat disetujui.",
        }

    # 3. Changed files must not be empty
    changed_files = res.get("actual_changed_files", []) or res.get("changed_files", [])
    if not changed_files:
        return {
            "valid": False,
            "error": "Daftar file yang diubah kosong (zero changed files). Approval ditolak.",
        }

    # 4. Test validation check
    val_exit_code = res.get("validation_exit_code", 0)
    val_cmds = res.get("validation_command_summary", [])
    if val_exit_code != 0 or any(c.get("exit_code") != 0 or c.get("status") != "passed" for c in val_cmds):
        return {
            "valid": False,
            "error": f"Validasi test gagal pada eksekusi (exit code: {val_exit_code}). Approval ditolak.",
        }

    # 5. Diff review status check
    diff_status = res.get("diff_check_status")
    indep_review = res.get("independent_review_status")
    if diff_status != "passed" or indep_review != "approved":
        return {
            "valid": False,
            "error": f"Diff review belum disetujui (diff_check: {diff_status}, independent_review: {indep_review}). Approval ditolak.",
        }

    # 6. Scope violations
    scope_violations = res.get("scope_violations", [])
    if scope_violations:
        return {
            "valid": False,
            "error": f"Terdapat scope violations pada eksekusi: {scope_violations}. Approval ditolak.",
        }

    # 7. Worktree existence check
    wt_id = res.get("worktree_identifier", "")
    if wt_id:
        wt_root = get_default_worktree_root(project_root)
        wt_path = wt_root / wt_id
        if not wt_path.is_dir():
            return {
                "valid": False,
                "error": f"Worktree '{wt_id}' tidak ditemukan di '{wt_path}'. Approval ditolak.",
            }

    # 8. Branch existence check
    branch_name = res.get("branch_name", "")
    if branch_name:
        res_br = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if not res_br.stdout.strip():
            return {
                "valid": False,
                "error": f"Branch '{branch_name}' tidak ditemukan di git repository. Approval ditolak.",
            }

    return {
        "valid": True,
        "error": None,
        "changed_files": changed_files,
        "test_summary": val_cmds,
        "worktree_identifier": wt_id,
        "branch_name": branch_name,
    }


def review_execution(
    project_root: Path,
    execution_file: Path,
    execution_data: dict[str, Any],
    decision: str,
    reviewer: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Execute formal human review on an execution artifact.

    Decisions:
    - 'approve': results in approved_for_integration
    - 'request-changes': results in changes_requested
    - 'reject': results in rejected
    """
    valid_decisions = {"approve", "request-changes", "reject"}
    if decision not in valid_decisions:
        raise ValueError(f"Keputusan review '{decision}' tidak valid. Harus salah satu dari: {sorted(valid_decisions)}")

    res = execution_data.get("result", {})
    current_status = execution_data.get("status")

    # Only execution_review_ready can be reviewed
    if current_status != "execution_review_ready":
        raise ValueError(
            f"Execution memiliki status '{current_status}'. "
            "Hanya execution dengan status 'execution_review_ready' yang dapat direview."
        )

    # For approval, enforce all strict integrity and verification gates
    if decision == "approve":
        validation = validate_execution_for_human_approval(project_root, execution_data)
        if not validation["valid"]:
            raise ValueError(f"Approval ditolak: {validation['error']}")
        new_status = "approved_for_integration"
    elif decision == "request-changes":
        new_status = "changes_requested"
    else:
        new_status = "rejected"

    # Resolve reviewer identity safely
    reviewer_identity = (
        (reviewer or "").strip()
        or os.getenv("REVIEWER_ID", "").strip()
        or os.getenv("REVIEWER_NAME", "").strip()
        or os.getenv("USER", "").strip()
        or "human_operator"
    )

    loaded_proj = load_project(project_root)
    artifact_store = ProjectArtifactStore(project_root, loaded_proj.manifest)

    exec_artifact_id = execution_data.get("artifact_id", execution_file.stem)
    clean_id = exec_artifact_id.replace("execution_", "").replace(".json", "")
    human_report_id = f"human_execution_review_{clean_id}"

    now_iso = datetime.now(timezone.utc).isoformat()
    changed_files = res.get("actual_changed_files", []) or res.get("changed_files", [])
    val_summary = res.get("validation_command_summary", [])
    source_rev = execution_data.get("source_revision", {})
    branch_name = res.get("branch_name", "")
    wt_id = res.get("worktree_identifier", "")
    proposal_path = res.get("source_proposal_path", "")

    # Human review report payload
    review_report_data = {
        "execution_id": exec_artifact_id,
        "proposal_id": proposal_path,
        "decision": decision,
        "reviewer": reviewer_identity,
        "timestamp": now_iso,
        "changed_files": changed_files,
        "test_summary": val_summary,
        "source_revision": source_rev,
        "worktree_branch": branch_name,
        "worktree_identifier": wt_id,
        "note": note.strip(),
        "review_type": "human_review",
        "status": new_status,
        "summary": f"Human review decision: {decision} ({new_status}) by {reviewer_identity}.",
    }

    report_file = artifact_store.write_report(human_report_id, review_report_data)

    # Update execution artifact status and human review metadata
    execution_data["status"] = new_status
    if "result" in execution_data and isinstance(execution_data["result"], dict):
        execution_data["result"]["final_status"] = new_status
        execution_data["result"]["human_review"] = {
            "decision": decision,
            "status": new_status,
            "reviewer": reviewer_identity,
            "timestamp": now_iso,
            "note": note.strip(),
            "report_path": str(report_file.relative_to(project_root)),
        }

    with open(execution_file, "w", encoding="utf-8") as f:
        json.dump(execution_data, f, ensure_ascii=False, indent=2)

    return {
        "decision": decision,
        "status": new_status,
        "reviewer": reviewer_identity,
        "execution_id": exec_artifact_id,
        "execution_file": execution_file,
        "report_file": report_file,
        "note": note.strip(),
    }
