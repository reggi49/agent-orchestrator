import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.config import get_settings
from agent_system.error_reporting import format_safe_error_report, sanitize_text
from agent_system.execution_package import (
    extract_approved_files_from_proposal,
    extract_validation_commands_from_proposal,
)
from agent_system.execution_reviewer import (
    locate_execution_artifact,
    review_execution,
)
from agent_system.executors.openhands_executor import OpenHandsExecutor
from agent_system.graph import (
    build_execution_graph,
    build_graph,
    build_implementation_graph,
    build_proposal_graph,
)
from agent_system.ceo_mode import CEORunner
from agent_system.integrator import (
    check_source_repository_state,
    integrate_execution,
)
from agent_system.orchestration_runner import OrchestrationRunner
from agent_system.project_initializer import initialize_project
from agent_system.project_inspector import format_inspection_report, inspect_project
from agent_system.project_loader import ProjectNotFoundError, load_project
from agent_system.worktree_manager import (
    generate_branch_name as wt_generate_branch_name,
    generate_worktree_name,
    get_default_worktree_root,
)



def cmd_init(args: argparse.Namespace) -> None:
    """Initialize portable project protocol."""
    if not args.project:
        print("ERROR: --project is required for init command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()

    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    print(f"Initializing project protocol at: {project_path}")
    print()

    result = initialize_project(
        project_root=project_path,
        apply=args.apply,
        force=args.force,
    )

    if not args.apply:
        print("=" * 70)
        print("DRY RUN - No files will be modified")
        print("=" * 70)
        print()
        print(f"Project Root: {result['project_root']}")
        print()
        print(f"Directories to create ({len(result['directories_to_create'])}):")
        for d in result["directories_to_create"]:
            print(f"  + {d}/")
        print()
        print(f"Files to create ({len(result['files_to_create'])}):")
        for f in result["files_to_create"]:
            print(f"  + {f}")
        print()

        if result["existing_files"]:
            print(f"Existing files that would be preserved ({len(result['existing_files'])}):")
            for f in result["existing_files"]:
                print(f"  = {f}")
            print()

        if result["warnings"]:
            print("Warnings:")
            for w in result["warnings"]:
                print(f"  ⚠ {w}")
            print()

        print()
        print(result["gitignore_recommendation"])
        print()
        print("To apply these changes, run:")
        print(f"  python main.py init --project {args.project} --apply")
    else:
        print("=" * 70)
        print("INITIALIZATION APPLIED")
        print("=" * 70)
        print()
        print(f"Project Root: {result['project_root']}")
        print()
        if result["created_directories"]:
            print(f"Created directories ({len(result['created_directories'])}):")
            for d in result["created_directories"]:
                print(f"  ✓ {d}/")
            print()
        if result["created_files"]:
            print(f"Created files ({len(result['created_files'])}):")
            for f in result["created_files"]:
                print(f"  ✓ {f}")
            print()
        if result["backed_up_files"]:
            print(f"Backed up files ({len(result['backed_up_files'])}):")
            for b in result["backed_up_files"]:
                print(f"  ↺ {b}")
            print()
        if result["skipped_files"]:
            print(f"Skipped existing files ({len(result['skipped_files'])}):")
            for s in result["skipped_files"]:
                print(f"  = {s}")
            print()
        print()
        print(result["gitignore_recommendation"])
        print()
        print("Project protocol initialized successfully.")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect project without modifying files or calling LLM."""
    if not args.project:
        print("ERROR: --project is required for inspect command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()

    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    try:
        report = inspect_project(project_path)
        print(format_inspection_report(report))
    except ProjectNotFoundError as e:
        print(f"ERROR: {e}")
        print()
        print("Run 'python main.py init --project <path> --apply' to initialize the project.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Inspection failed: {e}")
        sys.exit(1)


def cmd_plan(args: argparse.Namespace) -> None:
    """Run planning workflow with LLM."""
    if not args.project:
        print("ERROR: --project is required for plan command.")
        sys.exit(1)

    if not args.request:
        print("ERROR: --request is required for plan command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()

    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    try:
        validated_project = load_project(project_path)
    except ProjectNotFoundError as e:
        print(f"ERROR: {e}")
        print()
        print("Run 'python main.py init --project <path> --apply' to initialize the project.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load project: {e}")
        sys.exit(1)

    run_id = str(uuid.uuid4())

    initial_state = {
        "run_id": run_id,
        "user_request": args.request,
        "project_root": str(validated_project.root),
        "current_stage": "project_reader",
        "status": "running",
        "errors": [],
        "replan_count": 0,
        "max_replans": 2,
        "rejected_tasks": [],
    }

    print("=" * 70)
    print("AGENT ORCHESTRATOR - PLAN MODE")
    print("=" * 70)
    print(f"Run ID       : {run_id}")
    print(f"Project ID   : {validated_project.manifest.project.id}")
    print(f"Project Name : {validated_project.manifest.project.name}")
    print(f"Project Root : {validated_project.root}")
    print(f"Policy Mode  : {validated_project.policy.mode}")
    print(f"Request      : {args.request}")
    print()

    graph = build_graph()
    settings = get_settings()

    try:
        final_state = graph.invoke(
            initial_state,
            config={
                "recursion_limit": 10,
            },
        )
    except Exception as e:
        safe_report = format_safe_error_report(
            e,
            api_key=settings.llm_api_key,
            context="Workflow",
        )
        print(safe_report)

        err_type = type(e).__name__
        err_msg = str(e)
        if "AuthenticationError" in err_type or "401" in err_msg:
            print(
                "Petunjuk: Autentikasi LLM gagal. "
                "Periksa LLM_API_KEY dan pastikan key berasal dari provider yang sama dengan LLM_BASE_URL."
            )
        elif "APIConnectionError" in err_type or "Connection refused" in err_msg:
            print(
                "Petunjuk: Tidak dapat terhubung ke LLM gateway. "
                "Pastikan 9Router berjalan dan LLM_BASE_URL benar."
            )
        elif "NotFoundError" in err_type or "404" in err_msg:
            print(
                "Petunjuk: Model tidak ditemukan. "
                "Periksa LLM_MODEL terhadap daftar model dari 9Router."
            )
        elif "RateLimitError" in err_type or "429" in err_msg:
            print(
                "Petunjuk: Gateway atau provider sedang mencapai batas penggunaan."
            )
        elif "Timeout" in err_type or "timed out" in err_msg:
            print(
                "Petunjuk: Request ke LLM gateway timeout."
            )

        # Mark failed planning run as failed artifact
        try:
            artifact_store = ProjectArtifactStore(validated_project.root, validated_project.manifest)
            plan_id = f"plan_{run_id[:8]}"
            report_id = f"report_{run_id[:8]}"
            plan_data = {
                "initial_candidate": "",
                "status": "failed",
                "error_type": err_type,
                "error_message": sanitize_text(err_msg, api_key=settings.llm_api_key),
            }
            artifact_store.write_plan(plan_id, plan_data)
            report_data = {
                "request": args.request,
                "status": "failed",
                "error_type": err_type,
                "error_message": sanitize_text(err_msg, api_key=settings.llm_api_key),
            }
            artifact_store.write_report(report_id, report_data)
        except Exception:
            pass

        sys.exit(1)

    print()
    print("=" * 70)
    print("HASIL ANALISIS")
    print("=" * 70)

    analysis = final_state.get("analysis")

    if analysis:
        print(
            json.dumps(
                analysis,
                ensure_ascii=False,
                indent=2,
            )
        )

    print()
    print("=" * 70)
    print("STATUS")
    print("=" * 70)
    print(f"Status  : {final_state.get('status')}")
    print(f"Summary : {final_state.get('final_summary')}")
    print()

    # Save artifacts to project using ProjectArtifactStore
    try:
        artifact_store = ProjectArtifactStore(validated_project.root, validated_project.manifest)
        plan_id = f"plan_{run_id[:8]}"
        report_id = f"report_{run_id[:8]}"

        task_verification = final_state.get("task_verification", {})
        rejected_tasks = final_state.get("rejected_tasks", [])
        recommended_task = analysis.get("recommended_task", {}) if analysis else {}

        # Collect evidence paths
        evidence_results = final_state.get("repository_evidence", {}).get("results", [])
        evidence_paths = [
            r["path"] for r in evidence_results if isinstance(r, dict) and "path" in r
        ]

        # Initial candidate info
        if rejected_tasks:
            initial_candidate = rejected_tasks[0].get("title", "")
            initial_status = rejected_tasks[0].get("status", "")
        else:
            initial_candidate = recommended_task.get("title", "")
            initial_status = task_verification.get("status", "")

        missing_work = (
            task_verification.get("missing_work", [])
            if task_verification.get("status") == "partially_implemented"
            else []
        )

        plan_data = {
            "initial_candidate": initial_candidate,
            "initial_verification_status": initial_status,
            "rejected_tasks": rejected_tasks,
            "replan_count": final_state.get("replan_count", 0),
            "final_task": recommended_task.get("title", ""),
            "final_owner": recommended_task.get("owner", ""),
            "final_acceptance_criteria": recommended_task.get("acceptance_criteria", []),
            "evidence_paths": evidence_paths,
            "verification_status": task_verification.get("status", ""),
            "missing_work": missing_work,
            "status": final_state.get("status"),
        }
        plan_file = artifact_store.write_plan(plan_id, plan_data)
        print(f"Artifact Plan   : {plan_file.relative_to(validated_project.root)}")

        report_data = {
            "request": args.request,
            "initial_candidate": initial_candidate,
            "initial_verification_status": initial_status,
            "verification_status": task_verification.get("status", ""),
            "replan_count": final_state.get("replan_count", 0),
            "rejected_tasks_count": len(rejected_tasks),
            "verification_summary": task_verification.get("recommendation", ""),
            "summary": final_state.get("final_summary"),
            "status": final_state.get("status"),
            "evidence_paths": evidence_paths[:10],
        }
        report_file = artifact_store.write_report(report_id, report_data)
        print(f"Artifact Report : {report_file.relative_to(validated_project.root)}")
    except Exception as e:
        print(f"WARNING: Could not save artifacts: {e}")




def _is_valid_plan_payload(data: dict) -> bool:
    """Verify that plan artifact data is valid, not failed, and not partial."""
    if not isinstance(data, dict):
        return False
    if data.get("status") != "plan_ready":
        return False
    res = data.get("result", {})
    if not isinstance(res, dict):
        return False
    task = (
        res.get("final_task")
        or res.get("recommended_task")
        or (res.get("result", {}).get("recommended_task") if isinstance(res.get("result"), dict) else None)
        or (res.get("result", {}).get("final_task") if isinstance(res.get("result"), dict) else None)
        or data.get("final_task")
        or data.get("recommended_task")
    )
    return bool(task)


def locate_verified_plan(project_path: Path, plan_arg: str = "latest") -> tuple[Path, dict]:
    """Find and load verified plan with status plan_ready."""
    plans_dir = project_path / ".agent" / "artifacts" / "plans"
    if not plans_dir.is_dir():
        print("ERROR: Direktori rencana '.agent/artifacts/plans' tidak ditemukan.")
        print("Jalankan planning flow terlebih dahulu menggunakan 'python main.py plan'.")
        sys.exit(1)

    plan_files = sorted(plans_dir.glob("plan_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not plan_files:
        print("ERROR: Tidak ditemukan file plan di '.agent/artifacts/plans/'.")
        print("Jalankan planning flow terlebih dahulu menggunakan 'python main.py plan'.")
        sys.exit(1)

    if plan_arg == "latest":
        for pf in plan_files:
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
                if _is_valid_plan_payload(data):
                    return pf, data
            except Exception:
                continue

        print("ERROR: Tidak ditemukan verified plan dengan status 'plan_ready'.")
        print("Proposal implementasi memerlukan verified plan dengan status 'plan_ready'.")
        print("Pastikan tahap verifikasi task berhasil sebelum membuat proposal implementasi.")
        sys.exit(1)
    else:
        cand = project_path / plan_arg
        if not cand.is_file():
            cand = plans_dir / (plan_arg if plan_arg.endswith(".json") else f"{plan_arg}.json")
        if not cand.is_file():
            print(f"ERROR: Plan file '{plan_arg}' tidak ditemukan di '{plans_dir}'.")
            sys.exit(1)
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"ERROR: Plan file '{cand.name}' rusak atau bukan format JSON valid: {e}")
            sys.exit(1)
        if not isinstance(data, dict):
            print(f"ERROR: Plan file '{cand.name}' bukan objek JSON valid.")
            sys.exit(1)
        if not _is_valid_plan_payload(data):
            if data.get("status") != "plan_ready":
                print(f"ERROR: Plan '{cand.name}' memiliki status '{data.get('status')}', bukan 'plan_ready'.")
            else:
                print(f"ERROR: Plan '{cand.name}' tidak lengkap (parsial).")
            print("Proposal implementasi hanya boleh dibuat dari verified plan dengan status 'plan_ready'.")
            sys.exit(1)
        return cand, data

def cmd_propose(args: argparse.Namespace) -> None:
    """Run implementation proposal workflow from verified plan."""
    if not args.project:
        print("ERROR: --project is required for propose command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    try:
        validated_project = load_project(project_path)
    except ProjectNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load project: {e}")
        sys.exit(1)

    target_plan_file, target_plan_data = locate_verified_plan(project_path, getattr(args, "plan", "latest"))

    source_plan_rel = str(target_plan_file.relative_to(validated_project.root))
    res = target_plan_data.get("result", {})
    verified_task = res.get("recommended_task") or res.get("final_task") or {}
    if isinstance(verified_task, str):
        task_title = verified_task
        task_owner = res.get("final_owner", "backend")
        acceptance_criteria = res.get("final_acceptance_criteria", [])
    else:
        task_title = verified_task.get("title", "")
        task_owner = verified_task.get("owner", "backend")
        acceptance_criteria = verified_task.get("acceptance_criteria", [])

    run_id = str(uuid.uuid4())

    initial_state = {
        "run_id": run_id,
        "project_root": str(validated_project.root),
        "source_plan_path": source_plan_rel,
        "plan_data": target_plan_data,
        "task_verification": {
            "status": res.get("verification_status", "verified_pending"),
            "task_title": task_title,
            "evidence_for": target_plan_data.get("evidence_paths", []),
            "missing_work": res.get("missing_work", []),
        },
        "repository_evidence": {
            "results": [{"path": p} for p in target_plan_data.get("evidence_paths", [])],
        },
        "current_stage": "implementation_proposal",
        "status": "running",
        "errors": [],
        "proposal_revision_count": 0,
        "max_proposal_revisions": 2,
        "rejected_proposal_reasons": [],
        "proposal_hashes": [],
    }

    print("=" * 70)
    print("AGENT ORCHESTRATOR - IMPLEMENTATION PROPOSAL MODE")
    print("=" * 70)
    print(f"Run ID           : {run_id}")
    print(f"Project ID       : {validated_project.manifest.project.id}")
    print(f"Verified Plan    : {source_plan_rel}")
    print(f"Task Title       : {task_title}")
    print(f"Task Owner       : {task_owner}")
    print(f"Max Revisions    : 2")
    print()

    graph = build_proposal_graph()
    settings = get_settings()

    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 10},
        )
    except Exception as e:
        safe_report = format_safe_error_report(
            e,
            api_key=settings.llm_api_key,
            context="Workflow propose",
        )
        print(safe_report)
        sys.exit(1)

    print()
    print("=" * 70)
    print("STATUS PROPOSAL REVIEW")
    print("=" * 70)
    review = final_state.get("proposal_review", {})
    print(f"Review Status    : {review.get('status')}")
    print(f"Supervisor Status: {final_state.get('status')}")
    print(f"Summary          : {final_state.get('final_summary')}")
    print()

    try:
        artifact_store = ProjectArtifactStore(validated_project.root, validated_project.manifest)
        prop_id = f"proposal_{run_id[:8]}"
        review_id = f"proposal_review_{run_id[:8]}"

        proposal_data = final_state.get("implementation_proposal", {})
        rev_meta = {
            "source_plan_path": source_plan_rel,
            "proposal": proposal_data,
            "review_status": review.get("status"),
            "proposal_revision_count": final_state.get("proposal_revision_count", 0),
            "status": final_state.get("status"),
            "evidence_paths": proposal_data.get("evidence_paths", []),
        }
        prop_file = artifact_store.write_proposal(prop_id, rev_meta)
        print(f"Artifact Proposal: {prop_file.relative_to(validated_project.root)}")

        review_meta = {
            "source_plan_path": source_plan_rel,
            "proposal_id": proposal_data.get("proposal_id"),
            "review": review,
            "status": final_state.get("status"),
            "evidence_paths": proposal_data.get("evidence_paths", []),
        }
        review_file = artifact_store.write_report(review_id, review_meta)
        print(f"Artifact Review  : {review_file.relative_to(validated_project.root)}")
    except Exception as e:
        print(f"WARNING: Could not save proposal artifacts: {e}")

def locate_approved_proposal(project_path: Path, proposal_arg: str = "latest") -> tuple[Path, dict]:
    """Find and load approved proposal with status approved_for_implementation."""
    proposals_dir = project_path / ".agent" / "artifacts" / "proposals"
    if not proposals_dir.is_dir():
        print("ERROR: Direktori proposal '.agent/artifacts/proposals' tidak ditemukan.")
        sys.exit(1)

    prop_files = sorted(proposals_dir.glob("proposal_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not prop_files:
        print("ERROR: Tidak ditemukan file proposal di '.agent/artifacts/proposals/'.")
        sys.exit(1)

    if proposal_arg == "latest" or not proposal_arg:
        for pf in prop_files:
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
                if data.get("status") == "approved_for_implementation":
                    return pf, data
            except Exception:
                continue

        print("ERROR: Tidak ditemukan proposal dengan status 'approved_for_implementation'.")
        sys.exit(1)
    else:
        cand = project_path / proposal_arg
        if not cand.is_file():
            cand = proposals_dir / (proposal_arg if proposal_arg.endswith(".json") else f"{proposal_arg}.json")
        if not cand.is_file():
            print(f"ERROR: Proposal file '{proposal_arg}' tidak ditemukan.")
            sys.exit(1)
        data = json.loads(cand.read_text(encoding="utf-8"))
        if data.get("status") != "approved_for_implementation":
            print(
                f"ERROR: Proposal '{cand.name}' memiliki status '{data.get('status')}', bukan 'approved_for_implementation'."
            )
            sys.exit(1)
        return cand, data


def locate_proposal_review(project_path: Path, proposal_id: str) -> tuple[Path | None, dict | None]:
    """Locate review report for proposal in .agent/artifacts/reports."""
    reports_dir = project_path / ".agent" / "artifacts" / "reports"
    if not reports_dir.is_dir():
        return None, None
    clean_id = proposal_id.replace("prop_", "").replace("proposal_", "")
    cand = reports_dir / f"proposal_review_{clean_id}.json"
    if cand.is_file():
        try:
            return cand, json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            return None, None
    return None, None

def cmd_implement(args: argparse.Namespace) -> None:
    """Run controlled implementation workflow."""
    if not args.project:
        print("ERROR: --project is required for implement command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    try:
        validated_project = load_project(project_path)
    except ProjectNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load project: {e}")
        sys.exit(1)

    # 1. Load proposal
    prop_arg = getattr(args, "proposal", "latest") or "latest"
    target_prop_file, target_prop_data = locate_approved_proposal(project_path, prop_arg)
    source_proposal_rel = str(target_prop_file.relative_to(validated_project.root))

    # 2. Load proposal review
    prop_inner = target_prop_data.get("result", {}).get("proposal", {})
    prop_id = prop_inner.get("proposal_id") or target_prop_data.get("artifact_id", "")
    _, target_review_data = locate_proposal_review(project_path, prop_id)
    review_inner = (
        target_review_data.get("result", {}).get("review", {})
        if target_review_data
        else target_prop_data.get("result", {}).get("review", {})
    )

    # 3. Load verified plan
    plan_rel = target_prop_data.get("result", {}).get("source_plan_path", "latest")
    target_plan_file, target_plan_data = locate_verified_plan(project_path, plan_rel)
    source_plan_rel = str(target_plan_file.relative_to(validated_project.root))

    run_id = str(uuid.uuid4())
    is_apply = bool(getattr(args, "apply", False))
    dry_run = not is_apply
    isolation = getattr(args, "isolation", "git-worktree")
    worktree_root = getattr(args, "worktree_root", None)

    task_title = prop_inner.get("task_title", "Implementation Task")

    initial_state = {
        "run_id": run_id,
        "project_root": str(validated_project.root),
        "source_plan_path": source_plan_rel,
        "source_proposal_path": source_proposal_rel,
        "plan_data": target_plan_data,
        "proposal_data": target_prop_data,
        "proposal_review": review_inner,
        "current_stage": "implementation",
        "status": "running",
        "errors": [],
        "dry_run": dry_run,
        "apply": is_apply,
        "isolation": isolation,
        "worktree_root": worktree_root,
        "implementation_repair_count": 0,
        "max_implementation_repairs": 2,
        "implementation_failures": [],
    }

    print("=" * 70)
    print(f"AGENT ORCHESTRATOR - CONTROLLED IMPLEMENTATION ({'APPLY' if is_apply else 'DRY RUN'})")
    print("=" * 70)
    print(f"Run ID           : {run_id}")
    print(f"Project ID       : {validated_project.manifest.project.id}")
    print(f"Proposal         : {source_proposal_rel}")
    print(f"Verified Plan    : {source_plan_rel}")
    print(f"Task Title       : {task_title}")
    print(f"Mode             : {'APPLY (changes will be written)' if is_apply else 'DRY RUN (no changes)'}")
    print()

    graph = build_implementation_graph()
    settings = get_settings()

    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 15},
        )
    except Exception as e:
        safe_report = format_safe_error_report(
            e,
            api_key=settings.llm_api_key,
            context="Workflow implement",
        )
        print(safe_report)
        sys.exit(1)

    print()
    print("=" * 70)
    print("HASIL IMPLEMENTASI")
    print("=" * 70)

    preflight = final_state.get("preflight_result", {})
    impl_res = final_state.get("implementation_result", {})
    branch_name = final_state.get("branch_name", "N/A")
    final_status = final_state.get("status")

    files_to_mod = prop_inner.get("files_to_modify", [])
    files_to_crt = prop_inner.get("files_to_create", [])
    tests_to_mod = prop_inner.get("tests_to_modify", [])
    tests_to_crt = prop_inner.get("tests_to_create", [])
    target_files = sorted(list(dict.fromkeys([
        f["path"] if isinstance(f, dict) else f.path
        for f in (files_to_mod + files_to_crt + tests_to_mod + tests_to_crt)
    ])))
    val_cmds = prop_inner.get("validation_commands", [])

    print(f"Branch Direncanakan : {branch_name}")
    print(f"File Target ({len(target_files)}) : {', '.join(target_files) if target_files else '-'}")
    print(f"Validation Commands : {len(val_cmds)} command")
    for cmd in val_cmds:
        exe = cmd.get("executable") if isinstance(cmd, dict) else cmd.executable
        args_list = cmd.get("arguments", []) if isinstance(cmd, dict) else cmd.arguments
        cwd = cmd.get("working_directory", "") if isinstance(cmd, dict) else cmd.working_directory
        print(f"  - {exe} {' '.join(args_list)} (cwd: {cwd})")

    outside_scope = (
        impl_res.get("files_requested_outside_scope", [])
        or preflight.get("files_requested_outside_scope", [])
    )
    if outside_scope:
        print()
        print(f"File Tambahan Diperlukan (Di Luar Scope) ({len(outside_scope)}):")
        for f in outside_scope:
            print(f"  ⚠ {f}")

    print()
    print(f"Status Preflight    : {preflight.get('status', 'unknown')}")
    print(f"Status Akhir        : {final_status}")
    print(f"Ringkasan           : {final_state.get('final_summary')}")
    print()

    # Save artifacts
    try:
        artifact_store = ProjectArtifactStore(validated_project.root, validated_project.manifest)
        short_id = run_id[:8]
        wt_descriptor = final_state.get("worktree_descriptor") or f"{validated_project.root.name}-{short_id}"
        approved_files_list = final_state.get("approved_files", target_files)
        cmd_results = final_state.get("command_results", [])
        test_exit_code = cmd_results[0].get("exit_code") if cmd_results else 0
        diff_rev = final_state.get("diff_review", {})
        diff_check_status = "passed" if diff_rev.get("status") == "approved" else ("passed" if dry_run else "failed")
        diff_review_status = diff_rev.get("status") or ("dry_run" if dry_run else "unknown")

        # Review report
        review_id = f"implementation_review_{short_id}"
        report_data = {
            "executor_type": "LLMImplementationExecutor",
            "project_id": validated_project.manifest.project.id,
            "source_plan_path": source_plan_rel,
            "source_proposal_path": source_proposal_rel,
            "source_revision": preflight.get("source_revision", {}),
            "branch_name": branch_name,
            "worktree_descriptor": wt_descriptor,
            "approved_files": approved_files_list,
            "changed_files": impl_res.get("files_changed", []),
            "target_files": target_files,
            "files_requested_outside_scope": outside_scope,
            "validation_commands": val_cmds,
            "preflight_status": preflight.get("status"),
            "is_dry_run": dry_run,
            "test_command_summary": [
                {
                    "executable": c.get("executable"),
                    "arguments": c.get("arguments"),
                    "working_directory": c.get("working_directory"),
                    "exit_code": c.get("exit_code"),
                    "status": c.get("status"),
                }
                for c in cmd_results
            ],
            "test_exit_code": test_exit_code,
            "diff_check_status": diff_check_status,
            "diff_review_status": diff_review_status,
            "repair_count": final_state.get("implementation_repair_count", 0),
            "status": final_status,
            "summary": final_state.get("final_summary"),
        }
        report_file = artifact_store.write_report(review_id, report_data)
        print(f"Artifact Report     : {report_file.relative_to(validated_project.root)}")

        # Implementation artifact
        if is_apply and final_status == "implementation_ready":
            impl_id = f"implementation_{short_id}"
            impl_artifact_data = {
                "executor_type": "LLMImplementationExecutor",
                "project_id": validated_project.manifest.project.id,
                "source_plan_path": source_plan_rel,
                "source_proposal_path": source_proposal_rel,
                "source_revision": preflight.get("source_revision", {}),
                "branch_name": branch_name,
                "worktree_descriptor": wt_descriptor,
                "approved_files": approved_files_list,
                "changed_files": impl_res.get("files_changed", []),
                "file_hashes_before": final_state.get("file_hashes_before", {}),
                "file_hashes_after": final_state.get("file_hashes_after", {}),
                "test_command_summary": [
                    {
                        "executable": c.get("executable"),
                        "arguments": c.get("arguments"),
                        "working_directory": c.get("working_directory"),
                        "exit_code": c.get("exit_code"),
                        "status": c.get("status"),
                    }
                    for c in cmd_results
                ],
                "test_exit_code": test_exit_code,
                "diff_check_status": diff_check_status,
                "diff_review_status": diff_review_status,
                "repair_count": final_state.get("implementation_repair_count", 0),
                "acceptance_criteria_status": diff_rev.get("acceptance_criteria_status", []),
                "final_status": final_status,
                "status": final_status,
            }
            impl_file = artifact_store.write_implementation(impl_id, impl_artifact_data)
            print(f"Artifact Impl       : {impl_file.relative_to(validated_project.root)}")
    except Exception as e:
        print(f"WARNING: Gagal menyimpan artifact implementasi: {e}")


def cmd_execute(args: argparse.Namespace) -> None:
    """Run coding execution with OpenHands Software Agent SDK (default: dry-run)."""
    if not args.project:
        print("ERROR: --project is required for execute command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    try:
        validated_project = load_project(project_path)
    except ProjectNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load project: {e}")
        sys.exit(1)

    # 1. Validate OpenHands configuration
    executor = OpenHandsExecutor()
    config_validation = executor.validate_configuration()
    if not config_validation.get("valid"):
        print(f"ERROR: Konfigurasi OpenHands tidak valid: {config_validation.get('error')}")
        sys.exit(1)

    # 2. Locate and load approved proposal
    prop_arg = getattr(args, "proposal", "latest") or "latest"
    target_prop_file, target_prop_data = locate_approved_proposal(project_path, prop_arg)
    source_proposal_rel = str(target_prop_file.relative_to(validated_project.root))

    # 3. Locate proposal review
    prop_inner = target_prop_data.get("result", {}).get("proposal", {})
    prop_id = prop_inner.get("proposal_id") or target_prop_data.get("artifact_id", "")
    _, target_review_data = locate_proposal_review(project_path, prop_id)
    review_inner = (
        target_review_data.get("result", {}).get("review", {})
        if target_review_data
        else target_prop_data.get("result", {}).get("review", {})
    )

    # 4. Locate verified plan
    plan_rel = target_prop_data.get("result", {}).get("source_plan_path", "latest")
    target_plan_file, target_plan_data = locate_verified_plan(project_path, plan_rel)
    source_plan_rel = str(target_plan_file.relative_to(validated_project.root))

    # 5. Validate worktree planning
    is_apply = bool(getattr(args, "apply", False))
    dry_run = not is_apply

    run_id = str(uuid.uuid4())
    short_id = run_id[:8].lower()
    task_title = prop_inner.get("task_title", "Execution Task")
    task_words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in task_title.split()]
    task_tail = "-".join([w for w in task_words if w][-2:]) if task_words else "task"
    branch_name = f"agent/openhands-{task_tail}-{short_id}"
    wt_name = f"{validated_project.root.name}-openhands-{short_id}"
    wt_root = get_default_worktree_root(validated_project.root, custom_root=args.worktree_root)
    planned_wt_path = wt_root / wt_name

    # 6. Extract approved files & validation commands
    approved_files = extract_approved_files_from_proposal(prop_inner, review_inner)
    validation_commands = extract_validation_commands_from_proposal(prop_inner)

    # 7. Setup initial state for LangGraph execution graph
    initial_state = {
        "run_id": run_id,
        "project_root": str(validated_project.root),
        "main_project_root": str(validated_project.root),
        "source_plan_path": source_plan_rel,
        "source_proposal_path": source_proposal_rel,
        "plan_data": target_plan_data,
        "proposal_data": target_prop_data,
        "proposal_review": review_inner,
        "current_stage": "execution",
        "status": "running",
        "errors": [],
        "dry_run": dry_run,
        "apply": is_apply,
        "executor": "openhands",
        "isolation": args.isolation,
        "worktree_root": args.worktree_root,
        "approved_files": approved_files,
        "branch_name": branch_name,
        "worktree_descriptor": wt_name,
        "executor_retry_count": 0,
        "max_executor_retries": 1,
    }

    mode_label = "APPLY" if is_apply else "DRY RUN"
    print("=" * 70)
    print(f"AGENT ORCHESTRATOR - OPENHANDS EXECUTION ({mode_label})")
    print("=" * 70)
    print(f"Run ID              : {run_id}")
    print(f"Project ID          : {validated_project.manifest.project.id}")
    print(f"Executor            : openhands (OpenHands Software Agent SDK)")
    print(f"SDK Version         : {config_validation.get('sdk_version', 'unknown')}")
    print(f"Workspace Type      : LocalWorkspace (No Docker)")
    print(f"Docker Used         : Tidak (Docker dilarang)")
    print(f"Proposal Artifact   : {source_proposal_rel}")
    print(f"Task Title          : {task_title}")
    print(f"Isolation Mode      : {args.isolation}")
    print(f"Planned Branch      : {branch_name}")
    print(f"Planned Worktree    : {wt_name} (di bawah {wt_root})")
    print(f"Approved Files ({len(approved_files)})  :")
    for af in approved_files:
        print(f"  - {af}")
    print(f"Validation Commands ({len(validation_commands)}):")
    for vc in validation_commands:
        exe = vc.get("executable", "")
        args_str = " ".join(vc.get("arguments", []))
        cwd = vc.get("working_directory", ".")
        print(f"  - {exe} {args_str} (cwd: {cwd})")
    print()

    # Invoke execution graph
    graph = build_execution_graph()
    settings = get_settings()

    start_exec_time = time.time()
    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 15},
        )
    except Exception as e:
        safe_report = format_safe_error_report(
            e,
            api_key=settings.llm_api_key,
            context="Execution workflow",
        )
        print(safe_report)
        sys.exit(1)
    elapsed_total = round(time.time() - start_exec_time, 1)

    print("=" * 70)
    print(f"HASIL EXECUTION ({mode_label})")
    print("=" * 70)
    preflight = final_state.get("preflight_result", {})
    coding_res = final_state.get("coding_execution_result", {})
    diff_rev = final_state.get("diff_review", {})
    final_status = final_state.get("status", "unknown")

    if dry_run:
        print(f"Preflight Status    : {preflight.get('status', 'unknown')}")
        print(f"Worktree Dibuat     : Tidak (Dry-run: pembuatan worktree dilewati)")
        print(f"LLM Dipanggil       : Tidak (Dry-run: panggilan LLM dilewati)")
        print(f"Source Diubah       : Tidak (Dry-run: penulisan source code dilewati)")
        print(f"Command Dijalankan  : Tidak (Dry-run: eksekusi command dilewati)")
        print(f"Status Akhir        : {final_status}")
        print(f"Ringkasan           : {final_state.get('final_summary')}")
        print()
    else:
        wt_path_actual = final_state.get("worktree_path", str(planned_wt_path))
        changed_files_actual = diff_rev.get("changed_files", coding_res.get("changed_files", []))
        cmd_results = coding_res.get("validation_results", [])
        test_exit_code = cmd_results[0].get("exit_code") if cmd_results else 0

        print(f"Preflight Status    : {preflight.get('status', 'unknown')}")
        print(f"Worktree Dibuat     : Ya ({wt_name})")
        print(f"Lokasi Worktree     : {wt_path_actual}")
        print(f"Branch              : {branch_name}")
        print(f"OpenHands Iteration : {coding_res.get('iterations', 0)}")
        print(f"Waktu Eksekusi      : {elapsed_total}s")
        print(f"File Berubah ({len(changed_files_actual)})   : {', '.join(changed_files_actual) if changed_files_actual else '(none)'}")
        print(f"Test Status         : {'passed' if test_exit_code == 0 and cmd_results else 'failed'} (exit code {test_exit_code})")
        print(f"Diff Review Status  : {diff_rev.get('status', 'unknown')}")
        print(f"Status Akhir        : {final_status}")
        print(f"Ringkasan           : {final_state.get('final_summary')}")
        print()

        # Save artifacts to project target
        try:
            store = ProjectArtifactStore(validated_project.root, validated_project.manifest)
            exec_id = f"execution_{short_id}"
            review_id = f"execution_review_{short_id}"

            val_summary = [
                {
                    "executable": c.get("executable"),
                    "arguments": c.get("arguments"),
                    "working_directory": c.get("working_directory"),
                    "exit_code": c.get("exit_code"),
                    "status": c.get("status"),
                }
                for c in cmd_results
            ]

            diff_check_passed = diff_rev.get("status") == "approved"

            exec_data = {
                "executor": "openhands",
                "openhands_sdk_version": config_validation.get("sdk_version", "1.53.0"),
                "source_proposal_path": source_proposal_rel,
                "source_revision": preflight.get("source_revision", {}),
                "branch_name": branch_name,
                "worktree_identifier": wt_name,
                "approved_files": approved_files,
                "actual_changed_files": changed_files_actual,
                "iterations": coding_res.get("iterations", 0),
                "elapsed_time": elapsed_total,
                "validation_command_summary": val_summary,
                "validation_exit_code": test_exit_code,
                "diff_check_status": "passed" if diff_check_passed else "failed",
                "independent_review_status": diff_rev.get("status", "unknown"),
                "scope_violations": coding_res.get("scope_violations", []),
                "requested_additional_files": coding_res.get("requested_additional_files", []),
                "final_status": final_status,
                "status": final_status,
            }
            exec_file = store.write_execution(exec_id, exec_data)
            print(f"Artifact Execution  : {exec_file.relative_to(validated_project.root)}")

            review_data = {
                "executor": "openhands",
                "openhands_sdk_version": config_validation.get("sdk_version", "1.53.0"),
                "source_proposal_path": source_proposal_rel,
                "source_revision": preflight.get("source_revision", {}),
                "branch_name": branch_name,
                "worktree_identifier": wt_name,
                "approved_files": approved_files,
                "actual_changed_files": changed_files_actual,
                "iterations": coding_res.get("iterations", 0),
                "elapsed_time": elapsed_total,
                "validation_command_summary": val_summary,
                "validation_exit_code": test_exit_code,
                "diff_check_status": "passed" if diff_check_passed else "failed",
                "independent_review_status": diff_rev.get("status", "unknown"),
                "scope_violations": coding_res.get("scope_violations", []),
                "requested_additional_files": coding_res.get("requested_additional_files", []),
                "final_status": final_status,
                "status": final_status,
                "summary": final_state.get("final_summary"),
            }
            review_file = store.write_report(review_id, review_data)
            print(f"Artifact Review     : {review_file.relative_to(validated_project.root)}")

        except Exception as e:
            print(f"WARNING: Gagal menyimpan artifact eksekusi: {e}")

        # Human Review summary
        print()
        print("HUMAN REVIEW:")
        print(f"- Branch '{branch_name}' dan worktree '{wt_name}' dipertahankan untuk inspeksi manusia.")
        print(f"- Lokasi worktree: {wt_path_actual}")
        print("- Tidak ada commit, push, atau merge yang dilakukan.")
        print()


def cmd_review_execution(args: argparse.Namespace) -> None:
    """Execute formal human review on an execution artifact."""
    if not args.project:
        print("ERROR: --project is required.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    try:
        validated_project = load_project(project_path)
    except Exception as e:
        print(f"ERROR: Failed to load project: {e}")
        sys.exit(1)

    decisions = [
        ("approve", bool(args.approve)),
        ("request-changes", bool(args.request_changes)),
        ("reject", bool(args.reject)),
    ]
    chosen = [d[0] for d in decisions if d[1]]
    if len(chosen) != 1:
        print("ERROR: Tepat satu keputusan review wajib diberikan (--approve, --request-changes, atau --reject).")
        sys.exit(1)

    decision = chosen[0]

    try:
        execution_file, execution_data = locate_execution_artifact(validated_project.root, args.execution)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 70)
    print("AGENT ORCHESTRATOR - HUMAN EXECUTION REVIEW")
    print("=" * 70)
    print(f"Project ID          : {validated_project.manifest.project.id}")
    print(f"Execution Artifact  : {execution_file.relative_to(validated_project.root)}")
    print(f"Execution Status    : {execution_data.get('status')}")
    print(f"Decision Input      : {decision}")
    if args.note:
        print(f"Review Note         : {args.note}")
    print()

    try:
        review_result = review_execution(
            project_root=validated_project.root,
            execution_file=execution_file,
            execution_data=execution_data,
            decision=decision,
            reviewer=args.reviewer,
            note=args.note,
        )
    except ValueError as ve:
        print(f"ERROR: Human review ditolak: {ve}")
        sys.exit(1)
    except Exception as ex:
        print(f"ERROR: Terjadi kesalahan saat memproses review: {ex}")
        sys.exit(1)

    print("=" * 70)
    print("HASIL HUMAN REVIEW")
    print("=" * 70)
    print(f"Decision            : {review_result['decision']}")
    print(f"Status Baru         : {review_result['status']}")
    print(f"Reviewer Identity   : {review_result['reviewer']}")
    print(f"Execution Artifact  : {review_result['execution_file'].relative_to(validated_project.root)}")
    print(f"Report Artifact     : {review_result['report_file'].relative_to(validated_project.root)}")
    if review_result["status"] == "approved_for_integration":
        print()
        print("Langkah Berikutnya:")
        print(f"  Jalankan integrasi dry-run: python main.py integrate --project {validated_project.root} --execution {execution_file.name} --strategy cherry-pick --dry-run")
        print(f"  Jalankan integrasi apply  : python main.py integrate --project {validated_project.root} --execution {execution_file.name} --strategy cherry-pick --apply")
    print()


def cmd_integrate(args: argparse.Namespace) -> None:
    """Integrate approved execution changes into target branch."""
    if not args.project:
        print("ERROR: --project is required.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    try:
        validated_project = load_project(project_path)
    except Exception as e:
        print(f"ERROR: Failed to load project: {e}")
        sys.exit(1)

    if args.dry_run and args.apply:
        print("ERROR: Pilih salah satu antara --dry-run atau --apply.")
        sys.exit(1)

    # Default is dry-run
    is_apply = bool(args.apply)
    mode_label = "APPLY" if is_apply else "DRY RUN"

    try:
        execution_file, execution_data = locate_execution_artifact(validated_project.root, args.execution)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    strategy = getattr(args, "strategy", "cherry-pick") or "cherry-pick"

    print("=" * 70)
    print(f"AGENT ORCHESTRATOR - INTEGRATION ({mode_label})")
    print("=" * 70)
    print(f"Project ID          : {validated_project.manifest.project.id}")
    print(f"Execution Artifact  : {execution_file.relative_to(validated_project.root)}")
    print(f"Execution Status    : {execution_data.get('status')}")
    print(f"Strategy            : {strategy}")
    print()

    try:
        integration_result = integrate_execution(
            project_root=validated_project.root,
            execution_file=execution_file,
            execution_data=execution_data,
            strategy=strategy,
            apply=is_apply,
        )
    except ValueError as ve:
        print(f"ERROR: Integrasi ditolak: {ve}")
        sys.exit(1)
    except RuntimeError as re:
        print(f"ERROR: Integrasi gagal: {re}")
        sys.exit(1)
    except Exception as ex:
        print(f"ERROR: Terjadi kesalahan saat integrasi: {ex}")
        sys.exit(1)

    print("=" * 70)
    print(f"HASIL INTEGRASI ({mode_label})")
    print("=" * 70)
    if not is_apply:
        print(f"Source Project      : {integration_result['source_project']}")
        print(f"Current Branch      : {integration_result['current_branch']}")
        print(f"Agent Branch        : {integration_result['agent_branch']}")
        print(f"Worktree Path       : {integration_result['worktree_path']}")
        print(f"File Berubah ({len(integration_result['changed_files'])})   : {', '.join(integration_result['changed_files'])}")
        print(f"Git Clean           : {'Ya' if integration_result['is_git_clean'] else 'Tidak (Dirty)'}")
        if integration_result["has_runtime_artifacts"]:
            print(f"Runtime Artifacts   : Ditemukan {len(integration_result['runtime_artifacts'])} file pada .agent/ (bukan source changes)")
        if integration_result.get("gitignore_recommendation"):
            print(f"Rekomendasi .gitignore: {integration_result['gitignore_recommendation']}")
        print(f"Commit Message Rencana: {integration_result['planned_commit_message']}")
        print(f"Kelayakan Integrasi : {integration_result['integration_eligibility']}")
        print(f"Status              : {integration_result['status']}")
        print()
        print("Langkah Berikutnya:")
        print(f"  Jalankan integrasi apply: python main.py integrate --project {validated_project.root} --execution {execution_file.name} --strategy cherry-pick --apply")
        print()
    else:
        print(f"Status              : {integration_result['status']}")
        print(f"Target Branch       : {integration_result['target_branch']}")
        print(f"Source Agent Branch : {integration_result.get('source_agent_branch', integration_result['agent_branch'])}")
        print(f"Source Commit       : {integration_result.get('source_commit')}")
        print(f"Integrated Commit   : {integration_result.get('integrated_commit', integration_result['commit_hash'])}")
        print(f"Post-Validation     : {integration_result.get('post_integration_validation', {}).get('status', 'passed')}")
        print(f"File Berubah ({len(integration_result['changed_files'])})   : {', '.join(integration_result['changed_files'])}")
        print(f"Staged Diff Stat    :\n{integration_result['staged_diff_stat']}")
        print(f"Artifact Execution  : {integration_result['integration_artifact_file'].relative_to(validated_project.root)}")
        print(f"Artifact Report     : {integration_result['integration_report_file'].relative_to(validated_project.root)}")
        print()
        print("=" * 70)
        print("REKOMENDASI PEMBERSIHAN WORKTREE (MANUAL)")
        print("=" * 70)
        print("Worktree dan branch dipertahankan untuk referensi dan audit:")
        print(f"- Lokasi worktree : {integration_result['worktree_path']}")
        print(f"- Branch agent    : {integration_result.get('source_agent_branch', integration_result['agent_branch'])}")
        print()
        print("Jika Anda ingin membersihkan worktree dan branch secara manual:")
        print(f"  git worktree remove {integration_result['worktree_path']}")
        print(f"  git branch -d {integration_result.get('source_agent_branch', integration_result['agent_branch'])}")
        print()


def _print_run_summary(checkpoint: dict[str, Any], verbose: bool = False) -> None:
    """Print clean concise run summary according to orchestrator specification."""
    run_id = checkpoint.get("run_id", "unknown")
    status = checkpoint.get("overall_status", "unknown")
    risk = checkpoint.get("risk_level", "unknown")
    changed_files = checkpoint.get("changed_files", [])
    test_status = checkpoint.get("test_status", "none")
    commit = checkpoint.get("commit_hash") or "none"
    duration = checkpoint.get("duration_seconds", 0.0)

    print()
    print("=" * 60)
    print("ORCHESTRATION RUN SUMMARY")
    print("=" * 60)
    print(f"Run ID        : {run_id}")
    print(f"Status        : {status}")
    print(f"Risk          : {risk}")
    print(f"Changed Files : {len(changed_files)}")
    print(f"Tests         : {test_status}")
    if commit and commit != "none":
        print(f"Commit        : {commit[:8]}")
    if status == "awaiting_human_approval":
        print(f"Next Action   : {checkpoint.get('next_action')}")
    elif status == "dry_run_ready":
        print(f"Next Action   : {checkpoint.get('next_action')}")
    elif status in ("failed", "interrupted"):
        if checkpoint.get("error_message"):
            print(f"Error         : {checkpoint.get('error_message')}")
        if checkpoint.get("next_action"):
            print(f"Next Action   : {checkpoint.get('next_action')}")
    print(f"Duration      : {duration}s")
    print("=" * 60)
    print()


def cmd_run(args: argparse.Namespace) -> None:
    """Run end-to-end multi-agent orchestration lifecycle."""
    if not args.project:
        print("ERROR: --project is required for run command.")
        sys.exit(1)

    if not args.request:
        print("ERROR: --request is required for run command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    is_apply = bool(args.apply)
    runner = OrchestrationRunner(project_path, verbose=bool(args.verbose))
    res = runner.run(
        request=args.request,
        apply=is_apply,
        approval_policy=getattr(args, "approval_policy", "safe-auto") or "safe-auto",
        isolation=getattr(args, "isolation", "git-worktree") or "git-worktree",
        executor=getattr(args, "executor", "openhands") or "openhands",
    )

    _print_run_summary(res, verbose=bool(args.verbose))


def cmd_status(args: argparse.Namespace) -> None:
    """Inspect status of an orchestration run."""
    if not args.project:
        print("ERROR: --project is required for status command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    runner = OrchestrationRunner(project_path, verbose=bool(args.verbose))
    try:
        checkpoint = runner.get_status(getattr(args, "run", "latest") or "latest")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    run_id = checkpoint.get("run_id", "unknown")
    stage = checkpoint.get("current_stage", "unknown")
    status = checkpoint.get("overall_status", "unknown")
    risk = checkpoint.get("risk_level", "unknown")
    approval_req = checkpoint.get("approval_requirement", "none")
    changed_files = checkpoint.get("changed_files", [])
    test_status = checkpoint.get("test_status", "none")
    wt_id = checkpoint.get("worktree_identifier") or "none"
    commit = checkpoint.get("commit_hash") or "none"
    next_action = checkpoint.get("next_action") or "none"

    print("=" * 60)
    print(f"STATUS RUN: {run_id}")
    print("=" * 60)
    print(f"Stage         : {stage}")
    print(f"Status        : {status}")
    print(f"Risk Level    : {risk}")
    print(f"Approval Req  : {approval_req}")
    print(f"Changed Files : {len(changed_files)} ({', '.join(changed_files) if changed_files else 'none'})")
    print(f"Tests         : {test_status}")
    print(f"Worktree      : {wt_id}")
    print(f"Commit        : {commit[:8] if commit != 'none' else 'none'}")
    print(f"Next Action   : {next_action}")
    print("Artifacts     :")
    print(f"  - Plan      : {checkpoint.get('plan_artifact_id') or 'none'}")
    print(f"  - Proposal  : {checkpoint.get('proposal_artifact_id') or 'none'}")
    print(f"  - Execution : {checkpoint.get('execution_artifact_id') or 'none'}")
    print(f"  - Review    : {checkpoint.get('review_artifact_id') or 'none'}")
    print(f"  - Integrate : {checkpoint.get('integration_artifact_id') or 'none'}")
    print("=" * 60)
    print()


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume an orchestration run from its last checkpoint."""
    if not args.project:
        print("ERROR: --project is required for resume command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    runner = OrchestrationRunner(project_path, verbose=bool(args.verbose))
    try:
        res = runner.resume(
            run_arg=getattr(args, "run", "latest") or "latest",
            approve=bool(args.approve),
            reject=bool(args.reject),
            request_changes=bool(args.request_changes),
            apply=bool(args.apply),
            note=getattr(args, "note", "") or "",
        )
    except Exception as e:
        print(f"ERROR: Gagal me-resume run: {e}")
        sys.exit(1)

    _print_run_summary(res, verbose=bool(args.verbose))


def cmd_ceo(args: argparse.Namespace) -> None:
    """Run or resume high-level CEO program mode."""
    if not args.project:
        print("ERROR: --project is required for ceo command.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    runner = CEORunner(project_path, verbose=bool(args.verbose))
    try:
        runner.run_ceo(
            objective_file=getattr(args, "objective", None),
            request_text=getattr(args, "request", None),
            autonomous=bool(getattr(args, "autonomous", False)),
            max_workers=getattr(args, "max_workers", 1) or 1,
            program_id=getattr(args, "program", None),
            resume=bool(getattr(args, "resume", False)),
            approve=bool(getattr(args, "approve", False)),
            reject=bool(getattr(args, "reject", False)),
            request_changes=bool(getattr(args, "request_changes", False)),
            approval_policy=getattr(args, "approval_policy", "safe-auto") or "safe-auto",
            note=getattr(args, "note", "") or "",
        )
    except Exception as e:
        safe_report = format_safe_error_report(e, context="CEO mode")
        print(safe_report)
        sys.exit(1)


def cmd_program_lock_status(args: argparse.Namespace) -> None:
    """Read-only diagnostic for single-writer program lock."""
    if not args.project:
        print("ERROR: --project is required.")
        sys.exit(1)

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.is_dir():
        print(f"ERROR: Project directory does not exist: {project_path}")
        sys.exit(1)

    program_arg = getattr(args, "program", "latest") or "latest"
    if program_arg == "latest":
        try:
            from agent_system.program_manager import ProgramManager
            pm = ProgramManager(project_path)
            _, state = pm.load_program_state("latest")
            program_id = state.program_id
        except Exception:
            program_id = "default"
    else:
        program_id = program_arg

    from agent_system.program_lock import ProgramLock
    lock = ProgramLock(project_path, program_id=program_id)
    info = lock.inspect()

    print("=" * 60)
    print(f"PROGRAM LOCK STATUS: {program_id}")
    print("=" * 60)
    print(f"Lock Exists   : {info.get('exists')}")
    print(f"Lock Active   : {info.get('is_active')}")
    print(f"Lock Stale    : {info.get('is_stale')}")
    if info.get("exists"):
        print(f"Process PID   : {info.get('process_id')}")
        print(f"Hostname      : {info.get('hostname')}")
        print(f"Created At    : {info.get('created_at')}")
        print(f"Heartbeat At  : {info.get('heartbeat_at')}")
        print(f"Current Task  : {info.get('current_task_id') or 'none'}")
        print(f"Active Run ID : {info.get('active_run_id') or 'none'}")
    print("=" * 60)


def parse_arguments() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Multi-agent software orchestrator with portable project protocol"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize portable project protocol in target project",
    )
    init_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    init_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files (with backup)",
    )

    # inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect project protocol configuration (no LLM, no modifications)",
    )
    inspect_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )

    # plan command
    plan_parser = subparsers.add_parser(
        "plan",
        help="Run planning workflow with LLM",
    )
    plan_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    plan_parser.add_argument(
        "--request",
        required=True,
        help="Task atau permintaan dari pengguna",
    )

    # propose command
    propose_parser = subparsers.add_parser(
        "propose",
        help="Generate implementation proposal from verified plan",
    )
    propose_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    propose_parser.add_argument(
        "--plan",
        default="latest",
        help="Plan artifact ID, filename, or 'latest' (default: 'latest')",
    )

    # implement command
    implement_parser = subparsers.add_parser(
        "implement",
        help="Run controlled implementation workflow from approved proposal",
    )
    implement_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    implement_parser.add_argument(
        "--proposal",
        default="latest",
        help="Proposal artifact ID, filename, or 'latest' (default: 'latest')",
    )
    implement_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run without modifying files or running commands (default: True)",
    )
    implement_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to isolated branch and run tests",
    )
    implement_parser.add_argument(
        "--isolation",
        default="git-worktree",
        choices=["git-worktree", "branch"],
        help="Workspace isolation mechanism (default: git-worktree)",
    )
    implement_parser.add_argument(
        "--worktree-root",
        default=None,
        help="Custom root directory for git worktrees",
    )

    # execute command
    execute_parser = subparsers.add_parser(
        "execute",
        help="Run coding execution with OpenHands Software Agent SDK",
    )
    execute_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    execute_parser.add_argument(
        "--proposal",
        default="latest",
        help="Proposal artifact ID, filename, or 'latest' (default: 'latest')",
    )
    execute_parser.add_argument(
        "--executor",
        default="openhands",
        choices=["openhands"],
        help="Coding agent executor engine (default: openhands)",
    )
    execute_parser.add_argument(
        "--isolation",
        default="git-worktree",
        choices=["git-worktree", "branch"],
        help="Workspace isolation mechanism (default: git-worktree)",
    )
    execute_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry run without modifying files or running commands (default)",
    )
    execute_parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply changes: create worktree and run OpenHands execution",
    )
    execute_parser.add_argument(
        "--worktree-root",
        default=None,
        help="Custom root directory for git worktrees",
    )

    # review-execution command
    review_exec_parser = subparsers.add_parser(
        "review-execution",
        help="Perform formal human review on an execution artifact",
    )
    review_exec_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    review_exec_parser.add_argument(
        "--execution",
        default="latest",
        help="Execution artifact ID, filename, or 'latest' (default: 'latest')",
    )
    review_exec_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve execution for integration (status: approved_for_integration)",
    )
    review_exec_parser.add_argument(
        "--request-changes",
        action="store_true",
        help="Request changes on execution (status: changes_requested)",
    )
    review_exec_parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject execution (status: rejected)",
    )
    review_exec_parser.add_argument(
        "--note",
        default="",
        help="Optional human review note or feedback",
    )
    review_exec_parser.add_argument(
        "--reviewer",
        default=None,
        help="Optional reviewer identity override",
    )

    # integrate command
    integrate_parser = subparsers.add_parser(
        "integrate",
        help="Integrate approved execution changes into target branch",
    )
    integrate_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    integrate_parser.add_argument(
        "--execution",
        default="latest",
        help="Execution artifact ID, filename, or 'latest' (default: 'latest')",
    )
    integrate_parser.add_argument(
        "--strategy",
        default="cherry-pick",
        choices=["cherry-pick"],
        help="Integration strategy (default: cherry-pick)",
    )
    integrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run integration planning without making commits (default)",
    )
    integrate_parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply integration: stage, commit, cherry-pick, and validate",
    )

    # run command (user-facing orchestration facade)
    run_parser = subparsers.add_parser(
        "run",
        help="Run end-to-end multi-agent orchestration lifecycle",
    )
    run_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    run_parser.add_argument(
        "--request",
        required=True,
        help="User request description",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry run without modifying code or making commits (default)",
    )
    run_parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply changes end-to-end through integration",
    )
    run_parser.add_argument(
        "--approval-policy",
        default="safe-auto",
        choices=["safe-auto", "always-human", "never-auto"],
        help="Approval policy (default: safe-auto)",
    )
    run_parser.add_argument(
        "--isolation",
        default="git-worktree",
        choices=["git-worktree", "branch"],
        help="Workspace isolation mechanism (default: git-worktree)",
    )
    run_parser.add_argument(
        "--executor",
        default="openhands",
        choices=["openhands"],
        help="Coding executor engine (default: openhands)",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed internal agent progress logs",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Inspect status of an orchestration run",
    )
    status_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    status_parser.add_argument(
        "--run",
        default="latest",
        help="Run ID or 'latest' (default: latest)",
    )
    status_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show full checkpoint details",
    )

    # resume command
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an orchestration run from checkpoint",
    )
    resume_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    resume_parser.add_argument(
        "--run",
        default="latest",
        help="Run ID or 'latest' (default: latest)",
    )
    resume_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve changes at human approval gate",
    )
    resume_parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject changes at human approval gate",
    )
    resume_parser.add_argument(
        "--request-changes",
        action="store_true",
        help="Request changes at human approval gate",
    )
    resume_parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply integration after approval",
    )
    resume_parser.add_argument(
        "--note",
        default="",
        help="Optional human review note",
    )
    resume_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed progress logs",
    )

    # ceo command
    ceo_parser = subparsers.add_parser(
        "ceo",
        help="Run or resume high-level CEO program mode",
    )
    ceo_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    ceo_parser.add_argument(
        "--objective",
        default=None,
        help="Path to markdown objective file",
    )
    ceo_parser.add_argument(
        "--request",
        default=None,
        help="Inline objective text",
    )
    ceo_parser.add_argument(
        "--autonomous",
        action="store_true",
        default=False,
        help="Run autonomously until completion, approval required, or blocker",
    )
    ceo_parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel workers (default: 1)",
    )
    ceo_parser.add_argument(
        "--program",
        default=None,
        help="Program ID to resume or inspect",
    )
    ceo_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume existing program",
    )
    ceo_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve changes at CEO approval gate",
    )
    ceo_parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject changes at CEO approval gate",
    )
    ceo_parser.add_argument(
        "--request-changes",
        action="store_true",
        help="Request changes at CEO approval gate",
    )
    ceo_parser.add_argument(
        "--note",
        default="",
        help="Optional review note",
    )
    ceo_parser.add_argument(
        "--approval-policy",
        default="safe-auto",
        choices=["safe-auto", "always-human", "never-auto"],
        help="Approval policy (default: safe-auto)",
    )
    ceo_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed progress logs",
    )

    # program-lock-status command
    lock_status_parser = subparsers.add_parser(
        "program-lock-status",
        help="Read-only diagnostic for single-writer program lock",
    )
    lock_status_parser.add_argument(
        "--project",
        required=True,
        help="Path to target project directory",
    )
    lock_status_parser.add_argument(
        "--program",
        default="latest",
        help="Program ID or 'latest' (default: latest)",
    )

    # Backward compatibility: support old style without subcommand
    parser.add_argument(
        "--project",
        default=(
            str(settings.default_project_path)
            if settings.default_project_path
            else None
        ),
        help="[Legacy] Path repository target",
    )
    parser.add_argument(
        "--request",
        help="[Legacy] Task atau permintaan dari pengguna",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "propose":
        cmd_propose(args)
    elif args.command == "implement":
        cmd_implement(args)
    elif args.command == "execute":
        cmd_execute(args)
    elif args.command == "review-execution":
        cmd_review_execution(args)
    elif args.command == "integrate":
        cmd_integrate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "ceo":
        cmd_ceo(args)
    elif args.command == "program-lock-status":
        cmd_program_lock_status(args)
    else:
        # Backward compatibility mode
        if args.request:
            cmd_plan(args)
        else:
            print("ERROR: Command required. Use: init, inspect, plan, or propose")
            print("Run 'python main.py --help' for usage.")
            sys.exit(1)


if __name__ == "__main__":
    main()
