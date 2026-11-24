import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.error_reporting import sanitize_text
from agent_system.integration_lock import IntegrationLock, TargetHeadChangedError
from agent_system.project_loader import load_project
from agent_system.worktree_manager import get_default_worktree_root


def is_test_file(path_str: str) -> bool:
    """Check if file is a test file."""
    p = str(path_str).replace("\\", "/").lower()
    name = Path(p).name
    return (
        name.endswith("_test.go")
        or name.endswith("_test.py")
        or name.startswith("test_")
        or name.endswith(".test.ts")
        or name.endswith(".spec.ts")
        or name.endswith(".test.js")
        or name.endswith(".spec.js")
        or name.endswith("test.java")
        or "/tests/" in p
        or p.startswith("tests/")
        or "/test/" in p
        or p.startswith("test/")
    )


def is_doc_file(path_str: str) -> bool:
    """Check if file is documentation."""
    p = str(path_str).replace("\\", "/").lower()
    name = Path(p).name
    return (
        name.endswith(".md")
        or name.endswith(".rst")
        or name.endswith(".txt")
        or "/docs/" in p
        or p.startswith("docs/")
        or "/doc/" in p
        or p.startswith("doc/")
    )


def classify_commit_type(changed_files: list[str], task_title: str) -> str:
    """Classify conventional commit type based on actual changed files and task intent."""
    if not changed_files:
        return "chore"

    # 1. If only test files changed
    if all(is_test_file(f) for f in changed_files):
        return "test"

    # 2. If only docs changed
    if all(is_doc_file(f) for f in changed_files):
        return "docs"

    # 3. Production files or mixed
    title_lower = task_title.lower()
    if any(k in title_lower for k in ("fix", "bug", "perbaiki", "resolv", "patch", "error", "issue", "hotfix")):
        return "fix"
    if any(k in title_lower for k in ("refactor", "restructur", "clean", "reformat", "rapikan", "simplif")):
        return "refactor"
    return "feat"


def resolve_commit_scope(changed_files: list[str], default_scope: str = "backend") -> str:
    """Resolve commit scope from changed files."""
    if not changed_files:
        return default_scope or "core"

    scopes = set()
    for f in changed_files:
        clean = f.replace("\\", "/").lstrip("/")
        parts = clean.split("/")
        if len(parts) > 1 and parts[0] in ("backend", "frontend", "docs", "pkg", "cmd", "internal", "tests"):
            scopes.add(parts[0])
        elif len(parts) > 1:
            scopes.add(parts[0])

    if len(scopes) == 1:
        s = next(iter(scopes))
        return "test" if s == "tests" else s

    return default_scope or "core"


def generate_commit_message(
    changed_files: list[str],
    task_title: str,
    branch_name: str,
    task_owner: str | None = None,
) -> str:
    """Generate conventional commit message based on actual changed files."""
    commit_type = classify_commit_type(changed_files, task_title)
    scope = resolve_commit_scope(changed_files, default_scope=task_owner or "backend")
    clean_task_title = task_title.split("(")[0].strip()[:70]
    return f"{commit_type}({scope}): {clean_task_title} [{branch_name}]"


def check_source_repository_state(project_root: Path) -> dict[str, Any]:
    """Inspect Git working directory of source repository.

    Distinguishes runtime artifacts under .agent/ from source code modifications.
    Never claims repository is clean if git status produces any output.
    """
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    source_changes: list[tuple[str, str]] = []
    runtime_artifact_entries: list[tuple[str, str]] = []

    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            status_code, file_path = parts
        else:
            status_code = line[:2].strip()
            file_path = line[2:].strip()

        file_path_clean = file_path.strip().strip('"')
        if file_path_clean.startswith(".agent/") or file_path_clean == ".agent":
            runtime_artifact_entries.append((status_code, file_path_clean))
        else:
            source_changes.append((status_code, file_path_clean))

    is_completely_clean = len(lines) == 0
    has_uncommitted_source_changes = len(source_changes) > 0
    has_runtime_artifacts = len(runtime_artifact_entries) > 0

    return {
        "is_completely_clean": is_completely_clean,
        "has_uncommitted_source_changes": has_uncommitted_source_changes,
        "has_runtime_artifacts": has_runtime_artifacts,
        "source_changes": [sc[1] for sc in source_changes],
        "runtime_artifacts": [ra[1] for ra in runtime_artifact_entries],
        "gitignore_recommendation": (
            "Tambahkan '.agent/artifacts/' ke .gitignore untuk mengabaikan runtime artifacts "
            "tanpa mengabaikan konfigurasi versioned '.agent/' (seperti manifest.yaml, policy.yaml, workflow.yaml)."
        ) if has_runtime_artifacts else None,
        "raw_lines": lines,
    }


def integrate_execution(
    project_root: Path,
    execution_file: Path,
    execution_data: dict[str, Any],
    strategy: str = "cherry-pick",
    apply: bool = False,
) -> dict[str, Any]:
    """Integrate approved execution into main project target branch.

    Supports:
    - strategy: 'cherry-pick'
    - mode: dry-run (default, apply=False) or apply (apply=True)
    """
    if strategy != "cherry-pick":
        raise ValueError(f"Strategi integrasi '{strategy}' tidak didukung. Versi ini hanya mendukung 'cherry-pick'.")

    status = execution_data.get("status")
    if status != "approved_for_integration":
        raise ValueError(
            f"Execution '{execution_file.name}' memiliki status '{status}', bukan 'approved_for_integration'. "
            "Integrasi hanya dapat dijalankan setelah approval manusia ('python main.py review-execution --approve')."
        )

    res = execution_data.get("result", {})
    wt_id = res.get("worktree_identifier", "")
    branch_name = res.get("branch_name", "")
    approved_files = res.get("approved_files", [])
    approved_set = {str(f).replace("\\", "/").strip().lstrip("/") for f in approved_files}

    wt_root = get_default_worktree_root(project_root)
    wt_path = wt_root / wt_id
    if not wt_path.is_dir():
        raise FileNotFoundError(f"Worktree '{wt_id}' tidak ditemukan di '{wt_path}'.")

    # Verify branch exists
    res_br = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if not res_br.stdout.strip():
        raise ValueError(f"Branch '{branch_name}' tidak ditemukan di git repository.")

    # Target branch in main project
    res_curr = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    current_branch = res_curr.stdout.strip() or "main"

    # Check source repository state
    repo_state = check_source_repository_state(project_root)
    if repo_state["has_uncommitted_source_changes"]:
        raise ValueError(
            "Repository utama memiliki perubahan source code yang belum di-commit: "
            f"{', '.join(repo_state['source_changes'])}. "
            "Integrasi ditolak (fail-closed)."
        )

    exec_artifact_id = execution_data.get("artifact_id", execution_file.stem)
    clean_id = exec_artifact_id.replace("execution_", "").replace(".json", "")

    # Task title for commit message
    proposal_path = res.get("source_proposal_path", "")
    task_title = "Implementation changes"
    task_owner = res.get("owner", "backend")
    if proposal_path:
        prop_file = project_root / proposal_path
        if prop_file.is_file():
            try:
                p_data = json.loads(prop_file.read_text(encoding="utf-8"))
                prop_inner = p_data.get("result", {}).get("proposal", {})
                task_title = prop_inner.get("task_title") or task_title
                task_owner = prop_inner.get("owner") or task_owner
            except Exception:
                pass

    changed_files = res.get("actual_changed_files", []) or res.get("changed_files", [])
    planned_commit_message = generate_commit_message(
        changed_files=changed_files,
        task_title=task_title,
        branch_name=branch_name,
        task_owner=task_owner,
    )
    val_cmds = res.get("validation_command_summary", [])

    # =========================================================================
    # DRY RUN
    # =========================================================================
    if not apply:
        return {
            "mode": "dry-run",
            "source_project": str(project_root),
            "current_branch": current_branch,
            "execution_id": exec_artifact_id,
            "worktree_path": str(wt_path),
            "agent_branch": branch_name,
            "source_agent_branch": branch_name,
            "changed_files": changed_files,
            "test_result": val_cmds,
            "planned_commit_message": planned_commit_message,
            "validation_commands": val_cmds,
            "integration_eligibility": "Eligible (approved_for_integration)",
            "is_git_clean": repo_state["is_completely_clean"],
            "has_runtime_artifacts": repo_state["has_runtime_artifacts"],
            "runtime_artifacts": repo_state["runtime_artifacts"],
            "gitignore_recommendation": repo_state["gitignore_recommendation"],
            "status": "dry_run_ready",
        }

    # =========================================================================
    # APPLY
    # =========================================================================
    loaded_proj = load_project(project_root)
    artifact_store = ProjectArtifactStore(project_root, loaded_proj.manifest)

    # Check if a cherry-pick is already active on target repo
    cherry_pick_head_file = project_root / ".git" / "CHERRY_PICK_HEAD"
    if cherry_pick_head_file.is_file():
        raise RuntimeError(
            "FAIL-CLOSED: Cherry-pick sedang aktif atau belum selesai pada repository target. "
            "Selesaikan atau batalkan terlebih dahulu (git cherry-pick --abort)."
        )

    # Acquire Integration Lock
    res_expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    expected_target_head = res_expected_head.stdout.strip() if res_expected_head.returncode == 0 else None

    integ_lock = IntegrationLock(
        project_root=project_root,
        project_id=loaded_proj.manifest.project.id,
        target_branch=current_branch,
        program_id=execution_data.get("program_id", "default"),
    )
    integ_lock.acquire(
        task_id=execution_data.get("task_id"),
        run_id=clean_id,
        execution_id=exec_artifact_id,
        expected_target_head=expected_target_head,
    )

    try:
        integ_lock.update_phase("validating_worktree")
        # 1. Inspect actual worktree status and enforce scope
        res_wt_status = subprocess.run(
            ["git", "status", "--porcelain", "-uall"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        wt_status_lines = [l.strip() for l in res_wt_status.stdout.splitlines() if l.strip()]
        wt_changed_files: list[str] = []

        for line in wt_status_lines:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                f_path = parts[1].strip().strip('"')
                clean_f = f_path.replace("\\", "/").lstrip("/")
                wt_changed_files.append(clean_f)
                if clean_f not in approved_set:
                    # Record integration failure
                    now_iso = datetime.now(timezone.utc).isoformat()
                    fail_payload = {
                        "execution_id": exec_artifact_id,
                        "status": "integration_failed",
                        "strategy": strategy,
                        "target_branch": current_branch,
                        "source_agent_branch": branch_name,
                        "agent_branch": branch_name,
                        "timestamp": now_iso,
                        "error_message": f"Ditemukan file di luar approved scope: '{clean_f}'",
                        "scope_violations": [clean_f],
                    }
                    artifact_store.write_report(f"integration_failure_{clean_id}", fail_payload)
                    integ_lock.update_phase("failed")
                    raise ValueError(
                        f"Integrasi gagal (scope violation): File '{clean_f}' pada worktree tidak termasuk dalam approved_files."
                    )

        # Re-evaluate commit message using actual worktree changed files if present
        if wt_changed_files:
            planned_commit_message = generate_commit_message(
                changed_files=wt_changed_files,
                task_title=task_title,
                branch_name=branch_name,
                task_owner=task_owner,
            )

        # 2. Run validation commands in worktree
        wt_validation_results = []
        for cmd in val_cmds:
            exe = cmd.get("executable", "go")
            args = cmd.get("arguments", ["test", "./..."])
            cwd_rel = cmd.get("working_directory", "backend")
            target_cwd = (wt_path / cwd_rel).resolve()
            res_cmd = subprocess.run(
                [exe] + args,
                cwd=target_cwd,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
            passed = res_cmd.returncode == 0
            wt_validation_results.append({
                "executable": exe,
                "arguments": args,
                "working_directory": cwd_rel,
                "exit_code": res_cmd.returncode,
                "status": "passed" if passed else "failed",
            })
            if not passed:
                now_iso = datetime.now(timezone.utc).isoformat()
                fail_payload = {
                    "execution_id": exec_artifact_id,
                    "status": "integration_failed",
                    "strategy": strategy,
                    "target_branch": current_branch,
                    "source_agent_branch": branch_name,
                    "agent_branch": branch_name,
                    "timestamp": now_iso,
                    "error_message": f"Validation command dalam worktree gagal: {exe} {' '.join(args)} (exit code: {res_cmd.returncode})",
                    "validation_results": wt_validation_results,
                }
                artifact_store.write_report(f"integration_failure_{clean_id}", fail_payload)
                integ_lock.update_phase("failed")
                raise ValueError(f"Integrasi gagal: Validation command '{exe} {' '.join(args)}' dalam worktree gagal.")

        # 3. Stage only approved changed files in worktree
        for f in wt_changed_files:
            subprocess.run(["git", "add", f], cwd=wt_path, check=True, capture_output=True)

        res_staged_diff = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        staged_diff_stat = res_staged_diff.stdout.strip()

        # 4. Create commit on agent branch if changes are staged
        integ_lock.update_phase("creating_source_commit")
        res_check_staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=wt_path)
        has_uncommitted_staged = res_check_staged.returncode != 0

        if has_uncommitted_staged:
            res_commit = subprocess.run(
                ["git", "commit", "-m", planned_commit_message],
                cwd=wt_path,
                capture_output=True,
                text=True,
            )
            if res_commit.returncode != 0:
                now_iso = datetime.now(timezone.utc).isoformat()
                fail_payload = {
                    "execution_id": exec_artifact_id,
                    "status": "integration_failed",
                    "strategy": strategy,
                    "target_branch": current_branch,
                    "source_agent_branch": branch_name,
                    "agent_branch": branch_name,
                    "timestamp": now_iso,
                    "error_message": f"Gagal membuat commit pada worktree: {res_commit.stderr}",
                }
                artifact_store.write_report(f"integration_failure_{clean_id}", fail_payload)
                integ_lock.update_phase("failed")
                raise RuntimeError(f"Gagal membuat commit pada worktree: {res_commit.stderr}")

        # Get HEAD commit on agent branch (source_commit)
        res_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_path,
            check=True,
            capture_output=True,
            text=True,
        )
        source_commit = res_head.stdout.strip()

        # 5. Cherry-pick commit into main project target branch
        integ_lock.update_phase("cherry_picking")
        res_cp = subprocess.run(
            ["git", "cherry-pick", source_commit],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if res_cp.returncode != 0:
            now_iso = datetime.now(timezone.utc).isoformat()
            err_out = (res_cp.stderr or res_cp.stdout).strip()
            is_conflict = "conflict" in err_out.lower()

            conflicting_files = []
            res_st = subprocess.run(["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True)
            for line in res_st.stdout.splitlines():
                l_str = line.strip()
                if l_str.startswith(("UU ", "AA ", "UD ", "DU ", "UA ", "AU ", "DD ")):
                    conflicting_files.append(l_str.split(maxsplit=1)[-1].strip())

            fail_payload = {
                "execution_id": exec_artifact_id,
                "status": "integration_failed",
                "strategy": strategy,
                "target_branch": current_branch,
                "source_agent_branch": branch_name,
                "agent_branch": branch_name,
                "source_commit": source_commit,
                "integrated_commit": None,
                "cherry_pick_occurred": False,
                "is_conflict": is_conflict,
                "conflicting_files": conflicting_files,
                "expected_target_head": expected_target_head,
                "timestamp": now_iso,
                "error_message": sanitize_text(f"Cherry-pick gagal: {err_out}"),
                "recovery_instructions": (
                    "Untuk membatalkan cherry-pick: git cherry-pick --abort\n"
                    "Untuk menyelesaikan konflik: selesaikan konflik file, git add <file>, lalu git cherry-pick --continue"
                ),
            }
            # Abort cherry-pick if conflict was active to leave target repo safe
            abort_performed = False
            abort_error = None
            if is_conflict or (project_root / ".git" / "CHERRY_PICK_HEAD").is_file():
                res_abort = subprocess.run(["git", "cherry-pick", "--abort"], cwd=project_root, capture_output=True, text=True)
                abort_performed = res_abort.returncode == 0
                if not abort_performed:
                    abort_error = res_abort.stderr.strip()

            fail_payload["abort_scheduled"] = True
            fail_payload["abort_performed"] = abort_performed
            if abort_error:
                fail_payload["abort_error"] = abort_error

            artifact_store.write_report(f"integration_failure_{clean_id}", fail_payload)
            artifact_store.write_execution(f"integration_{clean_id}", fail_payload)

            integ_lock.update_phase("failed")

            raise RuntimeError(
                f"Cherry-pick gagal (exit code: {res_cp.returncode}): {err_out}\n"
                f"Conflicting files: {conflicting_files or 'none'}\n"
                "Instruksi recovery:\n"
                "  Batal: git cherry-pick --abort\n"
                "  Selesaikan: perbaiki konflik, git add <file>, git cherry-pick --continue"
            )

        # Get HEAD commit on target branch after cherry-pick (integrated_commit)
        res_main_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        integrated_commit = res_main_head.stdout.strip()

        # 6. Post-integration validation on main project
        integ_lock.update_phase("post_validation")
        post_validation_results = []
        all_post_passed = True
        for cmd in val_cmds:
            exe = cmd.get("executable", "go")
            args = cmd.get("arguments", ["test", "./..."])
            cwd_rel = cmd.get("working_directory", "backend")
            target_cwd = (project_root / cwd_rel).resolve()
            res_cmd = subprocess.run(
                [exe] + args,
                cwd=target_cwd,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
            passed = res_cmd.returncode == 0
            if not passed:
                all_post_passed = False
            post_validation_results.append({
                "executable": exe,
                "arguments": args,
                "working_directory": cwd_rel,
                "exit_code": res_cmd.returncode,
                "status": "passed" if passed else "failed",
            })

        if not all_post_passed:
            now_iso = datetime.now(timezone.utc).isoformat()
            fail_payload = {
                "execution_id": exec_artifact_id,
                "status": "integration_failed",
                "strategy": strategy,
                "target_branch": current_branch,
                "source_agent_branch": branch_name,
                "agent_branch": branch_name,
                "source_commit": source_commit,
                "integrated_commit": integrated_commit,
                "cherry_pick_occurred": True,
                "post_integration_validation": {
                    "status": "failed",
                    "validation_results": post_validation_results,
                },
                "post_validation_results": post_validation_results,
                "timestamp": now_iso,
                "error_message": "Post-integration validation pada repository utama gagal.",
                "recovery_instructions": (
                    f"Cherry-pick commit {integrated_commit[:8]} telah diterapkan pada branch {current_branch}, "
                    "namun post-integration validation gagal.\n"
                    "Instruksi recovery:\n"
                    f"  1. Periksa log kegagalan pengujian pada repository utama.\n"
                    f"  2. Untuk membatalkan commit integrasi: git revert --no-edit {integrated_commit[:8]}\n"
                    f"  3. Atau lakukan perbaikan manual dan commit perbaikan pada branch {current_branch}."
                ),
            }
            artifact_store.write_report(f"integration_failure_{clean_id}", fail_payload)
            artifact_store.write_execution(f"integration_{clean_id}", fail_payload)

            # Update execution artifact status
            execution_data["status"] = "integration_failed"
            if "result" in execution_data and isinstance(execution_data["result"], dict):
                execution_data["result"]["final_status"] = "integration_failed"
                execution_data["result"]["integration"] = fail_payload

            with open(execution_file, "w", encoding="utf-8") as f:
                json.dump(execution_data, f, ensure_ascii=False, indent=2)

            integ_lock.update_phase("failed")
            raise RuntimeError(
                f"Post-integration validation pada repository utama gagal. "
                f"Cherry-pick telah terjadi ({integrated_commit[:8]}). "
                "Status ditandai sebagai integration_failed."
            )

        # 7. Write integration artifact and report
        now_iso = datetime.now(timezone.utc).isoformat()
        integration_payload = {
            "execution_id": exec_artifact_id,
            "proposal_id": proposal_path,
            "status": "completed",
            "strategy": strategy,
            "target_branch": current_branch,
            "source_agent_branch": branch_name,
            "agent_branch": branch_name,
            "source_commit": source_commit,
            "integrated_commit": integrated_commit,
            "commit_hash": integrated_commit,
            "cherry_pick_occurred": True,
            "changed_files": changed_files or wt_changed_files,
            "worktree_identifier": wt_id,
            "staged_diff_stat": staged_diff_stat,
            "validation_results": wt_validation_results,
            "post_integration_validation": {
                "status": "passed",
                "validation_results": post_validation_results,
            },
            "post_validation_results": post_validation_results,
            "timestamp": now_iso,
            "summary": f"Integrasi berhasil via cherry-pick commit {source_commit[:8]} ke {current_branch} (integrated commit: {integrated_commit[:8]}).",
        }

        integration_artifact_file = artifact_store.write_execution(f"integration_{clean_id}", integration_payload)
        integration_report_file = artifact_store.write_report(f"integration_report_{clean_id}", integration_payload)

        # 8. Update execution artifact status to completed
        execution_data["status"] = "completed"
        if "result" in execution_data and isinstance(execution_data["result"], dict):
            execution_data["result"]["final_status"] = "completed"
            execution_data["result"]["integration"] = {
                "status": "completed",
                "strategy": strategy,
                "source_commit": source_commit,
                "integrated_commit": integrated_commit,
                "commit_hash": integrated_commit,
                "target_branch": current_branch,
                "source_agent_branch": branch_name,
                "post_integration_validation": {
                    "status": "passed",
                    "validation_results": post_validation_results,
                },
                "timestamp": now_iso,
                "integration_artifact": str(integration_artifact_file.relative_to(project_root)),
            }

        with open(execution_file, "w", encoding="utf-8") as f:
            json.dump(execution_data, f, ensure_ascii=False, indent=2)

        integ_lock.update_phase("completed")

        return {
            "mode": "apply",
            "status": "completed",
            "source_commit": source_commit,
            "integrated_commit": integrated_commit,
            "commit_hash": integrated_commit,
            "target_branch": current_branch,
            "source_agent_branch": branch_name,
            "agent_branch": branch_name,
            "worktree_path": str(wt_path),
            "changed_files": changed_files or wt_changed_files,
            "staged_diff_stat": staged_diff_stat,
            "integration_artifact_file": integration_artifact_file,
            "integration_report_file": integration_report_file,
            "post_integration_validation": {
                "status": "passed",
                "validation_results": post_validation_results,
            },
            "validation_results": post_validation_results,
        }
    finally:
        integ_lock.release()
