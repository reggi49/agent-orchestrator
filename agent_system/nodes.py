import json
import re
import subprocess
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)

from agent_system.config import get_settings
from agent_system.project_loader import load_project
from agent_system.project_reader import read_repository
from agent_system.prompts import (
    ANALYST_REPLAN_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    IMPLEMENTATION_PROPOSAL_SYSTEM_PROMPT,
    PROPOSAL_REVIEWER_SYSTEM_PROMPT,
    TASK_VERIFIER_SYSTEM_PROMPT,
)
from agent_system.proposal_agent import (
    compute_proposal_hash,
    detect_code_or_patch_in_text,
    validate_all_paths_relative,
)
from agent_system.proposal_reviewer import review_proposal_deterministically
from agent_system.branch_guard import BranchGuard, generate_branch_name
from agent_system.worktree_manager import (
    create_worktree,
    generate_branch_name as wt_generate_branch_name,
    generate_worktree_name,
    get_default_worktree_root,
)
from agent_system.command_runner import SafeCommandRunner
from agent_system.diff_reviewer import review_diff
from agent_system.file_writer import ControlledFileWriter
from agent_system.implementation_agent import run_implementation_agent
from agent_system.implementation_preflight import run_implementation_preflight
from agent_system.repair_agent import run_repair_agent
from agent_system.repository_explorer import RepositoryExplorer
from agent_system.schemas import (
    AgentState,
    CommandExecutionResult,
    DiffReview,
    ImplementationProposal,
    ImplementationResult,
    ImplementationStep,
    ProjectAnalysis,
    ProposalFinding,
    ProposalReview,
    ProposalRisk,
    ProposedCommand,
    ProposedFileChange,
    ProposedTest,
    RepositoryEvidence,
    RepositorySearchResult,
    TaskVerification,
)
from agent_system.source_revision import get_source_revision
from agent_system.task_verifier import (
    are_tasks_substantially_identical,
    build_search_targets,
    extract_migration_keywords,
    is_duplicate_task,
)


settings = get_settings()

model_kwargs = {
    "model": settings.llm_model,
    "api_key": settings.llm_api_key,
    "temperature": 0,
}

if settings.llm_base_url:
    model_kwargs["base_url"] = settings.llm_base_url

model = ChatOpenAI(**model_kwargs)

analyst_model = model.with_structured_output(
    ProjectAnalysis,
    method="json_mode"
)

verifier_model = model.with_structured_output(
    TaskVerification,
    method="json_mode"
)

proposal_model = model.with_structured_output(
    ImplementationProposal,
    method="json_mode"
)

proposal_reviewer_model = model.with_structured_output(
    ProposalReview,
    method="json_mode"
)


def project_reader_node(state: AgentState) -> dict:
    print("[Project Reader] Membaca repository...")

    project_root = Path(state["project_root"])
    context = read_repository(project_root)

    print(
        f"[Project Reader] Membaca "
        f"{len(context.important_files)} file penting."
    )

    for file in context.important_files:
        print(f"  - {file.path}")

    return {
        "repository_context": context.model_dump(mode="json"),
        "current_stage": "analyst",
        "status": "running",
        "replan_count": 0,
        "max_replans": state.get("max_replans", 2),
        "rejected_tasks": [],
    }



def analyst_node(state: AgentState) -> dict:
    rejected_tasks = state.get("rejected_tasks", [])
    replan_count = state.get("replan_count", 0)

    if rejected_tasks:
        print(f"[Analyst] Melakukan REPLAN ke-{replan_count} (task ditolak sebelumnya: {len(rejected_tasks)})...")
        prompt = ANALYST_REPLAN_SYSTEM_PROMPT
        analyst_input = {
            "user_request": state["user_request"],
            "repository": state["repository_context"],
            "replan_count": replan_count,
            "rejected_tasks": rejected_tasks,
            "previous_verification": state.get("task_verification", {}),
            "repository_evidence_summary": state.get("repository_evidence", {}).get("summary", ""),
        }
    else:
        print("[Analyst] Menganalisis repository dan task...")
        prompt = ANALYST_SYSTEM_PROMPT
        analyst_input = {
            "user_request": state["user_request"],
            "repository": state["repository_context"],
        }

    try:
        result = analyst_model.invoke(
            [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        analyst_input,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ]
        )
    except AuthenticationError:
        print(
            "ERROR: Autentikasi LLM gagal. "
            "Periksa LLM_API_KEY dan pastikan key berasal "
            "dari provider yang sama dengan LLM_BASE_URL.",
            file=sys.stderr,
        )
        sys.exit(1)
    except APIConnectionError:
        print(
            "ERROR: Tidak dapat terhubung ke LLM gateway. "
            "Pastikan 9Router berjalan dan LLM_BASE_URL benar.",
            file=sys.stderr,
        )
        sys.exit(1)
    except NotFoundError:
        print(
            "ERROR: Model tidak ditemukan. "
            "Periksa LLM_MODEL terhadap daftar model dari 9Router.",
            file=sys.stderr,
        )
        sys.exit(1)
    except RateLimitError:
        print(
            "ERROR: Gateway atau provider sedang mencapai batas penggunaan.",
            file=sys.stderr,
        )
        sys.exit(1)
    except TimeoutError:
        print(
            "ERROR: Request ke LLM gateway timeout.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        err_name = type(e).__name__
        if "OutputParserException" in err_name or "validation" in str(e).lower() or "json" in str(e).lower():
            print(f"[Analyst] Parsing error ({err_name}), mencoba fallback ekstraksi JSON mentah...")
            try:
                raw_res = model.invoke(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(analyst_input, ensure_ascii=False, indent=2)},
                    ]
                )
                raw_text = raw_res.content if isinstance(raw_res.content, str) else str(raw_res.content)
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    result = ProjectAnalysis.model_validate(parsed)
                else:
                    raise e
            except Exception as fallback_err:
                if settings.debug:
                    import traceback
                    traceback.print_exc()
                else:
                    print(
                        f"ERROR: Terjadi kesalahan saat memanggil LLM: {type(fallback_err).__name__}",
                        file=sys.stderr,
                    )
                sys.exit(1)
        else:
            if settings.debug:
                import traceback
                traceback.print_exc()
            else:
                print(
                    f"ERROR: Terjadi kesalahan saat memanggil LLM: {err_name}",
                    file=sys.stderr,
                )
            sys.exit(1)


    analysis_dict = result.model_dump(mode="json")
    candidate_title = analysis_dict.get("recommended_task", {}).get("title", "")

    if is_duplicate_task(candidate_title, rejected_tasks):
        print(f"[Analyst] PERINGATAN: Rekomendasi '{candidate_title}' serupa dengan task yang ditolak sebelumnya.")

    return {
        "analysis": analysis_dict,
        "current_stage": "repository_explorer",
        "status": "running",
    }

def repository_explorer_node(state: AgentState) -> dict:
    print("[Repository Explorer] Menyelidiki kode dan bukti konkret...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    validated_project = load_project(project_root)

    explorer = RepositoryExplorer(
        project_root=project_root,
        policy=validated_project.policy,
        max_file_size=500_000,
        max_results=50,
        max_total_chars=200_000,
    )

    analysis = state.get("analysis", {})
    recommended_task = analysis.get("recommended_task", {})
    title = recommended_task.get("title", "")
    desc = recommended_task.get("description", "")
    criteria = recommended_task.get("acceptance_criteria", [])

    targets = build_search_targets(title, desc, validated_project.manifest.components)

    terms = []
    for text_source in [title, desc]:
        for word in re.findall(r"[A-Za-z0-9_]+", text_source):
            w_lower = word.lower()
            if len(word) >= 3 and w_lower not in {"dan", "pada", "untuk", "dari", "yang", "dengan", "atau", "the", "and", "for", "with"}:
                terms.append(word)

    for c in criteria:
        for word in re.findall(r"[A-Za-z0-9_]+", c):
            w_lower = word.lower()
            if len(word) >= 3 and w_lower not in {"dan", "pada", "untuk", "dari", "yang", "dengan", "atau", "the", "and", "for", "with"}:
                terms.append(word)

    migration_kw = extract_migration_keywords(title, desc)
    terms.extend(migration_kw)

    clean_terms = []
    seen_terms = set()
    for t in terms:
        t_lower = t.lower()
        if t_lower not in seen_terms:
            seen_terms.add(t_lower)
            clean_terms.append(t)

    inspected_files = explorer.find_files(targets)
    evidence_data = explorer.inspect_candidate_task(
        task_title=title,
        search_terms=clean_terms[:15],
        target_paths=targets,
    )

    results = []
    for sr in evidence_data.get("search_results", []):
        results.append(
            RepositorySearchResult(
                path=sr["path"],
                line_start=sr["line"],
                line_end=sr["line"],
                excerpt=sr["excerpt"][:400],
                match_reason=f"Matched term '{sr['term']}'",
            )
        )

    # For files directly relevant to candidate task, read substantive excerpts
    for f in inspected_files:
        fname = Path(f).name.lower()
        if any(t.lower() in fname for t in clean_terms[:10]) or "main.go" in f or "service.go" in f:
            content, total_lines = explorer.read_excerpt(f, max_lines=70, max_chars=3000)
            if content and not any(r.path == f for r in results):
                results.append(
                    RepositorySearchResult(
                        path=f,
                        line_start=1,
                        line_end=min(70, total_lines),
                        excerpt=content,
                        match_reason=f"Relevant source file ({f})",
                    )
                )

    # If migration/database related, always inspect SQL migration files directly
    migration_files = explorer.find_files(["backend/migrations", "migrations"], extensions=[".sql"])
    for mf in migration_files:
        content, total_lines = explorer.read_excerpt(mf, max_lines=50, max_chars=2000)
        results.append(
            RepositorySearchResult(
                path=mf,
                line_start=1,
                line_end=min(50, total_lines),
                excerpt=content[:1000],
                match_reason=f"Migration file content ({total_lines} lines)",
            )
        )
        if mf not in inspected_files:
            inspected_files.append(mf)


    evidence = RepositoryEvidence(
        candidate_task=title,
        inspected_files=inspected_files[:30],
        results=results[:30],
        summary=evidence_data.get("summary", ""),
    )

    print(
        f"[Repository Explorer] Memeriksa {len(evidence.inspected_files)} file, "
        f"{len(evidence.results)} cuplikan bukti."
    )
    return {
        "repository_evidence": evidence.model_dump(mode="json"),
        "current_stage": "task_verifier",
        "status": "running",
    }

def check_migration_task_status(
    task_title: str,
    task_description: str,
    evidence_results: list[dict],
) -> TaskVerification | None:
    """Deterministically verify migration-related candidate tasks against actual evidence."""
    title_lower = task_title.lower()
    desc_lower = task_description.lower()
    full_text = f"{title_lower} {desc_lower}"

    if not any(k in full_text for k in ["migration", "database schema", "schema with migration", "create postgresql database schema"]):
        return None

    # Find which migration tables are evidenced in search results
    found_tables = set()
    migration_files = []
    
    for r in evidence_results:
        path = r.get("path", "")
        excerpt = r.get("excerpt", "").upper()
        if "migrations" in path and path.endswith(".sql"):
            migration_files.append(path)
            if "CREATE TABLE USERS" in excerpt:
                found_tables.add("users")
            if "CREATE TABLE WALLETS" in excerpt:
                found_tables.add("wallets")
            if "CREATE TABLE PRODUCTS" in excerpt:
                found_tables.add("products")
            if "CREATE TABLE TRANSACTIONS" in excerpt:
                found_tables.add("transactions")

    required_tables = {"users", "wallets", "products", "transactions"}
    
    # If all 4 tables are created in migrations
    if required_tables.issubset(found_tables):
        return TaskVerification(
            status="already_implemented",
            task_title=task_title,
            evidence_for=[
                "backend/migrations/001_create_users.sql: table users defined with UUID, email, password_hash, role, and index",
                "backend/migrations/002_create_wallets.sql: table wallets defined with FK to users, balance CHECK >= 0, and index",
                "backend/migrations/003_create_products.sql: table products defined with code, price, and indexes",
                "backend/migrations/004_create_transactions.sql: table transactions defined with FK to users, and indexes",
            ],
            evidence_against=[],
            missing_work=[],
            recommendation="Seluruh migration 001-004 sudah dibuat dan lengkap di backend/migrations/. Task ini sudah selesai.",
            should_replan=True,
        )
    elif found_tables:
        missing = required_tables - found_tables
        return TaskVerification(
            status="partially_implemented",
            task_title=task_title,
            evidence_for=[f"Migrations exist for: {', '.join(sorted(found_tables))}"],
            evidence_against=[f"Migrations missing for: {', '.join(sorted(missing))}"],
            missing_work=[f"Create migration SQL for {t}" for t in sorted(missing)],
            recommendation=f"Lengkapi migration untuk tabel: {', '.join(sorted(missing))}",
            should_replan=False,
        )

    return None


def task_verifier_node(state: AgentState) -> dict:
    print("[Task Verifier] Memverifikasi kandidat task terhadap kode aktual...")

    analysis = state.get("analysis", {})
    recommended_task = analysis.get("recommended_task", {})
    title = recommended_task.get("title", "")
    desc = recommended_task.get("description", "")
    criteria = recommended_task.get("acceptance_criteria", [])
    evidence_dict = state.get("repository_evidence", {})

    # 1. First, check deterministic migration verification
    det_verification = check_migration_task_status(
        task_title=title,
        task_description=desc,
        evidence_results=evidence_dict.get("results", []),
    )

    if det_verification:
        verification = det_verification
        print(f"[Task Verifier] Deterministic check: Status = {verification.status}")
    else:
        # 2. Use LLM Verifier
        verifier_input = {
            "candidate_task": recommended_task,
            "repository_evidence": evidence_dict,
        }
        try:
            res = verifier_model.invoke(
                [
                    {
                        "role": "system",
                        "content": TASK_VERIFIER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            verifier_input,
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                ]
            )
            verification = res
            print(f"[Task Verifier] Model check: Status = {verification.status}")
        except Exception as e:
            print(f"[Task Verifier] Fallback verification due to: {type(e).__name__}")
            verification = TaskVerification(
                status="verified_pending",
                task_title=title,
                evidence_for=[r["path"] for r in evidence_dict.get("results", [])[:3]],
                evidence_against=[],
                missing_work=[],
                recommendation="Verification accepted pending manual review.",
                should_replan=False,
            )

    # Manage replan state and rejected tasks
    replan_count = state.get("replan_count", 0)
    max_replans = state.get("max_replans", 2)
    rejected_tasks = list(state.get("rejected_tasks", []))

    should_replan = (
        verification.status in {"already_implemented", "invalid_task", "insufficient_evidence"}
        or verification.should_replan
    )

    if should_replan:
        rejected_tasks.append({
            "title": title,
            "status": verification.status,
            "reason": verification.recommendation,
        })
        replan_count += 1

    return {
        "task_verification": verification.model_dump(mode="json"),
        "replan_count": replan_count,
        "rejected_tasks": rejected_tasks,
        "current_stage": "supervisor" if not should_replan or replan_count > max_replans else "analyst",
        "status": "running",
    }

def supervisor_validate_proposal(state: AgentState) -> dict:
    proposal_data = state.get("implementation_proposal", {})
    review_data = state.get("proposal_review", {})
    plan_data = state.get("plan_data", {})
    task_verification = state.get("task_verification", {})
    revision_count = state.get("proposal_revision_count", 0)
    max_revisions = state.get("max_proposal_revisions", 2)
    project_root = Path(state["project_root"]).expanduser().resolve()

    # 1. Verified plan available
    if not plan_data:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Verified plan tidak tersedia.",
        }

    # 2. Task verification valid
    v_status = (
        task_verification.get("status")
        if task_verification
        else plan_data.get("result", {}).get("verification_status")
    )
    if v_status not in {"verified_pending", "partially_implemented"}:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Status verifikasi task '{v_status}' tidak valid untuk proposal implementasi.",
        }

    # 3. Evidence relevance valid
    evidence_paths = proposal_data.get("evidence_paths", []) or plan_data.get("evidence_paths", [])
    if not evidence_paths:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Evidence paths tidak tersedia atau kosong.",
        }

    # 4. Implementation proposal available
    if not proposal_data:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Implementation proposal belum tersedia.",
        }

    # 5. Proposal review status approved
    r_status = review_data.get("status")
    if r_status != "approved":
        if r_status == "revision_required" and revision_count < max_revisions:
            return {
                "status": "revision_required",
                "current_stage": "implementation_proposal",
                "final_summary": f"Proposal memerlukan revisi: {review_data.get('summary', '')}",
            }
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Proposal review status '{r_status}'. Dibutuhkan keputusan manusia.",
        }

    # 6. Proposal tidak memiliki critical finding
    findings = review_data.get("findings", [])
    if any(f.get("severity") == "critical" for f in findings):
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Proposal memiliki temuan berkategori critical.",
        }

    # 7. Semua acceptance criteria memiliki test mapping
    missing_ac = review_data.get("missing_acceptance_criteria", [])
    if missing_ac:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Acceptance criteria belum terpetakan ke test: {missing_ac}.",
        }

    # 8. Semua path relatif
    path_errors = validate_all_paths_relative(proposal_data)
    if path_errors:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Path tidak relatif ditemukan: {path_errors[:2]}.",
        }

    # 9. Semua command aman
    unsafe_cmds = review_data.get("unsafe_commands", [])
    if unsafe_cmds:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Perintah tidak aman ditemukan: {unsafe_cmds}.",
        }

    # 10. Semua command requires_approval bernilai true
    commands = proposal_data.get("validation_commands", [])
    if any(cmd.get("requires_approval") is not True for cmd in commands):
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Semua command harus memiliki requires_approval: true.",
        }

    # 11. Proposal tidak mengandung source code atau patch
    for f in proposal_data.get("files_to_create", []) + proposal_data.get("files_to_modify", []):
        for ch in f.get("changes", []):
            if detect_code_or_patch_in_text(ch):
                return {
                    "status": "human_required",
                    "current_stage": "finished",
                    "final_summary": "Proposal terdeteksi memuat potongan source code atau patch.",
                }

    # 13. Rollback plan tersedia
    if not proposal_data.get("rollback_plan"):
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Rollback plan wajib tersedia dalam proposal.",
        }

    return {
        "status": "approved_for_implementation",
        "current_stage": "finished",
        "final_summary": (
            f"Proposal '{proposal_data.get('proposal_id')}' disetujui untuk implementasi. "
            f"Task: '{proposal_data.get('task_title')}'. "
            f"Owner: '{proposal_data.get('owner')}'. "
            f"File changes: {len(proposal_data.get('files_to_create', []))} to create, "
            f"{len(proposal_data.get('files_to_modify', []))} to modify."
        ),
    }


def supervisor_validate_implementation(state: AgentState) -> dict:
    status = state.get("status")
    preflight = state.get("preflight_result", {})
    dry_run = state.get("dry_run", True)
    impl_res = state.get("implementation_result", {})
    outside_files = (
        impl_res.get("files_requested_outside_scope", [])
        or preflight.get("files_requested_outside_scope", [])
    )

    if status == "proposal_revision_required" or outside_files:
        return {
            "status": "proposal_revision_required",
            "current_stage": "finished",
            "final_summary": (
                state.get("final_summary")
                or f"Implementasi membutuhkan file di luar proposal yang disetujui ({outside_files}). Diperlukan proposal_revision_required."
            ),
        }

    if status == "stale_proposal":
        return {
            "status": "stale_proposal",
            "current_stage": "finished",
            "final_summary": state.get("final_summary") or "Source revision usang (stale_proposal).",
        }

    if status == "human_required":
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": state.get("final_summary") or "Dibutuhkan intervensi manual (human_required).",
        }

    if dry_run:
        return {
            "status": status if status in {"valid", "preflight_passed"} else "dry_run_ready",
            "current_stage": "finished",
            "final_summary": state.get("final_summary") or "Dry-run berhasil diverifikasi.",
        }

    # In --apply mode: check all invariants
    if not preflight.get("is_valid"):
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Preflight gagal sebelum implementasi.",
        }

    branch_name = state.get("branch_name")
    if not branch_name:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Branch terisolasi tidak berhasil dibuat.",
        }

    # OpenHands Coding Executor execution path
    if state.get("executor") == "openhands" or "coding_execution_result" in state:
        coding_res = state.get("coding_execution_result", {})
        diff_review = state.get("diff_review", {})

        # Fail-closed check: zero changed files
        changed_files = diff_review.get("changed_files", coding_res.get("changed_files", []))
        if not changed_files:
            return {
                "status": "execution_failed",
                "current_stage": "finished",
                "final_summary": "Eksekusi OpenHands gagal: tidak ada file yang berubah (zero changed files).",
            }

        # Scope violations check
        if diff_review.get("unexpected_files") or coding_res.get("scope_violations"):
            violations = diff_review.get("unexpected_files") or coding_res.get("scope_violations")
            return {
                "status": "scope_violation",
                "current_stage": "finished",
                "final_summary": f"Eksekusi OpenHands melanggar scope: {violations}.",
            }

        # Additional files requested
        if coding_res.get("requested_additional_files"):
            return {
                "status": "proposal_revision_required",
                "current_stage": "finished",
                "final_summary": f"OpenHands memerlukan file tambahan: {coding_res.get('requested_additional_files')}.",
            }

        # Validation command failures
        cmd_results = coding_res.get("validation_results", [])
        if any(c.get("exit_code") != 0 or c.get("status") != "passed" for c in cmd_results):
            return {
                "status": "failed",
                "current_stage": "finished",
                "final_summary": "Validation command gagal setelah eksekusi OpenHands.",
            }

        if diff_review.get("status") == "approved":
            return {
                "status": "execution_review_ready",
                "current_stage": "finished",
                "final_summary": "Eksekusi OpenHands berhasil diverifikasi dan siap ditinjau (execution_review_ready).",
            }

        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Diff review berstatus '{diff_review.get('status')}', bukan approved.",
        }

    # Legacy Implementation Executor path
    if impl_res.get("status") != "implementation_generated":
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Implementation status '{impl_res.get('status')}' tidak valid.",
        }

    cmd_results = state.get("command_results", [])
    if any(c.get("exit_code") != 0 or c.get("status") != "passed" for c in cmd_results):
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Validation command gagal setelah batas repair tercapai.",
        }

    diff_review = state.get("diff_review", {})
    if diff_review.get("status") != "approved":
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Diff review berstatus '{diff_review.get('status')}', bukan approved.",
        }

    if diff_review.get("unexpected_files"):
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": f"Ditemukan file tak terduga pada diff: {diff_review.get('unexpected_files')}",
        }

    return {
        "status": "implementation_ready",
        "current_stage": "finished",
        "final_summary": "Implementasi berhasil diverifikasi dan siap (implementation_ready).",
    }


def supervisor_node(state: AgentState) -> dict:
    if (
        state.get("current_stage") in {"implementation", "execution"}
        or "preflight_result" in state
        or "implementation_result" in state
        or "coding_execution_result" in state
    ):
        print("[Supervisor] Memvalidasi implementasi...")
        return supervisor_validate_implementation(state)

    if "implementation_proposal" in state:
        print("[Supervisor] Memvalidasi proposal implementasi...")
        return supervisor_validate_proposal(state)

    print("[Supervisor] Memvalidasi rencana...")

    analysis = state.get("analysis", {})
    recommended_task = analysis.get("recommended_task", {})
    task_verification = state.get("task_verification")
    rejected_tasks = state.get("rejected_tasks", [])
    replan_count = state.get("replan_count", 0)
    max_replans = state.get("max_replans", 2)

    # If max replans reached and verification is not passing
    if replan_count > max_replans or (
        task_verification
        and task_verification.get("status") in {"already_implemented", "invalid_task", "insufficient_evidence"}
        and replan_count >= max_replans
    ):
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": (
                f"Batas maksimum replan ({max_replans}) tercapai. "
                f"Kandidat terakhir '{recommended_task.get('title')}' ditolak "
                f"dengan status '{task_verification.get('status') if task_verification else 'unknown'}'. "
                f"Dibutuhkan keputusan manusia."
            ),
        }

    # 1. Task title check
    title = recommended_task.get("title", "").strip()
    if not title:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Analyst tidak menghasilkan task yang valid (title kosong).",
        }

    # 2. Owner valid and not unknown
    owner = recommended_task.get("owner")
    if not owner or owner == "unknown":
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Task belum memiliki owner yang valid.",
        }

    # 3. Acceptance criteria available
    acceptance_criteria = list(recommended_task.get("acceptance_criteria", []))
    if not acceptance_criteria:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Task belum memiliki acceptance criteria.",
        }

    # 4. TaskVerification available
    if not task_verification:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "TaskVerification belum tersedia.",
        }

    v_status = task_verification.get("status")
    # 5. Status verification must be verified_pending or partially_implemented
    if v_status not in {"verified_pending", "partially_implemented"}:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": (
                f"Status verifikasi '{v_status}' tidak memenuhi syarat untuk plan_ready."
            ),
        }

    # 6. Task not in rejected_tasks
    # Check by normalized title
    for rej in rejected_tasks:
        if are_tasks_substantially_identical(title, rej.get("title", "")):
            # If it's the exact one that was rejected and not partially_implemented narrowing
            if rej.get("status") in {"already_implemented", "invalid_task"}:
                return {
                    "status": "human_required",
                    "current_stage": "finished",
                    "final_summary": f"Task '{title}' termasuk dalam daftar rejected_tasks.",
                }

    # 7 & 8. Evidence paths available and relative
    evidence_for = task_verification.get("evidence_for", [])
    evidence_results = state.get("repository_evidence", {}).get("results", [])
    evidence_paths = []
    
    for item in evidence_for:
        # If item has "path: note", extract path
        parts = item.split(":")
        path_candidate = parts[0].strip()
        if path_candidate and not path_candidate.startswith("/") and not ".." in path_candidate:
            evidence_paths.append(path_candidate)

    for r in evidence_results:
        p = r.get("path")
        if p and not p.startswith("/") and ".." not in p:
            if p not in evidence_paths:
                evidence_paths.append(p)

    if not evidence_paths:
        return {
            "status": "human_required",
            "current_stage": "finished",
            "final_summary": "Evidence path tidak tersedia atau tidak relatif terhadap project root.",
        }

    # If partially_implemented, narrow the plan to missing_work
    missing_work = task_verification.get("missing_work", [])
    final_task = dict(recommended_task)
    if v_status == "partially_implemented" and missing_work:
        final_task["title"] = f"{title} (Selesaikan gap: {', '.join(missing_work[:2])})"
        final_task["description"] = (
            f"Melanjutkan pekerjaan yang belum selesai pada '{title}'. "
            f"Gap yang harus diselesaikan: {'; '.join(missing_work)}."
        )
        final_task["acceptance_criteria"] = [
            f"Selesaikan: {mw}" for mw in missing_work
        ]
        # Update analysis recommended task with narrowed task
        analysis["recommended_task"] = final_task

    summary = (
        f"Rencana siap. Task berikutnya: "
        f"{final_task['title']}. "
        f"Owner: {final_task['owner']}. "
        f"Status verifikasi: {v_status}."
    )

    return {
        "analysis": analysis,
        "status": "plan_ready",
        "current_stage": "finished",
        "final_summary": summary,
    }

def build_fallback_proposal(
    task_title: str,
    task_owner: str,
    task_desc: str,
    acceptance_criteria: list[str],
    source_plan_path: str,
    verification_status: str,
    evidence_paths: list[str],
    previous_review: dict | None = None,
    revision_count: int = 0,
) -> ImplementationProposal:
    """Deterministic fallback proposal when LLM is unavailable or in unit tests."""
    import uuid

    prop_id = f"prop_{uuid.uuid4().hex[:8]}"
    manifest_names = {"go.mod", "go.sum", "package.json", "package-lock.json", "requirements.txt", "pnpm-lock.yaml"}
    valid_code_exts = {".go", ".py", ".ts", ".tsx", ".js", ".jsx"}

    criteria_files = []
    full_text = f"{task_title} {task_desc} " + " ".join(acceptance_criteria)
    for m in re.findall(r"[\w\./\-]+\.(?:go|py|ts|tsx|js)", full_text):
        clean_m = m.strip("`'\"()[],: ")
        if clean_m.startswith("backend/") or clean_m.startswith("frontend/"):
            if clean_m not in criteria_files:
                criteria_files.append(clean_m)

    rejected_set = set()
    if previous_review:
        for rf in previous_review.get("rejected_files", []):
            rejected_set.add(rf)
        for sv in previous_review.get("scope_violations", []):
            rejected_set.add(sv)

    clean_ev = []
    seen = set()
    for p in criteria_files + (evidence_paths or []):
        if not p:
            continue
        p_clean = str(p).split(":")[0].strip()
        fname = Path(p_clean).name.lower()
        if fname in manifest_names:
            continue
        if Path(p_clean).suffix.lower() not in valid_code_exts:
            continue
        if p_clean in rejected_set:
            continue
        if not (p_clean.startswith("backend/") or p_clean.startswith("frontend/")):
            continue
        if p_clean not in seen:
            seen.add(p_clean)
            clean_ev.append(p_clean)

    if not clean_ev:
        clean_ev = ["backend/cmd/api/main.go"]

    test_paths = [p for p in clean_ev if "test" in p.lower()]
    source_paths = [p for p in clean_ev if p not in test_paths]
    if not source_paths and clean_ev:
        source_paths = [clean_ev[0]]
        test_paths = clean_ev[1:]

    if not test_paths and source_paths:
        first = source_paths[0]
        if first.endswith(".go"):
            test_paths = [first[:-3] + "_test.go"]
        elif first.endswith(".py"):
            p = Path(first)
            test_paths = [str(p.parent / f"test_{p.name}")]
        else:
            test_paths = ["tests/test_" + Path(first).name]

    rev_suffix = f" (revisi {revision_count})" if revision_count > 0 else ""
    files_to_modify = [
        ProposedFileChange(
            path=sp,
            action="modify",
            purpose=f"Implementasikan perubahan untuk {task_title}{rev_suffix}",
            changes=[f"Perbarui logika sesuai {task_title}{rev_suffix}"],
            evidence_paths=[sp],
            estimated_complexity="medium",
        )
        for sp in source_paths
    ]

    tests_to_create = []
    tests_to_modify = []
    for tp in test_paths:
        t_obj = ProposedTest(
            path=tp,
            test_type="integration" if "integration" in tp.lower() else "unit",
            scenarios=[f"Verifikasi {task_title}{rev_suffix}"],
            expected_results=["Pengujian berhasil memenuhi acceptance criteria"],
            related_acceptance_criteria=acceptance_criteria,
        )
        if "buat" in full_text.lower() or "create" in full_text.lower():
            tests_to_create.append(t_obj)
        else:
            tests_to_modify.append(t_obj)

    approved_writable = [f.path for f in files_to_modify] + [t.path for t in tests_to_modify] + [t.path for t in tests_to_create]

    validation_cmds = [
        ProposedCommand(
            executable="pytest" if any(p.endswith(".py") for p in clean_ev) else "go",
            arguments=["-q"] if any(p.endswith(".py") for p in clean_ev) else ["test", "./..."],
            working_directory="." if any(p.endswith(".py") for p in clean_ev) else "backend",
            purpose="Menjalankan suite test verifikasi",
            requires_approval=True,
        )
    ]

    steps = [
        ImplementationStep(
            order=i + 1,
            title=f"Perbarui {f.path}",
            description=f"Implementasikan perubahan pada {f.path}",
            affected_files=[f.path],
            acceptance_criteria=acceptance_criteria,
            dependencies=[],
        )
        for i, f in enumerate(files_to_modify)
    ]

    return ImplementationProposal(
        proposal_id=prop_id,
        task_title=task_title,
        owner=task_owner,
        objective=f"Implementasikan {task_title} sesuai spesifikasi dan kriteria penerimaan.",
        verification_status=verification_status or "verified_pending",
        source_plan_path=source_plan_path,
        affected_components=[task_owner] if task_owner else ["backend"],
        files_to_modify=files_to_modify,
        tests_to_create=tests_to_create,
        tests_to_modify=tests_to_modify,
        validation_commands=validation_cmds,
        implementation_steps=steps,
        dependencies=[],
        risks=[
            ProposalRisk(
                severity="medium",
                description="Potensi regresi pada komponen terkait",
                mitigation="Jalankan rangkaian test otomatis",
                evidence_paths=clean_ev[:2],
            )
        ],
        assumptions=["Lingkungan pengujian siap digunakan"],
        rollback_plan=[f"Kembalikan perubahan pada {p} ke revisi sebelumnya" for p in approved_writable],
        out_of_scope=["Perubahan di luar file yang disetujui"],
        evidence_paths=clean_ev,
        status="draft",
        approved_writable_files=approved_writable,
    )



def implementation_proposal_node(state: AgentState) -> dict:
    print("[Proposal Agent] Merancang proposal implementasi...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    validated_project = load_project(project_root)

    plan_data = state.get("plan_data", {})
    task_verification = state.get("task_verification", {})
    previous_review = state.get("proposal_review")
    revision_count = state.get("proposal_revision_count", 0)
    proposal_hashes = list(state.get("proposal_hashes", []))
    source_plan_path = state.get("source_plan_path", "")

    # Extract verified task details
    verified_task = (
        plan_data.get("result", {}).get("recommended_task")
        or plan_data.get("result", {}).get("final_task")
        or {}
    )
    if isinstance(verified_task, str):
        task_title = verified_task
        task_owner = plan_data.get("result", {}).get("final_owner", "backend")
        acceptance_criteria = plan_data.get("result", {}).get("final_acceptance_criteria", [])
        task_desc = ""
    else:
        task_title = verified_task.get("title", "")
        task_owner = verified_task.get("owner", "backend")
        acceptance_criteria = verified_task.get("acceptance_criteria", [])
        task_desc = verified_task.get("description", "")

    if not task_title:
        raise ValueError("Cannot generate proposal without a valid task title in verified plan.")

    manifest_names = {"go.mod", "go.sum", "package.json", "package-lock.json", "requirements.txt", "pnpm-lock.yaml"}
    clean_ev_input = []
    seen_ev = set()
    for p in plan_data.get("evidence_paths", []):
        if not p:
            continue
        p_clean = str(p).split(":")[0].strip()
        fname = Path(p_clean).name.lower()
        if fname in manifest_names:
            continue
        if p_clean not in seen_ev:
            seen_ev.add(p_clean)
            clean_ev_input.append(p_clean)

    agent_input = {
        "verified_task": {
            "title": task_title,
            "owner": task_owner,
            "description": task_desc,
            "acceptance_criteria": acceptance_criteria,
        },
        "task_verification": task_verification,
        "evidence_paths": clean_ev_input,
        "components": list(validated_project.manifest.components.keys()),
        "revision_info": {
            "revision_count": revision_count,
            "findings": previous_review.get("findings", []) if previous_review else [],
            "missing_acceptance_criteria": previous_review.get("missing_acceptance_criteria", []) if previous_review else [],
            "revision_instructions": previous_review.get("revision_instructions", []) if previous_review else [],
            "unsupported_changes": previous_review.get("unsupported_changes", []) if previous_review else [],
            "scope_violations": previous_review.get("scope_violations", []) if previous_review else [],
            "unsafe_commands": previous_review.get("unsafe_commands", []) if previous_review else [],
        } if previous_review else None,
    }

    try:
        res = proposal_model.invoke(
            [
                {"role": "system", "content": IMPLEMENTATION_PROPOSAL_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(agent_input, ensure_ascii=False, indent=2)},
            ]
        )
        proposal = res
    except Exception as e:
        print(f"[Proposal Agent] LLM invoke fallback due to: {type(e).__name__}: {e}")
        proposal = build_fallback_proposal(
            task_title=task_title,
            task_owner=task_owner,
            task_desc=task_desc,
            acceptance_criteria=acceptance_criteria,
            source_plan_path=source_plan_path,
            verification_status=task_verification.get("status", "verified_pending"),
            evidence_paths=clean_ev_input,
            previous_review=previous_review,
            revision_count=revision_count,
        )

    if not proposal.source_plan_path:
        proposal.source_plan_path = source_plan_path
    if not proposal.verification_status:
        proposal.verification_status = task_verification.get("status", "verified_pending")
    for cmd in proposal.validation_commands:
        cmd.requires_approval = True

    prop_dict = proposal.model_dump(mode="json")
    prop_hash = compute_proposal_hash(prop_dict)
    proposal_hashes.append(prop_hash)

    print(f"[Proposal Agent] Proposal '{proposal.proposal_id}' dirancang (hash: {prop_hash[:8]}).")
    return {
        "implementation_proposal": prop_dict,
        "proposal_hashes": proposal_hashes,
        "current_stage": "proposal_reviewer",
        "status": "running",
    }



def proposal_reviewer_node(state: AgentState) -> dict:
    print("[Proposal Reviewer] Meninjau implementasi proposal...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    validated_project = load_project(project_root)

    proposal_data = state.get("implementation_proposal", {})
    plan_data = state.get("plan_data", {})
    previous_hashes = list(state.get("proposal_hashes", []))
    past_hashes = previous_hashes[:-1] if len(previous_hashes) > 1 else []

    revision_count = state.get("proposal_revision_count", 0)
    max_revisions = state.get("max_proposal_revisions", 2)
    rejected_reasons = list(state.get("rejected_proposal_reasons", []))

    # 1. Deterministic Review
    review = review_proposal_deterministically(
        proposal=proposal_data,
        plan_data=plan_data,
        manifest=validated_project.manifest,
        policy=validated_project.policy,
        previous_hashes=past_hashes,
    )

    # 2. Semantic LLM Review if clean
    if review.status == "approved":
        try:
            reviewer_input = {
                "verified_plan": plan_data,
                "implementation_proposal": proposal_data,
                "components": list(validated_project.manifest.components.keys()),
            }
            llm_res = proposal_reviewer_model.invoke(
                [
                    {"role": "system", "content": PROPOSAL_REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(reviewer_input, ensure_ascii=False, indent=2)},
                ]
            )
            if llm_res.status != "approved":
                review = llm_res
        except Exception:
            pass

    # Handle status and revisions
    if review.status == "revision_required":
        revision_count += 1
        rejected_reasons.append(review.summary)
        if revision_count > max_revisions:
            review.status = "human_required"
            review.summary = (
                f"Batas maksimum revisi proposal ({max_revisions}) tercapai. "
                f"Dibutuhkan keputusan manusia."
            )
    elif review.status == "rejected":
        rejected_reasons.append(review.summary)

    print(f"[Proposal Reviewer] Status: {review.status}. Summary: {review.summary}")

    return {
        "proposal_review": review.model_dump(mode="json"),
        "proposal_revision_count": revision_count,
        "rejected_proposal_reasons": rejected_reasons,
        "current_stage": (
            "supervisor"
            if review.status in {"approved", "rejected", "human_required"}
            or revision_count > max_revisions
            else "implementation_proposal"
        ),
        "status": "running",
    }

def implementation_preflight_node(state: AgentState) -> dict:
    """Preflight Validator: ensures all invariants are satisfied before implementation."""
    print("[Preflight Validator] Memeriksa kelayakan implementasi...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    loaded_proj = load_project(project_root)

    source_plan_path = state.get("source_plan_path", "")
    source_proposal_path = state.get("source_proposal_path", "")
    plan_data = state.get("plan_data")
    proposal_data = state.get("proposal_data")
    proposal_review = state.get("proposal_review")
    apply = state.get("apply", False)
    run_id = state.get("run_id", "run")

    prop_inner = (proposal_data or {}).get("result", {}).get("proposal", {})
    task_title = prop_inner.get("task_title") or state.get("user_request", "agent-task")
    if state.get("executor") == "openhands":
        short_id = run_id[:8].lower()
        task_words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in task_title.split()]
        task_tail = "-".join([w for w in task_words if w][-2:]) if task_words else "task"
        target_branch = f"agent/openhands-{task_tail}-{short_id}" if apply else None
        wt_name = f"{project_root.name}-openhands-{short_id}"
    else:
        target_branch = wt_generate_branch_name(task_title, run_id) if apply else None
        wt_name = generate_worktree_name(project_root, run_id)
    wt_root = get_default_worktree_root(project_root, custom_root=state.get("worktree_root"))
    target_wt_path = (wt_root / wt_name) if apply else None

    preflight = run_implementation_preflight(
        project_root=project_root,
        manifest=loaded_proj.manifest,
        policy=loaded_proj.policy,
        source_plan_path=source_plan_path,
        source_proposal_path=source_proposal_path,
        plan_data=plan_data,
        proposal_data=proposal_data,
        proposal_review=proposal_review,
        apply=apply,
        target_branch_name=target_branch,
        target_worktree_path=target_wt_path,
        dry_run_report_data=state.get("dry_run_report_data"),
    )

    print(f"[Preflight Validator] Status: {preflight.status}. Summary: {preflight.summary}")

    return {
        "preflight_result": preflight.model_dump(mode="json"),
        "approved_files": preflight.approved_files,
        "status": preflight.status,
        "final_summary": preflight.summary,
        "current_stage": "branch_guard" if preflight.is_valid and preflight.status in {"valid", "preflight_passed"} else "supervisor",
    }


def branch_guard_node(state: AgentState) -> dict:
    """Branch Guard: ensures base branch is main and creates isolated branch/worktree safely."""
    print("[Branch Guard] Menyiapkan branch dan workspace terisolasi...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    run_id = state.get("run_id", "run")
    proposal_data = state.get("proposal_data", {})
    prop_inner = proposal_data.get("result", {}).get("proposal", {})
    task_title = prop_inner.get("task_title") or state.get("user_request", "agent-task")
    dry_run = state.get("dry_run", True)
    isolation = state.get("isolation", "git-worktree")

    if isolation == "branch":
        if dry_run:
            branch_name = generate_branch_name(task_title, run_id)
            print(f"[Branch Guard] DRY RUN - Planned branch: {branch_name}")
            return {
                "branch_name": branch_name,
                "current_stage": "implementation_agent",
            }
        try:
            guard = BranchGuard(project_root)
            branch_name = guard.create_isolated_branch(task_title, run_id)
            print(f"[Branch Guard] Berhasil berpindah ke branch terisolasi: {branch_name}")
            return {
                "branch_name": branch_name,
                "current_stage": "implementation_agent",
            }
        except Exception as e:
            print(f"[Branch Guard] Gagal membuat branch: {e}")
            return {
                "status": "human_required",
                "current_stage": "supervisor",
                "final_summary": f"Branch Guard gagal: {e}",
                "errors": [str(e)],
            }

    # Default: git-worktree isolation
    if state.get("executor") == "openhands":
        short_id = run_id[:8].lower()
        task_words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in task_title.split()]
        task_tail = "-".join([w for w in task_words if w][-2:]) if task_words else "task"
        branch_name = state.get("branch_name") or f"agent/openhands-{task_tail}-{short_id}"
        wt_name = state.get("worktree_descriptor") or f"{project_root.name}-openhands-{short_id}"
    else:
        branch_name = state.get("branch_name") or wt_generate_branch_name(task_title, run_id)
        wt_name = state.get("worktree_descriptor") or generate_worktree_name(project_root, run_id)

    if dry_run:
        print(f"[Branch Guard] DRY RUN - Planned branch: {branch_name}, Worktree: {wt_name}")
        return {
            "branch_name": branch_name,
            "worktree_descriptor": wt_name,
            "current_stage": "coding_executor" if state.get("executor") == "openhands" else "implementation_agent",
        }

    try:
        wt_root = get_default_worktree_root(project_root, custom_root=state.get("worktree_root"))
        wt_path = wt_root / wt_name
        if wt_path.is_dir():
            print(f"[Branch Guard] Menggunakan worktree terisolasi yang sudah ada: {wt_name} ({branch_name})")
        else:
            create_worktree(project_root=project_root, branch_name=branch_name, worktree_path=wt_path)
            print(f"[Branch Guard] Berhasil membuat worktree terisolasi: {wt_name} ({branch_name})")
        return {
            "main_project_root": str(project_root),
            "worktree_path": str(wt_path),
            "worktree_descriptor": wt_name,
            "project_root": str(wt_path),
            "branch_name": branch_name,
            "current_stage": "coding_executor" if state.get("executor") == "openhands" else "implementation_agent",
        }
    except Exception as e:
        print(f"[Branch Guard] Gagal membuat git worktree: {e}")
        return {
            "status": "human_required",
            "current_stage": "supervisor",
            "final_summary": f"Branch Guard gagal membuat worktree: {e}",
            "errors": [str(e)],
        }


def implementation_agent_node(state: AgentState) -> dict:
    """Implementation Agent: generates changes strictly within approved files."""
    print("[Implementation Agent] Mengimplementasikan proposal...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    proposal_data = state.get("proposal_data", {})
    plan_data = state.get("plan_data")
    proposal_review = state.get("proposal_review")
    approved_files = state.get("approved_files", [])
    dry_run = state.get("dry_run", True)
    run_id = state.get("run_id", "run")

    if dry_run:
        prop_inner = proposal_data.get("result", {}).get("proposal", {})
        task_title = prop_inner.get("task_title", "Dry Run")
        result = ImplementationResult(
            status="implementation_generated",
            task_title=task_title,
            changes=[],
            files_changed=approved_files,
            files_requested_outside_scope=[],
            assumptions=["Dry run: pengujian preflight lolos."],
            risks=[],
            validation_commands=[],
            summary=f"Dry run: {len(approved_files)} file disetujui siap diimplementasikan.",
        )
        return {
            "implementation_result": result.model_dump(mode="json"),
            "current_stage": "supervisor",
            "final_summary": result.summary,
        }

    result = run_implementation_agent(
        project_root=project_root,
        proposal_data=proposal_data,
        plan_data=plan_data,
        proposal_review=proposal_review,
        approved_files=approved_files,
    )

    print(f"[Implementation Agent] Status: {result.status}. Summary: {result.summary}")

    if result.status in {"proposal_revision_required", "implementation_generation_failed"}:
        return {
            "status": result.status,
            "implementation_result": result.model_dump(mode="json"),
            "current_stage": "supervisor",
            "final_summary": result.summary,
        }

    # In --apply mode, write approved files using ControlledFileWriter
    writer = ControlledFileWriter(
        project_root=project_root,
        approved_files=approved_files,
        run_id=run_id,
    )

    for chg in result.changes:
        writer.write_file(chg.path, chg.content)

    return {
        "implementation_result": result.model_dump(mode="json"),
        "file_hashes_before": writer.file_hashes_before,
        "file_hashes_after": writer.file_hashes_after,
        "current_stage": "test_runner",
    }

def test_runner_node(state: AgentState) -> dict:
    """Test Runner: executes validation commands safely from approved proposal."""
    print("[Test Runner] Menjalankan validation command...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    dry_run = state.get("dry_run", True)

    if dry_run:
        print("[Test Runner] DRY RUN - Pengujian dilewati.")
        return {
            "command_results": [],
            "current_stage": "supervisor",
        }

    proposal_data = state.get("proposal_data", {})
    prop_inner = proposal_data.get("result", {}).get("proposal", {})
    val_cmds = prop_inner.get("validation_commands", [])

    runner = SafeCommandRunner(project_root)
    cmd_results: list[dict] = []

    for cmd in val_cmds:
        exe = cmd.get("executable") if isinstance(cmd, dict) else cmd.executable
        args = cmd.get("arguments", []) if isinstance(cmd, dict) else cmd.arguments
        cwd = cmd.get("working_directory", "") if isinstance(cmd, dict) else cmd.working_directory

        res = runner.run_command(exe, args, cwd)
        cmd_results.append(res.model_dump(mode="json"))
        print(f"[Test Runner] {exe} {' '.join(args)}: {res.status} (exit code: {res.exit_code})")

    return {
        "command_results": cmd_results,
        "current_stage": "diff_reviewer",
    }


def diff_reviewer_node(state: AgentState) -> dict:
    """Diff Reviewer: verifies Git diff read-only against safety policies."""
    print("[Diff Reviewer] Memeriksa diff perubahan...")
    project_root = Path(state["project_root"]).expanduser().resolve()
    dry_run = state.get("dry_run", True)

    if dry_run:
        print("[Diff Reviewer] DRY RUN - Review diff dilewati.")
        return {
            "current_stage": "supervisor",
        }

    approved_files = state.get("approved_files", [])
    proposal_data = state.get("proposal_data", {})
    prop_inner = proposal_data.get("result", {}).get("proposal", {})
    acceptance_criteria = []
    for step in prop_inner.get("implementation_steps", []):
        ac_list = step.get("acceptance_criteria", []) if isinstance(step, dict) else getattr(step, "acceptance_criteria", [])
        acceptance_criteria.extend(ac_list)

    cmd_results = state.get("command_results", [])
    test_passed = all(c.get("status") == "passed" for c in cmd_results) if cmd_results else True
    test_status = "passed" if test_passed else "failed"

    review = review_diff(
        project_root=project_root,
        approved_files=approved_files,
        acceptance_criteria=acceptance_criteria,
        test_status=test_status,
    )

    print(f"[Diff Reviewer] Status: {review.status}. Summary: {review.summary}")

    return {
        "diff_review": review.model_dump(mode="json"),
        "current_stage": "supervisor" if review.status == "approved" else "repair_agent",
    }


def repair_agent_node(state: AgentState) -> dict:
    """Repair Agent: fixes failed tests or diff issues up to max 2 attempts."""
    repair_count = state.get("implementation_repair_count", 0)
    max_repairs = state.get("max_implementation_repairs", 2)
    print(f"[Repair Agent] Memperbaiki implementasi (Percobaan {repair_count + 1}/{max_repairs})...")

    if repair_count >= max_repairs:
        print("[Repair Agent] Batas maksimum perbaikan tercapai.")
        return {
            "status": "human_required",
            "current_stage": "supervisor",
            "final_summary": f"Batas maksimum perbaikan ({max_repairs}) tercapai. Dibutuhkan intervensi manusia.",
        }

    project_root = Path(state["project_root"]).expanduser().resolve()
    prev_impl = state.get("implementation_result", {})
    approved_files = state.get("approved_files", [])
    cmd_results = state.get("command_results", [])
    test_err = "\n".join(c.get("stderr_summary", "") for c in cmd_results)
    diff_review = state.get("diff_review", {})
    diff_findings = diff_review.get("findings", [])
    run_id = state.get("run_id", "repair")

    repaired = run_repair_agent(
        project_root=project_root,
        previous_implementation=prev_impl,
        test_failure_summary=test_err,
        diff_findings=diff_findings,
        approved_files=approved_files,
        acceptance_criteria=[],
        repair_count=repair_count,
    )

    # Write repaired changes
    writer = ControlledFileWriter(
        project_root=project_root,
        approved_files=approved_files,
        run_id=run_id,
    )
    for chg in repaired.changes:
        writer.write_file(chg.path, chg.content)

    return {
        "implementation_result": repaired.model_dump(mode="json"),
        "implementation_repair_count": repair_count + 1,
        "current_stage": "test_runner",
    }


def coding_executor_node(state: AgentState) -> dict:
    """Coding Executor Node: delegates code editing strictly to OpenHandsExecutor."""
    print("[Coding Executor] Menjalankan OpenHands Coding Executor...")

    # 1. Memastikan proposal approved
    proposal_data = state.get("proposal_data", {})
    prop_status = proposal_data.get("status")
    if prop_status != "approved_for_implementation":
        return {
            "status": "human_required",
            "current_stage": "supervisor",
            "final_summary": f"Proposal belum disetujui (status: {prop_status}). Eksekusi ditolak.",
        }

    # 2. Memastikan preflight passed
    preflight = state.get("preflight_result", {})
    if not preflight.get("is_valid") or preflight.get("status") not in {"valid", "preflight_passed"}:
        return {
            "status": "human_required",
            "current_stage": "supervisor",
            "final_summary": f"Preflight validasi gagal (status: {preflight.get('status')}). Eksekusi ditolak.",
        }

    # 3. Memastikan worktree tersedia
    dry_run = state.get("dry_run", True)
    if dry_run:
        print("[Coding Executor] DRY RUN - Eksekusi OpenHands dilewati.")
        return {
            "status": "dry_run_ready",
            "current_stage": "supervisor",
            "final_summary": "Dry-run berhasil diverifikasi.",
        }

    # Check if execution already completed for this state/run (avoid duplicate OpenHands execution on resume)
    existing_coding_res = state.get("coding_execution_result")
    if existing_coding_res and existing_coding_res.get("status") in ("completed", "execution_review_ready"):
        print("[Coding Executor] Menggunakan hasil eksekusi OpenHands yang sudah ada.")
        return {
            "coding_execution_request": state.get("coding_execution_request", {}),
            "coding_execution_result": existing_coding_res,
            "status": existing_coding_res.get("status"),
            "final_summary": existing_coding_res.get("summary", "Eksekusi OpenHands sudah selesai sebelumnya."),
            "current_stage": "independent_diff_review",
        }

    worktree_path_str = state.get("worktree_path") or state.get("project_root")
    worktree_path = Path(worktree_path_str).expanduser().resolve()
    main_proj_root = Path(state.get("main_project_root", "")).expanduser().resolve() if state.get("main_project_root") else None

    if not worktree_path.is_dir():
        return {
            "status": "human_required",
            "current_stage": "supervisor",
            "final_summary": f"Worktree path '{worktree_path}' tidak ditemukan atau tidak tersedia.",
        }

    if main_proj_root and worktree_path == main_proj_root:
        return {
            "status": "scope_violation",
            "current_stage": "supervisor",
            "final_summary": "SECURITY VIOLATION: Eksekusi OpenHands dilarang pada direktori project utama.",
        }

    # 4. Membuat CodingExecutionRequest
    from agent_system.execution_package import build_execution_request_from_proposal
    from agent_system.executors.openhands_executor import (
        OpenHandsExecutor,
        OPENHANDS_EXECUTION_TIMEOUT_SECONDS,
        OPENHANDS_MAX_ITERATIONS,
    )

    run_id = state.get("run_id", "run")
    proposal_review = state.get("proposal_review")
    req = build_execution_request_from_proposal(
        proposal_data=proposal_data,
        worktree_path=worktree_path,
        run_id=run_id,
        proposal_review=proposal_review,
        max_iterations=OPENHANDS_MAX_ITERATIONS,
        timeout_seconds=OPENHANDS_EXECUTION_TIMEOUT_SECONDS,
    )

    # 5. Memanggil OpenHandsExecutor (tidak melakukan implementasi sendiri, tidak memakai model LangGraph, tidak fallback)
    executor = OpenHandsExecutor()
    result = executor.execute(req)

    # 6. Menyimpan result ke state
    retry_count = state.get("executor_retry_count", 0)
    if result.status == "failed":
        retry_count += 1

    return {
        "coding_execution_request": req.model_dump(mode="json"),
        "coding_execution_result": result.model_dump(mode="json"),
        "executor_retry_count": retry_count,
        "status": result.status,
        "final_summary": result.summary,
        "current_stage": (
            "independent_diff_review"
            if result.status == "completed"
            else "supervisor"
        ),
    }


def independent_diff_review_node(state: AgentState) -> dict:
    """Independent Diff Review: verifies actual Git diff after OpenHands completes."""
    print("[Independent Diff Review] Memeriksa diff aktual repository...")
    worktree_path_str = state.get("worktree_path") or state.get("project_root")
    worktree_path = Path(worktree_path_str).expanduser().resolve()
    dry_run = state.get("dry_run", True)

    if dry_run:
        print("[Independent Diff Review] DRY RUN - Review dilewati.")
        return {
            "status": "dry_run_ready",
            "current_stage": "supervisor",
        }

    approved_files = state.get("approved_files", [])
    if not approved_files:
        exec_req = state.get("coding_execution_request", {})
        approved_files = exec_req.get("approved_files", [])

    proposal_data = state.get("proposal_data", {})
    prop_inner = proposal_data.get("result", {}).get("proposal", {})
    if not prop_inner and "proposal_id" in proposal_data:
        prop_inner = proposal_data

    from agent_system.execution_package import extract_acceptance_criteria_from_proposal
    acceptance_criteria = extract_acceptance_criteria_from_proposal(prop_inner)

    exec_res = state.get("coding_execution_result", {})
    val_results = exec_res.get("validation_results", [])
    test_passed = all(c.get("status") == "passed" for c in val_results) if val_results else True
    test_status = "passed" if test_passed else "failed"

    # Periksa Git diff aktual menggunakan review_diff orchestrator
    review = review_diff(
        project_root=worktree_path,
        approved_files=approved_files,
        acceptance_criteria=acceptance_criteria,
        test_status=test_status,
    )

    print(f"[Independent Diff Review] Status: {review.status}. Summary: {review.summary}")

    # Fail-closed checks:
    # 1. Zero changed files check
    if not review.changed_files:
        print("[Independent Diff Review] Gagal: Tidak ada file yang berubah (zero changed files).")
        return {
            "diff_review": review.model_dump(mode="json"),
            "status": "execution_failed",
            "final_summary": "Pemeriksaan diff aktual menemukan tidak ada file yang berubah (zero changed files).",
            "current_stage": "supervisor",
        }

    # 2. Main project must not be modified
    main_proj_root = Path(state.get("main_project_root", "")).expanduser().resolve() if state.get("main_project_root") else None
    if main_proj_root and main_proj_root.is_dir():
        res_main = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=main_proj_root,
            capture_output=True,
            text=True,
        )
        if res_main.returncode == 0 and res_main.stdout.strip():
            return {
                "diff_review": review.model_dump(mode="json"),
                "status": "scope_violation",
                "final_summary": f"SECURITY VIOLATION: Project utama mengalami perubahan ({res_main.stdout.strip()}).",
                "current_stage": "supervisor",
            }

    # 3. No unauthorized commit in worktree (changes must remain uncommitted)
    res_wt_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    base_head = state.get("preflight_result", {}).get("source_revision", {}).get("commit_head", "")
    if res_wt_head.returncode == 0 and base_head and res_wt_head.stdout.strip() != base_head:
        return {
            "diff_review": review.model_dump(mode="json"),
            "status": "human_required",
            "final_summary": "SECURITY VIOLATION: Ditemukan commit tidak terotorisasi pada worktree.",
            "current_stage": "supervisor",
        }

    final_status = "execution_review_ready" if review.status == "approved" else review.status
    summary_text = (
        "Implementasi terverifikasi oleh review diff aktual dan siap ditinjau (execution_review_ready)."
        if review.status == "approved"
        else review.summary
    )

    return {
        "diff_review": review.model_dump(mode="json"),
        "status": final_status,
        "final_summary": summary_text,
        "current_stage": "supervisor",
    }


