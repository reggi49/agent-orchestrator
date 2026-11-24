import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_system.artifact_store import ProjectArtifactStore, sanitize_artifact_payload
from agent_system.atomic_state import atomic_write_json, read_json_with_backup_recovery
from agent_system.config import get_settings
from agent_system.error_reporting import sanitize_text
from agent_system.objective_alignment import normalize_objective_to_requirements
from agent_system.project_inspector import inspect_project
from agent_system.project_loader import load_project
from agent_system.risk_classifier import classify_pre_execution_risk
from agent_system.schemas import ProgramEpic, ProgramState, ProgramTask
from agent_system.scope_and_overlap import analyze_backlog_overlaps_and_enforce_dependencies


def get_target_head_revision(project_root: Path) -> str:
    """Get target branch HEAD commit hash safely."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "HEAD"


def _categorize_task_files(
    task_text: str,
    components: list[str],
    project_root: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Separate files into readable_files, expected_writable_files, conditionally_writable_files."""
    expected_writable: list[str] = []
    conditionally_writable: list[str] = []
    readable: list[str] = []

    # Find explicitly mentioned files in text
    file_matches = re.findall(r"[\w\./\-]+\.(?:go|py|ts|tsx|js|json|yaml|yml|md)", task_text)
    for m in file_matches:
        clean = m.strip("`'\"()[],: ")
        if clean.endswith("_test.go") or clean.startswith("test_") or "/tests/" in clean:
            if clean not in expected_writable:
                expected_writable.append(clean)
        elif clean.endswith(".md") or clean.endswith(".yaml"):
            if clean not in readable:
                readable.append(clean)
        elif clean.startswith("backend/") or clean.startswith("frontend/"):
            if clean not in expected_writable:
                expected_writable.append(clean)

    # Defaults if none mentioned
    primary = components[0] if components else "backend"
    from agent_system.risk_classifier import DEPENDENCY_MANIFEST_NAMES, OBSERVABILITY_PATTERNS, READINESS_PATTERNS

    is_obs = any(pat.search(task_text) for pat in OBSERVABILITY_PATTERNS) or any(pat.search(task_text) for pat in READINESS_PATTERNS)

    if not expected_writable and project_root and project_root.is_dir():
        comp_dir = project_root / primary
        if comp_dir.is_dir():
            for p in sorted(comp_dir.glob("**/*.*")):
                if p.is_file() and not p.name.startswith("."):
                    rel = str(p.relative_to(project_root)).replace("\\", "/")
                    fname = p.name.lower()
                    if fname in DEPENDENCY_MANIFEST_NAMES or "migration" in rel.lower():
                        continue
                    if is_obs:
                        # For observability tasks, avoid financial/auth files
                        if any(k in rel.lower() for k in ("wallet", "balance", "transaction", "payment", "auth", "token")):
                            continue
                        # Prefer files related to handler, http, test, cmd
                        if any(k in rel.lower() for k in ("health", "ready", "ping", "metrics", "handler", "api", "cmd", "test")):
                            if rel not in expected_writable:
                                expected_writable.append(rel)
                    else:
                        if rel not in expected_writable:
                            expected_writable.append(rel)
                    if len(expected_writable) >= 3:
                        break

    if not expected_writable:
        if is_obs and primary == "backend":
            expected_writable.append("backend/internal/handler/http/health_handler.go")
            expected_writable.append("backend/internal/handler/http/health_handler_test.go")
        else:
            expected_writable.append(f"{primary}/cmd/api/main.go" if primary == "backend" else f"{primary}/src/app.py")
            conditionally_writable.append(f"{primary}/internal/service/" if primary == "backend" else f"{primary}/src/")

    return readable, expected_writable, conditionally_writable


class ProgramManager:
    """Program Manager orchestrates high-level CEO objectives into epics, tasks, and reports."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).expanduser().resolve()
        self.validated_project = load_project(self.project_root)
        self.artifact_store = ProjectArtifactStore(self.project_root, self.validated_project.manifest)

    def initialize_program(
        self,
        objective_text: str,
        program_id: str | None = None,
    ) -> ProgramState:
        """Initialize a new program: inspect repo, normalize objective, decompose, write reports."""
        now_iso = datetime.now(timezone.utc).isoformat()
        short_id = (program_id or uuid.uuid4().hex)[:8].lower()
        project_id = self.validated_project.manifest.project.id

        # 1. Inspect actual repository
        inspection = inspect_project(self.project_root)
        components = list(inspection.components.keys()) if hasattr(inspection, "components") else list(self.validated_project.manifest.components.keys())

        # 2. Normalize Objective to Requirement IDs
        clean_objective = sanitize_text(objective_text).strip()
        requirements = normalize_objective_to_requirements(clean_objective)
        req_ids = [r["id"] for r in requirements]

        # 3. Base revision
        base_head = get_target_head_revision(self.project_root)

        # 4. Decompose objective into Epics & Tasks
        epics, tasks = self._decompose_objective(
            objective=clean_objective,
            project_id=project_id,
            components=components,
            inspection=inspection,
            req_ids=req_ids,
            base_head=base_head,
        )

        # 5. Overlap Detection & Cycle Prevention on Backlog
        overlap_info, is_valid_dag, cycle_error = analyze_backlog_overlaps_and_enforce_dependencies(tasks)

        dependencies: dict[str, list[str]] = {}
        task_statuses: dict[str, str] = {}
        attempts: dict[str, int] = {}

        for t in tasks:
            dependencies[t.task_id] = list(t.dependencies)
            task_statuses[t.task_id] = t.status
            attempts[t.task_id] = 0

        # Refresh initial readiness
        self._refresh_statuses(tasks)
        for t in tasks:
            task_statuses[t.task_id] = t.status

        initial_status = "ready" if is_valid_dag else "blocked"
        blockers = [cycle_error] if cycle_error else []

        state = ProgramState(
            program_id=short_id,
            project_id=project_id,
            objective=clean_objective,
            status=initial_status,
            current_phase="Phase 1: Implementation & Verification",
            requirements=requirements,
            epics=epics,
            tasks=tasks,
            dependencies=dependencies,
            task_statuses=task_statuses,
            attempts=attempts,
            active_task=None,
            active_run_id=None,
            worktree_reference=None,
            last_checkpoint="Program initialized",
            next_action=(
                f"python main.py ceo --project {self.project_root} --program {short_id} --autonomous"
                if is_valid_dag
                else "Resolve dependency cycle blocker"
            ),
            blockers=blockers,
            overlap_analysis=overlap_info,
            state_revision=1,
            created_at=now_iso,
            updated_at=now_iso,
        )

        # 6. Persist program directory and reports
        self.save_program_state(state)
        self.write_all_markdown_reports(state)

        return state

    def get_program_dir(self, program_id: str) -> Path:
        """Get .agent/programs/<program_id>/ directory."""
        prog_dir = self.project_root / ".agent" / "programs" / program_id
        prog_dir.mkdir(parents=True, exist_ok=True)
        return prog_dir

    def save_program_state(self, state: ProgramState) -> Path:
        """Save machine-readable state atomically to .agent/artifacts/programs/program_<id>.json."""
        state.updated_at = datetime.now(timezone.utc).isoformat()
        for t in state.tasks:
            state.task_statuses[t.task_id] = t.status
            state.attempts[t.task_id] = t.attempt_count

        short_id = state.program_id[:8].lower()
        payload = sanitize_artifact_payload(state.model_dump(mode="json"))
        payload["status"] = state.status

        # Enforce atomic write with monotonic state_revision and fsync
        programs_dir = self.project_root / ".agent" / "artifacts" / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        target_file = programs_dir / f"program_{short_id}.json"

        atomic_write_json(target_file, payload, backup=True, monotonic_revision=True)
        return target_file

    def load_program_state(self, program_arg: str = "latest") -> tuple[Path, ProgramState]:
        """Load ProgramState from .agent/artifacts/programs/ with backup recovery."""
        progs_dir = self.project_root / ".agent" / "artifacts" / "programs"
        if not progs_dir.is_dir():
            raise FileNotFoundError(f"Direktori programs '{progs_dir}' tidak ditemukan.")

        prog_files = sorted(progs_dir.glob("program_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not prog_files:
            raise FileNotFoundError(f"Tidak ditemukan file program di '{progs_dir}'.")

        if program_arg in ("latest", "", None):
            target_file = prog_files[0]
        else:
            clean_name = program_arg if program_arg.endswith(".json") else f"{program_arg}.json"
            target_file = progs_dir / clean_name
            if not target_file.is_file() and not program_arg.startswith("program_"):
                target_file = progs_dir / f"program_{clean_name}"
            if not target_file.is_file():
                short = program_arg[:8].lower()
                short_file = progs_dir / f"program_{short}.json"
                if short_file.is_file():
                    target_file = short_file

        if not target_file.is_file():
            raise FileNotFoundError(f"Program artifact '{program_arg}' tidak ditemukan di '{progs_dir}'.")

        raw_data, _ = read_json_with_backup_recovery(target_file, program_id=program_arg)
        if isinstance(raw_data, dict) and "result" in raw_data and isinstance(raw_data["result"], dict):
            data = dict(raw_data["result"])
            data["status"] = raw_data.get("status") or data.get("status", "planning")
        else:
            data = raw_data
        state = ProgramState.model_validate(data)
        return target_file, state

    def _decompose_objective(
        self,
        objective: str,
        project_id: str,
        components: list[str],
        inspection: Any,
        req_ids: list[str],
        base_head: str,
    ) -> tuple[list[ProgramEpic], list[ProgramTask]]:
        """Decompose objective into Epics and Tasks using LLM with deterministic fallback."""
        settings = get_settings()
        if settings.has_api_key and settings.llm_model:
            try:
                from langchain_openai import ChatOpenAI
                from pydantic import BaseModel, Field

                class DecomposedTask(BaseModel):
                    task_id: str
                    title: str
                    description: str
                    acceptance_criteria: list[str] = Field(default_factory=list)
                    dependencies: list[str] = Field(default_factory=list)
                    owner_role: str = "backend"

                class DecomposedEpic(BaseModel):
                    epic_id: str
                    title: str
                    description: str
                    tasks: list[DecomposedTask] = Field(default_factory=list)

                class DecomposedProgram(BaseModel):
                    epics: list[DecomposedEpic] = Field(default_factory=list)

                prompt = (
                    f"Anda adalah Chief Program Manager untuk proyek {project_id}.\n"
                    f"Komponen proyek tersedia: {components}.\n"
                    f"Requirement IDs: {req_ids}.\n"
                    f"Tujuan CEO:\n{objective}\n\n"
                    "Pecah tujuan tersebut menjadi 1-3 Epic dan 2-5 Task yang konkret, dapat diuji secara bertahap, "
                    "dan memiliki dependensi yang jelas. Setiap task harus memiliki acceptance criteria yang spesifik."
                )

                model_kwargs: dict[str, Any] = {
                    "model": settings.llm_model,
                    "api_key": settings.llm_api_key,
                    "temperature": 0.0,
                }
                if settings.llm_base_url:
                    model_kwargs["base_url"] = settings.llm_base_url

                decomp_model = ChatOpenAI(**model_kwargs).with_structured_output(DecomposedProgram)
                res = decomp_model.invoke(prompt)

                if res and res.epics:
                    epics: list[ProgramEpic] = []
                    tasks: list[ProgramTask] = []
                    for e in res.epics:
                        t_ids = [t.task_id for t in e.tasks]
                        epics.append(ProgramEpic(
                            epic_id=e.epic_id,
                            title=e.title,
                            description=e.description,
                            status="pending",
                            tasks=t_ids,
                            mapped_requirement_ids=req_ids,
                        ))
                        for t in e.tasks:
                            r_files, w_files, c_files = _categorize_task_files(f"{t.title} {t.description}", components, self.project_root)
                            pre_risk = classify_pre_execution_risk(
                                request=f"{t.title} {t.description}",
                                expected_writable_files=w_files,
                            )["level"]

                            tasks.append(
                                ProgramTask(
                                    task_id=t.task_id,
                                    epic_id=e.epic_id,
                                    title=t.title,
                                    description=t.description,
                                    acceptance_criteria=t.acceptance_criteria or ["Implementasi berhasil memenuhi spesifikasi"],
                                    dependencies=t.dependencies,
                                    owner_role=t.owner_role or "backend",
                                    risk_level=pre_risk,
                                    pre_risk=pre_risk,
                                    status="pending",
                                    mapped_requirement_ids=req_ids[:2] or ["FR-001"],
                                    readable_files=r_files,
                                    expected_writable_files=w_files,
                                    conditionally_writable_files=c_files,
                                    planned_base_revision=base_head,
                                )
                            )
                    if tasks:
                        return epics, tasks
            except Exception:
                pass

        # Deterministic decomposition fallback
        return self._deterministic_decomposition(objective, components, req_ids, base_head)

    def _deterministic_decomposition(
        self,
        objective: str,
        components: list[str],
        req_ids: list[str],
        base_head: str,
    ) -> tuple[list[ProgramEpic], list[ProgramTask]]:
        """Reliable deterministic breakdown of objective into epics and sequential tasks."""
        primary_role = "backend" if "backend" in components else (components[0] if components else "backend")
        obj_clean = objective.strip()
        lines = [l.strip() for l in obj_clean.splitlines() if l.strip()]
        title_summary = lines[0] if lines else "CEO Objective Implementation"
        title_summary = title_summary.lstrip("#").strip()[:80]

        sub_items = []
        for line in lines:
            if line.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.")):
                clean_item = re.sub(r"^[-*\d\.]+\s*", "", line).strip()
                if clean_item and len(clean_item) > 5:
                    sub_items.append(clean_item)

        if not sub_items:
            sub_items = [
                f"Implementasi fitur inti: {title_summary}",
                f"Pengujian otomatis dan verifikasi black-box: {title_summary}",
            ]

        epic_id = "epic_01"
        epic = ProgramEpic(
            epic_id=epic_id,
            title=f"Core Delivery: {title_summary}",
            description=f"Implementasi menyeluruh untuk {title_summary}",
            status="pending",
            tasks=[],
            mapped_requirement_ids=req_ids,
        )

        tasks: list[ProgramTask] = []
        prev_task_id: str | None = None

        for idx, item in enumerate(sub_items, 1):
            t_id = f"task_{idx:02d}"
            epic.tasks.append(t_id)
            deps = [prev_task_id] if prev_task_id else []

            r_files, w_files, c_files = _categorize_task_files(item, components, self.project_root)
            pre_risk = classify_pre_execution_risk(
                request=item,
                expected_writable_files=w_files,
            )["level"]

            # Map to requirement IDs
            mapped_reqs = [req_ids[min(idx - 1, len(req_ids) - 1)]] if req_ids else ["FR-001"]

            tasks.append(
                ProgramTask(
                    task_id=t_id,
                    epic_id=epic_id,
                    title=f"Task {idx}: {item[:70]}",
                    description=item,
                    acceptance_criteria=[
                        f"Fitur '{item}' diimplementasikan sesuai spesifikasi",
                        "Seluruh test suite terkait lulus pengujian (pass)",
                    ],
                    dependencies=deps,
                    owner_role=primary_role,
                    risk_level=pre_risk,
                    pre_risk=pre_risk,
                    status="pending",
                    mapped_requirement_ids=mapped_reqs,
                    readable_files=r_files,
                    expected_writable_files=w_files,
                    conditionally_writable_files=c_files,
                    planned_base_revision=base_head,
                )
            )
            prev_task_id = t_id

        return [epic], tasks

    def refresh_task_statuses(self, state: ProgramState) -> None:
        """Evaluate dependencies and update status for all tasks in state."""
        self._refresh_statuses(state.tasks)
        for t in state.tasks:
            state.task_statuses[t.task_id] = t.status

        # Update epics status
        for epic in state.epics:
            epic_tasks = [t for t in state.tasks if t.epic_id == epic.epic_id]
            if epic_tasks and all(t.status == "completed" for t in epic_tasks):
                epic.status = "completed"
            elif any(t.status in ("in_progress", "ready", "awaiting_input") for t in epic_tasks):
                epic.status = "in_progress"
            else:
                epic.status = "pending"

    def _refresh_statuses(self, tasks: list[ProgramTask]) -> None:
        """Internal helper to unlock ready tasks whose dependencies are completed."""
        completed_ids = {t.task_id for t in tasks if t.status == "completed"}
        failed_ids = {t.task_id for t in tasks if t.status in ("failed", "cancelled")}

        for t in tasks:
            if t.status == "pending":
                if not t.dependencies or all(dep in completed_ids for dep in t.dependencies):
                    t.status = "ready"
                elif any(dep in failed_ids for dep in t.dependencies):
                    t.status = "blocked"

    def select_next_ready_task(self, state: ProgramState) -> ProgramTask | None:
        """Select the next executable task (in_progress, retrying, or ready)."""
        self.refresh_task_statuses(state)
        # 1. First priority: active task already in_progress or retrying
        for t in state.tasks:
            if t.status in ("in_progress", "retrying"):
                return t
        # 2. Next priority: ready task
        for t in state.tasks:
            if t.status == "ready":
                return t
        return None

    def write_all_markdown_reports(self, state: ProgramState) -> None:
        """Regenerate all 8 Program Manager Markdown documents from JSON state (Source of Truth)."""
        prog_dir = self.get_program_dir(state.program_id)

        # 1. CEO_OBJECTIVE.md
        ceo_obj_content = (
            f"# CEO Objective: Program {state.program_id}\n\n"
            f"**Project ID:** `{state.project_id}`  \n"
            f"**Created At:** {state.created_at}  \n"
            f"**Current Status:** `{state.status}`\n\n"
            "## Normalized Requirements\n\n"
        )
        for req in state.requirements:
            ceo_obj_content += f"- **`{req.get('id')}`** [{req.get('category')}]: {req.get('description')}\n"
        ceo_obj_content += f"\n## Raw Objective Description\n\n{state.objective}\n"
        (prog_dir / "CEO_OBJECTIVE.md").write_text(ceo_obj_content, encoding="utf-8")

        # 2. ROADMAP.md
        completed_cnt = sum(1 for t in state.tasks if t.status == "completed")
        total_cnt = len(state.tasks)
        pct = round((completed_cnt / total_cnt) * 100) if total_cnt else 0

        roadmap_content = (
            f"# Program Roadmap ({state.program_id})\n\n"
            f"**Phase:** {state.current_phase}  \n"
            f"**Progress:** {completed_cnt}/{total_cnt} tasks completed ({pct}%)\n\n"
            "## Epics & Milestones\n\n"
        )
        for e in state.epics:
            roadmap_content += f"### Epic `{e.epic_id}`: {e.title} [{e.status.upper()}]\n"
            roadmap_content += f"{e.description}\n\n"
            for t_id in e.tasks:
                t = next((task for task in state.tasks if task.task_id == t_id), None)
                if t:
                    icon = "[x]" if t.status == "completed" else "[ ]"
                    req_str = ", ".join(t.mapped_requirement_ids) if t.mapped_requirement_ids else "none"
                    roadmap_content += f"- {icon} `{t.task_id}`: {t.title} ({t.status}) [Reqs: {req_str}]\n"
            roadmap_content += "\n"
        (prog_dir / "ROADMAP.md").write_text(roadmap_content, encoding="utf-8")

        # 3. ARCHITECTURE.md
        arch_content = (
            f"# Program Architecture ({state.program_id})\n\n"
            f"**Project Target:** `{state.project_id}`  \n"
            f"**Root:** `{self.project_root}`\n\n"
            "## System Context\n\n"
            "Arsitektur program diatur sebagai pipeline terkelola multi-agent:\n"
            "- **Control Plane:** LangGraph Orchestrator\n"
            "- **Execution Engine:** OpenHands Software Agent SDK dalam LocalWorkspace\n"
            "- **Workspace Isolation:** Git worktrees (Sequential Lazy Creation)\n"
            "- **Model Gateway:** 9Router OpenAI-compatible\n\n"
            "## Components\n\n"
        )
        for c_name, c_info in self.validated_project.manifest.components.items():
            arch_content += f"- **`{c_name}`**: {c_info.language} ({c_info.kind}) at `{c_info.root}`\n"
        arch_content += "\n"
        (prog_dir / "ARCHITECTURE.md").write_text(arch_content, encoding="utf-8")

        # 4. BACKLOG.md
        backlog_content = (
            f"# Program Backlog ({state.program_id})\n\n"
            "Daftar backlog epics dan tasks beserta kriteria penerimaan, dependensi, dan scope files:\n\n"
        )
        for t in state.tasks:
            icon = "[x]" if t.status == "completed" else "[ ]"
            deps_str = ", ".join(t.dependencies) if t.dependencies else "none"
            reqs_str = ", ".join(t.mapped_requirement_ids) if t.mapped_requirement_ids else "none"
            exp_files = ", ".join(t.expected_writable_files) if t.expected_writable_files else "none"
            backlog_content += (
                f"### {icon} `{t.task_id}`: {t.title}\n"
                f"- **Epic:** `{t.epic_id}` | **Role:** `{t.owner_role}` | **Risk:** `{t.risk_level.upper()}` (pre={t.pre_risk}, post={t.post_risk or '-'})\n"
                f"- **Status:** `{t.status}` | **Attempts:** {t.attempt_count} | **Dependencies:** `{deps_str}` | **Requirements:** `{reqs_str}`\n"
                f"- **Expected Writable:** `{exp_files}`\n"
                f"- **Base Revision:** `{t.execution_base_revision or t.planned_base_revision or '-'}`\n"
                f"- **Description:** {t.description}\n"
                "- **Acceptance Criteria:**\n"
            )
            for ac in t.acceptance_criteria:
                backlog_content += f"  - [ ] {ac}\n"
            if t.changed_files:
                backlog_content += f"- **Changed Files:** {', '.join(t.changed_files)}\n"
            if t.completion_evidence:
                backlog_content += f"- **Evidence/Commit:** `{t.completion_evidence}`\n"
            backlog_content += "\n"
        (prog_dir / "BACKLOG.md").write_text(backlog_content, encoding="utf-8")

        # 5. CURRENT_STATUS.md
        cur_content = (
            f"# Current Status Dashboard ({state.program_id})\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| **Program ID** | `{state.program_id}` |\n"
            f"| **Overall Status** | `{state.status}` |\n"
            f"| **State Revision** | `{state.state_revision}` |\n"
            f"| **Current Phase** | {state.current_phase} |\n"
            f"| **Progress** | {completed_cnt}/{total_cnt} ({pct}%) |\n"
            f"| **Active Task** | `{state.active_task or 'none'}` |\n"
            f"| **Active Run ID** | `{state.active_run_id or 'none'}` |\n"
            f"| **Next Action** | `{state.next_action or 'none'}` |\n"
            f"| **Updated At** | {state.updated_at} |\n\n"
        )
        if state.blockers:
            cur_content += "## Blockers\n\n"
            for b in state.blockers:
                cur_content += f"- ⚠️ {b}\n"
            cur_content += "\n"
        (prog_dir / "CURRENT_STATUS.md").write_text(cur_content, encoding="utf-8")

        # 6. DECISIONS.md
        decisions_content = (
            f"# Architectural & Program Decisions ({state.program_id})\n\n"
            "Log keputusan arsitektur, pemilihan dependensi, dan mitigasi risiko:\n\n"
            "1. **Sequential Fresh-Base Worktrees:** Worktree dibuat secara lazy dari HEAD target branch terkini.\n"
            "2. **Single-Writer & Integration Locks:** Mencegah proses ganda dan konflik race condition saat integrasi.\n"
            "3. **Two-Phase Risk Reclassification:** Final risk adalah nilai tertinggi antara pre-risk dan post-risk (actual diff).\n"
            "4. **Fail-Closed Conflict Recovery:** Cherry-pick conflict otomatis membatalkan target state, mencatat artifact, dan me-replan maksimal 1 kali.\n"
        )
        (prog_dir / "DECISIONS.md").write_text(decisions_content, encoding="utf-8")

        # 7. TEST_REPORT.md
        test_content = (
            f"# Consolidated Test Report ({state.program_id})\n\n"
            "Ringkasan pengujian unit, integrasi, dan validasi post-integration:\n\n"
            "| Task ID | Task Title | Test Status | Evidence / Exit Code |\n"
            "|---|---|---|---|\n"
        )
        for t in state.tasks:
            t_res = t.test_result if t.test_result != "none" else ("PASS" if t.status == "completed" else "-")
            ev_str = str(t.completion_evidence or "-")[:12]
            test_content += f"| `{t.task_id}` | {t.title[:40]} | `{t_res}` | `{ev_str}` |\n"
        test_content += "\n"
        (prog_dir / "TEST_REPORT.md").write_text(test_content, encoding="utf-8")

        # 8. HANDOFF.md
        handoff_content = (
            f"# Final Black-Box Handoff Report for CEO ({state.program_id})\n\n"
            f"**Objective:** {state.objective}\n\n"
            f"**Status:** `{state.status.upper()}`\n\n"
            "## Summary of Deliverables\n\n"
        )
        for t in state.tasks:
            status_badge = "✅ COMPLETED" if t.status == "completed" else f"⏳ {t.status.upper()}"
            handoff_content += f"### {t.title} [{status_badge}]\n"
            handoff_content += f"- **Description:** {t.description}\n"
            if t.changed_files:
                handoff_content += f"- **Changed Files:** {', '.join(t.changed_files)}\n"
            if t.completion_evidence:
                handoff_content += f"- **Integrated Commit:** `{t.completion_evidence}`\n"
            handoff_content += "\n"

        handoff_content += (
            "## Black-Box Testing Guide\n\n"
            "Langkah-langkah untuk menguji hasil pekerjaan secara black-box:\n"
            "1. Jalankan server aplikasi utama pada environment staging/lokal.\n"
            "2. Jalankan test suite verifikasi: periksa endpoint atau fitur yang baru dibangun.\n"
            "3. Verifikasi bahwa tidak ada regresi pada endpoint yang sudah ada.\n\n"
            "## Verification Checklist\n\n"
        )
        for t in state.tasks:
            for ac in t.acceptance_criteria:
                mark = "[x]" if t.status == "completed" else "[ ]"
                handoff_content += f"- {mark} {ac} (Task `{t.task_id}`)\n"
        handoff_content += "\n"
        (prog_dir / "HANDOFF.md").write_text(handoff_content, encoding="utf-8")
