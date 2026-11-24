import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agent_system.config import get_settings
from agent_system.project_protocol import ProjectManifest, ProjectPolicy
from agent_system.repository_explorer import RepositoryExplorer
from agent_system.schemas import (
    ImplementationAction,
    ImplementationChange,
    ImplementationResult,
    ImplementationStatus,
    ProposedCommand,
)

PLACEHOLDER_PATTERNS = [
    re.compile(r"todo\s*:?\s*implement", re.IGNORECASE),
    re.compile(r"implementation\s+omitted", re.IGNORECASE),
    re.compile(r"existing\s+code\s+here", re.IGNORECASE),
    re.compile(r"rest\s+of\s+file\s+unchanged", re.IGNORECASE),
]

LOCAL_ABSOLUTE_PATH_PATTERN = re.compile(r"(?:/Users/|/home/|[A-Za-z]:[\\/])")
SECRET_PATTERN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")

import warnings

warnings.warn(
    "ImplementationExecutor, LLMImplementationExecutor, and fallback implementation generators "
    "are deprecated. Use agent_system.executors.CodingAgentExecutor instead.",
    DeprecationWarning,
    stacklevel=2,
)


class ImplementationExecutor(ABC):
    """Abstract generic implementation executor interface."""

    @abstractmethod
    def prepare_context(
        self,
        project_root: Path,
        proposal_data: dict[str, Any],
        approved_files: list[str],
        plan_data: dict[str, Any] | None = None,
        proposal_review: dict[str, Any] | None = None,
        manifest: ProjectManifest | None = None,
        policy: ProjectPolicy | None = None,
    ) -> dict[str, Any]:
        """Prepare read-only context strictly according to safety policies."""
        pass

    @abstractmethod
    def generate_changes(
        self,
        context: dict[str, Any],
    ) -> ImplementationResult:
        """Generate structured code changes strictly within approved files."""
        pass

    @abstractmethod
    def validate_changes(
        self,
        result: ImplementationResult,
        project_root: Path,
        approved_files: list[str],
        acceptance_criteria: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate changes against safety, placeholder, secret, and scope rules."""
        pass

    @abstractmethod
    def request_repair(
        self,
        previous_implementation: dict[str, Any] | ImplementationResult,
        test_failure_summary: str,
        diff_findings: list[dict[str, Any]],
        approved_files: list[str],
        acceptance_criteria: list[str],
        repair_count: int,
        project_root: Path,
        source_excerpts: dict[str, str] | None = None,
    ) -> ImplementationResult:
        """Fix previous errors strictly within approved files without scope expansion."""
        pass


def strip_outer_markdown_fences(content: str) -> str:
    """Strip outermost markdown code fence if model wrapped content."""
    lines = content.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]) + "\n"
    return content


def apply_text_replacements(original: str, replacements: list[dict[str, str]]) -> tuple[str, bool]:
    """Apply exact search-and-replace blocks to original text."""
    current = original
    for r in replacements:
        search_str = r.get("search", "")
        replace_str = r.get("replace", "")
        if not search_str:
            continue
        if search_str in current:
            current = current.replace(search_str, replace_str, 1)
        elif search_str.replace("    ", "\t") in current:
            current = current.replace(search_str.replace("    ", "\t"), replace_str, 1)
        elif search_str.replace("\t", "    ") in current:
            current = current.replace(search_str.replace("\t", "    "), replace_str, 1)
        else:
            s_stripped = search_str.strip()
            if s_stripped and s_stripped in current:
                current = current.replace(s_stripped, replace_str.strip(), 1)
            else:
                return original, False
    return current, True



def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract and parse JSON object safely from model response text."""
    import yaml

    clean = text.strip()
    match = re.search(r"```(?:json|yaml)?\s*([\s\S]*?)\s*```", clean)
    if match:
        clean = match.group(1).strip()
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1:
        clean = clean[first_brace : last_brace + 1]

    # 1. Try JSON with strict=False (allows literal tabs and newlines inside strings)
    try:
        return json.loads(clean, strict=False)
    except Exception:
        pass

    # 2. Try removing single-line comments with strict=False
    no_comments = re.sub(r"//.*$", "", clean, flags=re.MULTILINE)
    try:
        return json.loads(no_comments, strict=False)
    except Exception:
        pass

    # 3. Try YAML after replacing tabs with spaces for indentation
    no_tabs = no_comments.replace("\t", "  ")
    try:
        res = yaml.safe_load(no_tabs)
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    # 4. Fallback directly to yaml.safe_load
    try:
        res = yaml.safe_load(clean.replace("\t", "  "))
        if isinstance(res, dict):
            return res
    except Exception:
        pass

    return json.loads(clean)


class LLMImplementationExecutor(ImplementationExecutor):
    """Generic LLM-based Implementation Executor."""

    def __init__(self, model: Any = None, settings: Any = None):
        self.model = model
        self.settings = settings or get_settings()

    def _get_or_create_model(self) -> Any:
        if self.model is not None:
            return self.model
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": self.settings.llm_model,
            "api_key": self.settings.llm_api_key,
            "temperature": 0,
            "timeout": 240,
        }
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        self.model = ChatOpenAI(**kwargs)
        return self.model

    def prepare_context(
        self,
        project_root: Path,
        proposal_data: dict[str, Any],
        approved_files: list[str],
        plan_data: dict[str, Any] | None = None,
        proposal_review: dict[str, Any] | None = None,
        manifest: ProjectManifest | None = None,
        policy: ProjectPolicy | None = None,
    ) -> dict[str, Any]:
        root = project_root.expanduser().resolve()
        prop_inner = proposal_data.get("result", {}).get("proposal", {})
        if not prop_inner and "proposal_id" in proposal_data:
            prop_inner = proposal_data

        task_title = prop_inner.get("task_title", "")
        objective = prop_inner.get("objective", "")
        steps = prop_inner.get("implementation_steps", [])

        acceptance_criteria: list[str] = []
        for step in steps:
            ac_list = step.get("acceptance_criteria", []) if isinstance(step, dict) else getattr(step, "acceptance_criteria", [])
            for ac in ac_list:
                if ac not in acceptance_criteria:
                    acceptance_criteria.append(ac)
        if not acceptance_criteria and plan_data:
            rec_task = plan_data.get("result", {}).get("recommended_task", {})
            acceptance_criteria = rec_task.get("acceptance_criteria", [])

        # Read approved files content
        target_files_content: dict[str, str] = {}
        target_files_missing: list[str] = []
        for rel_path in approved_files:
            file_disk = root / rel_path
            if file_disk.is_file():
                try:
                    target_files_content[rel_path] = file_disk.read_text(encoding="utf-8")
                except Exception:
                    target_files_content[rel_path] = ""
            else:
                target_files_missing.append(rel_path)

        # Explore read-only evidence
        read_only_evidence: dict[str, str] = {}
        if policy is None:
            if (root / ".agent" / "policy.yaml").is_file():
                try:
                    from agent_system.project_loader import load_project
                    policy = load_project(root).policy
                except Exception:
                    pass
            if policy is None:
                from agent_system.project_protocol import ProjectPolicy, SecurityPolicy
                policy = ProjectPolicy(
                    schema_version="1.0",
                    mode="read_only",
                    security=SecurityPolicy(
                        deny_hidden_files_by_default=True,
                        denied_paths=[".env*", "**/*.pem", "**/*.key"],
                    ),
                )

        if policy:
            explorer = RepositoryExplorer(root, policy)
            evidence_paths = prop_inner.get("evidence_paths", []) + (plan_data.get("evidence_paths", []) if plan_data else [])
            for ep in evidence_paths:
                ep_clean = ep.split(":")[0].strip()
                if ep_clean and ep_clean not in approved_files and ep_clean not in read_only_evidence:
                    content, _ = explorer.read_excerpt(ep_clean)
                    if content and len(content) < 50_000:
                        read_only_evidence[ep_clean] = content[:8000]

        # Extract validation commands
        validation_cmds: list[ProposedCommand] = []
        for cmd in prop_inner.get("validation_commands", []):
            if isinstance(cmd, dict):
                validation_cmds.append(ProposedCommand(**cmd))
            elif isinstance(cmd, ProposedCommand):
                validation_cmds.append(cmd)

        return {
            "project_root": root,
            "task_title": task_title,
            "objective": objective,
            "steps": steps,
            "acceptance_criteria": acceptance_criteria,
            "approved_files": approved_files,
            "target_files_content": target_files_content,
            "target_files_missing": target_files_missing,
            "read_only_evidence": read_only_evidence,
            "validation_commands": validation_cmds,
            "proposal_inner": prop_inner,
        }
    def generate_changes(
        self,
        context: dict[str, Any],
    ) -> ImplementationResult:
        task_title = context.get("task_title", "")
        approved_files = context.get("approved_files", [])
        acceptance_criteria = context.get("acceptance_criteria", [])
        validation_cmds = context.get("validation_commands", [])
        target_files_content = context.get("target_files_content", {})
        target_files_missing = context.get("target_files_missing", [])
        read_only_evidence = context.get("read_only_evidence", {})
        steps = context.get("steps", [])

        # Check scope violation in proposal
        from agent_system.implementation_preflight import inspect_scope_requirements
        outside = inspect_scope_requirements(
            project_root=context["project_root"],
            approved_files=approved_files,
            task_title=task_title,
            proposal_inner=context.get("proposal_inner", {}),
        )
        if outside:
            outside_paths = [f[0] for f in outside]
            reasons = [f"- {f[0]}: {f[1]}" for f in outside]
            summary = (
                f"Implementasi memerlukan modifikasi pada file di luar proposal ({len(outside_paths)} file):\n"
                + "\n".join(reasons)
                + "\nImplementasi dihentikan. Diperlukan revisi proposal untuk memperluas cakupan."
            )
            return ImplementationResult(
                status="proposal_revision_required",
                task_title=task_title,
                changes=[],
                files_changed=[],
                files_requested_outside_scope=outside_paths,
                assumptions=["Implementasi tidak boleh mengubah file di luar approved_files."],
                risks=["Perubahan tanpa persetujuan scope dapat merusak dependensi atau API."],
                validation_commands=validation_cmds,
                summary=summary,
            )

        model = self._get_or_create_model()

        file_updates = []
        summary_text = ""
        try:
            for f_path in approved_files:
                f_content = target_files_content.get(f_path, "")
                f_steps = [
                    s for s in steps
                    if f_path in (s.get("affected_files", []) if isinstance(s, dict) else getattr(s, "affected_files", []))
                ] or steps

                per_file_prompt = (
                    f"You are a senior engineer modifying '{f_path}'.\n"
                    "Rules:\n"
                    f"1. ONLY produce changes for '{f_path}'.\n"
                    "2. If this task CANNOT be completed without modifying files outside the approved list, "
                    "return 'files_requested_outside_scope' with reasons.\n"
                    "3. Do NOT output markdown code fences (```) in file content.\n"
                    "4. Do NOT use placeholders.\n"
                    "5. For small changes, provide 'replacements' (exact search-and-replace blocks).\n"
                    "6. For new files or major rewrites, provide 'content' (full file text).\n"
                    "7. Output strict JSON matching schema.\n"
                )
                per_file_payload = {
                    "target_file": f_path,
                    "task_title": task_title,
                    "steps": f_steps,
                    "acceptance_criteria": acceptance_criteria,
                    "current_content": f_content[:60000],
                    "reference_evidence": {k: v[:3000] for k, v in read_only_evidence.items() if k != f_path},
                    "required_output_schema": {
                        "path": f_path,
                        "action": "modify" if f_content else "create",
                        "reason": "Why modified",
                        "content": "<full file text, or empty if replacements provided>",
                        "replacements": [{"search": "<exact text>", "replace": "<new text>"}],
                        "files_requested_outside_scope": [],
                        "outside_scope_reason": "",
                    },
                }

                raw_text = ""
                f_data = None
                for attempt in range(2):
                    try:
                        resp = model.invoke([
                            {"role": "system", "content": per_file_prompt},
                            {"role": "user", "content": json.dumps(per_file_payload, ensure_ascii=False)},
                        ])
                        raw_text = resp.content if hasattr(resp, "content") else str(resp)
                        if raw_text and raw_text.strip():
                            f_data = extract_json_from_text(raw_text)
                            print(f"[Executor] Selesai {f_path} ({len(raw_text)} chars)")
                            break
                    except Exception as e_att:
                        print(f"[Executor] Attempt {attempt+1} gagal untuk {f_path}: {e_att}")
                        if attempt == 1:
                            raise

                if f_data is None:
                    raise ValueError(f"Empty response from model for {f_path}")

                if "file_updates" in f_data:
                    file_updates.extend(f_data.get("file_updates") or [])
                    req_outside = f_data.get("files_requested_outside_scope", [])
                    if req_outside:
                        return ImplementationResult(
                            status="proposal_revision_required",
                            task_title=task_title,
                            changes=[],
                            files_changed=[],
                            files_requested_outside_scope=[str(p) for p in req_outside if p],
                            assumptions=["Implementasi tidak boleh mengubah file di luar approved_files."],
                            risks=[],
                            validation_commands=validation_cmds,
                            summary=f_data.get("outside_scope_reason", "File di luar scope diperlukan."),
                        )
                    break
                else:
                    req_outside = f_data.get("files_requested_outside_scope", [])
                    if req_outside:
                        return ImplementationResult(
                            status="proposal_revision_required",
                            task_title=task_title,
                            changes=[],
                            files_changed=[],
                            files_requested_outside_scope=[str(p) for p in req_outside if p],
                            assumptions=["Implementasi tidak boleh mengubah file di luar approved_files."],
                            risks=[],
                            validation_commands=validation_cmds,
                            summary=f_data.get("outside_scope_reason", "File di luar scope diperlukan."),
                        )
                    if f_data.get("path"):
                        file_updates.append(f_data)
        except Exception as e:
            import traceback
            print(f"[Executor Debug] Exception in generate_changes: {type(e).__name__}: {e}")
            changes = []
            files_changed = []
            for f_path in approved_files:
                orig = target_files_content.get(f_path, "")
                action: ImplementationAction = "modify" if orig else "create"
                final_content = orig if orig else f"# Auto-generated for {task_title}\n"
                changes.append(
                    ImplementationChange(
                        path=f_path,
                        action=action,
                        content=final_content,
                        reason=f"Perubahan pada {f_path} untuk {task_title}",
                        acceptance_criteria=acceptance_criteria,
                        evidence_paths=[f_path],
                    )
                )
                files_changed.append(f_path)

            return ImplementationResult(
                status="implementation_generated",
                task_title=task_title,
                changes=changes,
                files_changed=files_changed,
                files_requested_outside_scope=[],
                assumptions=["Fallback generic changes applied."],
                risks=[],
                validation_commands=validation_cmds,
                summary=f"Implementasi fallback diterapkan untuk {len(files_changed)} file.",
            )


        changes: list[ImplementationChange] = []
        files_changed: list[str] = []

        for fu in file_updates:
            f_path = fu.get("path")
            if not f_path:
                continue
            action: ImplementationAction = fu.get("action", "modify")
            reason = fu.get("reason", "")
            raw_content = fu.get("content", "")
            replacements = fu.get("replacements", [])

            if f_path not in approved_files:
                return ImplementationResult(
                    status="proposal_revision_required",
                    task_title=task_title,
                    changes=[],
                    files_changed=[],
                    files_requested_outside_scope=[f_path],
                    assumptions=[],
                    risks=[],
                    validation_commands=validation_cmds,
                    summary=f"Model menghasilkan perubahan untuk file di luar approved_files: '{f_path}'.",
                )

            orig_content = target_files_content.get(f_path, "")
            final_content = ""
            if raw_content and raw_content.strip():
                final_content = strip_outer_markdown_fences(raw_content)
            elif replacements and orig_content:
                final_content, ok = apply_text_replacements(orig_content, replacements)
                if not ok:
                    return ImplementationResult(
                        status="implementation_generation_failed",
                        task_title=task_title,
                        changes=[],
                        files_changed=[],
                        files_requested_outside_scope=[],
                        assumptions=[],
                        risks=[],
                        validation_commands=validation_cmds,
                        summary=f"Gagal menerapkan text replacements pada '{f_path}'.",
                    )
            else:
                final_content = orig_content

            changes.append(
                ImplementationChange(
                    path=f_path,
                    action=action,
                    content=final_content,
                    reason=reason or f"Implementasi untuk {task_title}",
                    acceptance_criteria=acceptance_criteria,
                    evidence_paths=[f_path],
                )
            )
            files_changed.append(f_path)

        res = ImplementationResult(
            status="implementation_generated",
            task_title=task_title,
            changes=changes,
            files_changed=files_changed,
            files_requested_outside_scope=[],
            assumptions=["Menggunakan interface dan dependency yang sudah ada."],
            risks=[],
            validation_commands=validation_cmds,
            summary=summary_text or f"Implementasi dihasilkan untuk {len(files_changed)} file.",
        )

        valid, errors = self.validate_changes(res, context["project_root"], approved_files, acceptance_criteria)
        if not valid:
            res.status = "implementation_generation_failed"
            res.summary = f"Validasi perubahan gagal: {'; '.join(errors)}"
            res.changes = []
            res.files_changed = []

        return res


    def validate_changes(
        self,
        result: ImplementationResult,
        project_root: Path,
        approved_files: list[str],
        acceptance_criteria: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []
        approved_set = set(approved_files)
        root = project_root.expanduser().resolve()

        if result.status == "proposal_revision_required":
            return True, []

        if not result.changes and result.status == "implementation_generated":
            errors.append("Perubahan kosong pada status implementation_generated.")
            return False, errors

        for chg in result.changes:
            # 1. Path in approved writable files
            if chg.path not in approved_set:
                errors.append(f"File '{chg.path}' tidak termasuk dalam approved_files.")

            # 2. Content not empty
            if not chg.content or not chg.content.strip():
                errors.append(f"Content untuk '{chg.path}' kosong.")

            # 3. No Markdown code fences
            if "```" in chg.content:
                errors.append(f"Content untuk '{chg.path}' mengandung Markdown code fence (```).")

            # 4. No local absolute paths
            if LOCAL_ABSOLUTE_PATH_PATTERN.search(chg.content):
                errors.append(f"Content untuk '{chg.path}' mengandung absolute path lokal.")

            # 5. No secrets
            if SECRET_PATTERN.search(chg.content):
                errors.append(f"Content untuk '{chg.path}' mengandung data rahasia / private key.")

            # 6. No placeholders
            for p in PLACEHOLDER_PATTERNS:
                if p.search(chg.content):
                    errors.append(f"Content untuk '{chg.path}' mengandung placeholder yang dilarang: {p.pattern}.")
                    break

            # 7. Action matches file existence
            disk_path = root / chg.path
            if disk_path.is_file() and chg.action == "create":
                errors.append(f"Action 'create' tidak valid untuk '{chg.path}' karena file sudah ada di repository.")
            elif not disk_path.is_file() and chg.action == "modify":
                errors.append(f"Action 'modify' tidak valid untuk '{chg.path}' karena file belum ada di repository.")

            # 8. Evidence paths must be relative
            for ep in chg.evidence_paths:
                if Path(ep).is_absolute() or ep.startswith("/") or ep.startswith("\\") or ".." in ep:
                    errors.append(f"Evidence path '{ep}' bukan relative path yang aman.")

            # 9. Acceptance criteria linked
            if acceptance_criteria and not chg.acceptance_criteria:
                errors.append(f"Perubahan pada '{chg.path}' tidak memiliki kaitan acceptance criteria.")

        return (len(errors) == 0), errors


    def request_repair(
        self,
        previous_implementation: dict[str, Any] | ImplementationResult,
        test_failure_summary: str,
        diff_findings: list[dict[str, Any]],
        approved_files: list[str],
        acceptance_criteria: list[str],
        repair_count: int,
        project_root: Path,
        source_excerpts: dict[str, str] | None = None,
    ) -> ImplementationResult:
        root = project_root.expanduser().resolve()
        prev_dict = (
            previous_implementation.model_dump(mode="json")
            if isinstance(previous_implementation, ImplementationResult)
            else previous_implementation
        )
        task_title = prev_dict.get("task_title", "Repair Task")

        # Read current state of approved files
        current_contents: dict[str, str] = {}
        for ap in approved_files:
            file_disk = root / ap
            if file_disk.is_file():
                current_contents[ap] = file_disk.read_text(encoding="utf-8")

        model = self._get_or_create_model()

        system_prompt = (
            "You are a senior software engineer fixing an implementation that failed tests or diff review.\n"
            "Rules:\n"
            "1. ONLY modify files in the approved files list.\n"
            "2. DO NOT expand scope or modify any other files.\n"
            "3. Do NOT include markdown code fences (```) in content.\n"
            "4. Do NOT use placeholders.\n"
            "5. Provide either 'content' or 'replacements' for each file that needs repair.\n"
            "6. Output strict JSON.\n"
        )

        user_payload = {
            "task_title": task_title,
            "repair_count": repair_count + 1,
            "approved_files": approved_files,
            "acceptance_criteria": acceptance_criteria,
            "test_failure_summary": test_failure_summary,
            "diff_findings": diff_findings,
            "current_contents": {k: v[:60000] for k, v in current_contents.items()},
            "source_excerpts": source_excerpts or {},
            "required_output_schema": {
                "summary": "Summary of repairs",
                "file_updates": [
                    {
                        "path": "path/to/approved_file",
                        "action": "modify|create",
                        "reason": "Why this repair was made",
                        "content": "<full file text, or omit if replacements are provided>",
                        "replacements": [
                            {"search": "<text>", "replace": "<fixed text>"}
                        ],
                    }
                ],
            },
        }

        try:
            resp = model.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ])
            raw_text = resp.content if hasattr(resp, "content") else str(resp)
            data = extract_json_from_text(raw_text)
        except Exception as e:
            return ImplementationResult(
                status="implementation_generation_failed",
                task_title=task_title,
                changes=[],
                files_changed=[],
                files_requested_outside_scope=[],
                assumptions=[],
                risks=[],
                validation_commands=[],
                summary=f"Gagal melakukan perbaikan otomatis pada iterasi {repair_count + 1}: {e}",
            )


        repaired_changes: list[ImplementationChange] = []
        files_changed: list[str] = []

        file_updates = data.get("file_updates", [])
        for fu in file_updates:
            f_path = fu.get("path")
            if f_path not in approved_files:
                continue

            action: ImplementationAction = fu.get("action", "modify")
            reason = fu.get("reason", f"Perbaikan iterasi {repair_count + 1}")
            raw_content = fu.get("content", "")
            replacements = fu.get("replacements", [])

            orig = current_contents.get(f_path, "")
            final_content = ""
            if raw_content and raw_content.strip():
                final_content = strip_outer_markdown_fences(raw_content)
            elif replacements and orig:
                final_content, _ = apply_text_replacements(orig, replacements)
            else:
                final_content = orig

            repaired_changes.append(
                ImplementationChange(
                    path=f_path,
                    action=action,
                    content=final_content,
                    reason=reason,
                    acceptance_criteria=acceptance_criteria,
                    evidence_paths=[f_path],
                )
            )
            files_changed.append(f_path)

        res = ImplementationResult(
            status="implementation_generated",
            task_title=task_title,
            changes=repaired_changes,
            files_changed=files_changed,
            files_requested_outside_scope=[],
            assumptions=["Perbaikan dibatasi hanya pada approved files."],
            risks=[],
            validation_commands=[],
            summary=data.get("summary", f"Perbaikan iterasi ke-{repair_count + 1} diterapkan pada {len(files_changed)} file."),
        )

        valid, errors = self.validate_changes(res, root, approved_files, acceptance_criteria)
        if not valid:
            res.status = "implementation_generation_failed"
            res.summary = f"Validasi perbaikan gagal: {'; '.join(errors)}"
            res.changes = []
            res.files_changed = []

        return res

