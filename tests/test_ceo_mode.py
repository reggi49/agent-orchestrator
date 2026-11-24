import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent_system.ceo_mode import CEORunner
from agent_system.program_manager import ProgramManager
from agent_system.project_initializer import initialize_project
from agent_system.schemas import ProgramState, ProgramTask


def setup_test_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True)
    (path / "README.md").write_text("# Test Repo\n")
    (path / "backend").mkdir(parents=True, exist_ok=True)
    (path / "backend" / "app.go").write_text("package main\n\nfunc Run() string { return \"ok\" }\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    initialize_project(path, apply=True)
    return path


@pytest.fixture(autouse=True)
def mock_langchain_openai():
    """Ensure no real LLM network calls are made during unit tests."""
    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_chat.return_value.with_structured_output.return_value.invoke.side_effect = RuntimeError("Offline mock")
        yield mock_chat


# ==============================================================================
# 1. Test Program Initialization from Markdown File & Inline Request
# ==============================================================================
def test_01_initialize_program_creates_all_reports_and_state(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    pm = ProgramManager(repo)

    objective_text = (
        "# Bangun Fitur Notifikasi\n\n"
        "- Implementasi notification service\n"
        "- Buat endpoint pengiriman pesan\n"
        "- Buat automated test verifikasi\n"
    )
    state = pm.initialize_program(objective_text, program_id="testprog1")

    assert state.program_id == "testprog"
    assert state.status == "ready"
    assert len(state.epics) >= 1
    assert len(state.tasks) >= 2

    prog_dir = repo / ".agent" / "programs" / "testprog"
    assert (prog_dir / "CEO_OBJECTIVE.md").is_file()
    assert (prog_dir / "ROADMAP.md").is_file()
    assert (prog_dir / "ARCHITECTURE.md").is_file()
    assert (prog_dir / "BACKLOG.md").is_file()
    assert (prog_dir / "CURRENT_STATUS.md").is_file()
    assert (prog_dir / "DECISIONS.md").is_file()
    assert (prog_dir / "TEST_REPORT.md").is_file()
    assert (prog_dir / "HANDOFF.md").is_file()

    # Machine-readable artifact check
    art_path = repo / ".agent" / "artifacts" / "programs" / "program_testprog.json"
    assert art_path.is_file()
    data = json.loads(art_path.read_text(encoding="utf-8"))
    res = data.get("result", data)
    assert res["program_id"] == "testprog"
    assert res["project_id"] == "main-repo"
    assert "current_phase" in res
    assert "dependencies" in res
    assert "task_statuses" in res


def test_02_ceo_runner_supports_objective_file_and_request(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    runner = CEORunner(repo, verbose=False)

    obj_file = tmp_path / "objective.md"
    obj_file.write_text("# Bangun CRM System\n- Buat customer database\n- Buat API customer\n")

    state_file = runner.run_ceo(objective_file=obj_file, autonomous=False)
    assert state_file.status == "ready"
    assert "Bangun CRM System" in state_file.objective

    # Inline request
    state_inline = runner.run_ceo(request_text="Bangun fitur billing", program_id="billing01", autonomous=False)
    assert state_inline.program_id == "billing0"
    assert "Bangun fitur billing" in state_inline.objective


# ==============================================================================
# 2. Test Task Model & Dependency Scheduler
# ==============================================================================
def test_03_dependency_scheduler_unlocks_tasks_sequentially(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    pm = ProgramManager(repo)

    objective_text = "- Step 1: Model\n- Step 2: Service\n- Step 3: Controller\n"
    state = pm.initialize_program(objective_text, program_id="deps01")

    # Initially: Task 1 ready, Task 2 & 3 pending
    t1 = state.tasks[0]
    t2 = state.tasks[1]
    t3 = state.tasks[2]

    assert t1.status == "ready"
    assert t2.status == "pending"
    assert t3.status == "pending"

    # Select next task -> should pick t1
    picked = pm.select_next_ready_task(state)
    assert picked is not None
    assert picked.task_id == t1.task_id

    # Mark t1 completed -> unlocks t2
    t1.status = "completed"
    pm.refresh_task_statuses(state)
    assert t2.status == "ready"
    assert t3.status == "pending"

    # Pick next -> should pick t2
    picked2 = pm.select_next_ready_task(state)
    assert picked2 is not None
    assert picked2.task_id == t2.task_id

    # Mark t2 completed -> unlocks t3
    t2.status = "completed"
    pm.refresh_task_statuses(state)
    assert t3.status == "ready"


# ==============================================================================
# 3. Test Autonomous Loop Execution
# ==============================================================================
def test_04_autonomous_loop_happy_path_completes_program(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_instance = MagicMock()
        mock_runner_cls.return_value = mock_instance

        # Mock runner.run returning completed
        mock_instance.run.return_value = {
            "overall_status": "completed",
            "run_id": "mock_run_123",
            "changed_files": ["backend/app.go"],
            "test_status": "passed",
            "commit_hash": "a" * 40,
            "worktree_identifier": "test-wt-1",
        }

        state = runner.run_ceo(
            request_text="- Step A\n- Step B\n",
            program_id="auto01",
            autonomous=True,
            approval_policy="safe-auto",
        )

        assert state.status == "completed"
        assert all(t.status == "completed" for t in state.tasks)
        assert state.current_phase == "Phase Completed: Handoff Ready"

        # Check HANDOFF.md was written with deliverables
        handoff_file = repo / ".agent" / "programs" / "auto01" / "HANDOFF.md"
        assert handoff_file.is_file()
        content = handoff_file.read_text(encoding="utf-8")
        assert "COMPLETED" in content
        assert "Black-Box Testing Guide" in content


# ==============================================================================
# 4. Test Approval Policy & Pausing at Awaiting CEO Input
# ==============================================================================
def test_05_autonomous_loop_pauses_for_high_medium_risk_awaiting_ceo_input(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_instance = MagicMock()
        mock_runner_cls.return_value = mock_instance

        # Runner returns awaiting_human_approval for medium/high risk task
        mock_instance.run.return_value = {
            "overall_status": "awaiting_human_approval",
            "run_id": "run_awaiting_456",
            "risk_level": "medium",
            "worktree_identifier": "test-wt-2",
        }

        state = runner.run_ceo(
            request_text="- Step 1: Update production payment service\n",
            program_id="pay01",
            autonomous=True,
            approval_policy="safe-auto",
        )

        assert state.status == "awaiting_ceo_input"
        assert state.active_run_id == "run_awaiting_456"
        assert any(t.status == "awaiting_input" for t in state.tasks)


# ==============================================================================
# 5. Test Durable Resume from Approval Gate
# ==============================================================================
def test_06_durable_resume_from_approval_gate(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_instance = MagicMock()
        mock_runner_cls.return_value = mock_instance

        # Step 1: Pauses at awaiting_ceo_input
        mock_instance.run.return_value = {
            "overall_status": "awaiting_human_approval",
            "run_id": "run_gate_789",
            "risk_level": "medium",
        }

        state = runner.run_ceo(
            request_text="- Step 1: Single critical task\n",
            program_id="gate01",
            autonomous=True,
            approval_policy="safe-auto",
        )
        assert state.status == "awaiting_ceo_input"

        # Step 2: Resume with --approve
        mock_instance.resume.return_value = {
            "overall_status": "completed",
            "changed_files": ["backend/app.go"],
            "test_status": "passed",
            "commit_hash": "c" * 40,
        }

        resumed_state = runner.run_ceo(
            program_id="gate01",
            resume=True,
            approve=True,
            autonomous=True,
        )

        assert resumed_state.status == "completed"
        assert resumed_state.tasks[0].status == "completed"
        assert resumed_state.tasks[0].completion_evidence == "c" * 40


# ==============================================================================
# 6. Test Crash Recovery & Checkpoint Integrity
# ==============================================================================
def test_07_crash_recovery_resumes_without_duplicating_completed_tasks(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    runner = CEORunner(repo, verbose=False)

    with patch("agent_system.ceo_mode.OrchestrationRunner") as mock_runner_cls:
        mock_instance = MagicMock()
        mock_runner_cls.return_value = mock_instance

        # Task 1 succeeds, Task 2 crashes
        call_count = 0
        def run_mock(request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "overall_status": "completed",
                    "run_id": "run_part_1",
                    "changed_files": ["backend/app.go"],
                    "test_status": "passed",
                    "commit_hash": "1" * 40,
                }
            raise RuntimeError("Unexpected terminal or crash event")

        mock_instance.run.side_effect = run_mock

        state = runner.run_ceo(
            request_text="- Step 1\n- Step 2\n",
            program_id="crash01",
            autonomous=True,
        )

        assert state.tasks[0].status == "completed"
        assert state.tasks[1].status == "retrying"

        # Resume after crash
        def run_mock_resume(request, **kwargs):
            return {
                "overall_status": "completed",
                "run_id": "run_part_2",
                "changed_files": ["backend/app_test.go"],
                "test_status": "passed",
                "commit_hash": "2" * 40,
            }

        mock_instance.run.side_effect = run_mock_resume
        resumed = runner.run_ceo(program_id="crash01", resume=True, autonomous=True)

        assert resumed.status == "completed"
        assert resumed.tasks[0].status == "completed"
        assert resumed.tasks[1].status == "completed"


# ==============================================================================
# 7. Test Sanitization of Program State and Reports
# ==============================================================================
def test_08_program_state_and_reports_are_sanitized(tmp_path: Path):
    repo = setup_test_git_repo(tmp_path / "main_repo")
    pm = ProgramManager(repo)

    dirty_objective = (
        "# Objective with secret\n"
        "Gunakan API key sk-1234567890abcdef dan token Bearer supersecrettoken999 untuk menghubungkan service."
    )

    state = pm.initialize_program(dirty_objective, program_id="sec01")

    # Check JSON
    art_path = repo / ".agent" / "artifacts" / "programs" / "program_sec01.json"
    content = art_path.read_text(encoding="utf-8")
    assert "sk-1234567890abcdef" not in content
    assert "supersecrettoken999" not in content
    assert "Bearer [REDACTED]" in content or "sk-[REDACTED]" in content
