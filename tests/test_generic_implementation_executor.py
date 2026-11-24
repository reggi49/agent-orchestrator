import json
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from agent_system.implementation_executor import (
    ImplementationExecutor,
    LLMImplementationExecutor,
)
from agent_system.project_initializer import initialize_project
from agent_system.schemas import ImplementationResult


ORCHESTRATOR_ROOT = Path(__file__).resolve().parent.parent


def get_runtime_python_files() -> list[Path]:
    files = [ORCHESTRATOR_ROOT / "main.py"]
    for d in ["agent_system", "scripts"]:
        dir_path = ORCHESTRATOR_ROOT / d
        if dir_path.is_dir():
            files.extend(dir_path.rglob("*.py"))
    return [f for f in files if f.is_file() and "__pycache__" not in f.parts]


# 1. Runtime source tidak mengandung PPOB
def test_01_runtime_source_does_not_contain_ppob():
    pattern = re.compile(r"\bppob\b", re.IGNORECASE)
    violations = []
    for f in get_runtime_python_files():
        content = f.read_text(encoding="utf-8")
        matches = pattern.findall(content)
        if matches:
            violations.append(f"{f.relative_to(ORCHESTRATOR_ROOT)}: {len(matches)} matches")
    assert not violations, f"Forbidden 'PPOB' found in runtime files: {violations}"


# 2. Runtime source tidak mengandung WalletService
def test_02_runtime_source_does_not_contain_wallet_service():
    pattern = re.compile(r"WalletService", re.IGNORECASE)
    violations = []
    for f in get_runtime_python_files():
        content = f.read_text(encoding="utf-8")
        if pattern.search(content):
            violations.append(str(f.relative_to(ORCHESTRATOR_ROOT)))
    assert not violations, f"Forbidden 'WalletService' found in runtime files: {violations}"


# 3. Runtime source tidak mengandung task topup
def test_03_runtime_source_does_not_contain_topup_task():
    pattern = re.compile(r"\btopup\b", re.IGNORECASE)
    violations = []
    for f in get_runtime_python_files():
        content = f.read_text(encoding="utf-8")
        if pattern.search(content):
            violations.append(str(f.relative_to(ORCHESTRATOR_ROOT)))
    assert not violations, f"Forbidden 'topup' found in runtime files: {violations}"


# Fixture A: Generic Go project
@pytest.fixture
def generic_go_project(tmp_path):
    proj = tmp_path / "go_proj"
    proj.mkdir()
    svc_dir = proj / "service"
    svc_dir.mkdir()
    (svc_dir / "greeter.go").write_text("package service\n\nfunc Greet(name string) string {\n\treturn \"Hi \" + name\n}\n")
    (svc_dir / "greeter_test.go").write_text("package service\n\nimport \"testing\"\n\nfunc TestGreet(t *testing.T) {}\n")
    return proj


# Fixture B: Generic Python project
@pytest.fixture
def generic_python_project(tmp_path):
    proj = tmp_path / "py_proj"
    proj.mkdir()
    src_dir = proj / "src"
    src_dir.mkdir()
    tests_dir = proj / "tests"
    tests_dir.mkdir()
    (src_dir / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    (tests_dir / "test_calc.py").write_text("from src.calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    return proj


# 5. Executor dapat menerima project Python fixture
def test_05_executor_accepts_python_fixture(generic_python_project):
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "summary": "Implemented multiply function",
        "files_requested_outside_scope": [],
        "file_updates": [
            {
                "path": "src/calc.py",
                "action": "modify",
                "content": "def add(a: int, b: int) -> int:\n    return a + b\n\ndef multiply(a: int, b: int) -> int:\n    return a * b\n",
                "reason": "Add multiply function",
            }
        ],
    })
    mock_model.invoke.return_value = mock_resp

    executor = LLMImplementationExecutor(model=mock_model)
    proposal = {
        "status": "approved_for_implementation",
        "result": {
            "proposal": {
                "proposal_id": "prop_py_01",
                "task_title": "Add multiplication support",
                "files_to_modify": [{"path": "src/calc.py", "action": "modify"}],
                "files_to_create": [],
                "approved_writable_files": ["src/calc.py"],
            }
        }
    }
    ctx = executor.prepare_context(generic_python_project, proposal, ["src/calc.py"])
    res = executor.generate_changes(ctx)

    assert res.status == "implementation_generated"
    assert "src/calc.py" in res.files_changed
    assert len(res.changes) == 1
    assert "def multiply" in res.changes[0].content


# 6. Executor dapat menerima project Go fixture
def test_06_executor_accepts_go_fixture(generic_go_project):
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "summary": "Updated greeting message",
        "files_requested_outside_scope": [],
        "file_updates": [
            {
                "path": "service/greeter.go",
                "action": "modify",
                "content": "package service\n\nfunc Greet(name string) string {\n\treturn \"Hello, \" + name\n}\n",
                "reason": "Change greeting prefix",
            }
        ],
    })
    mock_model.invoke.return_value = mock_resp

    executor = LLMImplementationExecutor(model=mock_model)
    proposal = {
        "status": "approved_for_implementation",
        "result": {
            "proposal": {
                "proposal_id": "prop_go_01",
                "task_title": "Enhance greeting format",
                "files_to_modify": [{"path": "service/greeter.go", "action": "modify"}],
                "files_to_create": [],
                "approved_writable_files": ["service/greeter.go"],
            }
        }
    }
    ctx = executor.prepare_context(generic_go_project, proposal, ["service/greeter.go"])
    res = executor.generate_changes(ctx)

    assert res.status == "implementation_generated"
    assert "service/greeter.go" in res.files_changed
    assert "Hello, " in res.changes[0].content


# 7. Executor tidak bergantung pada bahasa tertentu
def test_07_executor_is_language_agnostic(generic_python_project, generic_go_project):
    mock_model = MagicMock()
    executor = LLMImplementationExecutor(model=mock_model)

    # Process Python
    mock_model.invoke.return_value.content = json.dumps({
        "summary": "python change",
        "file_updates": [{"path": "src/calc.py", "action": "modify", "content": "# py\n"}],
    })
    ctx_py = executor.prepare_context(generic_python_project, {"task_title": "Py task"}, ["src/calc.py"])
    res_py = executor.generate_changes(ctx_py)
    assert res_py.status == "implementation_generated"

    # Process Go with same executor
    mock_model.invoke.return_value.content = json.dumps({
        "summary": "go change",
        "file_updates": [{"path": "service/greeter.go", "action": "modify", "content": "// go\n"}],
    })
    ctx_go = executor.prepare_context(generic_go_project, {"task_title": "Go task"}, ["service/greeter.go"])
    res_go = executor.generate_changes(ctx_go)
    assert res_go.status == "implementation_generated"



# 4. Runtime source tidak mengandung hardcoded approved file PPOB
def test_04_runtime_source_does_not_contain_hardcoded_approved_files_ppob():
    forbidden = [
        "wallet_service.go",
        "api_integration_test.go",
        "TransactionRepository",
        "transactionRepo",
    ]
    violations = []
    for f in get_runtime_python_files():
        content = f.read_text(encoding="utf-8")
        for term in forbidden:
            if term in content:
                violations.append(f"{f.relative_to(ORCHESTRATOR_ROOT)} contains '{term}'")
    assert not violations, f"Forbidden hardcoded terms found: {violations}"


# 8. Executor menggunakan approved proposal
def test_08_executor_uses_approved_proposal(generic_python_project):
    mock_model = MagicMock()
    executor = LLMImplementationExecutor(model=mock_model)
    proposal = {
        "status": "approved_for_implementation",
        "result": {
            "proposal": {
                "proposal_id": "prop_exact_123",
                "task_title": "Custom Task Title",
                "objective": "Objective from proposal",
                "files_to_modify": [{"path": "src/calc.py", "action": "modify"}],
                "implementation_steps": [
                    {"order": 1, "title": "Step 1", "acceptance_criteria": ["Criterion A"]}
                ],
                "validation_commands": [
                    {"executable": "pytest", "arguments": ["-q"], "working_directory": ".", "purpose": "Test"}
                ],
            }
        }
    }
    ctx = executor.prepare_context(generic_python_project, proposal, ["src/calc.py"])
    assert ctx["task_title"] == "Custom Task Title"
    assert ctx["objective"] == "Objective from proposal"
    assert "Criterion A" in ctx["acceptance_criteria"]
    assert len(ctx["validation_commands"]) == 1


# 9. Executor menggunakan repository evidence
def test_09_executor_uses_repository_evidence(tmp_path):
    from agent_system.project_loader import load_project
    proj = tmp_path / "repo_ev"
    proj.mkdir()
    initialize_project(proj, apply=True)
    val_proj = load_project(proj)
    (proj / "src").mkdir()
    (proj / "src" / "target.py").write_text("x = 1\n")
    (proj / "src" / "dep.py").write_text("API_CONSTANT = 42\n")

    executor = LLMImplementationExecutor()
    proposal = {
        "status": "approved_for_implementation",
        "result": {
            "proposal": {
                "task_title": "Task with evidence",
                "files_to_modify": [{"path": "src/target.py", "action": "modify"}],
                "evidence_paths": ["src/dep.py"],
            }
        }
    }
    ctx = executor.prepare_context(
        proj,
        proposal,
        ["src/target.py"],
        policy=val_proj.policy,
    )
    assert "src/dep.py" in ctx["read_only_evidence"]
    assert "API_CONSTANT = 42" in ctx["read_only_evidence"]["src/dep.py"]


# 10. Executor menolak file di luar scope
def test_10_executor_rejects_files_outside_scope(generic_python_project):
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "summary": "Attempting outside scope file",
        "file_updates": [
            {"path": "src/unapproved.py", "action": "create", "content": "print(1)"}
        ],
    })
    mock_model.invoke.return_value = mock_resp

    executor = LLMImplementationExecutor(model=mock_model)
    proposal = {
        "result": {
            "proposal": {
                "task_title": "Scope check",
                "files_to_modify": [{"path": "src/calc.py", "action": "modify"}],
            }
        }
    }
    ctx = executor.prepare_context(generic_python_project, proposal, ["src/calc.py"])
    res = executor.generate_changes(ctx)
    assert res.status == "proposal_revision_required"
    assert "src/unapproved.py" in res.files_requested_outside_scope


# 11. Executor menghasilkan proposal_revision_required untuk file tambahan
def test_11_executor_yields_proposal_revision_for_additional_files(generic_python_project):
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = json.dumps({
        "summary": "Need extra file",
        "files_requested_outside_scope": ["src/extra_helper.py"],
        "outside_scope_reason": "Helper module required for utility functions",
        "file_updates": [],
    })
    mock_model.invoke.return_value = mock_resp

    executor = LLMImplementationExecutor(model=mock_model)
    proposal = {
        "result": {
            "proposal": {
                "task_title": "Needs extra",
                "files_to_modify": [{"path": "src/calc.py", "action": "modify"}],
            }
        }
    }
    ctx = executor.prepare_context(generic_python_project, proposal, ["src/calc.py"])
    res = executor.generate_changes(ctx)
    assert res.status == "proposal_revision_required"
    assert "src/extra_helper.py" in res.files_requested_outside_scope
    assert res.changes == []



# 12. Executor tidak menjalankan command
def test_12_executor_does_not_run_commands(generic_python_project, monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))

    mock_model = MagicMock()
    mock_model.invoke.return_value.content = json.dumps({
        "summary": "No command execution",
        "file_updates": [{"path": "src/calc.py", "action": "modify", "content": "x = 10\n"}],
    })
    executor = LLMImplementationExecutor(model=mock_model)
    proposal = {"result": {"proposal": {"task_title": "t", "files_to_modify": [{"path": "src/calc.py"}]}}}
    ctx = executor.prepare_context(generic_python_project, proposal, ["src/calc.py"])
    executor.generate_changes(ctx)

    assert len(called) == 0, f"Commands were executed: {called}"


# 13. Executor tidak melakukan Git operation
def test_13_executor_does_not_do_git_operations(generic_python_project, monkeypatch):
    git_called = []
    orig_run = subprocess.run
    def mock_run(cmd, *a, **k):
        if isinstance(cmd, list) and cmd and cmd[0] == "git":
            git_called.append(cmd)
        return orig_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", mock_run)
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = json.dumps({
        "summary": "No git operations",
        "file_updates": [{"path": "src/calc.py", "action": "modify", "content": "y = 20\n"}],
    })
    executor = LLMImplementationExecutor(model=mock_model)
    proposal = {"result": {"proposal": {"task_title": "t", "files_to_modify": [{"path": "src/calc.py"}]}}}
    ctx = executor.prepare_context(generic_python_project, proposal, ["src/calc.py"])
    executor.generate_changes(ctx)

    assert len(git_called) == 0, f"Git commands were called: {git_called}"


# 14. Executor tidak menyimpan source template
def test_14_executor_does_not_store_source_templates():
    exec_source = (ORCHESTRATOR_ROOT / "agent_system" / "implementation_executor.py").read_text(encoding="utf-8")
    assert "package " not in exec_source
    assert "func " not in exec_source
    assert "def NewWallet" not in exec_source
    assert "class " in exec_source  # only python class definitions for the executor itself


# 15. Executor tidak mengandung conditional berdasarkan project ID
def test_15_executor_has_no_project_id_conditionals():
    exec_source = (ORCHESTRATOR_ROOT / "agent_system" / "implementation_executor.py").read_text(encoding="utf-8")
    assert "project_id ==" not in exec_source
    assert "project_id in" not in exec_source
    assert "project_name ==" not in exec_source
    assert "project_name in" not in exec_source

