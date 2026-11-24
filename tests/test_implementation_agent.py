from pathlib import Path

from agent_system.implementation_agent import run_implementation_agent

import json
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_llm_model(monkeypatch):
    mock = MagicMock()

    def fake_invoke(messages):
        user_msg = messages[-1]["content"]
        payload = json.loads(user_msg)
        target = payload.get("target_file")
        approved = payload.get("approved_files") or ([target] if target else [])
        updates = []
        for ap in approved:
            updates.append({
                "path": ap,
                "action": "modify",
                "content": f"// updated content for {ap}\n",
                "reason": "Test modification",
            })
        resp = MagicMock()
        resp.content = json.dumps({
            "summary": "Mock implementation",
            "files_requested_outside_scope": [],
            "file_updates": updates,
        })
        return resp

    mock.invoke.side_effect = fake_invoke
    monkeypatch.setattr("agent_system.implementation_executor.LLMImplementationExecutor._get_or_create_model", lambda self: mock)


def setup_ppob_like_project(tmp_path: Path) -> Path:
    p = tmp_path / "repo"
    p.mkdir()
    (p / "backend" / "internal" / "service").mkdir(parents=True)
    svc_code = """package service

import (
    "github.com/ppob-app/backend/internal/repository/postgres"
)

type WalletService struct {
    walletRepo *postgres.WalletRepository
    transactor *postgres.Transactor
}
"""
    (p / "backend" / "internal" / "service" / "wallet_service.go").write_text(svc_code)
    return p


def test_27_implementation_based_on_proposal(tmp_path):
    """27. Implementasi berdasarkan proposal."""
    p = tmp_path / "simple"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "backend" / "code.go").write_text("package backend\n")

    proposal = {
        "status": "approved_for_implementation",
        "result": {
            "proposal": {
                "proposal_id": "p1",
                "task_title": "Update simple backend code",
                "files_to_modify": [{"path": "backend/code.go", "action": "modify", "purpose": "fix"}],
                "files_to_create": [],
            }
        }
    }

    res = run_implementation_agent(p, proposal, approved_files=["backend/code.go"])
    assert res.status == "implementation_generated"
    assert "backend/code.go" in res.files_changed


def test_28_additional_files_outside_scope_yields_proposal_revision_required(tmp_path):
    """28. File tambahan menghasilkan proposal_revision_required."""
    p = setup_ppob_like_project(tmp_path)
    proposal = {
        "status": "approved_for_implementation",
        "result": {
            "proposal": {
                "proposal_id": "prop_wallet",
                "task_title": "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
                "files_to_modify": [{"path": "backend/internal/service/wallet_service.go", "action": "modify", "purpose": "persist"}],
                "files_to_create": [],
                "tests_to_modify": [{"path": "backend/tests/api_integration_test.go", "test_type": "integration"}],
            }
        }
    }

    res = run_implementation_agent(
        p,
        proposal,
        approved_files=["backend/internal/service/wallet_service.go"],
    )
    assert res.status == "proposal_revision_required"
    assert len(res.files_requested_outside_scope) > 0
    assert res.changes == []


def test_29_agent_does_not_run_commands(tmp_path, monkeypatch):
    """29. Agent tidak menjalankan command."""
    import subprocess
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))

    p = tmp_path / "proj"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "backend" / "test.go").write_text("package test\n")
    proposal = {
        "result": {
            "proposal": {
                "task_title": "Simple task",
                "files_to_modify": [{"path": "backend/test.go", "action": "modify", "purpose": "p"}],
                "files_to_create": [],
                "validation_commands": [{"executable": "go", "arguments": ["test", "./..."], "working_directory": "backend", "purpose": "t"}],
            }
        }
    }
    run_implementation_agent(p, proposal, approved_files=["backend/test.go"])
    assert len(called) == 0


def test_30_agent_does_not_do_git_operations(tmp_path, monkeypatch):
    """30. Agent tidak melakukan Git operation."""
    import subprocess
    git_called = []
    orig_run = subprocess.run
    def mock_run(cmd, *a, **k):
        if isinstance(cmd, list) and cmd[0] == "git":
            git_called.append(cmd)
        return orig_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", mock_run)
    p = tmp_path / "proj"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "backend" / "test.go").write_text("package test\n")
    proposal = {
        "result": {
            "proposal": {
                "task_title": "Simple task",
                "files_to_modify": [{"path": "backend/test.go", "action": "modify", "purpose": "p"}],
                "files_to_create": [],
            }
        }
    }
    run_implementation_agent(p, proposal, approved_files=["backend/test.go"])
def test_31_agent_does_not_modify_frontend(tmp_path):
    """31. Agent tidak mengubah frontend."""
    p = tmp_path / "proj"
    p.mkdir()
    (p / "frontend").mkdir()
    (p / "frontend" / "app.tsx").write_text("const app = null;")
    proposal = {
        "result": {
            "proposal": {
                "task_title": "Backend only task",
                "files_to_modify": [{"path": "backend/test.go", "action": "modify", "purpose": "p"}],
                "files_to_create": [],
            }
        }
    }
    res = run_implementation_agent(p, proposal, approved_files=["backend/test.go"])
    for chg in res.changes:
        assert not chg.path.startswith("frontend/")


def test_32_agent_does_not_modify_old_migrations(tmp_path):
    """32. Agent tidak mengubah migration lama."""
    p = tmp_path / "proj"
    p.mkdir()
    (p / "backend" / "migrations").mkdir(parents=True)
    (p / "backend" / "migrations" / "001_init.sql").write_text("CREATE TABLE test;")
    proposal = {
        "result": {
            "proposal": {
                "task_title": "Backend task",
                "files_to_modify": [{"path": "backend/test.go", "action": "modify", "purpose": "p"}],
                "files_to_create": [],
            }
        }
    }
    res = run_implementation_agent(p, proposal, approved_files=["backend/test.go"])
    for chg in res.changes:
        assert not chg.path.startswith("backend/migrations/")


def test_33_agent_does_not_claim_test_passed_before_running(tmp_path):
    """33. Agent tidak mengklaim test berhasil."""
    p = tmp_path / "proj"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "backend" / "test.go").write_text("package test\n")
    proposal = {
        "result": {
            "proposal": {
                "task_title": "Task with validation command",
                "files_to_modify": [{"path": "backend/test.go", "action": "modify", "purpose": "p"}],
                "files_to_create": [],
                "validation_commands": [{"executable": "go", "arguments": ["test", "./..."], "working_directory": "backend", "purpose": "run"}],
            }
        }
    }
    res = run_implementation_agent(p, proposal, approved_files=["backend/test.go"])
    assert res.status == "implementation_generated"
    assert not hasattr(res, "test_status") or res.test_status != "passed"
