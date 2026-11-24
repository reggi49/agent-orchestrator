import pytest
from unittest.mock import MagicMock, patch

from agent_system.graph import route_after_verification
from agent_system.nodes import (
    check_migration_task_status,
    supervisor_node,
    task_verifier_node,
)
from agent_system.schemas import AgentState, TaskVerification
from agent_system.task_verifier import (
    are_tasks_substantially_identical,
    is_duplicate_task,
    normalize_task_title,
)


def test_23_migration_already_implemented_when_complete():
    evidence_results = [
        {"path": "backend/migrations/001_create_users.sql", "excerpt": "CREATE TABLE USERS (\n  id UUID\n);"},
        {"path": "backend/migrations/002_create_wallets.sql", "excerpt": "CREATE TABLE WALLETS (\n  id UUID\n);"},
        {"path": "backend/migrations/003_create_products.sql", "excerpt": "CREATE TABLE PRODUCTS (\n  id UUID\n);"},
        {"path": "backend/migrations/004_create_transactions.sql", "excerpt": "CREATE TABLE TRANSACTIONS (\n  id UUID\n);"},
    ]
    verification = check_migration_task_status(
        task_title="Define and create PostgreSQL database schema with migrations",
        task_description="Create initial database migrations for users, wallets, products, and transactions.",
        evidence_results=evidence_results,
    )
    assert verification is not None
    assert verification.status == "already_implemented"
    assert verification.should_replan is True
    assert len(verification.evidence_for) == 4
    for ev in verification.evidence_for:
        assert ev.startswith("backend/migrations/")


def test_24_migration_partially_implemented_when_gap_exists():
    evidence_results = [
        {"path": "backend/migrations/001_create_users.sql", "excerpt": "CREATE TABLE USERS (\n  id UUID\n);"},
        {"path": "backend/migrations/002_create_wallets.sql", "excerpt": "CREATE TABLE WALLETS (\n  id UUID\n);"},
        {"path": "backend/migrations/003_create_products.sql", "excerpt": "CREATE TABLE PRODUCTS (\n  id UUID\n);"},
    ]
    verification = check_migration_task_status(
        task_title="Define and create PostgreSQL database schema with migrations",
        task_description="Create migrations",
        evidence_results=evidence_results,
    )
    assert verification is not None
    assert verification.status == "partially_implemented"
    assert "transactions" in verification.missing_work[0]


def test_25_gap_not_invented_without_evidence():
    # When all are found, missing_work is empty
    evidence_results = [
        {"path": "backend/migrations/001_create_users.sql", "excerpt": "CREATE TABLE USERS (\n  id UUID\n);"},
        {"path": "backend/migrations/002_create_wallets.sql", "excerpt": "CREATE TABLE WALLETS (\n  id UUID\n);"},
        {"path": "backend/migrations/003_create_products.sql", "excerpt": "CREATE TABLE PRODUCTS (\n  id UUID\n);"},
        {"path": "backend/migrations/004_create_transactions.sql", "excerpt": "CREATE TABLE TRANSACTIONS (\n  id UUID\n);"},
    ]
    verification = check_migration_task_status(
        task_title="Define and create PostgreSQL database schema with migrations",
        task_description="Create migrations",
        evidence_results=evidence_results,
    )
    assert verification.missing_work == []


def test_26_27_source_code_trumps_tasks_md_and_handoffs():
    # Even if TASKS.md had unchecked box, the existence of actual SQL table statements marks it already_implemented
    evidence_results = [
        {"path": "backend/migrations/001_create_users.sql", "excerpt": "CREATE TABLE USERS (id UUID);"},
        {"path": "backend/migrations/002_create_wallets.sql", "excerpt": "CREATE TABLE WALLETS (id UUID);"},
        {"path": "backend/migrations/003_create_products.sql", "excerpt": "CREATE TABLE PRODUCTS (id UUID);"},
        {"path": "backend/migrations/004_create_transactions.sql", "excerpt": "CREATE TABLE TRANSACTIONS (id UUID);"},
    ]
    res = check_migration_task_status(
        "Define and create PostgreSQL database schema with migrations",
        "Migration task",
        evidence_results,
    )
    assert res.status == "already_implemented"


def test_28_evidence_path_must_be_relative():
    evidence_results = [
        {"path": "backend/migrations/001_create_users.sql", "excerpt": "CREATE TABLE USERS (id UUID);"},
        {"path": "backend/migrations/002_create_wallets.sql", "excerpt": "CREATE TABLE WALLETS (id UUID);"},
        {"path": "backend/migrations/003_create_products.sql", "excerpt": "CREATE TABLE PRODUCTS (id UUID);"},
        {"path": "backend/migrations/004_create_transactions.sql", "excerpt": "CREATE TABLE TRANSACTIONS (id UUID);"},
    ]
    res = check_migration_task_status(
        "Define and create PostgreSQL database schema with migrations",
        "Migrations",
        evidence_results,
    )
    for ev in res.evidence_for:
        path = ev.split(":")[0].strip()
        assert not path.startswith("/")
def test_29_verified_pending_routes_to_supervisor():
    state: AgentState = {
        "task_verification": {"status": "verified_pending", "missing_work": []},
        "replan_count": 0,
        "max_replans": 2,
    }
    assert route_after_verification(state) == "supervisor"


def test_30_partially_implemented_with_missing_work_routes_to_supervisor():
    state: AgentState = {
        "task_verification": {"status": "partially_implemented", "missing_work": ["add 004 migration"]},
        "replan_count": 0,
        "max_replans": 2,
    }
    assert route_after_verification(state) == "supervisor"


def test_31_already_implemented_routes_to_analyst_replan():
    state: AgentState = {
        "task_verification": {"status": "already_implemented", "missing_work": []},
        "replan_count": 0,
        "max_replans": 2,
    }
    assert route_after_verification(state) == "analyst"


def test_32_invalid_task_routes_to_analyst_replan():
    state: AgentState = {
        "task_verification": {"status": "invalid_task", "missing_work": []},
        "replan_count": 0,
        "max_replans": 2,
    }
    assert route_after_verification(state) == "analyst"


def test_33_insufficient_evidence_routes_to_analyst_replan():
    state: AgentState = {
        "task_verification": {"status": "insufficient_evidence", "missing_work": []},
        "replan_count": 0,
        "max_replans": 2,
    }
    assert route_after_verification(state) == "analyst"


def test_34_max_replans_routes_to_supervisor_and_yields_human_required():
    state: AgentState = {
        "task_verification": {"status": "already_implemented", "missing_work": []},
        "replan_count": 2,
        "max_replans": 2,
        "analysis": {
            "recommended_task": {
                "title": "Define and create PostgreSQL database schema with migrations",
                "owner": "backend",
                "acceptance_criteria": ["run migrations"],
            }
        },
    }
    assert route_after_verification(state) == "supervisor"
    sup_res = supervisor_node(state)
    assert sup_res["status"] == "human_required"


def test_35_duplicate_task_rejected():
    rejected = [
        {"title": "Define and create PostgreSQL database schema with migrations", "status": "already_implemented"}
    ]
    assert is_duplicate_task("Define and create PostgreSQL database schema with migrations", rejected) is True
    assert is_duplicate_task("Create PostgreSQL database schema with migrations", rejected) is True
    assert is_duplicate_task("Setup Docker and Redis cache", rejected) is False


def test_36_supervisor_rejects_plan_without_verification():
    state: AgentState = {
        "analysis": {
            "recommended_task": {
                "title": "Implement auth",
                "owner": "backend",
                "acceptance_criteria": ["JWT working"],
            }
        },
        "task_verification": None,
        "replan_count": 0,
    }
    res = supervisor_node(state)
    assert res["status"] == "human_required"
    assert "TaskVerification belum tersedia" in res["final_summary"]


def test_37_supervisor_rejects_plan_without_evidence_path():
    state: AgentState = {
        "analysis": {
            "recommended_task": {
                "title": "Implement auth",
                "owner": "backend",
                "acceptance_criteria": ["JWT working"],
            }
        },
        "task_verification": {
            "status": "verified_pending",
            "evidence_for": [],
        },
        "repository_evidence": {"results": []},
        "replan_count": 0,
    }
    res = supervisor_node(state)
    assert res["status"] == "human_required"
    assert "Evidence path" in res["final_summary"]


def test_38_supervisor_narrows_partially_implemented_task():
    state: AgentState = {
        "analysis": {
            "recommended_task": {
                "title": "Setup database migrations",
                "owner": "backend",
                "acceptance_criteria": ["Create all 4 migrations"],
            }
        },
        "task_verification": {
            "status": "partially_implemented",
            "evidence_for": ["backend/migrations/001_create_users.sql: table users exists"],
            "missing_work": ["Create migration SQL for transactions"],
        },
        "repository_evidence": {
            "results": [{"path": "backend/migrations/001_create_users.sql"}]
        },
        "replan_count": 0,
    }
    res = supervisor_node(state)
    assert res["status"] == "plan_ready"
    narrowed_task = res["analysis"]["recommended_task"]
    assert "transactions" in narrowed_task["title"] or "gap" in narrowed_task["title"]
    assert any("transactions" in ac for ac in narrowed_task["acceptance_criteria"])


def test_39_rejected_task_saved_in_state():
    state: AgentState = {
        "analysis": {
            "recommended_task": {
                "title": "Define and create PostgreSQL database schema with migrations",
                "description": "Create schema",
            }
        },
        "repository_evidence": {
            "results": [
                {"path": "backend/migrations/001_create_users.sql", "excerpt": "CREATE TABLE USERS (\n id UUID\n);"},
                {"path": "backend/migrations/002_create_wallets.sql", "excerpt": "CREATE TABLE WALLETS (\n id UUID\n);"},
                {"path": "backend/migrations/003_create_products.sql", "excerpt": "CREATE TABLE PRODUCTS (\n id UUID\n);"},
                {"path": "backend/migrations/004_create_transactions.sql", "excerpt": "CREATE TABLE TRANSACTIONS (\n id UUID\n);"},
            ]
        },
        "replan_count": 0,
        "max_replans": 2,
        "rejected_tasks": [],
    }
    res = task_verifier_node(state)
    assert res["replan_count"] == 1
    assert len(res["rejected_tasks"]) == 1
    assert res["rejected_tasks"][0]["title"] == "Define and create PostgreSQL database schema with migrations"
    assert res["rejected_tasks"][0]["status"] == "already_implemented"


def test_40_no_network_called(monkeypatch):
    mock_model = MagicMock()
    mock_model.invoke.side_effect = AssertionError("Network/LLM must not be called")
    monkeypatch.setattr("agent_system.nodes.verifier_model", mock_model)

    state: AgentState = {
        "analysis": {
            "recommended_task": {
                "title": "Define and create PostgreSQL database schema with migrations",
                "description": "database migrations",
            }
        },
        "repository_evidence": {
            "results": [
                {"path": "backend/migrations/001_create_users.sql", "excerpt": "CREATE TABLE USERS (id UUID);"},
                {"path": "backend/migrations/002_create_wallets.sql", "excerpt": "CREATE TABLE WALLETS (id UUID);"},
                {"path": "backend/migrations/003_create_products.sql", "excerpt": "CREATE TABLE PRODUCTS (id UUID);"},
                {"path": "backend/migrations/004_create_transactions.sql", "excerpt": "CREATE TABLE TRANSACTIONS (id UUID);"},
            ]
        },
        "replan_count": 0,
        "max_replans": 2,
        "rejected_tasks": [],
    }
    res = task_verifier_node(state)
    assert res["task_verification"]["status"] == "already_implemented"
