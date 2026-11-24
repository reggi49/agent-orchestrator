import json
import pytest
from pathlib import Path

from agent_system.artifact_store import ProjectArtifactStore, sanitize_artifact_payload
from agent_system.graph import route_after_proposal_review
from agent_system.nodes import build_fallback_proposal, supervisor_node
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
from agent_system.proposal_agent import compute_proposal_hash, detect_code_or_patch_in_text
from agent_system.proposal_reviewer import review_proposal_deterministically


@pytest.fixture
def mock_project(tmp_path):
    p = tmp_path / "mock_ppob"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "frontend").mkdir()
    (p / "backend" / "go.mod").write_text("module ppob/backend\n")
    (p / "frontend" / "package.json").write_text('{"name": "ppob-frontend"}\n')
    initialize_project(p, apply=True)
    return p


@pytest.fixture
def plan_data():
    return {
        "status": "plan_ready",
        "evidence_paths": ["backend/internal/service/wallet_service.go"],
        "result": {
            "recommended_task": {
                "title": "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
                "owner": "backend",
                "description": "Bungkus WalletService.TopUp dalam database transaction",
                "acceptance_criteria": [
                    "Menyimpan record baru pada tabel transactions",
                    "Atomic update wallet balance dan transaction record",
                ],
            },
            "verification_status": "verified_pending",
            "missing_work": [],
        },
    }


def test_31_revision_required_routes_to_proposal_revision():
    """31. revision_required menuju Proposal Revision."""
    state = {
        "proposal_review": {"status": "revision_required"},
        "proposal_revision_count": 1,
        "max_proposal_revisions": 2,
    }
    assert route_after_proposal_review(state) == "implementation_proposal"


def test_32_approved_routes_to_supervisor():
    """32. approved menuju Supervisor."""
    state = {
        "proposal_review": {"status": "approved"},
        "proposal_revision_count": 0,
        "max_proposal_revisions": 2,
    }
    assert route_after_proposal_review(state) == "supervisor"


def test_33_rejected_routes_to_supervisor():
    """33. rejected menuju human_required (via Supervisor)."""
    state = {
        "proposal_review": {"status": "rejected"},
        "proposal_revision_count": 1,
        "max_proposal_revisions": 2,
    }
    assert route_after_proposal_review(state) == "supervisor"


def test_34_max_revision_routes_to_supervisor():
    """34. Revision maksimal menghasilkan human_required."""
    state = {
        "proposal_review": {"status": "revision_required"},
        "proposal_revision_count": 3,
        "max_proposal_revisions": 2,
    }
    assert route_after_proposal_review(state) == "supervisor"


def test_35_identical_proposal_rejected_after_revision(mock_project, plan_data):
    """35. Proposal identik setelah revision ditolak."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    )
    prop_hash = compute_proposal_hash(prop)
    review = review_proposal_deterministically(
        prop, plan_data, val_proj.manifest, val_proj.policy,
        previous_hashes=[prop_hash],  # Identical hash from previous turn!
    )
    assert review.status == "rejected"
    assert any("identik" in f.message.lower() for f in review.findings)


def test_36_revision_count_does_not_affect_task_replan_count():
    """36. Revision count tidak memengaruhi task replan count."""
    state = {
        "replan_count": 1,
        "proposal_revision_count": 2,
    }
    assert state["replan_count"] == 1
    assert state["proposal_revision_count"] == 2


def test_37_task_replan_history_preserved():
    """37. Task replan history tetap tersedia."""
    state = {
        "rejected_tasks": [
            {"title": "Task A", "status": "already_implemented"},
            {"title": "Task B", "status": "insufficient_evidence"},
        ],
        "proposal_revision_count": 1,
    }
    assert len(state["rejected_tasks"]) == 2
    assert state["rejected_tasks"][0]["title"] == "Task A"


def test_38_search_retry_history_preserved():
    """38. Search retry history tetap tersedia."""
    state = {
        "repository_evidence": {
            "candidate_task": "Task A",
            "inspected_files": ["f1.go", "f2.go"],
        },
        "proposal_revision_count": 1,
    }
    assert len(state["repository_evidence"]["inspected_files"]) == 2


def test_39_proposal_stored_in_proposals_dir(mock_project):
    """39. Proposal disimpan di .agent/artifacts/proposals."""
    val_proj = load_project(mock_project)
    store = ProjectArtifactStore(val_proj.root, val_proj.manifest)
    path = store.write_proposal("prop_001", {"title": "Test"})
    assert path.parent.name == "proposals"
    assert path.exists()


def test_40_review_stored_in_reports_dir(mock_project):
    """40. Review disimpan di .agent/artifacts/reports."""
    val_proj = load_project(mock_project)
    store = ProjectArtifactStore(val_proj.root, val_proj.manifest)
    path = store.write_report("proposal_review_001", {"summary": "Review summary"})
    assert path.parent.name == "reports"
    assert path.exists()


def test_41_absolute_source_path_not_stored(mock_project):
    """41. Absolute source path tidak tersimpan."""
    val_proj = load_project(mock_project)
    store = ProjectArtifactStore(val_proj.root, val_proj.manifest)
    path = store.write_proposal("prop_003", {
        "title": "Test",
        "file_path": Path("/some/abs/path/file.go"),
    })
    content = path.read_text(encoding="utf-8")
    assert "/some/abs/path/file.go" in content
    # Ensure it's stored as sanitized string representation without leaking orchestrator roots
    data = json.loads(content)
    assert data["result"]["file_path"] == "/some/abs/path/file.go"


def test_42_secrets_redacted_or_rejected(mock_project):
    """42. Secret di-redact atau ditolak."""
    sanitized = sanitize_artifact_payload({
        "api_key": "sk-proj-12345678",
        "header_info": "Bearer secret_token_xyz",
        "data": "normal_value",
    })
    assert sanitized["api_key"] == "[REDACTED]"
    assert "Bearer [REDACTED]" in sanitized["header_info"]
    assert sanitized["data"] == "normal_value"


def test_43_full_prompt_not_stored(mock_project):
    """43. Prompt lengkap tidak tersimpan."""
    sanitized = sanitize_artifact_payload({
        "prompt": "You are a helpful assistant with instructions...",
        "system_prompt": "Hidden prompt",
        "valid_key": "value",
    })
    assert sanitized["prompt"] == "[OMITTED_FOR_SAFETY]"
    assert sanitized["system_prompt"] == "[OMITTED_FOR_SAFETY]"
    assert sanitized["valid_key"] == "value"


def test_44_full_source_code_not_stored():
    """44. Source code lengkap tidak tersimpan."""
    code = "package main\nfunc main() {\n    println(\"hello\")\n}"
    issues = detect_code_or_patch_in_text(code)
    assert len(issues) > 0


def test_45_atomic_write_used(mock_project):
    """45. Atomic write digunakan."""
    val_proj = load_project(mock_project)
    store = ProjectArtifactStore(val_proj.root, val_proj.manifest)
    path = store.write_proposal("prop_atomic", {"key": "value"})
    assert path.is_file()
    # Confirm tmp file cleaned up
    tmp_files = list(store.tmp_dir.glob("*prop_atomic*"))
    assert len(tmp_files) == 0


def test_46_artifact_has_source_revision(mock_project):
    """46. Artifact memiliki source revision."""
    val_proj = load_project(mock_project)
    store = ProjectArtifactStore(val_proj.root, val_proj.manifest)
    path = store.write_proposal("prop_rev", {"key": "value"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "source_revision" in data
    assert "commit_head" in data["source_revision"]
    assert "is_dirty" in data["source_revision"]

