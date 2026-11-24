import pytest
from pathlib import Path

from agent_system.nodes import build_fallback_proposal, proposal_reviewer_node
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
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


def test_21_valid_proposal_results_in_approved(mock_project, plan_data):
    """21. Proposal valid menghasilkan approved."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    )
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status == "approved"


def test_22_overly_broad_proposal_results_in_revision_required(mock_project, plan_data):
    """22. Proposal terlalu luas menghasilkan revision_required."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["files_to_modify"].append({
        "path": "frontend/components/ui/button.tsx",
        "action": "modify",
        "purpose": "Broad change outside scope",
        "changes": ["Change UI button"],
    })
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status == "revision_required"
    assert "frontend/components/ui/button.tsx" in review.scope_violations


def test_23_proposal_with_absolute_path_rejected(mock_project, plan_data):
    """23. Proposal dengan absolute path ditolak."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["files_to_modify"].append({
        "path": "/etc/hosts",
        "action": "modify",
        "purpose": "Hacking",
        "changes": ["Add bad route"],
    })
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status in {"revision_required", "human_required"}
    assert any("Absolute path" in f.message for f in review.findings)


def test_24_proposal_with_dangerous_command_rejected(mock_project, plan_data):
    """24. Proposal dengan command berbahaya ditolak."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["validation_commands"].append({
        "executable": "bash",
        "arguments": ["-c", "rm -rf /"],
        "requires_approval": True,
    })
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status in {"revision_required", "human_required"}
    assert any(f.category == "command_safety" for f in review.findings)


def test_25_proposal_without_rollback_results_in_revision_required(mock_project, plan_data):
    """25. Proposal tanpa rollback menghasilkan revision_required."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["rollback_plan"] = []
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status == "revision_required"
    assert any(f.category == "rollback" for f in review.findings)


def test_26_proposal_without_test_results_in_revision_required(mock_project, plan_data):
    """26. Proposal tanpa test menghasilkan revision_required."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["tests_to_modify"] = []
    prop["tests_to_create"] = []
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status == "revision_required"
    assert any(f.category == "testing" for f in review.findings)


def test_27_proposal_critical_security_finding_results_in_human_required(mock_project, plan_data):
    """27. Proposal dengan critical security finding menghasilkan human_required."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["validation_commands"].append({
        "executable": "go",
        "arguments": ["test", "./... && rm -rf /"],
        "requires_approval": True,
    })
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status == "human_required"


def test_28_proposal_modifying_other_component_without_evidence_rejected(mock_project, plan_data):
    """28. Proposal yang mengubah component lain tanpa evidence ditolak."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["files_to_modify"].append({
        "path": "frontend/app/page.tsx",
        "action": "modify",
        "purpose": "Modify home page",
        "changes": ["Change title"],
    })
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert "frontend/app/page.tsx" in review.scope_violations
    assert "frontend/app/page.tsx" in review.rejected_files


def test_29_proposal_deleting_file_without_evidence_rejected(mock_project, plan_data):
    """29. Proposal yang menghapus file tanpa evidence ditolak."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["files_to_delete"].append({
        "path": "backend/cmd/api/main.go",
        "action": "delete",
        "purpose": "Delete main",
        "changes": ["Remove file"],
        "evidence_paths": [],
    })
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert "backend/cmd/api/main.go" in review.rejected_files


def test_30_reviewer_does_not_modify_source_code(mock_project, plan_data, monkeypatch):
    """30. Reviewer tidak memodifikasi source code."""
    from unittest.mock import MagicMock
    import agent_system.nodes as nodes
    monkeypatch.setattr(nodes, "proposal_reviewer_model", MagicMock())
    source_file = mock_project / "backend" / "go.mod"
    original_mtime = source_file.stat().st_mtime
    original_content = source_file.read_text()

    state = {
        "project_root": str(mock_project),
        "source_plan_path": ".agent/artifacts/plans/plan_test.json",
        "plan_data": plan_data,
        "implementation_proposal": build_fallback_proposal(
            "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
            "backend", "Desc",
            ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
            ".agent/artifacts/plans/plan_test.json", "verified_pending",
            ["backend/internal/service/wallet_service.go"],
        ).model_dump(mode="json"),
        "proposal_hashes": [],
    }
    proposal_reviewer_node(state)
    assert source_file.stat().st_mtime == original_mtime
    assert source_file.read_text() == original_content

