import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_system.artifact_store import ProjectArtifactStore
from agent_system.nodes import build_fallback_proposal, implementation_proposal_node
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
from agent_system.proposal_agent import (
    detect_code_or_patch_in_text,
    validate_all_paths_relative,
)
from agent_system.proposal_reviewer import (
    review_proposal_deterministically,
    validate_commands,
)
from agent_system.schemas import ImplementationProposal, ProposedCommand


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


def test_01_proposal_created_from_verified_plan(mock_project, plan_data):
    """1. Proposal dibuat dari verified plan."""
    state = {
        "project_root": str(mock_project),
        "source_plan_path": ".agent/artifacts/plans/plan_test.json",
        "plan_data": plan_data,
        "task_verification": {"status": "verified_pending"},
        "proposal_hashes": [],
    }
    with patch("agent_system.nodes.proposal_model") as mock_model:
        mock_model.invoke.side_effect = Exception("Fallback")
        out = implementation_proposal_node(state)
        prop = out["implementation_proposal"]
        assert prop["task_title"] == "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup"
        assert prop["owner"] == "backend"
        assert prop["source_plan_path"] == ".agent/artifacts/plans/plan_test.json"


def test_02_proposal_rejected_without_verified_plan(mock_project):
    """2. Proposal ditolak tanpa verified plan."""
    state = {
        "project_root": str(mock_project),
        "source_plan_path": ".agent/artifacts/plans/plan_test.json",
        "plan_data": {},
        "task_verification": {},
        "proposal_hashes": [],
    }
    with pytest.raises(ValueError, match="Cannot generate proposal without a valid task title"):
        implementation_proposal_node(state)


def test_03_all_paths_relative(plan_data):
    """3. Semua path relatif."""
    proposal = build_fallback_proposal(
        task_title="Test Task",
        task_owner="backend",
        task_desc="Desc",
        acceptance_criteria=["Criterion 1"],
        source_plan_path=".agent/artifacts/plans/plan_test.json",
        verification_status="verified_pending",
        evidence_paths=["backend/internal/service/wallet_service.go"],
    )
    assert validate_all_paths_relative(proposal) == []


def test_04_absolute_path_rejected():
    """4. Absolute path ditolak."""
    p = {"files_to_modify": [{"path": "/Users/someone/backend/service.go"}], "evidence_paths": []}
    assert any("Absolute path found" in err for err in validate_all_paths_relative(p))


def test_05_path_traversal_rejected():
    """5. Path traversal ditolak."""
    p = {"files_to_modify": [{"path": "../outside/service.go"}], "evidence_paths": []}
    assert any("Path traversal" in err for err in validate_all_paths_relative(p))


def test_06_file_change_matches_component_owner(mock_project, plan_data):
    """6. File perubahan sesuai component owner."""
    val_proj = load_project(mock_project)
    proposal = build_fallback_proposal(
        task_title="Test Task",
        task_owner="backend",
        task_desc="Desc",
        acceptance_criteria=["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        source_plan_path=".agent/artifacts/plans/plan_test.json",
        verification_status="verified_pending",
        evidence_paths=["backend/internal/service/wallet_service.go"],
    )
    review = review_proposal_deterministically(proposal, plan_data, val_proj.manifest, val_proj.policy)
    assert review.scope_violations == []
    assert "backend/internal/service/wallet_service.go" in review.approved_files


def test_07_unsupported_file_change_rejected(mock_project, plan_data):
    """7. Unsupported file change ditolak."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Test", "backend", "Desc", ["Criterion"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"]
    ).model_dump(mode="json")
    prop["files_to_modify"].append({"path": "backend/service/random.go", "action": "modify", "purpose": "", "changes": []})
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert "backend/service/random.go" in review.unsupported_changes


def test_08_proposal_no_full_source_code():
    """8. Proposal tidak berisi source code lengkap."""
    assert any("source code implementation" in i for i in detect_code_or_patch_in_text("package service\nfunc TopUp() {}"))


def test_09_proposal_no_patch():
    """9. Proposal tidak berisi patch."""
    assert any("git diff/patch" in i for i in detect_code_or_patch_in_text("--- a/w.go\n+++ b/w.go\n@@ -1 +1 @@"))


def test_10_proposal_no_diff():
    """10. Proposal tidak berisi diff."""
    assert any("git diff/patch" in i for i in detect_code_or_patch_in_text("diff --git a/w.go b/w.go"))


def test_11_every_criterion_has_test_mapping(mock_project, plan_data):
    """11. Setiap acceptance criterion memiliki test mapping."""
    val_proj = load_project(mock_project)
    proposal = build_fallback_proposal(
        "Test", "backend", "Desc",
        ["Menyimpan record baru pada tabel transactions", "Atomic update wallet balance dan transaction record"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    )
    review = review_proposal_deterministically(proposal, plan_data, val_proj.manifest, val_proj.policy)
    assert review.missing_acceptance_criteria == []


def test_12_missing_test_mapping_detected(mock_project, plan_data):
    """12. Missing test mapping ditemukan."""
    val_proj = load_project(mock_project)
    proposal = build_fallback_proposal(
        "Test", "backend", "Desc", ["Unmapped Criterion"],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    )
    review = review_proposal_deterministically(proposal, plan_data, val_proj.manifest, val_proj.policy)
    assert "Menyimpan record baru pada tabel transactions" in review.missing_acceptance_criteria


def test_13_rollback_plan_mandatory(mock_project, plan_data):
    """13. Rollback plan wajib tersedia."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Test", "backend", "Desc", [],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["rollback_plan"] = []
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert any(f.category == "rollback" for f in review.findings)


def test_14_out_of_scope_mandatory(mock_project, plan_data):
    """14. Out-of-scope wajib tersedia."""
    val_proj = load_project(mock_project)
    prop = build_fallback_proposal(
        "Test", "backend", "Desc", [],
        ".agent/artifacts/plans/plan_test.json", "verified_pending",
        ["backend/internal/service/wallet_service.go"],
    ).model_dump(mode="json")
    prop["out_of_scope"] = []
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert any(f.category == "scope" for f in review.findings)



def test_15_command_stored_as_executable_and_arguments():
    """15. Command disimpan sebagai executable dan arguments."""
    cmd = ProposedCommand(
        executable="go",
        arguments=["test", "./..."],
        working_directory="backend",
        purpose="Run tests",
        requires_approval=True,
    )
    assert cmd.executable == "go"
    assert cmd.arguments == ["test", "./..."]


def test_16_shell_command_string_rejected():
    """16. Shell command string ditolak."""
    commands = [{"executable": "go", "arguments": "test ./...", "requires_approval": True}]
    findings, _ = validate_commands(commands)
    assert any("arguments must be a list" in f.message for f in findings)


def test_17_command_destructive_rejected():
    """17. Command destructive ditolak."""
    commands = [{"executable": "rm", "arguments": ["-rf", "."], "requires_approval": True}]
    findings, _ = validate_commands(commands)
    assert any("Forbidden executable" in f.message for f in findings)


def test_18_command_requires_approval_must_be_true():
    """18. Command requires_approval harus true."""
    commands = [{"executable": "go", "arguments": ["test", "./..."], "requires_approval": False}]
    findings, _ = validate_commands(commands)
    assert any("requires_approval: true" in f.message for f in findings)


def test_19_proposal_artifact_in_target_project(mock_project):
    """19. Proposal artifact berada di project target."""
    val_proj = load_project(mock_project)
    store = ProjectArtifactStore(val_proj.root, val_proj.manifest)
    prop_path = store.write_proposal("prop_001", {"title": "Test Proposal"})
    assert prop_path.is_file()
    assert str(prop_path).startswith(str(mock_project))
    assert ".agent/artifacts/proposals" in str(prop_path)


def test_20_proposal_artifact_not_in_orchestrator(mock_project):
    """20. Proposal artifact tidak berada di orchestrator."""
    val_proj = load_project(mock_project)
    store = ProjectArtifactStore(val_proj.root, val_proj.manifest)
    prop_path = store.write_proposal("prop_002", {"title": "Test Proposal"})
    orch_agent = (Path(__file__).resolve().parent.parent / ".agent").resolve()
    assert not str(prop_path).startswith(str(orch_agent))

