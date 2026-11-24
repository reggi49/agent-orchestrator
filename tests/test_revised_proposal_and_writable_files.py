import json
import subprocess
from pathlib import Path
import pytest

from agent_system.graph import build_implementation_graph
from agent_system.implementation_preflight import (
    inspect_scope_requirements,
    run_implementation_preflight,
)
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project
from agent_system.proposal_agent import (
    compute_approved_writable_files,
    validate_all_paths_relative,
    validate_approved_writable_files,
)
from agent_system.proposal_reviewer import review_proposal_deterministically
from agent_system.schemas import (
    ImplementationProposal,
    ProposedCommand,
    ProposedFileChange,
    ProposedTest,
)


def get_git_head(p: Path) -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True, check=True)
    return res.stdout.strip()


@pytest.fixture
def mock_ppob_project(tmp_path):
    p = tmp_path / "mock_ppob"
    p.mkdir()

    subprocess.run(["git", "init", "-b", "main"], cwd=p, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "agent@orchestrator.local"], cwd=p, check=True)
    subprocess.run(["git", "config", "user.name", "Agent Orchestrator"], cwd=p, check=True)

    (p / "backend").mkdir()
    (p / "backend" / "cmd" / "api").mkdir(parents=True)
    (p / "backend" / "internal" / "service").mkdir(parents=True)
    (p / "backend" / "tests").mkdir(parents=True)

    (p / "backend" / "go.mod").write_text("module ppob/backend\n")
    (p / "backend" / "cmd" / "api" / "main.go").write_text("package main\nfunc main() {}\n")
    (p / "backend" / "internal" / "service" / "wallet_service.go").write_text(
        "package service\ntype WalletService struct {}\nfunc NewWalletService() *WalletService { return nil }\nfunc TopUp() {}\n"
    )
    (p / "backend" / "tests" / "api_integration_test.go").write_text(
        "package tests\nimport \"testing\"\nfunc TestWalletTopUp(t *testing.T) {}\n"
    )

    subprocess.run(["git", "add", "."], cwd=p, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=p, check=True)

    initialize_project(p, apply=True)
    val_proj = load_project(p)
    from agent_system.artifact_store import ProjectArtifactStore
    store = ProjectArtifactStore(p, val_proj.manifest)
    store.write_plan("plan_bca7ca8a", {
        "status": "plan_ready",
        "result": {
            "recommended_task": {
                "title": "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
                "owner": "backend",
            }
        }
    })
    store.write_proposal("proposal_revised", {
        "proposal_id": "prop_revised_001",
        "task_title": "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
    })
    return p


@pytest.fixture
def plan_data():
    return {
        "status": "plan_ready",
        "evidence_paths": [
            "backend/internal/service/wallet_service.go",
            "backend/cmd/api/main.go",
            "backend/tests/api_integration_test.go",
        ],
        "result": {
            "recommended_task": {
                "title": "Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
                "owner": "backend",
                "description": "Bungkus WalletService.TopUp dalam database transaction",
                "acceptance_criteria": [
                    "Menyimpan record baru pada tabel transactions",
                    "Atomic update wallet balance dan transaction record",
                    "Rollback jika gagal",
                    "Verifikasi test",
                ],
            },
            "verification_status": "verified_pending",
            "missing_work": [],
        },
    }


def build_valid_revised_proposal() -> ImplementationProposal:
    criteria = [
        "Menyimpan record baru pada tabel transactions",
        "Atomic update wallet balance dan transaction record",
        "Rollback jika gagal",
        "Verifikasi test",
    ]
    return ImplementationProposal(
        proposal_id="prop_revised_001",
        task_title="Implementasi Persistensi Riwayat Transaksi pada Wallet Topup",
        owner="backend",
        objective="Implementasikan Persistensi Riwayat Transaksi pada Wallet Topup sesuai spesifikasi.",
        verification_status="verified_pending",
        source_plan_path=".agent/artifacts/plans/plan_bca7ca8a.json",
        original_proposal_path=".agent/artifacts/proposals/proposal_4513cd15.json",
        dry_run_report_path=".agent/artifacts/reports/implementation_review_2ad3d64e.json",
        revision_reason="Preflight menemukan kebutuhan dependency wiring di main.go dan pengujian di api_integration_test.go.",
        file_scope_before=["backend/internal/service/wallet_service.go"],
        file_scope_after=[
            "backend/cmd/api/main.go",
            "backend/internal/service/wallet_service.go",
            "backend/tests/api_integration_test.go",
        ],
        affected_components=["backend"],
        files_to_modify=[
            ProposedFileChange(
                path="backend/internal/service/wallet_service.go",
                action="modify",
                purpose="Injeksi TransactionRepository dan atomisitas penyimpanan transaksi topup",
                changes=[
                    "Tambahkan dependency TransactionRepository pada WalletService",
                    "Panggil Transactor.Begin untuk membungkus update saldo dan create transaksi",
                ],
                evidence_paths=["backend/internal/service/wallet_service.go"],
                estimated_complexity="medium",
            ),
            ProposedFileChange(
                path="backend/cmd/api/main.go",
                action="modify",
                purpose="Dependency wiring untuk WalletService dengan TransactionRepository",
                changes=[
                    "Injeksi transactionRepo ke NewWalletService pada inisialisasi API server",
                ],
                evidence_paths=["backend/cmd/api/main.go"],
                estimated_complexity="low",
            ),
        ],
        tests_to_modify=[
            ProposedTest(
                path="backend/tests/api_integration_test.go",
                test_type="integration",
                scenarios=[
                    "Topup berhasil dan menyimpan record transaksi di database",
                    "Rollback saldo jika penyimpanan riwayat transaksi gagal menggunakan trigger",
                ],
                expected_results=[
                    "Record baru tersimpan pada tabel transactions dengan status success",
                    "Saldo tidak bertambah jika terjadi kegagalan transaksi",
                ],
                related_acceptance_criteria=criteria,
            )
        ],
        approved_writable_files=[
            "backend/cmd/api/main.go",
            "backend/internal/service/wallet_service.go",
            "backend/tests/api_integration_test.go",
        ],
        validation_commands=[
            ProposedCommand(
                executable="go",
                arguments=["test", "./..."],
                working_directory="backend",
                purpose="Menjalankan suite test backend",
                requires_approval=True,
            )
        ],
        rollback_plan=[
            "Kembalikan perubahan pada backend/internal/service/wallet_service.go ke revisi sebelumnya",
            "Kembalikan perubahan pada backend/cmd/api/main.go ke revisi sebelumnya",
            "Kembalikan perubahan pada backend/tests/api_integration_test.go ke revisi sebelumnya",
        ],
        out_of_scope=["Fitur transaksi lain dan perubahan pada frontend"],
        evidence_paths=[
            "backend/internal/service/wallet_service.go",
            "backend/cmd/api/main.go",
            "backend/tests/api_integration_test.go",
        ],
        status="draft",
    )


def test_01_tests_to_modify_included_in_approved_writable_files():
    """1. tests_to_modify masuk approved writable files."""
    prop = build_valid_revised_proposal()
    computed = compute_approved_writable_files(prop)
    assert "backend/tests/api_integration_test.go" in computed


def test_02_tests_to_create_included_in_approved_writable_files():
    """2. tests_to_create masuk approved writable files."""
    prop = build_valid_revised_proposal().model_dump(mode="json")
    prop["tests_to_create"] = [
        {
            "path": "backend/tests/wallet_unit_test.go",
            "test_type": "unit",
            "scenarios": ["Unit test topup"],
            "expected_results": ["Pass"],
            "related_acceptance_criteria": [],
        }
    ]
    computed = compute_approved_writable_files(prop)
    assert "backend/tests/wallet_unit_test.go" in computed


def test_03_approved_test_file_not_considered_outside_scope(mock_ppob_project, plan_data):
    """3. File test yang sudah disetujui tidak dianggap out-of-scope."""
    prop = build_valid_revised_proposal()
    val_proj = load_project(mock_ppob_project)
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status == "approved"
    assert "backend/tests/api_integration_test.go" in review.approved_writable_files

    outside = inspect_scope_requirements(
        project_root=mock_ppob_project,
        approved_files=review.approved_writable_files,
        task_title=prop.task_title,
        proposal_inner=prop.model_dump(mode="json"),
    )
    assert outside == []


def test_04_unapproved_test_file_rejected():
    """4. File test tanpa approval ditolak."""
    prop = build_valid_revised_proposal().model_dump(mode="json")
    prop["approved_writable_files"] = [
        "backend/internal/service/wallet_service.go",
        "backend/cmd/api/main.go",
    ]
    errors = validate_approved_writable_files(prop)
    assert any("Proposed test to modify 'backend/tests/api_integration_test.go' is not in" in e for e in errors)


def test_05_approved_file_without_proposed_file_change_rejected():
    """5. File biasa tanpa ProposedFileChange ditolak."""
    prop = build_valid_revised_proposal().model_dump(mode="json")
    prop["approved_writable_files"].append("backend/internal/service/unrelated_service.go")
    errors = validate_approved_writable_files(prop)
    assert any("unrelated_service.go" in e and "does not correspond" in e for e in errors)


def test_06_duplicate_approved_file_normalized():
    """6. Duplicate approved file dinormalisasi."""
    prop = build_valid_revised_proposal().model_dump(mode="json")
    prop["files_to_create"] = [
        {"path": "backend/cmd/api/main.go", "action": "create", "purpose": "dup", "changes": ["dup"]}
    ]
    computed = compute_approved_writable_files(prop)
    assert computed.count("backend/cmd/api/main.go") == 1
    assert computed == sorted(list(set(computed)))


def test_07_absolute_path_rejected():
    """7. Absolute path ditolak."""
    prop = build_valid_revised_proposal().model_dump(mode="json")
    prop["approved_writable_files"].append("/etc/passwd")
    prop["files_to_modify"].append({
        "path": "/etc/passwd",
        "action": "modify",
        "purpose": "bad",
        "changes": ["bad"],
    })
    errors = validate_approved_writable_files(prop)
    assert any("absolute path found" in e.lower() for e in errors)
    path_errors = validate_all_paths_relative(prop)
    assert any("Absolute path" in pe for pe in path_errors)


def test_08_path_traversal_rejected():
    """8. Path traversal ditolak."""
    prop = build_valid_revised_proposal().model_dump(mode="json")
    prop["approved_writable_files"].append("../escape.go")
    prop["files_to_modify"].append({
        "path": "../escape.go",
        "action": "modify",
        "purpose": "bad",
        "changes": ["bad"],
    })
    errors = validate_approved_writable_files(prop)
    assert any("traversal" in e.lower() for e in errors)
    path_errors = validate_all_paths_relative(prop)
    assert any("Path traversal" in pe for pe in path_errors)


def test_09_proposal_revision_stores_original_proposal_ref():
    """9. Proposal revision menyimpan reference proposal awal."""
    prop = build_valid_revised_proposal()
    assert prop.original_proposal_path == ".agent/artifacts/proposals/proposal_4513cd15.json"
    dump = prop.model_dump(mode="json")
    assert dump["original_proposal_path"] == ".agent/artifacts/proposals/proposal_4513cd15.json"


def test_10_proposal_revision_stores_dry_run_report_ref():
    """10. Proposal revision menyimpan reference dry-run report."""
    prop = build_valid_revised_proposal()
    assert prop.dry_run_report_path == ".agent/artifacts/reports/implementation_review_2ad3d64e.json"
    dump = prop.model_dump(mode="json")
    assert dump["dry_run_report_path"] == ".agent/artifacts/reports/implementation_review_2ad3d64e.json"


def test_11_revised_proposal_review_yields_approved(mock_ppob_project, plan_data):
    """11. Revised proposal review dapat menghasilkan approved."""
    prop = build_valid_revised_proposal()
    val_proj = load_project(mock_ppob_project)
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)
    assert review.status == "approved"
    assert len(review.approved_writable_files) == 3


def test_12_revised_proposal_dry_run_yields_preflight_passed(mock_ppob_project, plan_data):
    """12. Revised proposal dry-run dapat menghasilkan preflight_passed."""
    head = get_git_head(mock_ppob_project)
    prop = build_valid_revised_proposal()
    val_proj = load_project(mock_ppob_project)
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)

    res = run_implementation_preflight(
        project_root=mock_ppob_project,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=".agent/artifacts/plans/plan_bca7ca8a.json",
        source_proposal_path=".agent/artifacts/proposals/proposal_revised.json",
        plan_data=plan_data,
        proposal_data={
            "status": "approved_for_implementation",
            "result": {"proposal": prop.model_dump(mode="json")},
            "source_revision": {"commit_head": head, "is_dirty": False},
        },
        proposal_review=review.model_dump(mode="json"),
        apply=False,
    )
    assert res.is_valid is True
    assert res.status == "preflight_passed"
    assert res.files_requested_outside_scope == []


def test_13_dry_run_does_not_write_source_code(mock_ppob_project, plan_data):
    """13. Dry-run tidak menulis source code."""
    head = get_git_head(mock_ppob_project)
    prop = build_valid_revised_proposal()
    val_proj = load_project(mock_ppob_project)
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)

    target_file = mock_ppob_project / "backend" / "internal" / "service" / "wallet_service.go"
    original_mtime = target_file.stat().st_mtime
    original_content = target_file.read_text(encoding="utf-8")

    graph = build_implementation_graph()
    initial_state = {
        "project_root": str(mock_ppob_project),
        "source_plan_path": ".agent/artifacts/plans/plan_bca7ca8a.json",
        "source_proposal_path": ".agent/artifacts/proposals/proposal_revised.json",
        "plan_data": plan_data,
        "proposal_data": {
            "status": "approved_for_implementation",
            "result": {"proposal": prop.model_dump(mode="json")},
            "source_revision": {"commit_head": head, "is_dirty": False},
        },
        "proposal_review": review.model_dump(mode="json"),
        "dry_run": True,
        "apply": False,
        "run_id": "testrun13",
    }
    final_state = graph.invoke(initial_state)

    assert target_file.stat().st_mtime == original_mtime
    assert target_file.read_text(encoding="utf-8") == original_content
    assert final_state.get("status") == "preflight_passed"


def test_14_dry_run_does_not_create_branch(mock_ppob_project, plan_data):
    """14. Dry-run tidak membuat branch."""
    head = get_git_head(mock_ppob_project)
    prop = build_valid_revised_proposal()
    val_proj = load_project(mock_ppob_project)
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)

    branches_before = subprocess.run(["git", "branch"], cwd=mock_ppob_project, capture_output=True, text=True).stdout

    graph = build_implementation_graph()
    initial_state = {
        "project_root": str(mock_ppob_project),
        "source_plan_path": ".agent/artifacts/plans/plan_bca7ca8a.json",
        "source_proposal_path": ".agent/artifacts/proposals/proposal_revised.json",
        "plan_data": plan_data,
        "proposal_data": {
            "status": "approved_for_implementation",
            "result": {"proposal": prop.model_dump(mode="json")},
            "source_revision": {"commit_head": head, "is_dirty": False},
        },
        "proposal_review": review.model_dump(mode="json"),
        "dry_run": True,
        "apply": False,
        "run_id": "testrun14",
    }
    graph.invoke(initial_state)

    branches_after = subprocess.run(["git", "branch"], cwd=mock_ppob_project, capture_output=True, text=True).stdout
    assert branches_before == branches_after


def test_15_dry_run_does_not_execute_commands(mock_ppob_project, plan_data, monkeypatch):
    """15. Dry-run tidak menjalankan command."""
    head = get_git_head(mock_ppob_project)
    prop = build_valid_revised_proposal()
    val_proj = load_project(mock_ppob_project)
    review = review_proposal_deterministically(prop, plan_data, val_proj.manifest, val_proj.policy)

    executed_cmds = []

    def fake_run(*args, **kwargs):
        executed_cmds.append(args)
        raise RuntimeError("Command execution was attempted in dry-run mode!")

    import agent_system.command_runner as cmd_runner
    monkeypatch.setattr(cmd_runner.SafeCommandRunner, "run_command", fake_run)

    graph = build_implementation_graph()
    initial_state = {
        "project_root": str(mock_ppob_project),
        "source_plan_path": ".agent/artifacts/plans/plan_bca7ca8a.json",
        "source_proposal_path": ".agent/artifacts/proposals/proposal_revised.json",
        "plan_data": plan_data,
        "proposal_data": {
            "status": "approved_for_implementation",
            "result": {"proposal": prop.model_dump(mode="json")},
            "source_revision": {"commit_head": head, "is_dirty": False},
        },
        "proposal_review": review.model_dump(mode="json"),
        "dry_run": True,
        "apply": False,
        "run_id": "testrun15",
    }
    final_state = graph.invoke(initial_state)

    assert len(executed_cmds) == 0
    assert final_state.get("status") == "preflight_passed"


