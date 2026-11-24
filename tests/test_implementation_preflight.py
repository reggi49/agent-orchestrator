import json
import os
import subprocess
from pathlib import Path

import pytest

from agent_system.implementation_preflight import run_implementation_preflight
from agent_system.project_initializer import initialize_project
from agent_system.project_loader import load_project


def setup_test_project(tmp_path: Path) -> Path:
    p = tmp_path / "repo"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "frontend").mkdir()
    (p / "backend" / "go.mod").write_text("module test/backend\n")
    (p / "frontend" / "package.json").write_text('{"name": "frontend"}\n')

    svc_dir = p / "backend" / "internal" / "service"
    svc_dir.mkdir(parents=True)
    (svc_dir / "wallet_service.go").write_text("package service\n\n// wallet service\n")

    initialize_project(p, apply=True)

    subprocess.run(["git", "init", "-b", "main"], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=p, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=p, check=True, capture_output=True)
    return p


def get_head_rev(p: Path) -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def build_test_artifacts(project_root: Path, commit_head: str, proposal_status: str = "approved_for_implementation", review_status: str = "approved"):
    plan_path = ".agent/artifacts/plans/plan_test.json"
    plan_file = project_root / plan_path
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_data = {
        "status": "plan_ready",
        "result": {
            "recommended_task": {"title": "Update wallet", "owner": "backend", "acceptance_criteria": ["ac1"]},
            "verification_status": "verified_pending",
        },
    }
    plan_file.write_text(json.dumps(plan_data))

    prop_path = ".agent/artifacts/proposals/proposal_test.json"
    prop_file = project_root / prop_path
    prop_file.parent.mkdir(parents=True, exist_ok=True)
    prop_data = {
        "status": proposal_status,
        "artifact_id": "proposal_test",
        "source_revision": {"commit_head": commit_head, "is_dirty": False},
        "result": {
            "source_plan_path": plan_path,
            "review_status": review_status,
            "proposal": {
                "proposal_id": "prop_test",
                "task_title": "Update wallet",
                "owner": "backend",
                "files_to_modify": [{"path": "backend/internal/service/wallet_service.go", "action": "modify", "purpose": "update"}],
                "files_to_create": [],
                "validation_commands": [{"executable": "go", "arguments": ["test", "./..."], "working_directory": "backend", "purpose": "test"}],
            },
        },
    }
    prop_file.write_text(json.dumps(prop_data))

    review_data = {
        "status": review_status,
        "proposal_id": "prop_test",
        "approved_files": ["backend/internal/service/wallet_service.go"],
    }
    return plan_path, prop_path, plan_data, prop_data, review_data


def test_01_approved_proposal_accepted(tmp_path):
    """1. Approved proposal diterima."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is True
    assert res.status in {"valid", "preflight_passed"}


def test_02_unapproved_proposal_rejected(tmp_path):
    """2. Proposal tanpa approval ditolak."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head, proposal_status="under_review", review_status="revision_required")
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "rejected"


def test_03_stale_source_revision_rejected(tmp_path):
    """3. Stale source revision ditolak."""
    p = setup_test_project(tmp_path)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, "deadbeef1234567890abcdefdeadbeef12345678")
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "stale_proposal"

def test_04_dirty_target_file_yields_human_required(tmp_path):
    """4. Dirty target file menghasilkan human_required."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)
    val_proj = load_project(p)

    (p / "backend" / "internal" / "service" / "wallet_service.go").write_text("dirty content")

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "human_required"


def test_05_absolute_path_rejected(tmp_path):
    """5. Absolute path ditolak."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)
    prop_d["result"]["proposal"]["files_to_modify"] = [{"path": "/etc/passwd", "action": "modify", "purpose": "hack"}]
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "rejected"


def test_06_path_traversal_rejected(tmp_path):
    """6. Path traversal ditolak."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)
    prop_d["result"]["proposal"]["files_to_modify"] = [{"path": "../outside.go", "action": "modify", "purpose": "bad"}]
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "rejected"


def test_07_file_outside_component_rejected(tmp_path):
    """7. File di luar component ditolak."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)
    (p / "unknown_comp").mkdir()
    (p / "unknown_comp" / "file.txt").write_text("content")
    prop_d["result"]["proposal"]["files_to_modify"] = [{"path": "unknown_comp/file.txt", "action": "modify", "purpose": "bad"}]
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
def test_08_secret_file_rejected(tmp_path):
    """8. Secret file ditolak."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)
    (p / "backend" / ".env").write_text("SECRET=1")
    prop_d["result"]["proposal"]["files_to_modify"] = [{"path": "backend/.env", "action": "modify", "purpose": "bad"}]
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "rejected"


def test_09_symlink_escaping_rejected(tmp_path):
    """9. Symlink keluar project ditolak."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)

    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret")
    symlink_file = p / "backend" / "sym_link.go"
    try:
        os.symlink(outside, symlink_file)
    except OSError:
        pytest.skip("Symlink not supported")

    prop_d["result"]["proposal"]["files_to_modify"] = [{"path": "backend/sym_link.go", "action": "modify", "purpose": "bad"}]
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "rejected"


def test_10_unknown_command_rejected(tmp_path):
    """10. Unknown command ditolak."""
    p = setup_test_project(tmp_path)
    head = get_head_rev(p)
    plan_p, prop_p, plan_d, prop_d, rev_d = build_test_artifacts(p, head)
    prop_d["result"]["proposal"]["validation_commands"] = [
        {"executable": "malicious_runner", "arguments": ["run"], "working_directory": "backend", "purpose": "hack"}
    ]
    val_proj = load_project(p)

    res = run_implementation_preflight(
        project_root=p,
        manifest=val_proj.manifest,
        policy=val_proj.policy,
        source_plan_path=plan_p,
        source_proposal_path=prop_p,
        plan_data=plan_d,
        proposal_data=prop_d,
        proposal_review=rev_d,
    )
    assert res.is_valid is False
    assert res.status == "rejected"

    assert res.status == "rejected"
