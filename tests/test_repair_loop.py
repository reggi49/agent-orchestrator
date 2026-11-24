from pathlib import Path

from agent_system.graph import (
    route_after_diff_reviewer,
    route_after_implementation_agent,
    route_after_test_runner,
)
from agent_system.repair_agent import run_repair_agent


def test_54_test_failure_routes_to_repair_agent():
    """54. Test failure menuju Repair Agent."""
    state = {
        "command_results": [{"status": "failed", "exit_code": 1}],
        "implementation_repair_count": 0,
        "max_implementation_repairs": 2,
    }
    dest = route_after_test_runner(state)
    assert dest == "repair_agent"


def test_55_diff_repair_required_routes_to_repair_agent():
    """55. Diff repair_required menuju Repair Agent."""
    state = {
        "diff_review": {"status": "repair_required", "unexpected_files": []},
        "implementation_repair_count": 1,
        "max_implementation_repairs": 2,
    }
    dest = route_after_diff_reviewer(state)
    assert dest == "repair_agent"


def test_56_max_repairs_exhausted_routes_to_supervisor():
    """56. Repair maksimal menghasilkan supervisor (human_required)."""
    # From test runner
    state_test = {
        "command_results": [{"status": "failed", "exit_code": 1}],
        "implementation_repair_count": 2,
        "max_implementation_repairs": 2,
    }
    assert route_after_test_runner(state_test) == "supervisor"

    # From diff reviewer
    state_diff = {
        "diff_review": {"status": "repair_required"},
        "implementation_repair_count": 2,
        "max_implementation_repairs": 2,
    }
    assert route_after_diff_reviewer(state_diff) == "supervisor"


def test_57_files_outside_scope_routes_to_supervisor():
    """57. File tambahan menghasilkan proposal_revision_required (langsung ke supervisor)."""
    state = {
        "status": "proposal_revision_required",
    }
    dest = route_after_implementation_agent(state)
    assert dest == "supervisor"


def test_58_passed_test_and_approved_diff_routes_to_supervisor():
    """58. Passed test dan approved diff menuju Supervisor."""
    state_test = {
        "command_results": [{"status": "passed", "exit_code": 0}],
        "implementation_repair_count": 0,
        "max_implementation_repairs": 2,
    }
    assert route_after_test_runner(state_test) == "diff_reviewer"

    state_diff = {
        "diff_review": {"status": "approved", "unexpected_files": []},
    }
    assert route_after_diff_reviewer(state_diff) == "supervisor"


def test_59_repair_does_not_expand_scope(tmp_path):
    """59. Repair tidak memperluas scope."""
    p = tmp_path / "proj"
    p.mkdir()
    (p / "backend").mkdir()
    (p / "backend" / "test.go").write_text("package test\n// TODO: implement\n")

    prev_impl = {
        "task_title": "Fix something",
        "changes": [{"path": "backend/test.go", "action": "modify", "content": "package test\n// TODO: implement\n"}],
    }

    repaired = run_repair_agent(
        project_root=p,
        previous_implementation=prev_impl,
        test_failure_summary="",
        diff_findings=[],
        approved_files=["backend/test.go"],
        acceptance_criteria=["ac1"],
        repair_count=0,
    )

    assert repaired.files_requested_outside_scope == []
    for chg in repaired.changes:
        assert chg.path == "backend/test.go"
