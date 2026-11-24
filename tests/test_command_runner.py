import subprocess
from pathlib import Path

import pytest

from agent_system.command_runner import SafeCommandRunner, redact_sensitive_text, summarize_output


def setup_runner_project(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    p.mkdir()
    (p / "backend").mkdir()
    return p


def test_34_go_test_in_backend_accepted(tmp_path, monkeypatch):
    """34. go test ./... di backend diterima."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)

    # Mock subprocess to avoid real network/compilation in unit tests
    def mock_run(cmd, cwd, shell, capture_output, text, timeout):
        assert cmd == ["go", "test", "./..."]
        assert shell is False
        assert cwd == p / "backend"
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    res = runner.run_command("go", ["test", "./..."], "backend")
    assert res.status == "passed"
    assert res.exit_code == 0


def test_35_shell_operator_rejected(tmp_path):
    """35. shell operator ditolak."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)

    res1 = runner.run_command("go", ["test", ";", "rm"], "backend")
    assert res1.status == "rejected"

    res2 = runner.run_command("go", ["test", "|", "cat"], "backend")
    assert res2.status == "rejected"

    res3 = runner.run_command("go", ["test", "&&", "echo"], "backend")
    assert res3.status == "rejected"


def test_36_sh_rejected(tmp_path):
    """36. sh ditolak."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)
    res = runner.run_command("sh", ["-c", "echo 1"], "backend")
    assert res.status == "rejected"


def test_37_bash_rejected(tmp_path):
    """37. bash ditolak."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)
    res = runner.run_command("bash", ["-c", "echo 1"], "backend")
    assert res.status == "rejected"


def test_38_sudo_rejected(tmp_path):
    """38. sudo ditolak."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)
    res = runner.run_command("sudo", ["go", "test"], "backend")
    assert res.status == "rejected"


def test_39_rm_rejected(tmp_path):
    """39. rm ditolak."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)
    res = runner.run_command("rm", ["-rf", "."], "backend")
def test_40_cwd_outside_project_rejected(tmp_path):
    """40. cwd di luar project ditolak."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)
    res = runner.run_command("go", ["test"], "../outside")
    assert res.status == "rejected"


def test_41_timeout_handled(tmp_path, monkeypatch):
    """41. Timeout ditangani."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)

    def mock_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["go", "test"], timeout=1.0)

    monkeypatch.setattr(subprocess, "run", mock_timeout)
    res = runner.run_command("go", ["test"], "backend", timeout=1.0)
    assert res.timed_out is True
    assert res.status == "timed_out"


def test_42_output_truncated(tmp_path):
    """42. Output dibatasi ukurannya."""
    huge_text = "line " * 5000
    summary = summarize_output(huge_text, max_length=1000)
    assert len(summary) < 2000
    assert "TRUNCATED" in summary


def test_43_secret_redacted():
    """43. Secret di-redact."""
    text = "Authorization: Bearer my_secret_token_12345 api_key=xyz987"
    redacted = redact_sensitive_text(text)
    assert "my_secret_token_12345" not in redacted
    assert "xyz987" not in redacted
    assert "[REDACTED]" in redacted


def test_44_shell_false_enforced(tmp_path, monkeypatch):
    """44. shell=False digunakan."""
    p = setup_runner_project(tmp_path)
    runner = SafeCommandRunner(p)

    captured_kwargs = {}
    def mock_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    runner.run_command("go", ["test"], "backend")
    assert captured_kwargs.get("shell") is False
