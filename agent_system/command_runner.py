import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from agent_system.path_safety import resolve_project_path, validate_relative_pattern
from agent_system.schemas import CommandExecutionResult

ALLOWED_EXECUTABLES = {
    "go",
    "pytest",
    "python",
    "python3",
    "npm",
    "pnpm",
    "yarn",
    "cargo",
    "mvn",
    "gradle",
    "make",
}

FORBIDDEN_EXECUTABLES = {
    "sh",
    "bash",
    "zsh",
    "sudo",
    "rm",
    "curl",
    "wget",
    "nc",
    "git",
    "dd",
    "chmod",
    "chown",
}

FORBIDDEN_ARG_COMBINATIONS = [
    ("go", "install"),
    ("pip", "install"),
    ("npm", "install"),
    ("pnpm", "install"),
    ("yarn", "add"),
]

SHELL_OPERATORS_PATTERN = re.compile(r"[|;&><`$\n\r]")
SENSITIVE_OUTPUT_PATTERNS = [
    re.compile(r"bearer\s+[a-z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"api[_-]?key[:=]\s*[a-z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"token[:=]\s*[a-z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"secret[:=]\s*[a-z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"password[:=]\s*[a-z0-9_\-\.]+", re.IGNORECASE),
]


def redact_sensitive_text(text: str) -> str:
    """Redact sensitive credentials from command output."""
    res = text
    for pat in SENSITIVE_OUTPUT_PATTERNS:
        res = pat.sub("[REDACTED]", res)
    return res


def summarize_output(text: str, max_length: int = 2000) -> str:
    """Summarize output and cap to max_length."""
    redacted = redact_sensitive_text(text)
    if len(redacted) <= max_length:
        return redacted
    half = max_length // 2
    return f"{redacted[:half]}\n\n[... OUTPUT TRUNCATED FOR SAFETY ...]\n\n{redacted[-half:]}"

class SafeCommandRunner:
    """Safely executes validation commands with strict sandboxing and allowlisting."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.expanduser().resolve()

    def run_command(
        self,
        executable: str,
        arguments: list[str],
        working_directory: str,
        timeout: float = 60.0,
    ) -> CommandExecutionResult:
        """Run validation command safely without shell=True."""
        exe_clean = str(executable).strip()

        # 1. Allowlist validation
        if not exe_clean or exe_clean.lower() not in ALLOWED_EXECUTABLES:
            return CommandExecutionResult(
                executable=exe_clean,
                arguments=arguments,
                working_directory=working_directory,
                exit_code=-1,
                stdout_summary="",
                stderr_summary=f"Executable '{exe_clean}' ditolak: di luar allowlist {sorted(ALLOWED_EXECUTABLES)}.",
                timed_out=False,
                duration_seconds=0.0,
                status="rejected",
            )

        if exe_clean.lower() in FORBIDDEN_EXECUTABLES:
            return CommandExecutionResult(
                executable=exe_clean,
                arguments=arguments,
                working_directory=working_directory,
                exit_code=-1,
                stdout_summary="",
                stderr_summary=f"Executable '{exe_clean}' secara eksplisit dilarang.",
                timed_out=False,
                duration_seconds=0.0,
                status="rejected",
            )

        # 2. Argument safety & package install rejection
        for arg in arguments:
            arg_str = str(arg)
            if SHELL_OPERATORS_PATTERN.search(arg_str):
                return CommandExecutionResult(
                    executable=exe_clean,
                    arguments=arguments,
                    working_directory=working_directory,
                    exit_code=-1,
                    stdout_summary="",
                    stderr_summary=f"Argument '{arg_str}' mengandung operator shell terlarang.",
                    timed_out=False,
                    duration_seconds=0.0,
                    status="rejected",
                )

        for bad_exe, bad_sub in FORBIDDEN_ARG_COMBINATIONS:
            if exe_clean.lower() == bad_exe and bad_sub in [a.lower() for a in arguments]:
                return CommandExecutionResult(
                    executable=exe_clean,
                    arguments=arguments,
                    working_directory=working_directory,
                    exit_code=-1,
                    stdout_summary="",
                    stderr_summary=f"Subcommand package installation '{exe_clean} {bad_sub}' ditolak.",
                    timed_out=False,
                    duration_seconds=0.0,
                    status="rejected",
                )

        # 3. Working directory validation
        try:
            validate_relative_pattern(working_directory)
            cwd_path = resolve_project_path(self.project_root, working_directory)
            cwd_path.relative_to(self.project_root)
            if not cwd_path.is_dir():
                return CommandExecutionResult(
                    executable=exe_clean,
                    arguments=arguments,
                    working_directory=working_directory,
                    exit_code=-1,
                    stdout_summary="",
                    stderr_summary=f"Working directory '{working_directory}' tidak ditemukan di disk.",
                    timed_out=False,
                    duration_seconds=0.0,
                    status="rejected",
                )
        except Exception as e:
            return CommandExecutionResult(
                executable=exe_clean,
                arguments=arguments,
                working_directory=working_directory,
                exit_code=-1,
                stdout_summary="",
                stderr_summary=f"Working directory '{working_directory}' tidak valid atau keluar project root: {e}",
                timed_out=False,
                duration_seconds=0.0,
                status="rejected",
            )
        cmd = [exe_clean] + [str(a) for a in arguments]
        start_time = time.monotonic()

        try:
            res = subprocess.run(
                cmd,
                cwd=cwd_path,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = round(time.monotonic() - start_time, 2)
            stdout_sum = summarize_output(res.stdout)
            stderr_sum = summarize_output(res.stderr)
            status = "passed" if res.returncode == 0 else "failed"

            return CommandExecutionResult(
                executable=exe_clean,
                arguments=arguments,
                working_directory=working_directory,
                exit_code=res.returncode,
                stdout_summary=stdout_sum,
                stderr_summary=stderr_sum,
                timed_out=False,
                duration_seconds=duration,
                status=status,
            )
        except subprocess.TimeoutExpired as te:
            duration = round(time.monotonic() - start_time, 2)
            stdout_sum = summarize_output(te.stdout or "" if isinstance(te.stdout, str) else "")
            stderr_sum = summarize_output(te.stderr or "" if isinstance(te.stderr, str) else "")
            return CommandExecutionResult(
                executable=exe_clean,
                arguments=arguments,
                working_directory=working_directory,
                exit_code=-1,
                stdout_summary=stdout_sum,
                stderr_summary=f"Command timeout setelah {timeout} detik.\n{stderr_sum}",
                timed_out=True,
                duration_seconds=duration,
                status="timed_out",
            )
        except Exception as e:
            duration = round(time.monotonic() - start_time, 2)
            return CommandExecutionResult(
                executable=exe_clean,
                arguments=arguments,
                working_directory=working_directory,
                exit_code=-1,
                stdout_summary="",
                stderr_summary=f"Eksekusi command gagal: {e}",
                timed_out=False,
                duration_seconds=duration,
                status="failed",
            )

