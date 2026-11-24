import json
import random
import re
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from agent_system.atomic_state import atomic_write_json
from agent_system.error_reporting import sanitize_text


class FailureCategory(str, Enum):
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    RATE_LIMIT = "RATE_LIMIT"
    TRANSIENT_GATEWAY = "TRANSIENT_GATEWAY"
    AUTH_CONFIGURATION = "AUTH_CONFIGURATION"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    PERMANENT_REQUEST = "PERMANENT_REQUEST"
    CODE_OR_VALIDATION_FAILURE = "CODE_OR_VALIDATION_FAILURE"


def classify_failure(exc: Any, message: str = "") -> FailureCategory:
    """Classify failure into one of 7 distinct categories with fail-closed precision.

    Never treats code or validation failures as gateway transient failures.
    APPROVAL_REQUIRED is a non-error approval category and never retried.
    """
    exc_type = type(exc).__name__ if isinstance(exc, Exception) else str(exc)
    full_msg = f"{exc_type} {str(exc)} {message}".lower()

    # 0. Approval Required (Non-error category: human_required, approval gate)
    if any(k in full_msg for k in (
        "human_required",
        "approval_required",
        "approval required",
        "awaiting_human_approval",
        "awaiting_ceo_input",
        "dibutuhkan keputusan manusia",
        "memerlukan keputusan manusia",
        "proposal review tidak disetujui (status: human_required)",
    )):
        return FailureCategory.APPROVAL_REQUIRED

    # 1. Code or Validation Failure check (Highest priority - never classify as gateway error!)
    if any(k in full_msg for k in (
        "validation command gagal",
        "test failed",
        "scope violation",
        "unexpected file",
        "security blocker",
        "security violation",
        "diff review rejected",
        "go test",
        "pytest",
        "unit test failed",
        "compile error",
        "syntax error in",
    )):
        return FailureCategory.CODE_OR_VALIDATION_FAILURE

    # 2. Rate Limit (HTTP 429)
    if any(k in full_msg for k in ("429", "rate limit", "ratelimit", "too many requests", "quota exceeded")):
        return FailureCategory.RATE_LIMIT

    # 3. Auth Configuration (HTTP 401, 403, invalid api key)
    if any(k in full_msg for k in (
        "401", "403", "unauthorized", "authenticationerror",
        "invalid api key", "forbidden", "permission denied to model",
        "invalid credentials", "access denied",
    )):
        return FailureCategory.AUTH_CONFIGURATION

    # 4. Model Output parsing / Structured Output / Schema validation failure
    if any(k in full_msg for k in (
        "output parsing error", "schema validation error", "jsondecodeerror",
        "validationerror", "pydantic", "structured output",
        "failed to parse", "missing required fields", "truncated response",
    )):
        return FailureCategory.MODEL_OUTPUT

    # 5. Permanent Request (Context length exceeded, unsupported operation)
    if any(k in full_msg for k in (
        "context length", "maximum context length", "too long",
        "prompt is too long", "unsupported operation", "invalid request error",
        "not allowed for this model", "malformed request",
    )):
        return FailureCategory.PERMANENT_REQUEST

    # 6. Transient Gateway (502, 503, 504, timeout, connection reset, temporary network)
    if any(k in full_msg for k in (
        "502", "503", "504", "bad gateway", "service unavailable", "gateway timeout",
        "timeout", "timed out", "connection reset", "connection refused",
        "connecterror", "network error", "dns resolution", "apiconnectionerror",
        "remote end closed",
    )):
        return FailureCategory.TRANSIENT_GATEWAY

    # Default fallback for unknown exception
    return FailureCategory.TRANSIENT_GATEWAY


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Persistent circuit breaker per provider or model alias."""

    def __init__(
        self,
        project_root: Path,
        provider_id: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.provider_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", provider_id)
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock or time.time
        self.cb_dir = self.project_root / ".agent" / "artifacts" / "circuit_breakers"
        self.cb_dir.mkdir(parents=True, exist_ok=True)
        self.cb_file = self.cb_dir / f"circuit_{self.provider_id}.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.cb_file.is_file():
            try:
                data = json.loads(self.cb_file.read_text(encoding="utf-8"))
                self.state = CircuitState(data.get("state", "closed"))
                self.consecutive_failures = data.get("consecutive_failures", 0)
                self.cooldown_until = data.get("cooldown_until", 0.0)
                self.half_open_probes = data.get("half_open_probes", 0)
                return
            except Exception:
                pass
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.cooldown_until = 0.0
        self.half_open_probes = 0

    def _persist(self) -> None:
        payload = {
            "provider_id": self.provider_id,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_until": self.cooldown_until,
            "half_open_probes": self.half_open_probes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.cb_file, payload, backup=False, monotonic_revision=False)

    def is_available(self) -> bool:
        """Check if circuit breaker allows requests."""
        now = self.clock()
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if now >= self.cooldown_until:
                # Enter half-open
                self.state = CircuitState.HALF_OPEN
                self.half_open_probes = 0
                self._persist()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # In half-open: allow exactly 1 probe request
            if self.half_open_probes < 1:
                return True
            return False
        return False

    def record_success(self) -> None:
        """Record successful call, closing circuit if half-open."""
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED
        self.half_open_probes = 0
        self._persist()

    def record_failure(self, failure_category: FailureCategory) -> None:
        """Record failure, opening circuit if threshold exceeded."""
        if failure_category in (FailureCategory.TRANSIENT_GATEWAY, FailureCategory.RATE_LIMIT):
            self.consecutive_failures += 1
            if self.state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.cooldown_until = self.clock() + self.cooldown_seconds
                self._persist()


class BackoffManager:
    """Manages checkpointed retry backoff with jitter and injectable sleeper."""

    def __init__(
        self,
        base_delays: list[float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.base_delays = base_delays or [10.0, 30.0, 90.0]
        self.sleeper = sleeper or time.sleep
        self.clock = clock or time.time

    def calculate_delay(self, attempt: int, retry_after: float | None = None, add_jitter: bool = True) -> float:
        """Calculate backoff delay for given attempt."""
        if retry_after is not None and retry_after > 0:
            return min(float(retry_after), 300.0)

        idx = min(max(0, attempt - 1), len(self.base_delays) - 1)
        base = self.base_delays[idx]
        if add_jitter:
            jitter = random.uniform(-0.1, 0.1) * base
            return max(0.1, base + jitter)
        return base

    def wait(self, delay: float) -> None:
        """Execute sleep using configured sleeper."""
        self.sleeper(delay)


class ModelFailoverManager:
    """Manages failover across primary -> secondary -> tertiary models."""

    def __init__(self, model_chain: list[str] | None = None):
        self.model_chain = model_chain or []

    def get_model_for_attempt(self, attempt: int) -> str | None:
        """Get model alias for current attempt count."""
        if not self.model_chain:
            return None
        idx = min(attempt - 1, len(self.model_chain) - 1)
        return self.model_chain[idx]
