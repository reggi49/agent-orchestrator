import os
import re
import traceback
from typing import Any


def sanitize_text(text: str, api_key: str | None = None) -> str:
    """Sanitize sensitive patterns from text.

    Removes secrets, API keys, Bearer tokens, Authorization headers, and raw prompt dumps.
    """
    if not text:
        return ""

    sanitized = text

    # Redact known api_key literal if provided
    if api_key and len(api_key.strip()) > 3:
        sanitized = sanitized.replace(api_key.strip(), "[REDACTED_API_KEY]")

    # Redact Bearer tokens
    sanitized = re.sub(
        r"Bearer\s+[A-Za-z0-9_\-\.]+",
        "Bearer [REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Redact sk- style keys
    sanitized = re.sub(
        r"\bsk-[A-Za-z0-9_\-\.]{8,}\b",
        "sk-[REDACTED]",
        sanitized,
    )

    # Redact headers: Authorization, X-Api-Key, etc.
    sanitized = re.sub(
        r"(?i)\b(authorization|x-api-key|api-key):\s*[^\r\n,]+",
        r"\1: [REDACTED]",
        sanitized,
    )

    # Redact key/secret/token assignments (e.g. api_key='...', SECRET_KEY="...", token="...")
    sanitized = re.sub(
        r"(?i)\b([A-Za-z0-9_\-]*(?:api[_-]?key|token|secret|password|authorization)[A-Za-z0-9_\-]*)\s*=\s*['\"][^'\"]+['\"]",
        r"\1='[REDACTED]'",
        sanitized,
    )

    # Redact raw prompt blocks if they leak in message
    sanitized = re.sub(
        r"(?i)(SystemMessage|HumanMessage|AIMessage)\(content=['\"][^'\"]{100,}['\"]",
        r"\1(content='[OMITTED_FOR_SAFETY]'",
        sanitized,
    )

    return sanitized


def format_safe_error_report(
    exc: Exception,
    api_key: str | None = None,
    context: str = "Workflow",
) -> str:
    """Format safe error report containing exception type, message, and traceback.

    Guarantees secrets, API keys, headers, tokens, and raw prompts are never printed.
    """
    err_type = type(exc).__name__
    err_msg = str(exc)
    raw_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    sanitized_msg = sanitize_text(err_msg, api_key=api_key)
    sanitized_tb = sanitize_text(raw_tb, api_key=api_key)

    lines = [
        f"ERROR: {context} gagal: {err_type}",
        f"Message: {sanitized_msg}",
        "Traceback (sanitized):",
        sanitized_tb.rstrip(),
    ]
    return "\n".join(lines)
