import re
from typing import Any

PROHIBITED_HIGH_RISK_PATTERNS = [
    re.compile(r"\b(wallet|balance|saldo|ledger)\b", re.IGNORECASE),
    re.compile(r"\b(settlement|reconciliation|rekonsiliasi)\b", re.IGNORECASE),
    re.compile(r"\b(payment|callback|pembayaran|invoice)\b", re.IGNORECASE),
    re.compile(r"\b(auth|login|token|jwt|password|credential|secret|role|permission)\b", re.IGNORECASE),
    re.compile(r"\b(migration|database schema|drop table|truncate|delete from)\b", re.IGNORECASE),
    re.compile(r"\b(infrastructure|production|k8s|kubernetes|terraform)\b", re.IGNORECASE),
]

ALLOWED_FALLBACK_SCOPES = [
    "manifest_reading",
    "dag_task_selection",
    "document_skeleton",
    "schema_normalization",
    "validation_command_selection",
    "markdown_regeneration",
    "status_reporting",
]

PROHIBITED_DECISIONS = [
    "financial_behavior",
    "wallet_balance",
    "ledger_semantics",
    "payment_callback_behavior",
    "settlement",
    "reconciliation",
    "authentication",
    "authorization",
    "complex_migration",
    "destructive_data_operation",
    "production_infrastructure",
    "secret_management",
    "ambiguous_business_rules",
]


def evaluate_deterministic_fallback_safety(
    context_text: str,
    risk_level: str,
    fallback_type: str,
) -> dict[str, Any]:
    """Evaluate whether a deterministic fallback is safe or must be blocked / escalated.

    Policy:
    - LOW: deterministic fallback allowed to proceed if guards pass.
    - MEDIUM: fallback can create draft, but CANNOT final-integrate without valid model or human approval.
    - HIGH: fallback CANNOT make design decisions or integrate; program must pause for model or CEO input.
    """
    text = (context_text or "").lower()
    is_prohibited_domain = any(p.search(text) for p in PROHIBITED_HIGH_RISK_PATTERNS)

    if risk_level == "high" or is_prohibited_domain:
        return {
            "allowed": False,
            "requires_ceo_escalation": True,
            "requires_human_approval": True,
            "fallback_used": True,
            "fallback_type": fallback_type,
            "allowed_scope": ALLOWED_FALLBACK_SCOPES,
            "prohibited_decisions": PROHIBITED_DECISIONS,
            "reason": (
                "Deterministic fallback dilarang membuat keputusan final untuk domain HIGH-risk "
                "(finansial/keamanan/migrasi/infrastruktur). Program wajib jeda (pause) untuk input CEO."
            ),
        }

    if risk_level == "medium":
        return {
            "allowed": True,
            "requires_ceo_escalation": False,
            "requires_human_approval": True,
            "fallback_used": True,
            "fallback_type": fallback_type,
            "allowed_scope": ALLOWED_FALLBACK_SCOPES,
            "prohibited_decisions": PROHIBITED_DECISIONS,
            "reason": (
                "Deterministic fallback diperbolehkan untuk membuat draf pada level MEDIUM, "
                "namun dilarang melakukan integrasi final tanpa persetujuan manusia."
            ),
        }

    # LOW risk
    return {
        "allowed": True,
        "requires_ceo_escalation": False,
        "requires_human_approval": False,
        "fallback_used": True,
        "fallback_type": fallback_type,
        "allowed_scope": ALLOWED_FALLBACK_SCOPES,
        "prohibited_decisions": PROHIBITED_DECISIONS,
        "reason": "Deterministic fallback diperbolehkan untuk level LOW karena seluruh guard terpenuhi.",
    }
