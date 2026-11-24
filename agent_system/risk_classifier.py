import re
from pathlib import Path
from typing import Any

from agent_system.integrator import is_doc_file, is_test_file

RISK_ORDER = {"low": 1, "medium": 2, "high": 3}

HIGH_RISK_FINANCIAL_PATTERNS = [
    re.compile(r"\b(wallet|balance|saldo|ledger)\b", re.IGNORECASE),
    re.compile(r"\b(transaction|transaksi|deposit|credit|transfer|settlement|reconciliation|rekonsiliasi)\b", re.IGNORECASE),
    re.compile(r"\b(payment|pembayaran|checkout|billing|invoice|top-up)\b", re.IGNORECASE),
    re.compile(r"\b(gateway|midtrans|xendit|stripe|paypal|callback provider)\b", re.IGNORECASE),
]

HIGH_RISK_SECURITY_PATTERNS = [
    re.compile(r"\b(auth|login|logout|token|jwt|password|credential|secret|permission|role|oauth|session)\b", re.IGNORECASE),
    re.compile(r"\b(encryption|cipher|hash|private_key|api_key|credential)\b", re.IGNORECASE),
]

HIGH_RISK_INFRA_PATTERNS = [
    re.compile(r"\b(migration|migrations|db/migrate|database schema)\b", re.IGNORECASE),
    re.compile(r"\b(docker|dockerfile|kubernetes|k8s|terraform|helm|ci/cd)\b", re.IGNORECASE),
    re.compile(r"\.github/workflows/", re.IGNORECASE),
    re.compile(r"\b(drop table|truncate|delete from|destructive operation)\b", re.IGNORECASE),
]

DEPENDENCY_MANIFEST_NAMES = {
    "go.mod",
    "go.sum",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
}

READINESS_PATTERNS = [
    re.compile(r"\breadiness\b", re.IGNORECASE),
    re.compile(r"/ready\b", re.IGNORECASE),
]

OBSERVABILITY_PATTERNS = [
    re.compile(r"\b(health|ping|liveness|metrics|observability)\b", re.IGNORECASE),
    re.compile(r"/(health|ping|live|metrics)\b", re.IGNORECASE),
]


def strip_negative_constraints(text: str) -> str:
    """Strip negative constraints and out-of-scope clauses before checking risk patterns.

    Avoids triggering high risk when text says 'Jangan mengubah wallet' or 'Out of scope: payment'.
    """
    clean = re.sub(
        r"(?i)\b(?:jangan|dilarang|tidak)\s+(?:mengubah|menyentuh|membuat|memodifikasi|pakai|gunakan)[^.\n;\r]+",
        "",
        text,
    )
    clean = re.sub(
        r"(?i)\b(?:do\s+not|never)\s+(?:modify|change|touch|use|create)[^.\n;\r]+",
        "",
        clean,
    )
    clean = re.sub(
        r"(?i)\b(?:out\s+of\s+scope|excluded|prohibited)\s*:[^.\n;\r]+",
        "",
        clean,
    )
    return clean


def _detect_high_risk_reasons(text: str, files: list[str]) -> list[str]:
    reasons: list[str] = []
    clean_text = strip_negative_constraints(text)

    for pat in HIGH_RISK_FINANCIAL_PATTERNS:
        match = pat.search(clean_text)
        if match:
            reasons.append(f"Menyentuh domain finansial / wallet / transaksi / payment ('{match.group(0)}')")
            break

    for pat in HIGH_RISK_SECURITY_PATTERNS:
        match = pat.search(clean_text)
        if match:
            reasons.append(f"Menyentuh domain keamanan / autentikasi / otorisasi ('{match.group(0)}')")
            break

    for pat in HIGH_RISK_INFRA_PATTERNS:
        match = pat.search(clean_text)
        if match:
            reasons.append(f"Menyentuh migrasi database / infrastruktur / operasi destruktif ('{match.group(0)}')")
            break

    for f in files:
        fname = Path(f).name.lower()
        clean_path = str(f).replace("\\", "/").lower()
        if fname in DEPENDENCY_MANIFEST_NAMES:
            reasons.append(f"Mengubah manifest dependensi ('{f}')")
            break
        if "migration" in clean_path or "schema" in clean_path:
            reasons.append(f"Menyentuh migration / schema file ('{f}')")
            break
        if "wallet" in clean_path or "balance" in clean_path or "transaction" in clean_path or "payment" in clean_path:
            reasons.append(f"Menyentuh berkas domain sensitif finansial ('{f}')")
            break
        if "auth" in clean_path or "token" in clean_path or "secret" in clean_path:
            reasons.append(f"Menyentuh berkas domain keamanan ('{f}')")
            break

    return reasons


def classify_pre_execution_risk(
    request: str = "",
    task_title: str = "",
    task_desc: str = "",
    acceptance_criteria: list[str] | None = None,
    mapped_requirements: list[dict | str] | None = None,
    expected_writable_files: list[str] | None = None,
    conditionally_writable_files: list[str] | None = None,
    proposed_commands: list[dict] | None = None,
) -> dict[str, Any]:
    """Classify risk before coding based on intent, requirements, and expected files."""
    expected_writable_files = expected_writable_files or []
    conditionally_writable_files = conditionally_writable_files or []
    all_target_files = expected_writable_files + conditionally_writable_files

    ac_text = " ".join(acceptance_criteria or [])
    req_text = " ".join([r["description"] if isinstance(r, dict) else str(r) for r in (mapped_requirements or [])])

    text_to_check = f"{task_title} {task_desc} {ac_text} {req_text} " + " ".join(all_target_files)

    # Check if this task is an observability task
    is_observability = any(pat.search(text_to_check) for pat in OBSERVABILITY_PATTERNS)
    is_readiness = any(pat.search(text_to_check) for pat in READINESS_PATTERNS)

    # Check if target files touch any sensitive domains
    has_sensitive_file = any(
        any(k in str(f).lower() for k in ("wallet", "balance", "transaction", "payment", "auth", "token", "migration", "schema"))
        or Path(f).name.lower() in DEPENDENCY_MANIFEST_NAMES
        for f in all_target_files
    )

    if (is_observability or is_readiness) and not has_sensitive_file:
        # Observability task without sensitive files is NEVER HIGH (Requirement 11 & 13)
        if is_readiness:
            return {
                "level": "medium",
                "reasons": ["Task readiness endpoint."],
                "summary": "Pre-risk: MEDIUM (readiness endpoint).",
            }
        return {
            "level": "low",
            "reasons": ["Observability endpoint tanpa modifikasi skema/keamanan."],
            "summary": "Pre-risk: LOW (observability).",
        }

    reasons = _detect_high_risk_reasons(text_to_check, all_target_files)
    if reasons:
        return {
            "level": "high",
            "reasons": reasons,
            "summary": f"Pre-risk: HIGH ({'; '.join(reasons)}).",
        }

    if not is_readiness:
        if all(is_test_file(f) for f in all_target_files) and all_target_files:
            return {"level": "low", "reasons": ["Expected files hanya berkas pengujian."], "summary": "Pre-risk: LOW (test-only)."}
        if all(is_doc_file(f) for f in all_target_files) and all_target_files:
            return {"level": "low", "reasons": ["Expected files hanya berkas dokumentasi."], "summary": "Pre-risk: LOW (docs-only)."}
        if is_observability and len(all_target_files) <= 3:
            return {"level": "low", "reasons": ["Observability endpoint tanpa modifikasi skema."], "summary": "Pre-risk: LOW (observability)."}

    med_reasons = ["Task readiness endpoint."] if is_readiness else ["Perubahan kode produksi/service layer non-kritis."]
    return {
        "level": "medium",
        "reasons": med_reasons,
        "summary": f"Pre-risk: MEDIUM ({'; '.join(med_reasons)}).",
    }


def classify_post_execution_risk(
    actual_changed_files: list[str] | None = None,
    diff_summary: str = "",
    test_status: str = "passed",
    diff_review_status: str = "approved",
    scope_violations: list[str] | None = None,
    has_raw_db_error: bool = False,
    has_secret_exposure: bool = False,
) -> dict[str, Any]:
    """Classify risk after coding based on actual git diff and validation evidence."""
    actual_changed_files = actual_changed_files or []
    scope_violations = scope_violations or []
    text_to_check = f"{diff_summary} " + " ".join(actual_changed_files)

    reasons = _detect_high_risk_reasons(text_to_check, actual_changed_files)
    if has_secret_exposure:
        reasons.append("Terdeteksi potensi paparan secret pada diff.")
    if has_raw_db_error:
        reasons.append("Terdeteksi raw database error pada response.")

    if reasons:
        return {
            "level": "high",
            "reasons": reasons,
            "summary": f"Post-risk: HIGH ({'; '.join(reasons)}).",
        }

    is_safe = (test_status == "passed" and diff_review_status == "approved" and not scope_violations and actual_changed_files)
    is_readiness = any(pat.search(text_to_check) for pat in READINESS_PATTERNS)
    is_observability = any(pat.search(text_to_check) for pat in OBSERVABILITY_PATTERNS)

    if is_safe and not is_readiness:
        if all(is_test_file(f) for f in actual_changed_files):
            return {"level": "low", "reasons": ["Actual diff hanya mengubah berkas test."], "summary": "Post-risk: LOW (test-only diff)."}
        if all(is_doc_file(f) for f in actual_changed_files):
            return {"level": "low", "reasons": ["Actual diff hanya mengubah berkas docs."], "summary": "Post-risk: LOW (docs-only diff)."}
        if is_observability and len(actual_changed_files) <= 3:
            return {"level": "low", "reasons": ["Actual diff adalah observability endpoint sederhana."], "summary": "Post-risk: LOW (observability diff)."}

    med_reasons = []
    if is_readiness:
        med_reasons.append("Actual diff menyentuh readiness endpoint.")
    elif scope_violations:
        med_reasons.append(f"Terdapat scope violations: {scope_violations}")
    elif test_status != "passed":
        med_reasons.append(f"Test status: {test_status}")
    else:
        med_reasons.append("Actual diff mengubah berkas produksi.")

    return {
        "level": "medium",
        "reasons": med_reasons,
        "summary": f"Post-risk: MEDIUM ({'; '.join(med_reasons)}).",
    }


def classify_final_risk(
    pre_risk_info: dict[str, Any],
    post_risk_info: dict[str, Any],
) -> dict[str, Any]:
    """Calculate final risk: highest level between pre_risk and post_risk.

    Risk can NEVER decrease after coding.
    """
    pre_level = pre_risk_info.get("level", "low").lower()
    post_level = post_risk_info.get("level", "low").lower()

    pre_val = RISK_ORDER.get(pre_level, 1)
    post_val = RISK_ORDER.get(post_level, 1)

    if post_val >= pre_val:
        final_level = post_level
        reasons = post_risk_info.get("reasons", []) or pre_risk_info.get("reasons", [])
    else:
        # Risk cannot decrease
        final_level = pre_level
        reasons = pre_risk_info.get("reasons", [])

    escalated = post_val > pre_val

    return {
        "level": final_level,
        "pre_risk": pre_level,
        "post_risk": post_level,
        "escalated": escalated,
        "reasons": reasons,
        "summary": f"Final risk: {final_level.upper()} (pre={pre_level.upper()}, post={post_level.upper()}).",
    }


def classify_risk(
    request: str = "",
    task_title: str = "",
    task_desc: str = "",
    changed_files: list[str] | None = None,
    approved_files: list[str] | None = None,
    scope_violations: list[str] | None = None,
    test_status: str = "passed",
    diff_review_status: str = "approved",
    has_raw_db_error: bool = False,
) -> dict[str, Any]:
    """Backward-compatible composite risk classifier."""
    changed_files = changed_files or []
    pre_info = classify_pre_execution_risk(
        request=request,
        task_title=task_title,
        task_desc=task_desc,
        expected_writable_files=approved_files or changed_files,
    )
    post_info = classify_post_execution_risk(
        actual_changed_files=changed_files,
        diff_summary=f"{request} {task_title}",
        test_status=test_status,
        diff_review_status=diff_review_status,
        scope_violations=scope_violations,
        has_raw_db_error=has_raw_db_error,
    )
    return classify_final_risk(pre_info, post_info)


def evaluate_approval_policy(
    risk_level: str,
    approval_policy: str = "safe-auto",
) -> dict[str, Any]:
    """Evaluate whether changes can be automatically approved or require human approval."""
    policy = (approval_policy or "safe-auto").lower().strip()

    if policy in ("always-human", "never-auto"):
        return {
            "approval_required": True,
            "can_auto_approve": False,
            "policy": policy,
            "risk_level": risk_level,
            "action": "awaiting_human_approval",
            "reason": f"Kebijakan '{policy}' mewajibkan approval manusia untuk semua tingkatan risiko.",
        }

    # Default policy: safe-auto
    if risk_level == "low":
        return {
            "approval_required": False,
            "can_auto_approve": True,
            "policy": "safe-auto",
            "risk_level": "low",
            "action": "auto_approve",
            "reason": "Kebijakan safe-auto: Perubahan berisiko LOW disetujui otomatis setelah test dan diff review lolos.",
        }
    elif risk_level == "medium":
        return {
            "approval_required": True,
            "can_auto_approve": False,
            "policy": "safe-auto",
            "risk_level": "medium",
            "action": "awaiting_human_approval",
            "reason": "Kebijakan safe-auto: Perubahan berisiko MEDIUM membutuhkan approval manusia sebelum integrasi.",
        }
    else:  # high
        return {
            "approval_required": True,
            "can_auto_approve": False,
            "policy": "safe-auto",
            "risk_level": "high",
            "action": "awaiting_human_approval",
            "reason": "Kebijakan safe-auto: Perubahan berisiko HIGH wajib mendapatkan persetujuan manusia.",
        }
