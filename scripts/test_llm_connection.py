import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)

from agent_system.config import get_settings


def check_models_endpoint(base_url: str, api_key: str, configured_model: str) -> bool:
    """Check /models endpoint if available."""
    models_url = f"{base_url}/models"
    req = urllib.request.Request(models_url)
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                model_items = data.get("data", [])
                model_ids = [m.get("id") for m in model_items if isinstance(m, dict) and "id" in m]
                print(f"Models endpoint: OK (200)")
                if model_ids:
                    print(f"Daftar Model IDs ({len(model_ids)}): {', '.join(model_ids[:10])}")
                    if configured_model not in model_ids:
                        print(
                            f"PERINGATAN: LLM_MODEL '{configured_model}' tidak ditemukan dalam daftar model dari 9Router."
                        )
                        print(f"Model ID yang tersedia antara lain: {model_ids[:5]}")
                        return False
                return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("ERROR: 9Router key ditolak oleh gateway lokal (HTTP 401).")
            sys.exit(1)
        elif e.code == 404:
            print("Catatan: Endpoint /v1/models tidak ditemukan (HTTP 404). Melanjutkan ke chat test.")
            return True
        else:
            print(f"Catatan: Endpoint /v1/models mengembalikan HTTP {e.code}.")
            return True
    except urllib.error.URLError as e:
        # Check if connection refused
        reason_str = str(e.reason).lower()
        if "connection refused" in reason_str:
            print("ERROR: 9Router belum berjalan atau tidak dapat dijangkau.")
            sys.exit(1)
        print(f"Catatan: Pengecekan /models mengalami timeout/URLError ({e.reason}). Melanjutkan ke chat test.")
        return True
    except Exception as e:
        print(f"Catatan: Pengecekan /models dilewati: {type(e).__name__}")
        return True


def main() -> None:
    try:
        settings = get_settings()
    except Exception as e:
        print(f"ERROR: Konfigurasi gagal dimuat: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Provider: {settings.llm_provider}")
    print(f"Base URL: {settings.llm_base_url}")
    print(f"Model: {settings.llm_model}")
    print(f"Key tersedia: {settings.has_api_key}")

    if not settings.has_api_key:
        print("ERROR: API key tidak tersedia.", file=sys.stderr)
        sys.exit(1)

    # Check models endpoint if base_url is configured
    if settings.llm_base_url:
        model_valid = check_models_endpoint(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
        )
        if not model_valid:
            print("ERROR: Konfigurasi LLM_MODEL tidak cocok dengan model 9Router.", file=sys.stderr)
            sys.exit(1)

    # Simple chat test without structured output
    model_kwargs = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": 0,
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url

    try:
        model = ChatOpenAI(**model_kwargs)
        response = model.invoke("Jawab hanya dengan teks: AUTH_OK")
        resp_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        print(f"Response: {resp_text}")
        sys.exit(0)
    except AuthenticationError:
        print(
            "ERROR: Autentikasi LLM gagal. "
            "Periksa LLM_API_KEY dan pastikan key berasal dari provider yang sama dengan LLM_BASE_URL.",
            file=sys.stderr,
        )
        sys.exit(1)
    except APIConnectionError:
        print(
            "ERROR: Tidak dapat terhubung ke LLM gateway. "
            "Pastikan 9Router berjalan dan LLM_BASE_URL benar.",
            file=sys.stderr,
        )
        sys.exit(1)
    except NotFoundError:
        print(
            "ERROR: Model tidak ditemukan. "
            "Periksa LLM_MODEL terhadap daftar model dari 9Router.",
            file=sys.stderr,
        )
        sys.exit(1)
    except RateLimitError:
        print(
            "ERROR: Gateway atau provider sedang mencapai batas penggunaan.",
            file=sys.stderr,
        )
        sys.exit(1)
    except TimeoutError:
        print(
            "ERROR: Request ke LLM gateway timeout.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        if settings.debug:
            import traceback
            traceback.print_exc()
        else:
            print(
                f"ERROR: Koneksi gagal: {type(e).__name__}",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
