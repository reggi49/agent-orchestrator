import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _get_first_available(*env_vars: str) -> str:
    """Get first available environment variable value."""
    for var in env_vars:
        value = os.getenv(var, "").strip()
        if value:
            return value
    return ""


def _normalize_base_url(url: str) -> str:
    """Normalize base URL by removing trailing slashes."""
    if not url:
        return url
    while url.endswith("/"):
        url = url[:-1]
    return url


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str = field(repr=False)
    llm_base_url: str | None
    llm_model: str
    default_project_path: Path | None
    max_repair_attempts: int
    max_file_characters: int
    debug: bool

    @property
    def has_api_key(self) -> bool:
        """Check if API key is available without exposing it."""
        return bool(self.llm_api_key)

    @property
    def config_summary(self) -> dict:
        """Safe configuration summary without secrets."""
        return {
            "provider": self.llm_provider,
            "base_url": self.llm_base_url,
            "model": self.llm_model,
            "has_api_key": self.has_api_key,
            "debug": self.debug,
        }


def get_settings() -> Settings:
    # Provider with fallback
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    # API key with fallback to legacy env vars
    api_key = _get_first_available(
        "LLM_API_KEY",
        "NINE_ROUTER_API_KEY",
        "OPENAI_API_KEY"
    )

    # Model with fallback
    model = _get_first_available(
        "LLM_MODEL",
        "OPENAI_MODEL"
    )

    # Base URL with fallback
    base_url = _get_first_available(
        "LLM_BASE_URL",
        "NINE_ROUTER_BASE_URL"
    )

    # Default base URL for 9router
    if provider == "9router" and not base_url:
        base_url = "http://localhost:20128/v1"

    # Normalize base URL
    if base_url:
        base_url = _normalize_base_url(base_url)
        # Ensure 9Router base URL ends with /v1
        if provider == "9router" and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

    # Other settings
    default_project = os.getenv("DEFAULT_PROJECT_PATH", "").strip()
    debug = os.getenv("DEBUG", "false").strip().lower() in ("true", "1", "yes")

    # Validations
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY belum tersedia. "
            "Tambahkan ke file .env."
        )

    if not model:
        raise RuntimeError(
            "LLM_MODEL belum tersedia. "
            "Tambahkan ke file .env."
        )

    if provider == "9router" and not base_url:
        raise RuntimeError(
            "LLM_BASE_URL diperlukan untuk provider 9router. "
            "Tambahkan ke file .env."
        )

    return Settings(
        llm_provider=provider,
        llm_api_key=api_key,
        llm_base_url=base_url if base_url else None,
        llm_model=model,
        default_project_path=(
            Path(default_project).expanduser().resolve()
            if default_project
            else None
        ),
        max_repair_attempts=int(
            os.getenv("MAX_REPAIR_ATTEMPTS", "3")
        ),
        max_file_characters=int(
            os.getenv("MAX_FILE_CHARACTERS", "30000")
        ),
        debug=debug,
    )