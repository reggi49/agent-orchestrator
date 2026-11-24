import os
from unittest.mock import patch

import pytest

from agent_system.config import _normalize_base_url, get_settings


def test_9router_uses_default_base_url():
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "9router",
            "LLM_API_KEY": "dummy-key",
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "",
            "NINE_ROUTER_BASE_URL": "",
        },
        clear=True,
    ):
        settings = get_settings()
        assert settings.llm_provider == "9router"
        assert settings.llm_base_url == "http://localhost:20128/v1"


def test_llm_base_url_overrides_default():
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "9router",
            "LLM_API_KEY": "dummy-key",
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "http://custom-host:9999/v1",
        },
        clear=True,
    ):
        settings = get_settings()
        assert settings.llm_base_url == "http://custom-host:9999/v1"


def test_llm_api_key_prioritized_over_openai_api_key():
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "9router",
            "LLM_API_KEY": "priority-key",
            "OPENAI_API_KEY": "fallback-key",
            "LLM_MODEL": "test-model",
        },
        clear=True,
    ):
        settings = get_settings()
        assert settings.llm_api_key == "priority-key"


def test_llm_model_prioritized_over_openai_model():
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "priority-model",
            "OPENAI_MODEL": "fallback-model",
        },
        clear=True,
    ):
        settings = get_settings()
        assert settings.llm_model == "priority-model"


def test_missing_key_raises_runtime_error():
    with patch.dict(
        os.environ,
        {
            "LLM_MODEL": "test-model",
        },
        clear=True,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            get_settings()
        assert "LLM_API_KEY" in str(exc_info.value)


def test_missing_model_raises_runtime_error():
    with patch.dict(
        os.environ,
        {
            "LLM_API_KEY": "dummy-key",
        },
        clear=True,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            get_settings()
        assert "LLM_MODEL" in str(exc_info.value)


def test_api_key_not_in_settings_repr():
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "9router",
            "LLM_API_KEY": "super-secret-key-12345",
            "LLM_MODEL": "test-model",
        },
        clear=True,
    ):
        settings = get_settings()
        repr_str = repr(settings)
        str_str = str(settings)
        assert "super-secret-key-12345" not in repr_str
        assert "super-secret-key-12345" not in str_str


def test_base_url_normalized_trailing_slashes():
    url = _normalize_base_url("http://example.com/v1///")
    assert url == "http://example.com/v1"

    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "9router",
            "LLM_API_KEY": "dummy-key",
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "http://localhost:20128/v1///",
        },
        clear=True,
    ):
        settings = get_settings()
        assert settings.llm_base_url == "http://localhost:20128/v1"


def test_config_summary_does_not_print_secret():
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "9router",
            "LLM_API_KEY": "super-secret-key-12345",
            "LLM_MODEL": "test-model",
        },
        clear=True,
    ):
        settings = get_settings()
        summary = settings.config_summary
        assert "super-secret-key-12345" not in str(summary)
        assert summary["has_api_key"] is True
