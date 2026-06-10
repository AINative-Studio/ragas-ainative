"""
Tests for ragas-ainative configuration.

Refs #3954
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from ragas_ainative.config import (
    configure_ragas,
    get_llm_config,
    get_embeddings_config,
    get_model,
    store_results,
    MODELS,
    DEFAULT_MODEL,
    API_BASE,
    EMBEDDINGS_MODEL,
    MEMORY_API_BASE,
)
from ragas_ainative.provision import (
    resolve_api_key,
    _load_credentials,
    _auto_provision,
    _save_credentials,
    ZERODB_DIR,
    CREDS_PATH,
    CONFIG_PATH,
)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestGetModel:
    """Tests for model alias resolution."""

    def test_llama_alias(self):
        assert get_model("llama") == "meta-llama/Llama-3.3-70B-Instruct"

    def test_qwen_alias(self):
        assert get_model("qwen") == "qwen3-coder-flash"

    def test_deepseek_alias(self):
        assert get_model("deepseek") == "deepseek-4-flash"

    def test_kimi_alias(self):
        assert get_model("kimi") == "kimi-k2"

    def test_passthrough_full_id(self):
        assert get_model("meta-llama/Llama-3.3-70B-Instruct") == "meta-llama/Llama-3.3-70B-Instruct"

    def test_unknown_returns_as_is(self):
        assert get_model("custom-model") == "custom-model"


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

class TestGetLlmConfig:
    """Tests for LLM judge config generation."""

    @patch("ragas_ainative.config.resolve_api_key", return_value="test-key")
    def test_default_config(self, mock_resolve):
        config = get_llm_config()
        assert config["OPENAI_API_KEY"] == "test-key"
        assert config["OPENAI_API_BASE"] == API_BASE
        assert config["model"] == DEFAULT_MODEL

    @patch("ragas_ainative.config.resolve_api_key", return_value="test-key")
    def test_custom_model(self, mock_resolve):
        config = get_llm_config(model="qwen")
        assert config["model"] == "qwen3-coder-flash"

    @patch("ragas_ainative.config.resolve_api_key", return_value="test-key")
    def test_custom_base(self, mock_resolve):
        config = get_llm_config(api_base="https://custom.api/v1")
        assert config["OPENAI_API_BASE"] == "https://custom.api/v1"


# ---------------------------------------------------------------------------
# Embeddings config
# ---------------------------------------------------------------------------

class TestGetEmbeddingsConfig:
    """Tests for embeddings config generation."""

    @patch("ragas_ainative.config.resolve_api_key", return_value="emb-key")
    def test_default_embeddings_config(self, mock_resolve):
        config = get_embeddings_config()
        assert config["OPENAI_API_KEY"] == "emb-key"
        assert config["embeddings_model"] == EMBEDDINGS_MODEL

    @patch("ragas_ainative.config.resolve_api_key", return_value="emb-key")
    def test_custom_embeddings_model(self, mock_resolve):
        config = get_embeddings_config(model="custom-emb")
        assert config["embeddings_model"] == "custom-emb"


# ---------------------------------------------------------------------------
# configure_ragas
# ---------------------------------------------------------------------------

class TestConfigureRagas:
    """Tests for the main configure_ragas() function."""

    @patch("ragas_ainative.config.resolve_api_key", return_value="ragas-key-123456789")
    def test_sets_env_vars(self, mock_resolve):
        with patch.dict(os.environ, {}, clear=False):
            result = configure_ragas(verbose=False)
            assert result == "ragas-key-123456789"
            assert os.environ["OPENAI_API_KEY"] == "ragas-key-123456789"
            assert os.environ["OPENAI_API_BASE"] == API_BASE
            assert os.environ["AINATIVE_API_KEY"] == "ragas-key-123456789"

    @patch("ragas_ainative.config.resolve_api_key", return_value="ragas-key-123456789")
    def test_verbose_output(self, mock_resolve, capsys):
        configure_ragas(verbose=True)
        captured = capsys.readouterr()
        assert "ragas-ainative configured" in captured.err

    @patch("ragas_ainative.config.resolve_api_key", return_value="ragas-key-123456789")
    def test_returns_api_key(self, mock_resolve):
        result = configure_ragas(verbose=False)
        assert result == "ragas-key-123456789"


# ---------------------------------------------------------------------------
# store_results
# ---------------------------------------------------------------------------

class TestStoreResults:
    """Tests for storing evaluation results to ZeroDB."""

    @patch("ragas_ainative.config.resolve_api_key", return_value="store-key")
    @patch("ragas_ainative.config.requests")
    def test_store_results_success(self, mock_requests, mock_resolve):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "mem-1", "status": "stored"}
        mock_requests.post.return_value = mock_resp

        result = store_results(
            results={"faithfulness": 0.85, "answer_relevancy": 0.92},
            dataset_name="my-rag-test",
        )

        assert result is not None
        assert result["id"] == "mem-1"
        mock_requests.post.assert_called_once()
        call_kwargs = mock_requests.post.call_args
        assert "remember" in call_kwargs[0][0]

    @patch("ragas_ainative.config.resolve_api_key", return_value="store-key")
    @patch("ragas_ainative.config.requests")
    def test_store_results_failure_returns_none(self, mock_requests, mock_resolve):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_requests.post.return_value = mock_resp

        result = store_results(results={"score": 0.5})
        assert result is None

    @patch("ragas_ainative.config.resolve_api_key", return_value="store-key")
    @patch("ragas_ainative.config.requests")
    def test_store_results_network_error_returns_none(self, mock_requests, mock_resolve):
        import requests as real_requests
        mock_requests.RequestException = real_requests.RequestException
        mock_requests.post.side_effect = real_requests.ConnectionError("fail")

        result = store_results(results={"score": 0.5})
        assert result is None


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    """Tests for the multi-source API key resolution."""

    def test_explicit_key_takes_priority(self):
        result = resolve_api_key(explicit_key="explicit-key")
        assert result == "explicit-key"

    @patch.dict(os.environ, {"AINATIVE_API_KEY": "env-key"}, clear=False)
    def test_ainative_env_var(self):
        result = resolve_api_key()
        assert result == "env-key"

    @patch.dict(os.environ, {"ZERODB_API_KEY": "zerodb-key"}, clear=False)
    def test_zerodb_env_var(self):
        env = os.environ.copy()
        env.pop("AINATIVE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["ZERODB_API_KEY"] = "zerodb-key"
            result = resolve_api_key()
            assert result == "zerodb-key"


# ---------------------------------------------------------------------------
# Credential file loading
# ---------------------------------------------------------------------------

class TestLoadCredentials:
    """Tests for credential file loading."""

    def test_returns_none_when_no_files(self, tmp_path):
        with patch("ragas_ainative.provision.CREDS_PATH", tmp_path / "missing.json"):
            with patch("ragas_ainative.provision.CONFIG_PATH", tmp_path / "missing2.json"):
                result = _load_credentials()
                assert result is None

    def test_reads_credentials_json(self, tmp_path):
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps({"api_key": "file-key"}))

        with patch("ragas_ainative.provision.CREDS_PATH", creds_file):
            with patch("ragas_ainative.provision.CONFIG_PATH", tmp_path / "missing.json"):
                result = _load_credentials()
                assert result == "file-key"


# ---------------------------------------------------------------------------
# Auto-provisioning
# ---------------------------------------------------------------------------

class TestAutoProvision:
    """Tests for the auto-provisioning endpoint."""

    @patch("ragas_ainative.provision.requests")
    @patch("ragas_ainative.provision._save_credentials")
    @patch("ragas_ainative.provision._print_success")
    def test_successful_provision(self, mock_print, mock_save, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "api_key": "new-key-12345678",
            "project_id": "proj-1",
            "claim_url": "https://ainative.studio/claim/abc",
        }
        mock_requests.post.return_value = mock_resp

        result = _auto_provision()

        assert result == "new-key-12345678"
        mock_save.assert_called_once()

    @patch("ragas_ainative.provision.requests")
    def test_rate_limited(self, mock_requests):
        import requests as real_requests
        mock_requests.RequestException = real_requests.RequestException
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_requests.post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Rate limited"):
            _auto_provision()

    @patch("ragas_ainative.provision.requests")
    def test_server_error(self, mock_requests):
        import requests as real_requests
        mock_requests.RequestException = real_requests.RequestException
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_requests.post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Provisioning failed"):
            _auto_provision()

    @patch("ragas_ainative.provision.requests")
    def test_network_error(self, mock_requests):
        import requests as real_requests
        mock_requests.RequestException = real_requests.RequestException
        mock_requests.post.side_effect = real_requests.ConnectionError("fail")

        with pytest.raises(RuntimeError, match="Network error"):
            _auto_provision()


# ---------------------------------------------------------------------------
# Credential persistence
# ---------------------------------------------------------------------------

class TestSaveCredentials:
    """Tests for credential persistence."""

    def test_saves_to_file(self, tmp_path):
        creds_dir = tmp_path / ".zerodb"
        creds_path = creds_dir / "credentials.json"

        with patch("ragas_ainative.provision.ZERODB_DIR", creds_dir):
            with patch("ragas_ainative.provision.CREDS_PATH", creds_path):
                _save_credentials({
                    "api_key": "saved-key",
                    "project_id": "proj-1",
                    "claim_url": "https://ainative.studio/claim/abc",
                })

        assert creds_path.exists()
        data = json.loads(creds_path.read_text())
        assert data["api_key"] == "saved-key"
        assert data["source"] == "ragas-ainative"
