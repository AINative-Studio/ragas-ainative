"""
ragas-ainative — Configuration

Sets environment variables so ragas uses AINative's OpenAI-compatible
endpoint for LLM-based evaluation and embeddings.

Ragas reads OPENAI_API_KEY and OPENAI_API_BASE to configure its default
LLM judge and embeddings. We set these to AINative values.

Refs #3954
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from ragas_ainative.provision import resolve_api_key

API_BASE = "https://api.ainative.studio/api/v1"
MEMORY_API_BASE = "https://api.ainative.studio/api/v1/public/memory/v2"
EMBEDDINGS_ENDPOINT = "https://api.ainative.studio/api/v1/embeddings"

# Model aliases -> full model identifiers
MODELS: Dict[str, str] = {
    # Meta Llama
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    # Qwen
    "qwen": "qwen3-coder-flash",
    "qwen-coder": "qwen3-coder-flash",
    # DeepSeek
    "deepseek": "deepseek-4-flash",
    "deepseek-flash": "deepseek-4-flash",
    # Kimi
    "kimi": "kimi-k2",
}

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
EMBEDDINGS_MODEL = "bge-m3"


def get_model(alias: str) -> str:
    """
    Resolve a model alias to its full identifier.

    Args:
        alias: Short alias (e.g. "llama", "qwen") or full model ID.

    Returns:
        Full model identifier string.
    """
    return MODELS.get(alias, alias)


def get_llm_config(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> dict:
    """
    Return env var overrides for ragas LLM judge.

    Args:
        model: Model alias or full ID. Defaults to Llama 3.3 70B.
        api_key: Explicit API key. Auto-provisions if not provided.
        api_base: Override API base URL.

    Returns:
        Dict with OPENAI_API_KEY, OPENAI_API_BASE, model name.
    """
    key = resolve_api_key(api_key)
    base = api_base or API_BASE
    model_id = get_model(model) if model else DEFAULT_MODEL

    return {
        "OPENAI_API_KEY": key,
        "OPENAI_API_BASE": base,
        "model": model_id,
    }


def get_embeddings_config(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Return config for AINative embeddings used by ragas.

    Args:
        api_key: Explicit API key. Uses resolved key if not provided.
        api_base: Override API base URL.
        model: Embeddings model name. Defaults to bge-m3.

    Returns:
        Dict with embeddings API configuration.
    """
    key = resolve_api_key(api_key)
    base = api_base or API_BASE
    emb_model = model or EMBEDDINGS_MODEL

    return {
        "OPENAI_API_KEY": key,
        "OPENAI_API_BASE": base,
        "embeddings_model": emb_model,
    }


def configure_ragas(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Configure ragas to use AINative's free LLMs and embeddings.

    Sets environment variables that ragas reads for its LLM judge
    and embeddings:
    - OPENAI_API_KEY -> AINative API key
    - OPENAI_API_BASE -> AINative OpenAI-compatible endpoint

    Also sets AINATIVE_API_KEY for ecosystem compatibility.

    Args:
        model: Model alias or full ID for the judge LLM.
            Defaults to Llama 3.3 70B.
        api_key: Explicit API key. Auto-provisions if not provided.
        api_base: Override API base URL.
        verbose: Print configuration summary to stderr.

    Returns:
        The API key being used.

    Example:
        from ragas_ainative import configure_ragas
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas import evaluate

        configure_ragas()
        results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    """
    config = get_llm_config(model=model, api_key=api_key, api_base=api_base)

    # Set env vars for ragas (uses OpenAI SDK under the hood)
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["OPENAI_API_BASE"] = config["OPENAI_API_BASE"]

    # Also set AINATIVE_API_KEY for ecosystem tools
    os.environ["AINATIVE_API_KEY"] = config["OPENAI_API_KEY"]

    if verbose:
        key_preview = config["OPENAI_API_KEY"][:12] + "..."
        print(
            f"\n  ragas-ainative configured!\n"
            f"\n"
            f"  Judge LLM:   {config['model']}\n"
            f"  Embeddings:  {EMBEDDINGS_MODEL}\n"
            f"  API Base:    {config['OPENAI_API_BASE']}\n"
            f"  API Key:     {key_preview}\n"
            f"\n"
            f"  All ragas evaluations will now use AINative's free LLMs.\n",
            file=sys.stderr,
        )

    return config["OPENAI_API_KEY"]


def store_results(
    results: dict,
    dataset_name: str = "ragas-eval",
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """
    Store ragas evaluation results to ZeroDB for tracking over time.

    Sends the evaluation scores as a memory entry to ZeroDB's
    /remember endpoint, making it searchable and trackable.

    Args:
        results: Ragas evaluation results dict (scores per metric).
        dataset_name: Name to tag the evaluation. Defaults to "ragas-eval".
        api_key: Explicit API key. Uses resolved key if not provided.

    Returns:
        ZeroDB response dict, or None if storage fails.
    """
    key = resolve_api_key(api_key)

    payload = {
        "entity": f"ragas-eval:{dataset_name}",
        "content": json.dumps({
            "type": "ragas_evaluation",
            "dataset": dataset_name,
            "scores": results,
            "evaluated_at": datetime.now(tz=None).isoformat(),
        }),
        "metadata": {
            "source": "ragas-ainative",
            "dataset": dataset_name,
        },
    }

    try:
        resp = requests.post(
            f"{MEMORY_API_BASE}/remember",
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if resp.status_code in (200, 201):
            return resp.json()
        return None
    except requests.RequestException:
        return None
