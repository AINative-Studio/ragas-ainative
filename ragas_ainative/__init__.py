"""
ragas-ainative — Configure ragas to use AINative's free LLMs + embeddings.

No OpenAI key needed. Evaluate your RAG pipelines for free with Llama, Qwen,
DeepSeek, and Kimi models.

Usage:
    from ragas_ainative import configure_ragas
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas import evaluate

    configure_ragas()  # sets AINative as eval LLM + embeddings

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

Refs #3954
"""

from ragas_ainative.config import (
    configure_ragas,
    get_llm_config,
    get_embeddings_config,
    store_results,
    MODELS,
    DEFAULT_MODEL,
    API_BASE,
    EMBEDDINGS_MODEL,
)
from ragas_ainative.provision import resolve_api_key

__all__ = [
    "configure_ragas",
    "get_llm_config",
    "get_embeddings_config",
    "store_results",
    "resolve_api_key",
    "MODELS",
    "DEFAULT_MODEL",
    "API_BASE",
    "EMBEDDINGS_MODEL",
]
__version__ = "0.1.0"
