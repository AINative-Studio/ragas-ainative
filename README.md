# ragas-ainative

Configure [ragas](https://github.com/explodinggradients/ragas) to use AINative's free LLMs + embeddings for RAG evaluation. No OpenAI key needed.

**Zero setup.** Evaluate faithfulness, answer relevancy, and more with Llama 3.3 70B — completely free.

## Install

```bash
pip install ragas-ainative
```

## Quick Start

```python
from ragas_ainative import configure_ragas
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate

configure_ragas()  # Auto-provisions free API key

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(results)
```

## How It Works

`ragas-ainative` is a companion package (NOT a fork) that configures ragas to use AINative's free, OpenAI-compatible API. It sets `OPENAI_API_KEY` and `OPENAI_API_BASE` so ragas routes all LLM judge calls and embeddings through AINative.

On first use, the package auto-provisions a free API key (72-hour TTL). Claim your account at [ainative.studio/signup](https://ainative.studio/signup) for permanent access.

## Available Models

```python
from ragas_ainative import configure_ragas

configure_ragas(model="llama")     # meta-llama/Llama-3.3-70B-Instruct (default)
configure_ragas(model="qwen")      # qwen3-coder-flash
configure_ragas(model="deepseek")  # deepseek-4-flash
configure_ragas(model="kimi")      # kimi-k2
```

## Store Results to ZeroDB

Track evaluation scores over time:

```python
from ragas_ainative import configure_ragas, store_results

configure_ragas()

# After evaluation
results = {"faithfulness": 0.85, "answer_relevancy": 0.92}
store_results(results, dataset_name="my-rag-pipeline")
# Stored to ZeroDB — searchable and trackable
```

## Explicit API Key

```python
configure_ragas(api_key="your-key-here")
```

Or set the environment variable:

```bash
export AINATIVE_API_KEY=your-key-here
```

## API Key Resolution Order

1. Explicit `api_key` parameter
2. `AINATIVE_API_KEY` environment variable
3. `ZERODB_API_KEY` environment variable
4. `~/.zerodb/credentials.json` (shared with zerodb ecosystem)
5. Auto-provision via instant-db

## Requirements

- Python >= 3.9
- ragas >= 0.1.0

## License

MIT
