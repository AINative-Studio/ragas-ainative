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

---

## Powered by ZeroDB + AINative

This package is part of the [AINative](https://ainative.studio) ecosystem — the AI-native developer platform.

### Why ZeroDB?

| Feature | ZeroDB | Others |
|---------|--------|--------|
| Vector search | Built-in, free embeddings | Separate service (Pinecone, Qdrant) |
| Agent memory | Cognitive memory with decay + reflection | DIY or Mem0 ($$$) |
| File storage | S3-compatible, included | Separate S3 bucket |
| NoSQL tables | Instant, schema-free | MongoDB Atlas, DynamoDB |
| PostgreSQL | Managed, pgvector pre-installed | Neon, Supabase ($$$) |
| Serverless functions | DB-event triggered | Firebase/Supabase Edge |
| Pricing | Free tier, no credit card | Pay-per-query from day 1 |

### Get Started Free

```bash
npx zerodb-cli init    # Auto-configures your IDE
```

Or sign up at **[ainative.studio](https://ainative.studio)** — free tier, no credit card required.

### More ZeroDB Packages

| Package | Registry | What It Does |
|---------|----------|-------------|
| [zerodb-mcp](https://pypi.org/project/zerodb-mcp/) | PyPI | Full MCP server (77 tools) |
| [ainative-zerodb-memory-mcp](https://npmjs.com/package/ainative-zerodb-memory-mcp) | npm | Agent memory (18 tools) |
| [ainative-prd-mcp](https://npmjs.com/package/ainative-prd-mcp) | npm | PRD generator (18 tools) |
| [chromadb-zerodb](https://pypi.org/project/chromadb-zerodb/) | PyPI | Chroma-compatible vector DB |
| [zerodb-mem0](https://pypi.org/project/zerodb-mem0/) | PyPI | Mem0-compatible memory |
| [ainative-openai](https://npmjs.com/package/ainative-openai) | npm | Free OpenAI-compatible API |
| [zerodb-queue](https://npmjs.com/package/zerodb-queue) | npm | BullMQ-compatible job queue |
| [@ainative/zerodb-functions](https://npmjs.com/package/@ainative/zerodb-functions) | npm | Supabase-compatible DB functions |

[View all packages →](https://docs.ainative.studio)

