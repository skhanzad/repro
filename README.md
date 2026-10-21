# repro

A framework-free agentic evaluation harness. Just the `openai` SDK, pydantic and asyncio.

## Setup

```bash
uv sync
cp .env.example .env   # set OPENAI_API_KEY (not needed for Ollama-only runs)
```

## Run

```bash
uv run repro run data/example.jsonl
uv run repro run data/example.jsonl --model gpt-5 --max-steps 15 --concurrency 4 --limit 2

# Local models via Ollama (OpenAI-compatible endpoint, default http://localhost:11434/v1)
uv run repro run data/example.jsonl --provider ollama --model gemma4:e4b \
    --judge-provider ollama --concurrency 2
```

Each run writes `results/<timestamp>-<model>/results.jsonl` (full trajectories) and `summary.json`.

### Providers

Providers are `openai` and `ollama`. Both use the same `openai` SDK client; Ollama just gets a
different base URL. The agent and the LLM judge pick their provider independently, so you can
evaluate a local model with an OpenAI judge. If `--model` is omitted, each provider has a default
model (see `DEFAULT_MODELS` in `config.py`).

Not every model that Ollama says supports tools emits real tool calls. For example,
`qwen2.5-coder` writes the tool-call JSON as plain text, so it never calls a tool. Check that
`mean_tool_calls` is above zero before trusting the scores.

## Layout

| File | Purpose |
|---|---|
| `src/repro/agent.py` | The agent loop: chat completion → run tool calls → repeat until a plain answer or `max_steps` |
| `src/repro/tools.py` | `ToolRegistry`: register a function taking one pydantic model; its schema becomes the tool spec |
| `src/repro/example_tools.py` | Example `calculator` and `lookup` tools; replace with your own |
| `src/repro/graders.py` | `exact_match`, `contains`, `llm_judge`; add more with `@grader("name")` |
| `src/repro/runner.py` | Loads JSONL tasks, runs them concurrently, grades, summarizes, saves |
| `src/repro/schemas.py` | `Task`, `Trajectory`, `Step`, `Grade`, `EvalResult` |
| `src/repro/config.py` | Settings from `REPRO_*` env vars / `.env`, providers and default models |
| `src/repro/llm.py` | `get_client(provider)`: one cached `AsyncOpenAI` client per provider |

## Dataset format

One JSON object per line:

```json
{"id": "calc-1", "prompt": "What is 17*23+4?", "expected": "395", "grader": "exact_match", "metadata": {}}
```

## Tests

```bash
uv run pytest      # offline, uses a scripted fake client
uv run ruff check .
```
