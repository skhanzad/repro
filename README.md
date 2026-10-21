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

## Scripts

`scripts/scrape_github.py` collects repositories and their issues into `data/github/`
(`repos.jsonl` and `issues.jsonl`, one issue per line; pull requests are excluded).

```bash
uv run scripts/scrape_github.py astral-sh/uv openai/openai-python --max-issues 50 --comments
uv run scripts/scrape_github.py --search "language:python stars:>5000" --max-repos 10 \
    --state closed --labels bug --min-body-chars 200
```

Set `GITHUB_TOKEN` in `.env` to raise the API limit from 60 to 5000 requests/hour. Without a
token, `--comments` uses up the limit quickly, since each issue that has comments costs at least
one request. The script waits out rate limits (up to 15 minutes) and skips repos that can't be
fetched.

`scripts/clone_repos.py` clones every repo in `repos.jsonl` into `data/repos/<owner>__<name>`
and writes `data/repos/manifest.jsonl` with each repo's path, branch, commit and status.

```bash
uv run scripts/clone_repos.py                 # shallow clones (latest commit only)
uv run scripts/clone_repos.py --history       # full history, file contents fetched on demand
uv run scripts/clone_repos.py --update        # bring existing clones up to date
```

Use `--history` when you need the code as it was when an issue was filed
(`git checkout <commit>`). Repos that are already cloned are skipped unless you pass `--update`.

`scripts/top_repos_pipeline.py` does it all in one go: it finds the repos with the most open
issues, scrapes their issues and clones them. Every step is cached in `data/top_repos/`, so
rerunning (or raising `--top-n`) continues from where the last run stopped.

```bash
uv run scripts/top_repos_pipeline.py --top-n 20
uv run scripts/top_repos_pipeline.py --top-n 50 --query "language:python stars:>1000" \
    --pool 1000 --max-issues 200 --state closed --comments
```

GitHub search can't sort by issue count, so the script searches `--query` for up to `--pool`
candidates (sorted by stars), then ranks them by open issues, which include open PRs. Pass
`--refresh` to search again. Outputs are `data/top_repos/{repos,issues,manifest}.jsonl` and the
clones in `data/repos/`.

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
