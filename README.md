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

`scripts/watch_bugs.py` watches repos for bug reports and tracks each one until it's fixed:
new bugs, new comments, closed and reopened, and the fixing PR or commit. By default it
watches the pipeline's selected repos (`data/top_repos/repos.jsonl`). State is kept in
`data/bug_tracker/` (`bugs.json` snapshot, `events.jsonl` change log, `cursor.json` per-repo
poll times), so each poll only fetches what changed since the last one.

```bash
uv run scripts/watch_bugs.py --once                       # one poll (cron-friendly)
uv run scripts/watch_bugs.py --interval 600               # keep polling every 10 minutes
uv run scripts/watch_bugs.py --issue astral-sh/uv#22805   # track one specific bug
uv run scripts/watch_bugs.py --report                     # list tracked bugs and fixes
```

A bug is any issue whose labels match `--label-pattern` (default
`bug|defect|regression|crash`). To find a fix, the script looks for the commit that closed the
issue, then a merged PR that references it, then a commit in the same repo that mentions it,
which is mapped to its PR when there is one.

### Bug classifier

Labels are free, but many repos leave issues unlabelled or label them inconsistently.
`repro.bug_classifier` decides "bug or not" in three tiers, each reached only when the
previous one can't decide:

1. **Rules** (free): labels (`bug`, `type/bug`, `enhancement`, `question`, ...) and title tags
   (`[Bug]:`, `Feature request:`). Conflicting signals fall through.
2. **Cheap model** (`gpt-5-nano` / `gemma4:e4b`): title + first 3000 chars of the body ->
   category, whether it is ambiguous, and a confidence.
3. **Escalation model** (`gpt-5-mini` by default): only when the cheap model flags the issue
   as ambiguous or reports confidence below `REPRO_ESCALATE_BELOW`. Set
   `REPRO_ESCALATION_MODEL=` (empty) to turn it off.

Verdicts are cached by issue content, so unchanged issues are never re-classified.

```bash
uv run scripts/watch_bugs.py --classify model                         # OpenAI
uv run scripts/watch_bugs.py --classify model --classifier-provider ollama \
    --escalation-provider ollama --escalation-model qwen2.5-coder:14b  # fully local
uv run scripts/eval_bug_classifier.py --limit 100                     # measure accuracy
```

`eval_bug_classifier.py` uses issues with clear labels as ground truth, hides the labels,
and reports accuracy, precision, recall and token use for the cheap model alone and with
escalation. Small models report high confidence even when wrong, so the ambiguity flag is
usually the more useful escalation trigger.

## Bug reproduction agent

`scripts/run_repro_agent.py` gives the agent a bug report plus the repo as it was just before
the fix. The agent must write a pytest test that reproduces the bug. The loop runs on the
host, because it calls the model API. Every tool call runs in a sandbox.

- **Tools:** `search_code`, `read_file`, `write_test` (new `test_*.py` / `*_test.py` files
  only; it can't change existing files or write outside the repo) and `run_pytest`.
  Tool output is clipped, and every command has a timeout.
- **Limits:** `--max-steps`, `--max-cost` (USD, from `MODEL_PRICES` in `config.py`) and
  `--max-tokens` per task. These are checked after each model call, so a run can go over by
  at most one call. Models not in the price table have no cost limit, only the token limit.
- **Sandbox (`--sandbox docker`):** `--network=none`, `--cap-drop=ALL`,
  `no-new-privileges`, a non-root user, memory/CPU/process limits, and no environment
  variables from the host. The repo is copied in, never mounted. Each repo's dependencies
  are installed once into an image, and that build is the only step with network access.
- **Grading ("fail to pass"):** in fresh sandboxes, the test must fail with an assertion on
  the buggy commit (not with an import or collection error) and pass on the fix commit.
- **What the agent sees:** only the issue title and body, never the comments or the fixing
  PR, which could give the answer away.

```bash
uv run scripts/run_repro_agent.py --limit 5 --model gpt-5-mini --max-cost 0.25
uv run scripts/run_repro_agent.py --issue NousResearch/hermes-agent#46310
uv run scripts/run_repro_agent.py --tasks tasks.jsonl --sandbox local-unsafe --provider ollama
```

Tasks come from `data/bug_tracker/bugs.json` (Python repos with a known fix) or from a JSONL
file of `{repo, number, title, body, clone_path, fix_commit}`. `--sandbox local-unsafe` runs
tools on the host with a scrubbed environment but **no isolation**. It's for development
without Docker only.

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
| `src/repro/bug_classifier.py` | Rules → cheap model → escalation cascade for "is this issue a bug?" |
| `src/repro/bug_repro.py` | Bug-repro task: sandboxed tools, limits, fail-to-pass grading |
| `src/repro/sandbox.py` | `DockerSandbox` (isolated), per-repo dependency images, `LocalSandbox` (dev only) |
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
