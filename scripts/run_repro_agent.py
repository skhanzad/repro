"""Run the bug-reproduction agent on tracked bugs, each in its own sandbox.

For each bug the agent gets the issue title and body and the repo checked out just
before the fix, and must write a pytest test that fails there. The test is then
graded in fresh sandboxes: it must fail on the buggy tree and pass on the fix commit.

Bugs come from watch_bugs.py's bugs.json (only Python repos, and by default only bugs
with a known fix, since those can be verified) or from --tasks, a JSONL file with
repo, number, title, body, clone_path and fix_commit and/or buggy_commit.

Sandboxes:
    docker        (default) no network, no secrets, non-root, CPU/memory/pid limits.
                  The first run per repo builds an image with its dependencies, which
                  is the only step that uses the network.
    local-unsafe  runs tools directly on this machine with a scrubbed env. No isolation
                  at all; for development without Docker only.

Examples:
    uv run scripts/run_repro_agent.py --limit 5
    uv run scripts/run_repro_agent.py --issue NousResearch/hermes-agent#134889 \
        --model gpt-5-mini --max-cost 0.25 --max-steps 25
    uv run scripts/run_repro_agent.py --tasks my_tasks.jsonl --sandbox local-unsafe \
        --provider ollama
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

from dotenv import load_dotenv
from scrape_github import GitHub, log

from repro.agent import Agent
from repro.bug_repro import (
    SYSTEM_PROMPT,
    BugTask,
    Limits,
    ReproResult,
    ensure_commit,
    resolve_commit,
    run_task,
    task_from_bug,
    tree_tar,
)
from repro.config import Provider, settings
from repro.sandbox import DockerSandbox, LocalSandbox, ensure_env_image
from repro.schemas import Trajectory


def slug(repo: str) -> str:
    return repo.replace("/", "__")


async def fix_sha(gh: GitHub, fix: dict[str, Any]) -> str | None:
    if fix.get("sha"):
        return fix["sha"]
    if fix.get("type") == "pull_request":
        pr = (await gh.get(f"/repos/{fix['repo']}/pulls/{fix['number']}")).json()
        return pr.get("merge_commit_sha") if pr.get("merged_at") else None
    return None


async def tasks_from_bugs(args: argparse.Namespace) -> list[BugTask]:
    bugs: dict[str, dict] = json.loads(Path(args.bugs).read_text())
    if args.issue:
        bugs = {k: v for k, v in bugs.items() if k in set(args.issue)}
    if args.repo:
        bugs = {k: v for k, v in bugs.items() if v["repo"] in set(args.repo)}
    if not args.allow_unfixed:
        bugs = {k: v for k, v in bugs.items() if "fix" in v}

    manifest = {}
    if Path(args.repos_file).exists():
        for line in Path(args.repos_file).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                manifest[r["full_name"]] = r
    gh = GitHub(os.environ.get("GITHUB_TOKEN"), 8)
    tasks: list[BugTask] = []
    try:
        for key, bug in bugs.items():
            if args.limit and len(tasks) >= args.limit:
                break
            lang = manifest.get(bug["repo"], {}).get("language")
            if lang and lang != "Python":
                continue
            clone = Path(args.clone_dir) / slug(bug["repo"])
            if not (clone / ".git").exists():
                log(f"skip {key}: no clone at {clone} (run clone_repos.py)")
                continue
            try:
                fields = task_from_bug(bug, clone)
                if "fix" in bug:
                    sha = await fix_sha(gh, bug["fix"])
                    if not sha:
                        log(f"skip {key}: fix has no merge commit")
                        continue
                    await ensure_commit(str(clone), sha)
                    fields["fix_commit"] = sha
                    fields["buggy_commit"] = await resolve_commit(str(clone), f"{sha}^1")
                else:
                    fields["buggy_commit"] = await resolve_commit(str(clone), "HEAD")
                tasks.append(BugTask(**fields))
            except Exception as e:  # noqa: BLE001 - one bad bug should not stop the batch
                log(f"skip {key}: {e}")
    finally:
        await gh.close()
    return tasks


async def tasks_from_file(path: str) -> list[BugTask]:
    tasks = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = json.loads(line)
        clone = fields["clone_path"]
        if fields.get("fix_commit") and not fields.get("buggy_commit"):
            fields["buggy_commit"] = f"{fields['fix_commit']}^1"
        fields["buggy_commit"] = await resolve_commit(clone, fields.get("buggy_commit", "HEAD"))
        if fields.get("fix_commit"):
            fields["fix_commit"] = await resolve_commit(clone, fields["fix_commit"])
        tasks.append(BugTask(**fields))
    return tasks


def sandbox_factory(kind: str):
    if kind == "local-unsafe":

        async def local(task: BugTask):
            return await LocalSandbox().start()

        return local

    images: dict[str, str] = {}
    locks: dict[str, asyncio.Lock] = {}

    async def docker(task: BugTask):
        lock = locks.setdefault(task.repo, asyncio.Lock())
        async with lock:  # build each repo's environment image once
            if task.repo not in images:
                log(f"preparing environment image for {task.repo} (uses network)")
                tar = await tree_tar(task.clone_path, task.buggy_commit)
                images[task.repo] = await ensure_env_image(slug(task.repo), tar)
        return await DockerSandbox(images[task.repo]).start()

    return docker


def result_status(r: ReproResult) -> str:
    if r.error:
        return "error"
    if r.fail_to_pass:
        return "FAIL->PASS"
    if r.fails_before:
        return "fails (unverified)" if r.task.fix_commit is None else "fails, fix fails"
    return r.before.outcome if r.before else "no test"


def print_summary(results: list[ReproResult]) -> dict[str, Any]:
    n = len(results) or 1
    costs = [r.trajectory.cost_usd for r in results]
    summary = {
        "n": len(results),
        "test_written": sum(r.test_path is not None for r in results),
        "fails_before": sum(r.fails_before for r in results),
        "fail_to_pass": sum(r.fail_to_pass for r in results),
        "verifiable": sum(r.task.fix_commit is not None for r in results),
        "mean_steps": sum(len(r.trajectory.steps) for r in results) / n,
        "total_tokens": sum(r.trajectory.total_tokens for r in results),
        "total_cost_usd": None if None in costs else round(sum(costs), 4),
        "stop_reasons": {
            reason: sum(r.trajectory.stop_reason == reason for r in results)
            for reason in ("final_answer", "max_steps", "budget", "error")
        },
        "errors": sum(r.error is not None for r in results),
    }
    print(json.dumps(summary, indent=2))
    return summary


async def main_async(args: argparse.Namespace) -> None:
    tasks = await (tasks_from_file(args.tasks) if args.tasks else tasks_from_bugs(args))
    if not tasks:
        raise SystemExit("No runnable tasks.")
    if args.sandbox == "local-unsafe":
        log("WARNING: local-unsafe sandbox has NO isolation and full network access")

    limits = Limits(
        max_steps=args.max_steps,
        max_cost_usd=args.max_cost,
        max_total_tokens=args.max_tokens,
        pytest_timeout_s=args.pytest_timeout,
    )

    def make_agent(tools):
        return Agent(
            tools,
            provider=args.provider,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            max_steps=limits.max_steps,
            max_cost_usd=limits.max_cost_usd,
            max_total_tokens=limits.max_total_tokens,
            sequential_tools=True,
        )

    model = make_agent(None).model
    log(f"Running {len(tasks)} tasks with {model} in {args.sandbox} sandboxes")
    factory = sandbox_factory(args.sandbox)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(task: BugTask) -> ReproResult:
        async with sem:
            try:
                result = await run_task(task, make_agent, factory, limits)
            except Exception as e:  # noqa: BLE001 - record the failure, keep the batch going
                result = ReproResult(
                    task=task, trajectory=Trajectory(task_id=task.id, model=model), error=str(e)
                )
        status = result_status(result)
        cost = result.trajectory.cost_usd
        cost_str = f"${cost:.4f}" if cost is not None else "cost n/a"
        log(
            f"{status:<16} {task.id}  steps={len(result.trajectory.steps)} "
            f"stop={result.trajectory.stop_reason} {cost_str}"
        )
        return result

    results = await asyncio.gather(*(one(t) for t in tasks))

    out = Path(settings.results_dir) / f"repro-{datetime.now():%Y%m%d-%H%M%S}"
    (out / "tests").mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
            if r.test_content:
                name = f"{slug(r.task.repo)}__{r.task.number}__{Path(r.test_path).name}"
                (out / "tests" / name).write_text(r.test_content)
    summary = print_summary(results)
    (out / "summary.json").write_text(json.dumps({"model": model, **summary}, indent=2))
    log(f"Saved to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_argument_group("tasks")
    src.add_argument("--bugs", default="data/bug_tracker/bugs.json")
    src.add_argument("--tasks", help="JSONL of explicit tasks instead of --bugs")
    src.add_argument("--issue", action="append", default=[], help="owner/name#123")
    src.add_argument("--repo", action="append", default=[], help="Only bugs in owner/name")
    src.add_argument("--allow-unfixed", action="store_true", help="Include bugs without a fix")
    src.add_argument("--limit", type=int, default=None)
    src.add_argument("--clone-dir", default="data/repos")
    src.add_argument("--repos-file", default="data/top_repos/repos.jsonl")

    run = p.add_argument_group("agent and limits")
    run.add_argument("--provider", choices=get_args(Provider), default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--max-cost", type=float, default=0.50, help="USD per task")
    run.add_argument("--max-tokens", type=int, default=400_000, help="Tokens per task")
    run.add_argument("--pytest-timeout", type=float, default=300)
    run.add_argument("--sandbox", choices=["docker", "local-unsafe"], default="docker")
    run.add_argument("--concurrency", type=int, default=2)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
