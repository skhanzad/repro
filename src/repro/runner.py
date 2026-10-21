"""Run an agent over a dataset concurrently, grade, and persist results."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from repro.agent import Agent
from repro.config import settings
from repro.graders import grade
from repro.schemas import EvalResult, Task


def load_tasks(path: str | Path) -> list[Task]:
    with open(path) as f:
        return [Task.model_validate_json(line) for line in f if line.strip()]


async def run_eval(
    agent: Agent,
    tasks: list[Task],
    *,
    concurrency: int | None = None,
    on_result=None,
) -> list[EvalResult]:
    sem = asyncio.Semaphore(concurrency or settings.concurrency)

    async def run_one(task: Task) -> EvalResult:
        async with sem:
            traj = await agent.run(task.prompt, task_id=task.id)
            result = EvalResult(task=task, trajectory=traj, grade=await grade(task, traj))
        if on_result:
            on_result(result)
        return result

    return await asyncio.gather(*(run_one(t) for t in tasks))


def summarize(results: list[EvalResult]) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "accuracy": sum(r.grade.passed for r in results) / n,
        "mean_score": sum(r.grade.score for r in results) / n,
        "mean_steps": sum(len(r.trajectory.steps) for r in results) / n,
        "mean_tool_calls": sum(r.trajectory.num_tool_calls for r in results) / n,
        "total_tokens": sum(r.trajectory.total_tokens for r in results),
        "errors": sum(r.trajectory.stop_reason == "error" for r in results),
        "max_steps_hit": sum(r.trajectory.stop_reason == "max_steps" for r in results),
    }


def save_results(results: list[EvalResult], summary: dict, model: str) -> Path:
    run_name = f"{datetime.now():%Y%m%d-%H%M%S}-{model.replace('/', '_')}"
    run_dir = Path(settings.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"model": model, **summary}, indent=2))
    return run_dir
