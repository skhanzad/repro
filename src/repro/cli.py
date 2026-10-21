import argparse
import asyncio

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from repro.agent import Agent
from repro.config import settings
from repro.example_tools import registry
from repro.runner import load_tasks, run_eval, save_results, summarize
from repro.schemas import EvalResult

console = Console()


def _print_result(r: EvalResult) -> None:
    mark = "[green]PASS[/]" if r.grade.passed else "[red]FAIL[/]"
    console.print(
        f"{mark} {r.task.id}  steps={len(r.trajectory.steps)} "
        f"tools={r.trajectory.num_tool_calls}  answer={r.trajectory.final_answer!r}"
    )


async def _run(args: argparse.Namespace) -> None:
    tasks = load_tasks(args.dataset)
    if args.limit:
        tasks = tasks[: args.limit]
    agent = Agent(registry, model=args.model, max_steps=args.max_steps)

    console.print(f"Running [bold]{len(tasks)}[/] tasks with [bold]{agent.model}[/]")
    results = await run_eval(agent, tasks, concurrency=args.concurrency, on_result=_print_result)

    summary = summarize(results)
    table = Table(title="Summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in summary.items():
        table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    console.print(table)
    console.print(f"Saved to {save_results(results, summary, agent.model)}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="repro", description="Agentic evaluation runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an eval over a JSONL dataset")
    run.add_argument("dataset", help="Path to a JSONL file of tasks")
    run.add_argument("--model", default=settings.agent_model)
    run.add_argument("--max-steps", type=int, default=settings.max_steps)
    run.add_argument("--concurrency", type=int, default=settings.concurrency)
    run.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(_run(args))
