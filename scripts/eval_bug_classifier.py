"""Measure how well the cheap model separates bugs from non-bugs.

Ground truth comes from issue labels: issues the label rules classify unambiguously
(e.g. "bug" vs "enhancement") are kept, everything else is dropped. The classifier
then sees each issue *without* its labels, so it has to decide from title and body.

Reports accuracy / precision / recall for the cheap model alone and for the full
cascade (cheap -> escalation when unsure), plus token usage, and writes per-issue
results to results/bug_classifier-<timestamp>.jsonl.

Examples:
    uv run scripts/eval_bug_classifier.py --limit 100
    uv run scripts/eval_bug_classifier.py --provider ollama --no-escalation
    uv run scripts/eval_bug_classifier.py --provider ollama --escalation-provider ollama \
        --escalation-model qwen2.5-coder:14b
    uv run scripts/eval_bug_classifier.py --model gpt-5-mini --escalate-below 0.8
"""

import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv

from repro.bug_classifier import BugClassifier, Classification, rule_classify
from repro.config import Provider, settings


def load_labelled(path: str, limit: int | None, seed: int) -> list[tuple[dict, bool]]:
    with open(path) as f:
        issues = [json.loads(line) for line in f if line.strip()]
    labelled = []
    for issue in issues:
        truth = rule_classify({"title": "", "labels": issue.get("labels", [])})
        if truth is not None and truth.source == "rule_label":
            labelled.append((issue, truth.is_bug))
    random.Random(seed).shuffle(labelled)
    return labelled[:limit] if limit else labelled


def metrics(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    tp = sum(p and t for p, t in pairs)
    fp = sum(p and not t for p, t in pairs)
    fn = sum(not p and t for p, t in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": sum(p == t for p, t in pairs) / len(pairs) if pairs else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


async def main_async(args: argparse.Namespace) -> None:
    data = load_labelled(args.issues, args.limit, args.seed)
    n_bugs = sum(t for _, t in data)
    print(f"{len(data)} labelled issues ({n_bugs} bugs, {len(data) - n_bugs} non-bugs)")
    if not data:
        raise SystemExit("No issues with unambiguous bug / non-bug labels.")

    common = {"use_rules": False, "use_labels": False, "concurrency": args.concurrency}
    if args.provider:
        common["provider"] = args.provider
    cheap = BugClassifier(model=args.model, escalation_model="", **common)
    cheap_results: list[Classification] = await asyncio.gather(
        *(cheap.classify(issue) for issue, _ in data)
    )

    # Re-ask the escalation model only where the cheap one was unsure.
    final_results = list(cheap_results)
    escalation_provider = args.escalation_provider or settings.escalation_provider
    escalation_model = args.escalation_model or settings.resolved_escalation_model()
    if args.no_escalation:
        escalation_model = ""
    unsure = [
        i for i, r in enumerate(cheap_results) if r.ambiguous or r.confidence < args.escalate_below
    ]
    if escalation_model and unsure:
        strong = BugClassifier(
            provider=escalation_provider,
            model=escalation_model,
            escalation_model="",
            use_rules=False,
            use_labels=False,
            concurrency=args.concurrency,
        )
        escalated = await asyncio.gather(*(strong.classify(data[i][0]) for i in unsure))
        for i, r in zip(unsure, escalated, strict=True):
            final_results[i] = r.model_copy(
                update={"source": "escalation_model", "tokens": r.tokens + cheap_results[i].tokens}
            )

    truths = [t for _, t in data]
    rows = [
        ("cheap model only", cheap.model, cheap_results),
        (
            "cascade (ambiguous/unsure)",
            f"{cheap.model} -> {escalation_model or 'off'}",
            final_results,
        ),
    ]
    print(
        f"\n{'setup':<28} {'models':<28} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'tokens':>8}"
    )
    for name, models, results in rows:
        m = metrics([(r.is_bug, t) for r, t in zip(results, truths, strict=True)])
        tokens = sum(r.tokens for r in results)
        print(
            f"{name:<28} {models:<28} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {tokens:>8}"
        )
    print(f"\nescalated {len(unsure) if escalation_model else 0}/{len(data)} issues")

    out = Path(settings.results_dir) / f"bug_classifier-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for (issue, truth), c, r in zip(data, cheap_results, final_results, strict=True):
            row = {
                "issue": f"{issue['repo']}#{issue['number']}",
                "title": issue["title"],
                "labels": issue["labels"],
                "truth_is_bug": truth,
                "cheap": c.model_dump(),
                "final": r.model_dump(),
            }
            f.write(json.dumps(row) + "\n")
    print(f"per-issue results: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--issues", default="data/top_repos/issues.jsonl")
    p.add_argument("--provider", choices=get_args(Provider), default=None)
    p.add_argument("--model", default=None, help="Cheap model (default per provider)")
    p.add_argument("--escalate-below", type=float, default=settings.escalate_below)
    p.add_argument("--escalation-provider", choices=get_args(Provider), default=None)
    p.add_argument("--escalation-model", default=None)
    p.add_argument("--no-escalation", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
