"""End to end: find the repos with the most issues, scrape their issues, clone them.

Steps, each cached under --work-dir so a rerun continues where the last one stopped:

1. Candidates  GitHub search for --query, up to --pool repos, saved to candidates.json.
               Reused on later runs unless the query changes or --refresh is passed.
2. Rank        Candidates sorted by open issue count; the top --top-n are selected.
3. Issues      Per repo, written to issues/<owner>__<name>.jsonl. Repos with a file
               are skipped. Files are written atomically, so a crash never leaves a
               truncated one.
4. Clone       Into --clone-dir/<owner>__<name>. Repos already cloned are skipped.

Raising --top-n on a later run only does the work for the newly included repos.
At the end it writes repos.jsonl (ranked), issues.jsonl (combined) and manifest.jsonl
(clone status) to --work-dir.

GitHub's search can't sort by issue count, so the candidate pool is searched by
--sort (stars by default) and then ranked by `open_issues_count`. That count includes
open pull requests. A bigger --pool finds more of the high-issue repos (max 1000).

Examples:
    uv run scripts/top_repos_pipeline.py --top-n 20
    uv run scripts/top_repos_pipeline.py --top-n 50 --query "language:python stars:>1000" \
        --max-issues 200 --state closed --comments --history
"""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from clone_repos import clone_one
from dotenv import load_dotenv
from scrape_github import GitHub, log, scrape_repo


def slug(full_name: str) -> str:
    return full_name.replace("/", "__")


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


async def load_candidates(gh: GitHub, args: argparse.Namespace) -> list[dict[str, Any]]:
    cache = Path(args.work_dir) / "candidates.json"
    key = {"query": args.query, "sort": args.sort, "pool": args.pool}
    if cache.exists() and not args.refresh:
        cached = json.loads(cache.read_text())
        if cached["key"] == key:
            log(f"candidates: using cache from {cached['fetched_at']} ({len(cached['repos'])})")
            return cached["repos"]
        log("candidates: query changed, refetching")

    log(f"candidates: searching '{args.query}' (sort={args.sort}, pool={args.pool})")
    found = await gh.paginate(
        "/search/repositories", params={"q": args.query, "sort": args.sort}, limit=args.pool
    )
    repos = [
        {
            "full_name": r["full_name"],
            "html_url": r["html_url"],
            "default_branch": r["default_branch"],
            "open_issues_count": r["open_issues_count"],
            "stargazers_count": r["stargazers_count"],
            "language": r["language"],
            "has_issues": r["has_issues"],
            "archived": r["archived"],
        }
        for r in found
    ]
    payload = {"key": key, "fetched_at": datetime.now(UTC).isoformat(), "repos": repos}
    write_atomic(cache, json.dumps(payload))
    return repos


def rank(candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    eligible = [c for c in candidates if c["has_issues"] and not c["archived"]]
    eligible.sort(key=lambda c: c["open_issues_count"], reverse=True)
    if len(eligible) < top_n:
        log(f"warning: only {len(eligible)} eligible candidates for top {top_n}; raise --pool")
    return [{"rank": i + 1, **c} for i, c in enumerate(eligible[:top_n])]


async def scrape_cached(
    gh: GitHub, repo: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any] | None:
    """Return repo metadata, scraping issues only if not cached. None on failure."""
    issues_path = Path(args.work_dir) / "issues" / f"{slug(repo['full_name'])}.jsonl"
    meta_path = issues_path.with_suffix(".meta.json")
    if issues_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    try:
        meta, issues = await scrape_repo(gh, repo["full_name"], args)
    except httpx.HTTPError as e:
        log(f"issues failed: {repo['full_name']}: {e}")
        return None
    write_atomic(issues_path, "".join(json.dumps(i) + "\n" for i in issues))
    write_atomic(meta_path, json.dumps(meta))  # written last: marks the repo as done
    return meta


async def process(
    gh: GitHub, repo: dict[str, Any], args: argparse.Namespace, sem: asyncio.Semaphore
) -> dict[str, Any]:
    async with sem:
        meta = await scrape_cached(gh, repo, args)
        clone = await clone_one(repo, argparse.Namespace(**vars(args), out_dir=args.clone_dir))
    return {
        "rank": repo["rank"],
        "full_name": repo["full_name"],
        "issues_ok": meta is not None,
        **{k: clone.get(k) for k in ("path", "branch", "commit", "commit_date", "error")},
        "clone_status": clone["status"],
    }


def write_outputs(work_dir: Path, selected: list[dict[str, Any]], manifest: list[dict]) -> int:
    issues_dir = work_dir / "issues"
    repo_lines, n_issues = [], 0
    with open(work_dir / "issues.jsonl", "w") as issues_f:
        for repo in selected:
            s = slug(repo["full_name"])
            meta_path = issues_dir / f"{s}.meta.json"
            if not meta_path.exists():
                continue
            repo_lines.append(
                json.dumps({"rank": repo["rank"], **json.loads(meta_path.read_text())})
            )
            with open(issues_dir / f"{s}.jsonl") as f:
                for line in f:
                    issues_f.write(line)
                    n_issues += 1
    write_atomic(work_dir / "repos.jsonl", "".join(line + "\n" for line in repo_lines))
    manifest.sort(key=lambda m: m["rank"])
    write_atomic(work_dir / "manifest.jsonl", "".join(json.dumps(m) + "\n" for m in manifest))
    return n_issues


async def main_async(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    (work_dir / "issues").mkdir(parents=True, exist_ok=True)
    Path(args.clone_dir).mkdir(parents=True, exist_ok=True)

    gh = GitHub(os.environ.get("GITHUB_TOKEN"), args.concurrency)
    if "Authorization" not in gh.client.headers:
        log("GITHUB_TOKEN not set: unauthenticated limit is 60 requests/hour")
    try:
        selected = rank(await load_candidates(gh, args), args.top_n)
        log(f"selected top {len(selected)} repos by open issues:")
        for r in selected:
            log(f"  {r['rank']:>4}. {r['full_name']}  ({r['open_issues_count']} open)")

        sem = asyncio.Semaphore(args.repo_concurrency)
        manifest = await asyncio.gather(*(process(gh, r, args, sem) for r in selected))
    finally:
        await gh.close()

    n_issues = write_outputs(work_dir, selected, manifest)
    issues_failed = sum(not m["issues_ok"] for m in manifest)
    clones_failed = sum(m["clone_status"] == "failed" for m in manifest)
    log(
        f"Done: {len(selected)} repos, {n_issues} issues "
        f"({issues_failed} issue scrapes failed, {clones_failed} clones failed). "
        "Rerun the same command to retry failures."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--top-n", type=int, default=10, help="How many repos to take (by open issues)")
    p.add_argument("--query", default="stars:>1000 archived:false", help="GitHub repo search")
    p.add_argument("--sort", default="stars", choices=["stars", "forks", "updated"])
    p.add_argument("--pool", type=int, default=300, help="Candidates to rank (max 1000)")
    p.add_argument("--refresh", action="store_true", help="Refetch candidates even if cached")
    # issue scraping (same meaning as in scrape_github.py)
    p.add_argument("--max-issues", type=int, default=100, help="Max issues per repo")
    p.add_argument("--state", default="all", choices=["open", "closed", "all"])
    p.add_argument("--labels")
    p.add_argument("--since")
    p.add_argument("--min-body-chars", type=int, default=0)
    p.add_argument("--comments", action="store_true")
    # cloning (same meaning as in clone_repos.py)
    p.add_argument("--history", action="store_true", help="Blobless clone with full history")
    p.add_argument("--update", action="store_true", help="Fetch latest for existing clones")
    # io / concurrency
    p.add_argument("--work-dir", default="data/top_repos")
    p.add_argument("--clone-dir", default="data/repos")
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent GitHub API requests")
    p.add_argument("--repo-concurrency", type=int, default=4, help="Repos processed at once")
    args = p.parse_args()
    if not 1 <= args.pool <= 1000:
        p.error("--pool must be between 1 and 1000 (GitHub search limit)")
    return args


def main() -> None:
    load_dotenv()
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
