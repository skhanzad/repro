"""Scrape GitHub repositories and their issues into JSONL.

Repos come from explicit `owner/name` arguments and/or a GitHub search query.
Writes two files to --out-dir:
    repos.jsonl   one line per repository
    issues.jsonl  one line per issue (pull requests excluded), tagged with its repo

Set GITHUB_TOKEN (env or .env) to raise the rate limit from 60 to 5000 requests/hour.

Examples:
    uv run scripts/scrape_github.py astral-sh/uv openai/openai-python --max-issues 50
    uv run scripts/scrape_github.py --search "language:python stars:>5000" --max-repos 10 \
        --state closed --comments
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

API = "https://api.github.com"
MAX_RATE_LIMIT_WAIT_S = 15 * 60

REPO_FIELDS = [
    "full_name", "description", "html_url", "language", "topics", "stargazers_count",
    "forks_count", "open_issues_count", "license", "default_branch", "created_at",
    "updated_at", "pushed_at", "archived",
]  # fmt: skip


class GitHub:
    def __init__(self, token: str | None, concurrency: int) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.AsyncClient(base_url=API, headers=headers, timeout=30)
        self.sem = asyncio.Semaphore(concurrency)

    async def close(self) -> None:
        await self.client.aclose()

    async def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with retries on rate limiting and transient server errors."""
        for attempt in range(5):
            async with self.sem:
                resp = await self.client.get(url, params=params)
            if resp.status_code in (403, 429) and _is_rate_limited(resp):
                wait = _rate_limit_wait(resp)
                if wait > MAX_RATE_LIMIT_WAIT_S:
                    resp.raise_for_status()
                log(f"rate limited, sleeping {wait:.0f}s")
                await asyncio.sleep(wait)
                continue
            if resp.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp

    async def paginate(
        self, url: str, params: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Follow Link headers until `limit` items are collected or pages run out."""
        items: list[dict[str, Any]] = []
        params = {"per_page": 100, **(params or {})}
        next_url: str | None = url
        while next_url and (limit is None or len(items) < limit):
            resp = await self.get(next_url, params=params)
            params = None  # the next link already carries the query string
            data = resp.json()
            items.extend(data["items"] if isinstance(data, dict) else data)
            next_url = resp.links.get("next", {}).get("url")
        return items[:limit] if limit is not None else items


def _is_rate_limited(resp: httpx.Response) -> bool:
    return resp.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in resp.headers


def _rate_limit_wait(resp: httpx.Response) -> float:
    if retry_after := resp.headers.get("retry-after"):
        return float(retry_after)
    reset = float(resp.headers.get("x-ratelimit-reset", time.time() + 60))
    return max(reset - time.time(), 0) + 1


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def slim_repo(repo: dict[str, Any]) -> dict[str, Any]:
    out = {k: repo.get(k) for k in REPO_FIELDS}
    out["license"] = (repo.get("license") or {}).get("spdx_id")
    return out


def slim_issue(issue: dict[str, Any], repo: str) -> dict[str, Any]:
    return {
        "repo": repo,
        "number": issue["number"],
        "title": issue["title"],
        "body": issue.get("body") or "",
        "state": issue["state"],
        "state_reason": issue.get("state_reason"),
        "labels": [label["name"] for label in issue.get("labels", [])],
        "author": (issue.get("user") or {}).get("login"),
        "comments_count": issue.get("comments", 0),
        "created_at": issue["created_at"],
        "updated_at": issue["updated_at"],
        "closed_at": issue.get("closed_at"),
        "html_url": issue["html_url"],
    }


async def fetch_comments(gh: GitHub, repo: str, number: int) -> list[dict[str, Any]]:
    raw = await gh.paginate(f"/repos/{repo}/issues/{number}/comments")
    return [
        {
            "author": (c.get("user") or {}).get("login"),
            "body": c.get("body") or "",
            "created_at": c["created_at"],
        }
        for c in raw
    ]


async def scrape_repo(gh: GitHub, repo: str, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    meta = slim_repo((await gh.get(f"/repos/{repo}")).json())

    params: dict[str, Any] = {"state": args.state, "sort": "created", "direction": "desc"}
    if args.labels:
        params["labels"] = args.labels
    if args.since:
        params["since"] = args.since

    # The issues endpoint also returns PRs; over-fetch a little and filter them out.
    issues: list[dict[str, Any]] = []
    next_url: str | None = f"/repos/{repo}/issues"
    page_params: dict[str, Any] | None = {"per_page": 100, **params}
    while next_url and len(issues) < args.max_issues:
        resp = await gh.get(next_url, params=page_params)
        page_params = None
        issues.extend(
            slim_issue(i, repo)
            for i in resp.json()
            if "pull_request" not in i and len(i.get("body") or "") >= args.min_body_chars
        )
        next_url = resp.links.get("next", {}).get("url")
    issues = issues[: args.max_issues]

    if args.comments:
        comment_lists = await asyncio.gather(
            *(
                fetch_comments(gh, repo, i["number"]) if i["comments_count"] else _empty()
                for i in issues
            )
        )
        for issue, comments in zip(issues, comment_lists, strict=True):
            issue["comments"] = comments

    log(f"{repo}: {len(issues)} issues")
    return meta, issues


async def _empty() -> list:
    return []


async def resolve_repos(gh: GitHub, args: argparse.Namespace) -> list[str]:
    repos = list(args.repos)
    if args.search:
        found = await gh.paginate(
            "/search/repositories",
            params={"q": args.search, "sort": args.sort},
            limit=args.max_repos,
        )
        repos.extend(r["full_name"] for r in found)
    bad = [r for r in repos if not re.fullmatch(r"[\w.-]+/[\w.-]+", r)]
    if bad:
        raise SystemExit(f"Invalid repo names (expected owner/name): {bad}")
    return list(dict.fromkeys(repos))  # dedupe, keep order


async def main_async(args: argparse.Namespace) -> None:
    gh = GitHub(os.environ.get("GITHUB_TOKEN"), args.concurrency)
    if "Authorization" not in gh.client.headers:
        log("GITHUB_TOKEN not set: unauthenticated limit is 60 requests/hour")
    try:
        repos = await resolve_repos(gh, args)
        if not repos:
            raise SystemExit("No repos given. Pass owner/name arguments or --search.")
        log(f"Scraping {len(repos)} repos")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_issues = 0
        with (
            open(out_dir / "repos.jsonl", "w") as repos_f,
            open(out_dir / "issues.jsonl", "w") as issues_f,
        ):
            for coro in asyncio.as_completed([scrape_repo(gh, r, args) for r in repos]):
                try:
                    meta, issues = await coro
                except httpx.HTTPStatusError as e:
                    log(f"skipping: {e.request.url} -> {e.response.status_code}")
                    continue
                repos_f.write(json.dumps(meta) + "\n")
                issues_f.writelines(json.dumps(i) + "\n" for i in issues)
                n_issues += len(issues)
        log(f"Wrote {n_issues} issues to {out_dir}")
    finally:
        await gh.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("repos", nargs="*", help="Repositories as owner/name")
    p.add_argument("--search", help="GitHub repo search query, e.g. 'language:python stars:>1000'")
    p.add_argument("--sort", default="stars", choices=["stars", "forks", "updated"])
    p.add_argument("--max-repos", type=int, default=10, help="Max repos taken from --search")
    p.add_argument("--max-issues", type=int, default=100, help="Max issues per repo")
    p.add_argument("--state", default="all", choices=["open", "closed", "all"])
    p.add_argument("--labels", help="Comma-separated label filter, e.g. 'bug,good first issue'")
    p.add_argument("--since", help="Only issues updated after this ISO 8601 timestamp")
    p.add_argument("--min-body-chars", type=int, default=0, help="Drop issues with short bodies")
    p.add_argument("--comments", action="store_true", help="Also fetch each issue's comments")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out-dir", default="data/github")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
