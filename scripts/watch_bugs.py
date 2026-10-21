"""Watch repositories for bug reports and track each bug until it is fixed.

Every poll fetches issues updated since the previous poll, keeps the bug reports (plus any
issue pinned with --issue), and records what changed:
new bug, new comments, closed, reopened, and the fix (merged PR or closing commit).

State lives in --state-dir and survives restarts:
    bugs.json     current snapshot of every tracked bug, keyed "owner/name#123"
    events.jsonl  append-only log of every change seen
    cursor.json   per-repo timestamp of the last successful poll

Repos come from owner/name arguments and/or --repos-file. With no repos and no
--issue, the pipeline's selected repos (data/top_repos/repos.jsonl) are watched.
On a repo's first poll only the last --lookback-days are scanned.

Examples:
    uv run scripts/watch_bugs.py --once                         # one poll, e.g. from cron
    uv run scripts/watch_bugs.py --interval 600                 # poll every 10 minutes
    uv run scripts/watch_bugs.py --issue astral-sh/uv#22805     # track one bug report
    uv run scripts/watch_bugs.py --report                       # print tracked bugs
    uv run scripts/watch_bugs.py --classify model --classifier-provider ollama \
        --escalation-provider ollama --escalation-model qwen2.5-coder:14b

What counts as a bug:
    --classify labels   label names matching --label-pattern (free, misses unlabelled bugs)
    --classify model    label/title rules first, then a cheap model for undecided issues,
                        escalating to a stronger model when it is unsure. Verdicts are
                        cached in classifications.json and stored on each bug.
"""

import argparse
import asyncio
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, get_args

import httpx
import openai
from dotenv import load_dotenv
from scrape_github import GitHub, log, slim_issue

from repro.bug_classifier import BugClassifier
from repro.config import Provider

DEFAULT_REPOS_FILE = "data/top_repos/repos.jsonl"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def parse_issue_ref(ref: str) -> tuple[str, int]:
    m = re.fullmatch(r"([\w.-]+/[\w.-]+)#(\d+)", ref)
    if not m:
        raise SystemExit(f"Invalid --issue '{ref}', expected owner/name#123")
    return m[1], int(m[2])


class Tracker:
    def __init__(self, state_dir: Path) -> None:
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bugs: dict[str, dict[str, Any]] = self._load("bugs.json")
        self.cursor: dict[str, str] = self._load("cursor.json")

    def _load(self, name: str) -> dict:
        path = self.dir / name
        return json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        write_atomic(self.dir / "bugs.json", self.bugs)
        write_atomic(self.dir / "cursor.json", self.cursor)

    def emit(self, key: str, event: str, **details: Any) -> None:
        record = {"at": now_iso(), "bug": key, "event": event, **details}
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        extra = " ".join(f"{k}={v}" for k, v in details.items())
        log(f"[{event}] {key} {extra}".rstrip())

    def update(self, issue: dict[str, Any]) -> bool:
        """Merge a fresh issue snapshot. Returns True if the bug needs a fix lookup."""
        key = f"{issue['repo']}#{issue['number']}"
        old = self.bugs.get(key)
        bug = {**(old or {}), **issue, "last_seen": now_iso()}
        bug.setdefault("first_seen", bug["last_seen"])
        bug.setdefault("history", [])

        if old is None:
            self.emit(key, "new", state=issue["state"], title=issue["title"])
        else:
            if issue["comments_count"] > old["comments_count"]:
                new = issue["comments_count"] - old["comments_count"]
                self.emit(key, "comments", new=new, total=issue["comments_count"])
            if issue["state"] != old["state"]:
                event = "closed" if issue["state"] == "closed" else "reopened"
                self.emit(key, event, reason=issue.get("state_reason"))
                bug["history"].append({"at": bug["last_seen"], "state": issue["state"]})
                if event == "reopened":
                    bug.pop("fix", None)  # the earlier fix did not hold
        if old is None:
            bug["history"].append({"at": issue["created_at"], "state": "open"})
            if issue["state"] == "closed":
                bug["history"].append({"at": issue["closed_at"], "state": "closed"})

        self.bugs[key] = bug
        # Look up the fix once per issue revision, and only for bugs closed as completed.
        return (
            bug["state"] == "closed"
            and bug.get("state_reason") not in ("not_planned", "duplicate")
            and "fix" not in bug
            and bug.get("fix_checked_at") != issue["updated_at"]
        )


def _commit_repo(commit_url: str | None, default_repo: str) -> str:
    """The repo a commit lives in, taken from its API url (it may be a fork or another repo)."""
    m = re.search(r"/repos/([\w.-]+/[\w.-]+)/commits/", commit_url or "")
    return m[1] if m else default_repo


async def _commit_fix(gh: GitHub, repo: str, sha: str) -> dict[str, Any]:
    """Describe a fixing commit, upgraded to its merged PR when GitHub knows one."""
    url = f"https://github.com/{repo}/commit/{sha}"
    fix: dict[str, Any] = {"type": "commit", "repo": repo, "sha": sha, "url": url}
    try:
        pulls = (await gh.get(f"/repos/{repo}/commits/{sha}/pulls")).json()
    except httpx.HTTPError:
        return fix
    merged = [p for p in pulls if p.get("merged_at")]
    if not merged:
        return fix
    pr = merged[0]
    return {
        "type": "pull_request",
        "repo": repo,
        "number": pr["number"],
        "title": pr["title"],
        "url": pr["html_url"],
        "merged_at": pr["merged_at"],
        "sha": sha,
    }


async def find_fix(gh: GitHub, repo: str, number: int) -> dict[str, Any] | None:
    """Find the PR or commit that fixed an issue, from strongest evidence to weakest:

    1. the commit that closed it ("closed" event with a commit_id)
    2. the latest merged PR that cross-referenced it
    3. the latest commit in the issue's own repo that referenced it before it was closed

    Fine-grained tokens can hide cross-reference events, which is why 3 exists. For 1
    and 3 the commit is mapped to its merged PR when there is one.
    """
    timeline = await gh.paginate(f"/repos/{repo}/issues/{number}/timeline")
    merged_prs, referenced = [], []
    for ev in timeline:
        event = ev.get("event")
        if event == "closed" and ev.get("commit_id"):
            return await _commit_fix(gh, _commit_repo(ev.get("commit_url"), repo), ev["commit_id"])
        if event == "closed":
            break  # ignore anything that happened after the issue was closed
        # Commits in forks or unrelated repos that merely mention the issue are not fixes.
        if event == "referenced" and _commit_repo(ev.get("commit_url"), repo) == repo:
            referenced.append(ev)
        source = (ev.get("source") or {}).get("issue") or {}
        pr = source.get("pull_request") or {}
        if event == "cross-referenced" and pr.get("merged_at"):
            merged_prs.append(
                {
                    "type": "pull_request",
                    "repo": (source.get("repository") or {}).get("full_name", repo),
                    "number": source["number"],
                    "title": source["title"],
                    "url": source["html_url"],
                    "merged_at": pr["merged_at"],
                }
            )
    if merged_prs:
        return max(merged_prs, key=lambda p: p["merged_at"])
    if referenced:
        return {
            **await _commit_fix(gh, repo, referenced[-1]["commit_id"]),
            "evidence": "referenced",
        }
    return None


async def poll_repo(
    gh: GitHub, tracker: Tracker, repo: str, args: argparse.Namespace
) -> list[tuple[str, int]]:
    started = now_iso()
    since = tracker.cursor.get(repo) or (
        datetime.now(UTC) - timedelta(days=args.lookback_days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = await gh.paginate(
        f"/repos/{repo}/issues",
        params={"state": "all", "since": since, "sort": "updated", "direction": "desc"},
    )
    issues = [slim_issue(item, repo) for item in raw if "pull_request" not in item]
    tracked = [i for i in issues if f"{repo}#{i['number']}" in tracker.bugs]
    new = [i for i in issues if f"{repo}#{i['number']}" not in tracker.bugs]

    if args.classify == "model":
        # Raises on model errors so the cursor stays put and the next poll retries.
        verdicts = await asyncio.gather(*(args.classifier.classify(i) for i in new))
        new = [
            {**i, "classification": v.model_dump(exclude={"tokens"})}
            for i, v in zip(new, verdicts, strict=True)
            if v.is_bug
        ]
    else:
        label_re = re.compile(args.label_pattern, re.IGNORECASE)
        new = [i for i in new if any(label_re.search(lb) for lb in i["labels"])]

    needs_fix = [(repo, i["number"]) for i in tracked + new if tracker.update(i)]
    tracker.cursor[repo] = started
    return needs_fix


async def poll_once(gh: GitHub, tracker: Tracker, repos: list[str], args) -> None:
    needs_fix: list[tuple[str, int]] = []

    async def safe(coro, what: str):
        try:
            return await coro
        except (httpx.HTTPError, openai.APIError) as e:
            log(f"poll failed for {what}: {e}")
            return None

    results = await asyncio.gather(*(safe(poll_repo(gh, tracker, r, args), r) for r in repos))
    for r in results:
        needs_fix.extend(r or [])

    for ref in args.issue:
        repo, number = parse_issue_ref(ref)
        resp = await safe(gh.get(f"/repos/{repo}/issues/{number}"), ref)
        if resp is not None and tracker.update(slim_issue(resp.json(), repo)):
            needs_fix.append((repo, number))

    targets = list(dict.fromkeys(needs_fix))
    if targets:
        log(f"looking up fixes for {len(targets)} closed bugs")
    fixes = await asyncio.gather(
        *(safe(find_fix(gh, repo, number), f"{repo}#{number} timeline") for repo, number in targets)
    )
    for (repo, number), fix in zip(targets, fixes, strict=True):
        key = f"{repo}#{number}"
        tracker.bugs[key]["fix_checked_at"] = tracker.bugs[key]["updated_at"]
        if fix:
            tracker.bugs[key]["fix"] = fix
            tracker.emit(key, "fixed", by=fix["url"])
    tracker.save()
    if args.classifier:
        args.classifier.save_cache()


def load_repos(args: argparse.Namespace) -> list[str]:
    repos = list(args.repos)
    # An explicit --repos-file is always used; the default only when nothing else was given.
    path = Path(args.repos_file or DEFAULT_REPOS_FILE)
    if (args.repos_file or not (repos or args.issue)) and path.exists():
        with open(path) as f:
            repos = [json.loads(line)["full_name"] for line in f if line.strip()]
    return list(dict.fromkeys(repos))


def report(tracker: Tracker) -> None:
    bugs = sorted(tracker.bugs.values(), key=lambda b: b["updated_at"], reverse=True)
    open_bugs = [b for b in bugs if b["state"] == "open"]
    fixed = [b for b in bugs if "fix" in b]
    print(f"{len(bugs)} tracked bugs: {len(open_bugs)} open, {len(fixed)} with a known fix\n")
    for b in bugs:
        fix = b.get("fix", {}).get("url", "")
        print(f"{b['state']:<6}  {b['repo']}#{b['number']:<6}  {b['title'][:70]:<70}  {fix}")


async def main_async(args: argparse.Namespace) -> None:
    tracker = Tracker(Path(args.state_dir))
    if args.report:
        report(tracker)
        return
    repos = load_repos(args)
    if not repos and not args.issue:
        raise SystemExit("Nothing to watch: pass owner/name, --repos-file or --issue.")
    log(f"Watching {len(repos)} repos and {len(args.issue)} pinned issues")

    args.classifier = None
    if args.classify == "model":
        args.classifier = BugClassifier(
            provider=args.classifier_provider,
            model=args.classifier_model,
            escalation_provider=args.escalation_provider,
            escalation_model=args.escalation_model,
            cache_path=Path(args.state_dir) / "classifications.json",
        )
        escalation = args.classifier.escalation_model or "off"
        log(f"Classifying with rules -> {args.classifier.model} -> {escalation}")

    gh = GitHub(os.environ.get("GITHUB_TOKEN"), args.concurrency)
    if "Authorization" not in gh.client.headers:
        log("GITHUB_TOKEN not set: unauthenticated limit is 60 requests/hour")
    try:
        while True:
            log(f"--- poll at {now_iso()}")
            await poll_once(gh, tracker, repos, args)
            if args.once:
                break
            await asyncio.sleep(args.interval)
    finally:
        await gh.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("repos", nargs="*", help="Repositories as owner/name")
    p.add_argument(
        "--repos-file",
        default=None,
        help=f"JSONL with full_name per line (default {DEFAULT_REPOS_FILE} if nothing else given)",
    )
    p.add_argument("--issue", action="append", default=[], help="Always track owner/name#123")
    p.add_argument(
        "--label-pattern",
        default=r"bug|defect|regression|crash",
        help="Regex matched against label names to decide what counts as a bug",
    )
    p.add_argument(
        "--classify",
        default="labels",
        choices=["labels", "model"],
        help="labels: --label-pattern only. model: label/title rules, then a cheap model\n"
        "for issues the rules can't decide (see repro.bug_classifier)",
    )
    p.add_argument("--classifier-provider", choices=get_args(Provider), default=None)
    p.add_argument("--classifier-model", default=None, help="Default: the provider's cheap model")
    p.add_argument("--escalation-provider", choices=get_args(Provider), default=None)
    p.add_argument("--escalation-model", default=None, help="'' disables escalation")
    p.add_argument("--lookback-days", type=int, default=30, help="Window for a repo's first poll")
    p.add_argument("--interval", type=int, default=600, help="Seconds between polls")
    p.add_argument("--once", action="store_true", help="Poll once and exit")
    p.add_argument("--report", action="store_true", help="Print tracked bugs and exit")
    p.add_argument("--state-dir", default="data/bug_tracker")
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    try:
        asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        log("stopped")


if __name__ == "__main__":
    main()
