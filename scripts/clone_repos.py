"""Clone the repositories listed in a repos.jsonl (from scrape_github.py) to local disk.

Each repo lands in <out-dir>/<owner>__<name>. A manifest.jsonl next to them records
the path, branch and checked-out commit for every repo, including failures.

By default clones are shallow (latest commit only). Use --history for a blobless
partial clone: full commit history, file contents downloaded on demand. That lets
you check out the code as it was when an issue was filed.

Examples:
    uv run scripts/clone_repos.py
    uv run scripts/clone_repos.py data/github/repos.jsonl --history --concurrency 8
    uv run scripts/clone_repos.py --update   # fetch latest for repos already cloned
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def git_env() -> dict[str, str]:
    # Never block on a credentials prompt for private or deleted repos.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if token := os.environ.get("GITHUB_TOKEN"):
        # Pass the token as an auth header via env-based config, so it never ends up
        # in the clone's .git/config or in the process arguments.
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env |= {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
        }
    return env


async def git(*args: str, cwd: Path | None = None, timeout: float = 1800) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=git_env(),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"git {args[0]} timed out after {timeout:.0f}s") from None
    if proc.returncode != 0:
        lines = err.decode().strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"git {args[0]} failed")
    return out.decode().strip()


def clone_url(repo: dict[str, Any]) -> str:
    return (repo.get("html_url") or f"https://github.com/{repo['full_name']}") + ".git"


async def clone_one(repo: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    name = repo["full_name"]
    dest = Path(args.out_dir) / name.replace("/", "__")
    branch = repo.get("default_branch")
    record: dict[str, Any] = {"full_name": name, "path": str(dest), "branch": branch}

    try:
        if (dest / ".git").exists():
            if args.update:
                await git("fetch", "--quiet", "origin", cwd=dest)
                await git("reset", "--quiet", "--hard", "FETCH_HEAD", cwd=dest)
                record["status"] = "updated"
            else:
                record["status"] = "exists"
        else:
            cmd = ["clone", "--quiet", "--no-tags"]
            cmd += ["--filter=blob:none"] if args.history else ["--depth", "1"]
            if branch:
                cmd += ["--branch", branch]
            # Clone into a temp dir and rename when done, so an interrupted run never
            # leaves a half-cloned repo that looks complete on the next run.
            partial = dest.with_name(dest.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            await git(*cmd, clone_url(repo), str(partial))
            partial.rename(dest)
            record["status"] = "cloned"
        record["commit"] = await git("rev-parse", "HEAD", cwd=dest)
        record["commit_date"] = await git("log", "-1", "--format=%cI", cwd=dest)
    except RuntimeError as e:
        record["status"] = "failed"
        # Strip any token that git may echo back in its error message.
        record["error"] = str(e).replace(os.environ.get("GITHUB_TOKEN") or "\0", "***")

    log(f"{record['status']:>8}  {name}" + (f"  ({record['error']})" if "error" in record else ""))
    return record


async def main_async(args: argparse.Namespace) -> None:
    with open(args.repos_file) as f:
        repos = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        repos = repos[: args.limit]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log(f"Cloning {len(repos)} repos into {args.out_dir}")

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(repo: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await clone_one(repo, args)

    records = await asyncio.gather(*(bounded(r) for r in repos))

    manifest = Path(args.out_dir) / "manifest.jsonl"
    with open(manifest, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    failed = sum(r["status"] == "failed" for r in records)
    log(f"Done: {len(records) - failed} ok, {failed} failed. Manifest: {manifest}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("repos_file", nargs="?", default="data/github/repos.jsonl")
    p.add_argument("--out-dir", default="data/repos")
    p.add_argument("--history", action="store_true", help="Blobless clone with full history")
    p.add_argument("--update", action="store_true", help="Fetch latest for existing clones")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Only the first N repos")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
