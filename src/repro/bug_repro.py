"""Bug reproduction task: given a bug report and a repo, write a failing pytest test.

The agent loop runs on the host (it needs the model API). Every tool call runs in a
sandbox (see sandbox.py) that has no network and no secrets. The agent only sees the
code as it was *before* the fix and only the issue title and body, never its comments
or the fixing PR.

Grading is SWE-bench style "fail to pass", in fresh sandboxes:
    1. on the buggy tree (fix commit's parent) the test must fail with test failures,
       not collection or import errors;
    2. on the fix commit the same test must pass.
A test that fails on the buggy tree but has no fix commit to check against counts as
"fails_before" only.
"""

import asyncio
import io
import os
import posixpath
import re
import tarfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from repro.agent import Agent
from repro.sandbox import ExecResult, Sandbox, run
from repro.schemas import Trajectory
from repro.tools import ToolRegistry

TEST_NAME = re.compile(r"(^|/)(test_[\w-]*|[\w-]*_test)\.py$")

SYSTEM_PROMPT = """You reproduce bugs. You get a bug report and a Python repository checked out
at a version that still has the bug. Your job: write ONE pytest test file that fails on this code
because of the reported bug, and that would pass once the bug is fixed.

Work like this:
1. Use search_code and read_file to find the code the report is about.
2. Write a small, focused test with write_test. Test the behavior described in the report,
   through the project's real code. Do not mock away the buggy code.
3. Run it with run_pytest. It must FAIL with an assertion about the buggy behavior. An
   ImportError, SyntaxError or fixture error does not count: fix the test and run it again.
4. When the test fails for the right reason, stop calling tools and reply with the test
   file path and one sentence on why it fails.

You cannot modify the project's source files, only write new test files. There is no network
access, so do not try to install packages."""

TASK_PROMPT = """Bug report for {repo}, issue #{number}

Title: {title}

{body}"""


class Limits(BaseModel):
    max_steps: int = 30
    max_cost_usd: float | None = 0.50
    max_total_tokens: int | None = 400_000
    tool_timeout_s: float = 120
    pytest_timeout_s: float = 300
    max_output_chars: int = 8_000
    max_read_lines: int = 400
    max_test_bytes: int = 50_000


class BugTask(BaseModel):
    repo: str
    number: int
    title: str
    body: str = ""
    clone_path: str
    buggy_commit: str  # tree the agent works on
    fix_commit: str | None = None  # tree the test must pass on, when known

    @property
    def id(self) -> str:
        return f"{self.repo}#{self.number}"


class TestRun(BaseModel):
    exit_code: int
    outcome: str  # passed | failed | collection_error | no_tests | timeout | missing
    output: str = ""


class ReproResult(BaseModel):
    task: BugTask
    trajectory: Trajectory
    test_path: str | None = None
    test_content: str | None = None
    before: TestRun | None = None
    after: TestRun | None = None
    fails_before: bool = False
    fail_to_pass: bool = False
    error: str | None = None


# --- host-side git -----------------------------------------------------------


async def git(clone: str, *args: str, timeout: float = 600) -> ExecResult:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return await run(["git", "-C", clone, *args], timeout=timeout, env=env)


async def ensure_commit(clone: str, sha: str) -> None:
    """Make sure a commit (and its parent) exist locally; shallow clones may lack them."""
    # sha^1 existing implies sha exists too.
    if (await git(clone, "cat-file", "-e", f"{sha}^1^{{commit}}")).exit_code == 0:
        return
    res = await git(clone, "fetch", "--quiet", "--depth", "2", "origin", sha)
    if res.exit_code != 0:
        raise RuntimeError(f"cannot fetch {sha}: {res.output.strip()}")


async def resolve_commit(clone: str, rev: str) -> str:
    res = await git(clone, "rev-parse", "--verify", f"{rev}^{{commit}}")
    if res.exit_code != 0:
        raise RuntimeError(f"unknown revision {rev}: {res.output.strip()}")
    return res.output.strip()


async def tree_tar(clone: str, commit: str) -> bytes:
    """The repo's files at a commit, as a tar (blobless clones fetch blobs here)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", clone, "archive", "--format=tar", commit,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    out, err = await asyncio.wait_for(proc.communicate(), 900)
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {commit} failed: {err.decode().strip()}")
    return out


# --- tools -------------------------------------------------------------------


class SearchArgs(BaseModel):
    pattern: str = Field(description="Extended regex (grep -E) to search for")
    path: str = Field(".", description="Directory or file to search, relative to repo root")
    glob: str | None = Field(None, description="Only files matching this glob, e.g. '*.py'")


class ReadArgs(BaseModel):
    path: str = Field(description="File or directory, relative to repo root")
    start_line: int = Field(1, ge=1)
    end_line: int | None = Field(None, description="Inclusive; default start_line + 199")


class WriteTestArgs(BaseModel):
    path: str = Field(description="New test file, e.g. tests/test_issue_123.py")
    content: str = Field(description="Full contents of the test file")


class PytestArgs(BaseModel):
    path: str = Field(description="Test file or directory to run")
    k: str | None = Field(None, description="Optional -k expression to select tests")


def _clip(text: str, limit: int) -> str:
    """Keep head and tail: pytest puts the useful summary at the end."""
    if len(text) <= limit:
        return text
    head = limit // 3
    return f"{text[:head]}\n... [{len(text) - limit} chars clipped] ...\n{text[-(limit - head) :]}"


def _rel(path: str) -> str:
    """Normalize a repo-relative path and refuse anything outside the repo."""
    norm = posixpath.normpath(path.strip().lstrip("/") or ".")
    if norm == ".." or norm.startswith("../"):
        raise ValueError(f"path escapes the repository: {path}")
    return norm


def pytest_cmd(path: str, k: str | None = None) -> list[str]:
    cmd = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "-rfE"]
    cmd += ["-o", "addopts=", path]  # ignore repo addopts that may need plugins we lack
    return cmd + (["-k", k] if k else [])


def outcome(res: ExecResult) -> str:
    # pytest exit codes: 0 ok, 1 tests failed, 2 interrupted/collection error,
    # 3 internal error, 4 usage error, 5 no tests collected
    if res.timed_out:
        return "timeout"
    return {0: "passed", 1: "failed", 5: "no_tests"}.get(res.exit_code, "collection_error")


def make_tools(
    sandbox: Sandbox, limits: Limits, original_files: set[str]
) -> tuple[ToolRegistry, dict[str, str]]:
    """Tools bound to one sandbox, plus a dict of the test files the agent wrote."""
    registry = ToolRegistry()
    written: dict[str, str] = {}

    async def search_code(args: SearchArgs) -> str:
        cmd = ["grep", "-rnIE", "--exclude-dir=.git", "-m", "20"]
        if args.glob:
            cmd.append(f"--include={args.glob}")
        res = await sandbox.exec([*cmd, "-e", args.pattern, "--", _rel(args.path)], 60)
        if res.exit_code == 1:
            return "No matches."
        return _clip(res.output, limits.max_output_chars)

    async def read_file(args: ReadArgs) -> str:
        path = _rel(args.path)
        end = args.end_line or args.start_line + 199
        end = min(end, args.start_line + limits.max_read_lines - 1)
        script = (
            'if [ -d "$1" ]; then ls -la "$1"; '
            'else awk -v s="$2" -v e="$3" \'NR>=s && NR<=e {printf "%6d  %s\\n", NR, $0} '
            'END {if (NR > e) printf "... (%d lines total)\\n", NR}\' "$1"; fi'
        )
        cmd = ["sh", "-c", script, "sh", path, str(args.start_line), str(end)]
        res = await sandbox.exec(cmd, 30)
        return _clip(res.output, limits.max_output_chars) or "(empty)"

    async def write_test(args: WriteTestArgs) -> str:
        path = _rel(args.path)
        if not TEST_NAME.search(path):
            raise ValueError("test files must be named test_*.py or *_test.py")
        if path in original_files:
            raise ValueError(f"{path} is an existing project file; write a new test file")
        if len(args.content.encode()) > limits.max_test_bytes:
            raise ValueError(f"test file larger than {limits.max_test_bytes} bytes")
        await sandbox.write_file(path, args.content)
        written[path] = args.content
        return f"Wrote {path} ({len(args.content.splitlines())} lines)."

    async def run_pytest(args: PytestArgs) -> str:
        res = await sandbox.exec(pytest_cmd(_rel(args.path), args.k), limits.pytest_timeout_s)
        header = f"[pytest exit code {res.exit_code}: {outcome(res)}]\n"
        return header + _clip(res.output, limits.max_output_chars)

    search_code.__doc__ = "Search the repository with an extended regex (like grep -rnE)."
    read_file.__doc__ = "Read a file with line numbers (or list a directory)."
    write_test.__doc__ = (
        "Create or overwrite a NEW pytest file (test_*.py or *_test.py). "
        "Existing project files cannot be modified."
    )
    run_pytest.__doc__ = "Run pytest on a file or directory and return its output."
    for fn in (search_code, read_file, write_test, run_pytest):
        registry.tool(fn)
    return registry, written


# --- running and grading -----------------------------------------------------

SandboxFactory = Callable[[BugTask], Awaitable[Sandbox]]


def _list_files(tar_bytes: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        return {m.name for m in tar.getmembers() if m.isfile()}


async def _run_test_on(
    factory: SandboxFactory, task: BugTask, tar_bytes: bytes, path: str, content: str, limits
) -> TestRun:
    sandbox = await factory(task)
    try:
        await sandbox.put_tree(tar_bytes)
        await sandbox.write_file(path, content)
        res = await sandbox.exec(pytest_cmd(path), limits.pytest_timeout_s)
        return TestRun(
            exit_code=res.exit_code, outcome=outcome(res), output=_clip(res.output, 4000)
        )
    finally:
        await sandbox.close()


def _pick_test(written: dict[str, str], final_answer: str | None) -> str | None:
    """The test file the agent named in its answer, else the last one it wrote."""
    for path in reversed(list(written)):
        if final_answer and path in final_answer:
            return path
    return next(reversed(written), None) if written else None


async def run_task(
    task: BugTask, make_agent: Callable[[ToolRegistry], Agent], factory: SandboxFactory,
    limits: Limits,
) -> ReproResult:  # fmt: skip
    buggy_tar = await tree_tar(task.clone_path, task.buggy_commit)
    original = _list_files(buggy_tar)

    sandbox = await factory(task)
    try:
        await sandbox.put_tree(buggy_tar)
        tools, written = make_tools(sandbox, limits, original)
        agent = make_agent(tools)
        prompt = TASK_PROMPT.format(
            repo=task.repo, number=task.number, title=task.title, body=task.body or "(no body)"
        )
        traj = await agent.run(prompt, task_id=task.id)
    finally:
        await sandbox.close()

    result = ReproResult(task=task, trajectory=traj)
    path = _pick_test(written, traj.final_answer)
    if path is None:
        result.before = TestRun(exit_code=-1, outcome="missing")
        return result
    result.test_path, result.test_content = path, written[path]

    result.before = await _run_test_on(factory, task, buggy_tar, path, result.test_content, limits)
    result.fails_before = result.before.outcome == "failed"
    if result.fails_before and task.fix_commit:
        fix_tar = await tree_tar(task.clone_path, task.fix_commit)
        result.after = await _run_test_on(factory, task, fix_tar, path, result.test_content, limits)
        result.fail_to_pass = result.after.outcome == "passed"
    return result


def task_from_bug(bug: dict[str, Any], clone_path: str | Path) -> dict[str, Any]:
    """Fields for a BugTask from a watch_bugs.py record; commits are resolved later."""
    return {
        "repo": bug["repo"],
        "number": bug["number"],
        "title": bug["title"],
        "body": bug.get("body", ""),
        "clone_path": str(clone_path),
    }
