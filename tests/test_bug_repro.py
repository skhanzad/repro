"""Offline tests for the bug-repro task: tools, guardrails, limits and grading.

Uses LocalSandbox (no Docker needed) and a scripted fake model client.
"""

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from repro.agent import Agent
from repro.bug_repro import BugTask, Limits, make_tools, run_task, tree_tar
from repro.sandbox import LocalSandbox

BUGGY = "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n"
FIXED = (
    "def median(xs):\n    s = sorted(xs)\n    m = len(s) // 2\n"
    "    return (s[m - 1] + s[m]) / 2 if len(s) % 2 == 0 else s[m]\n"
)
TEST = "from stats import median\n\n\ndef test_even():\n    assert median([1, 2, 3, 4]) == 2.5\n"


@pytest.fixture
def toy_repo(tmp_path):
    repo = tmp_path / "toy"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "stats.py").write_text(BUGGY)
    git("add", "-A")
    git("commit", "-qm", "initial")
    (repo / "stats.py").write_text(FIXED)
    git("commit", "-qam", "fix")
    return repo


def _rev(repo, rev):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", rev], text=True).strip()


def _tools(repo):
    async def setup():
        sandbox = LocalSandbox()
        await sandbox.put_tree(await tree_tar(str(repo), "HEAD~1"))
        return sandbox

    sandbox = asyncio.run(setup())
    tools, written = make_tools(sandbox, Limits(), {"stats.py"})
    return sandbox, tools, written


def _call(tools, name, **args):
    return asyncio.run(tools.execute(name, json.dumps(args)))


def test_tools_and_guardrails(toy_repo):
    sandbox, tools, written = _tools(toy_repo)
    try:
        out, err = _call(tools, "search_code", pattern="def median")
        assert not err and "stats.py:1" in out
        out, err = _call(tools, "read_file", path="stats.py")
        assert "return s[len(s) // 2]" in out

        # Only new test files, inside the repo.
        assert _call(tools, "write_test", path="stats.py", content="x")[1]
        assert _call(tools, "write_test", path="helper.py", content="x")[1]
        assert _call(tools, "write_test", path="../test_escape.py", content="x")[1]
        assert _call(tools, "read_file", path="../../etc/passwd")[1]

        out, err = _call(tools, "write_test", path="tests/test_median.py", content=TEST)
        assert not err and written == {"tests/test_median.py": TEST}
        out, _ = _call(tools, "run_pytest", path="tests/test_median.py")
        assert "exit code 1: failed" in out and "assert 3 == 2.5" in out
    finally:
        asyncio.run(sandbox.close())


def test_local_sandbox_hides_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    sandbox = LocalSandbox()
    res = asyncio.run(sandbox.exec(["env"]))
    asyncio.run(sandbox.close())
    assert "sk-secret" not in res.output


def _response(tool_calls=None, content=None, tokens=(1000, 100)):
    calls = [
        SimpleNamespace(
            id=f"c{i}", type="function", function=SimpleNamespace(name=n, arguments=json.dumps(a))
        )
        for i, (n, a) in enumerate(tool_calls or [])
    ]
    msg = SimpleNamespace(content=content, tool_calls=calls or None)
    msg.model_dump = lambda exclude_none=True: {"role": "assistant", "content": content}
    usage = SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeClient:
    def __init__(self, script):
        self.script = iter(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        return next(self.script)


def test_run_task_grades_fail_to_pass(toy_repo):
    script = [
        _response([("write_test", {"path": "tests/test_median.py", "content": TEST})]),
        _response([("run_pytest", {"path": "tests/test_median.py"})]),
        _response(content="tests/test_median.py fails: median([1,2,3,4]) returns 3."),
    ]
    task = BugTask(
        repo="toy/toy", number=1, title="median wrong", clone_path=str(toy_repo),
        buggy_commit=_rev(toy_repo, "HEAD~1"), fix_commit=_rev(toy_repo, "HEAD"),
    )  # fmt: skip

    async def factory(_task):
        return await LocalSandbox().start()

    def make_agent(tools):
        return Agent(
            tools, provider="openai", model="gpt-5-mini", client=FakeClient(script),
            sequential_tools=True,
        )  # fmt: skip

    result = asyncio.run(run_task(task, make_agent, factory, Limits()))
    assert result.test_path == "tests/test_median.py"
    assert (result.before.outcome, result.after.outcome) == ("failed", "passed")
    assert result.fail_to_pass
    # 3 steps x (1000 in, 100 out) at gpt-5-mini prices
    assert result.trajectory.cost_usd == pytest.approx(3 * (1000 * 0.25 + 100 * 2.0) / 1e6)


def test_agent_stops_at_cost_budget(toy_repo):
    sandbox, tools, _ = _tools(toy_repo)
    looping = [_response([("search_code", {"pattern": "x"})], tokens=(100_000, 0))] * 10
    agent = Agent(
        tools, provider="openai", model="gpt-5-mini", client=FakeClient(looping),
        max_cost_usd=0.06, max_steps=10,
    )  # fmt: skip
    traj = asyncio.run(agent.run("go"))
    asyncio.run(sandbox.close())
    # each step costs $0.025, so the third step crosses $0.06
    assert traj.stop_reason == "budget" and len(traj.steps) == 3
