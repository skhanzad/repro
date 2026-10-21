"""Offline tests: a scripted fake client stands in for the OpenAI API."""

import asyncio
import json
from types import SimpleNamespace

from repro.agent import Agent
from repro.example_tools import registry
from repro.graders import grade
from repro.schemas import Task


def _message(content=None, tool_calls=None):
    calls = [
        SimpleNamespace(
            id=f"call_{i}",
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        for i, (name, args) in enumerate(tool_calls or [])
    ]
    msg = SimpleNamespace(content=content, tool_calls=calls or None)
    msg.model_dump = lambda exclude_none=True: {"role": "assistant", "content": content}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class FakeClient:
    def __init__(self, script):
        self._script = iter(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        return next(self._script)


def test_agent_uses_tool_then_answers():
    client = FakeClient(
        [
            _message(tool_calls=[("calculator", {"expression": "17 * 23 + 4"})]),
            _message(content="395"),
        ]
    )
    agent = Agent(registry, model="fake", client=client)
    traj = asyncio.run(agent.run("compute", task_id="t1"))

    assert traj.stop_reason == "final_answer"
    assert traj.final_answer == "395"
    assert traj.steps[0].tool_calls[0].output == "395"
    assert traj.total_tokens == 30

    g = asyncio.run(grade(Task(id="t1", prompt="compute", expected="395"), traj))
    assert g.passed


def test_agent_stops_at_max_steps():
    loop_forever = [_message(tool_calls=[("lookup", {"key": "x"})]) for _ in range(5)]
    agent = Agent(registry, model="fake", client=FakeClient(loop_forever), max_steps=3)
    traj = asyncio.run(agent.run("loop"))
    assert traj.stop_reason == "max_steps"
    assert len(traj.steps) == 3


def test_tool_errors_are_returned_not_raised():
    output, is_error = asyncio.run(registry.execute("calculator", '{"expression": "1/0"}'))
    assert is_error and "ZeroDivisionError" in output
    output, is_error = asyncio.run(registry.execute("nope", "{}"))
    assert is_error and "unknown tool" in output
