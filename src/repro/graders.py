"""Graders turn (task, trajectory) into a Grade. Register new ones with @grader."""

import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from repro.config import settings
from repro.llm import get_client
from repro.schemas import Grade, Task, Trajectory

GraderFn = Callable[[Task, Trajectory], Awaitable[Grade]]
GRADERS: dict[str, GraderFn] = {}


def grader(name: str) -> Callable[[GraderFn], GraderFn]:
    def register(fn: GraderFn) -> GraderFn:
        GRADERS[name] = fn
        return fn

    return register


async def grade(task: Task, traj: Trajectory) -> Grade:
    if traj.final_answer is None:
        return Grade(score=0.0, passed=False, reason=f"no final answer ({traj.stop_reason})")
    fn = GRADERS.get(task.grader)
    if fn is None:
        raise KeyError(f"Unknown grader '{task.grader}'. Available: {sorted(GRADERS)}")
    return await fn(task, traj)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


@grader("exact_match")
async def exact_match(task: Task, traj: Trajectory) -> Grade:
    ok = _normalize(traj.final_answer or "") == _normalize(task.expected or "")
    return Grade(score=float(ok), passed=ok)


@grader("contains")
async def contains(task: Task, traj: Trajectory) -> Grade:
    ok = _normalize(task.expected or "") in _normalize(traj.final_answer or "")
    return Grade(score=float(ok), passed=ok)


class _JudgeVerdict(BaseModel):
    reasoning: str
    correct: bool


JUDGE_PROMPT = """You are grading an AI agent's answer.

Question:
{prompt}

Reference answer:
{expected}

Agent's answer:
{answer}

Decide whether the agent's answer is correct with respect to the reference answer.
Minor formatting differences are fine; factual or numerical differences are not."""


@grader("llm_judge")
async def llm_judge(task: Task, traj: Trajectory) -> Grade:
    response = await get_client().chat.completions.parse(
        model=settings.judge_model,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    prompt=task.prompt, expected=task.expected, answer=traj.final_answer
                ),
            }
        ],
        response_format=_JudgeVerdict,
    )
    verdict = response.choices[0].message.parsed
    if verdict is None:
        return Grade(score=0.0, passed=False, reason="judge returned no parseable verdict")
    return Grade(score=float(verdict.correct), passed=verdict.correct, reason=verdict.reasoning)
