"""Core data types shared across the agent, graders and runner."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single evaluation example."""

    id: str
    prompt: str
    expected: str | None = None
    grader: str = "exact_match"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    output: str
    error: bool = False


class Step(BaseModel):
    """One model turn: what the model said and which tools it called."""

    index: int
    content: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class Trajectory(BaseModel):
    task_id: str
    model: str
    steps: list[Step] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str | None = None
    stop_reason: Literal["final_answer", "max_steps", "budget", "error"] = "final_answer"
    cost_usd: float | None = None  # None when the model's price is unknown
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    @property
    def num_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)


class Grade(BaseModel):
    score: float
    passed: bool
    reason: str = ""


class EvalResult(BaseModel):
    task: Task
    trajectory: Trajectory
    grade: Grade
