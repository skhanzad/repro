"""The agent loop: call the model, execute tool calls, repeat until it answers."""

import asyncio
import json
import time
from typing import Any

from openai import AsyncOpenAI

from repro.config import DEFAULT_MODELS, Provider, model_cost, settings
from repro.llm import get_client
from repro.schemas import Step, ToolCallRecord, Trajectory
from repro.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful agent. Use the available tools when they help. "
    "When you are done, reply with the final answer only, without calling any tools."
)


class Agent:
    def __init__(
        self,
        tools: ToolRegistry,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int | None = None,
        client: AsyncOpenAI | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        max_cost_usd: float | None = None,
        max_total_tokens: int | None = None,
        sequential_tools: bool = False,
    ) -> None:
        """max_cost_usd / max_total_tokens are checked after every step, so a run can
        overshoot by at most one model call. sequential_tools runs a step's tool calls
        in order instead of concurrently (needed when tools have side effects)."""
        self.tools = tools
        self.provider = provider or settings.agent_provider
        self.model = model or settings.agent_model or DEFAULT_MODELS[self.provider]
        self.system_prompt = system_prompt
        self.max_steps = max_steps or settings.max_steps
        self.client = client or get_client(self.provider)
        self.completion_kwargs = completion_kwargs or {}
        self.max_cost_usd = max_cost_usd
        self.max_total_tokens = max_total_tokens
        self.sequential_tools = sequential_tools

    async def run(self, prompt: str, task_id: str = "adhoc") -> Trajectory:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        traj = Trajectory(task_id=task_id, model=self.model, messages=messages)
        cost: float | None = 0.0

        try:
            for i in range(self.max_steps):
                step = await self._step(i, messages)
                traj.steps.append(step)
                step_cost = model_cost(
                    self.provider, self.model, step.prompt_tokens, step.completion_tokens
                )
                cost = None if cost is None or step_cost is None else cost + step_cost
                traj.cost_usd = cost
                if not step.tool_calls:
                    traj.final_answer = step.content
                    traj.stop_reason = "final_answer"
                    return traj
                if self._over_budget(traj.total_tokens, cost):
                    traj.stop_reason = "budget"
                    return traj
            traj.stop_reason = "max_steps"
        except Exception as e:  # noqa: BLE001 - a failed run is a data point, not a crash
            traj.stop_reason = "error"
            traj.error = f"{type(e).__name__}: {e}"
        return traj

    def _over_budget(self, tokens: int, cost: float | None) -> bool:
        if self.max_total_tokens is not None and tokens >= self.max_total_tokens:
            return True
        return self.max_cost_usd is not None and cost is not None and cost >= self.max_cost_usd

    async def _step(self, index: int, messages: list[dict[str, Any]]) -> Step:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.completion_kwargs,
        }
        if len(self.tools):
            kwargs["tools"] = self.tools.specs()

        t0 = time.perf_counter()
        response = await self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        step = Step(index=index, content=msg.content, latency_s=latency)
        if response.usage:
            step.prompt_tokens = response.usage.prompt_tokens
            step.completion_tokens = response.usage.completion_tokens

        calls = [c for c in (msg.tool_calls or []) if c.type == "function"]
        if self.sequential_tools:
            outputs = [
                await self.tools.execute(c.function.name, c.function.arguments) for c in calls
            ]
        else:
            outputs = await asyncio.gather(
                *(self.tools.execute(c.function.name, c.function.arguments) for c in calls)
            )
        for call, (output, is_error) in zip(calls, outputs, strict=True):
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            step.tool_calls.append(
                ToolCallRecord(
                    id=call.id,
                    name=call.function.name,
                    arguments=_safe_json(call.function.arguments),
                    output=output,
                    error=is_error,
                )
            )
        return step


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except json.JSONDecodeError:
        return {"_raw": raw}
