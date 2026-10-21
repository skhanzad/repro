"""The agent loop: call the model, execute tool calls, repeat until it answers."""

import asyncio
import json
import time
from typing import Any

from openai import AsyncOpenAI

from repro.config import settings
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
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int | None = None,
        client: AsyncOpenAI | None = None,
        completion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tools = tools
        self.model = model or settings.agent_model
        self.system_prompt = system_prompt
        self.max_steps = max_steps or settings.max_steps
        self.client = client or get_client()
        self.completion_kwargs = completion_kwargs or {}

    async def run(self, prompt: str, task_id: str = "adhoc") -> Trajectory:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        traj = Trajectory(task_id=task_id, model=self.model, messages=messages)

        try:
            for i in range(self.max_steps):
                step = await self._step(i, messages)
                traj.steps.append(step)
                if not step.tool_calls:
                    traj.final_answer = step.content
                    traj.stop_reason = "final_answer"
                    return traj
            traj.stop_reason = "max_steps"
        except Exception as e:  # noqa: BLE001 - a failed run is a data point, not a crash
            traj.stop_reason = "error"
            traj.error = f"{type(e).__name__}: {e}"
        return traj

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
