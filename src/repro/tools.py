"""Minimal tool registry.

A tool is a plain (sync or async) function whose single argument is a pydantic
model. The model's JSON schema becomes the OpenAI function `parameters`.

    class AddArgs(BaseModel):
        a: float
        b: float

    @registry.tool
    def add(args: AddArgs) -> str:
        "Add two numbers."
        return str(args.a + args.b)
"""

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError

ToolFn = Callable[[Any], str | Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    async def __call__(self, raw_arguments: str) -> str:
        args = self.args_model.model_validate(json.loads(raw_arguments or "{}"))
        result = self.fn(args)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else json.dumps(result)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, fn: ToolFn | None = None, *, name: str | None = None):
        def register(f: ToolFn) -> ToolFn:
            hints = get_type_hints(f)
            params = [p for p in inspect.signature(f).parameters if p != "return"]
            if len(params) != 1 or not issubclass(hints.get(params[0], object), BaseModel):
                raise TypeError(f"Tool {f.__name__} must take exactly one pydantic model argument")
            self.add(
                Tool(
                    name=name or f.__name__,
                    description=inspect.getdoc(f) or "",
                    args_model=hints[params[0]],
                    fn=f,
                )
            )
            return f

        return register(fn) if fn else register

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    async def execute(self, name: str, raw_arguments: str) -> tuple[str, bool]:
        """Run a tool. Returns (output, is_error). Errors are returned to the model, not raised."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'", True
        try:
            return await tool(raw_arguments), False
        except (json.JSONDecodeError, ValidationError) as e:
            return f"Error: invalid arguments for '{name}': {e}", True
        except Exception as e:  # noqa: BLE001 - surface any tool failure to the agent
            return f"Error: {type(e).__name__}: {e}", True
