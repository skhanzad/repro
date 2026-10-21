"""Example tools. Replace or extend these with the tools your eval needs."""

import ast
import operator

from pydantic import BaseModel, Field

from repro.tools import ToolRegistry

registry = ToolRegistry()


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(3 + 4) * 2'")


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@registry.tool
def calculator(args: CalculatorArgs) -> str:
    """Evaluate an arithmetic expression and return the result."""
    result = _eval(ast.parse(args.expression, mode="eval").body)
    return str(int(result) if float(result).is_integer() else result)


class LookupArgs(BaseModel):
    key: str = Field(description="Name of the fact to look up")


_FACTS = {
    "capital_of_france": "Paris",
    "speed_of_light_m_s": "299792458",
    "boiling_point_water_c": "100",
}


@registry.tool
def lookup(args: LookupArgs) -> str:
    """Look up a fact from a small key-value knowledge base.

    Keys: capital_of_france, speed_of_light_m_s, boiling_point_water_c.
    """
    return _FACTS.get(args.key, f"No fact found for '{args.key}'")
