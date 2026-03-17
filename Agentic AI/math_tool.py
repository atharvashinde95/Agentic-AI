"""
tools/math_tool.py
-------------------
ONE tool that handles any math expression.

Examples it can evaluate:
  "1 + 1"
  "a + b + c * d"   → the LLM fills in the numbers before calling
  "128 + 374"
  "3.14 * 2"
  "100 / 4 + 50 - 3 * 2"
  "(5 + 3) * (10 - 4)"

Safety: uses Python's ast module instead of eval()
so only math is allowed — no code execution.
"""

import ast
import math
import operator
from langchain_core.tools import tool


# Safe allowed operators
OPERATORS = {
    ast.Add:    operator.add,
    ast.Sub:    operator.sub,
    ast.Mult:   operator.mul,
    ast.Div:    operator.truediv,
    ast.Pow:    operator.pow,
    ast.Mod:    operator.mod,
    ast.USub:   operator.neg,   # unary minus e.g. -5
    ast.UAdd:   operator.pos,
}


def _safe_eval(node):
    """
    Recursively evaluate an AST node using only allowed operators.
    Raises ValueError for anything that isn't pure arithmetic.
    """
    if isinstance(node, ast.Constant):          # numbers: 3, 3.14
        return node.value

    if isinstance(node, ast.BinOp):             # a OP b
        op = OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        left  = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):           # -5, +3
        op = OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unary operator not allowed.")
        return op(_safe_eval(node.operand))

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate any mathematical expression and return the result.

    Use this tool for ALL math questions — simple or complex.
    The expression can include:
      + (addition), - (subtraction), * (multiplication),
      / (division), ** (power), % (modulo), () (brackets)

    Examples:
      "1 + 1"
      "128 + 374"
      "3.14 * 2"
      "100 / 4 + 50 - 3 * 2"
      "(5 + 3) * (10 - 4)"
      "2 ** 10"
      "100 % 7"

    Always pass a clean numeric expression — no variables, no words.
    """
    try:
        # Clean the input
        expression = expression.strip()

        # Parse into AST (never runs code directly)
        tree = ast.parse(expression, mode="eval")

        # Evaluate safely
        result = _safe_eval(tree.body)

        # Format: show int if whole number, else float
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"{expression} = {result}"

    except ZeroDivisionError as e:
        return f"Error: {e}"
    except (ValueError, SyntaxError) as e:
        return f"Error: Invalid expression — {e}"
    except Exception as e:
        return f"Error: {e}"
