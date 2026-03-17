"""
tools/math_tool.py  —  Single calculate() tool
"""
import ast
import operator
from langchain_core.tools import tool

OPERATORS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op = OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = OPERATORS.get(type(node.op))
        if not op:
            raise ValueError("Unary operator not allowed.")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

@tool
def calculate(expression: str) -> str:
    """
    Evaluate any mathematical expression and return the result.
    Use for ALL math: simple or complex, single or multi-operation.
    Supports: + - * / ** % and brackets ().
    Examples: "1+1", "128+374", "100/4+50-3*2", "(5+3)*(10-4)", "2**10"
    Always pass a clean numeric expression — no words or variables.
    """
    try:
        expression = expression.strip()
        tree   = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result}"
    except ZeroDivisionError as e:
        return f"Error: {e}"
    except (ValueError, SyntaxError) as e:
        return f"Error: Invalid expression — {e}"
    except Exception as e:
        return f"Error: {e}"
