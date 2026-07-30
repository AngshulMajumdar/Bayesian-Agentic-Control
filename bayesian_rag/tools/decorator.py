"""The `@tool` decorator: turn a plain function into an agent-ready tool.

    @tool
    def search(query: str) -> str:
        \"\"\"Search the web.\"\"\"
        return "..."

The decorator reads the function's name, docstring, and signature, so nothing
has to be declared twice. Two optional hints matter for inference:

    reliability -- prior belief this tool returns correct results
    appeal      -- how attractive it is before evidence (cheap/fast tools rate
                   higher). Kept separate from reliability because the whole
                   failure mode this library addresses is a tool that is
                   attractive and wrong.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

JsonDict = Dict[str, Any]

# Parameter names the default proposer knows how to fill automatically.
QUERY_PARAMS = ("query", "q", "question", "input", "prompt")
CLAIM_PARAMS = ("text", "claim", "answer", "statement", "content")


class ToolKind:
    """What role a tool plays, which determines when it can be proposed."""

    ACTION = "action"  # produces an answer
    CHECKER = "checker"  # validates a previous answer
    TERMINAL = "terminal"  # produces a final answer and ends the episode


@dataclass
class Tool:
    """A callable wrapped with the metadata inference needs.

    Satisfies the runner's tool protocol via `invoke`, and normalises whatever
    the underlying function returns into the dict shape the scorers expect.
    """

    func: Callable[..., Any]
    name: str
    description: str = ""
    kind: str = ToolKind.ACTION
    reliability: float = 0.5
    appeal: float = 0.5
    params: List[str] = field(default_factory=list)
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(
                f"Tool {self.name!r}: reliability must lie in [0, 1], got {self.reliability}"
            )
        if not 0.0 <= self.appeal <= 1.0:
            raise ValueError(
                f"Tool {self.name!r}: appeal must lie in [0, 1], got {self.appeal}"
            )
        if self.kind not in (ToolKind.ACTION, ToolKind.CHECKER, ToolKind.TERMINAL):
            raise ValueError(f"Tool {self.name!r}: unknown kind {self.kind!r}")

    # -- invocation ----------------------------------------------------------

    def invoke(self, inp: JsonDict) -> JsonDict:
        """Call the wrapped function and normalise its result."""
        kwargs = {k: v for k, v in inp.items() if k in self.params} if self.params else {}
        result = self.func(**kwargs) if kwargs or not self.params else self.func(**inp)
        return self._normalize(result)

    def _normalize(self, result: Any) -> JsonDict:
        """Accept the natural return type for each kind of tool.

        Checkers may return a bare bool; action tools may return a bare string.
        Returning a dict gives full control over `confidence` and `verified`.
        """
        if isinstance(result, dict):
            out = dict(result)
        elif isinstance(result, bool):
            out = {"ok": result}
        elif result is None:
            out = {}
        else:
            out = {"answer": str(result)}

        if self.kind == ToolKind.CHECKER:
            # A checker's verdict must never be mistaken for a user-facing
            # answer, or the agent will happily return the word "consistent".
            if "ok" not in out:
                out["ok"] = bool(out.pop("answer", True))
            out.pop("answer", None)
        else:
            out.setdefault("confidence", self.reliability)

        return out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calling the decorated object still calls the original function."""
        return self.func(*args, **kwargs)

    # -- introspection -------------------------------------------------------

    @property
    def is_checker(self) -> bool:
        return self.kind == ToolKind.CHECKER

    @property
    def is_terminal(self) -> bool:
        return self.kind == ToolKind.TERMINAL

    def wants_query(self) -> bool:
        """True when this tool takes the user's query."""
        return any(p in QUERY_PARAMS for p in self.params)

    def wants_claim(self) -> bool:
        """True when this tool takes a claim to inspect."""
        return any(p in CLAIM_PARAMS for p in self.params)

    def query_param(self) -> Optional[str]:
        """Name of the parameter that receives the user query, if any."""
        return next((p for p in self.params if p in QUERY_PARAMS), None)

    def claim_param(self) -> Optional[str]:
        """Name of the parameter that receives a claim to inspect, if any."""
        return next((p for p in self.params if p in CLAIM_PARAMS), None)

    def __repr__(self) -> str:
        return (
            f"Tool({self.name!r}, kind={self.kind!r}, "
            f"reliability={self.reliability}, appeal={self.appeal}, "
            f"deterministic={self.deterministic})"
        )


def tool(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    kind: str = ToolKind.ACTION,
    reliability: float = 0.5,
    appeal: Optional[float] = None,
    deterministic: bool = True,
) -> Any:
    """Register a function as a tool. Usable bare or with arguments.

        @tool
        def search(query: str) -> str: ...

        @tool(reliability=0.95, appeal=0.4)
        def verified_search(query: str) -> str: ...

    Args:
        name: defaults to the function name.
        description: defaults to the first line of the docstring.
        kind: "action", "checker", or "terminal".
        reliability: prior probability this tool is correct, in [0, 1].
        appeal: prior attractiveness before evidence, in [0, 1]. Defaults to
            `reliability`. Set it higher than reliability for tools that are
            cheap and tempting but unreliable -- that gap is exactly what
            distinguishes the inference regimes.
        deterministic: whether identical arguments always yield identical
            results. Deterministic tools are memoised within an episode, so N
            particles proposing the same call cost one invocation.

            Set this False for anything whose repeated calls may differ -- a
            sampled model, a live API, a flaky or adversarial source. Leaving it
            True for such a tool silently caps accuracy: every particle receives
            the same cached draw, so the cloud collapses to a single sample and
            retrying cannot produce a different outcome. That removes the main
            mechanism by which a filter beats an unreliable tool.
    """

    def decorate(func: Callable) -> Tool:
        sig = inspect.signature(func)
        params = [
            p.name
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        doc = (func.__doc__ or "").strip().split("\n")[0]

        wrapped = Tool(
            func=func,
            name=name or func.__name__,
            description=description or doc,
            kind=kind,
            reliability=reliability,
            appeal=appeal if appeal is not None else reliability,
            params=params,
            deterministic=deterministic,
        )
        functools.update_wrapper(wrapped, func, updated=[])
        return wrapped

    return decorate if _func is None else decorate(_func)


def checker(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    reliability: float = 0.8,
    deterministic: bool = True,
) -> Any:
    """Register a validation tool.

        @checker
        def fact_check(text: str) -> bool: ...

    A checker's verdict feeds back onto whichever tool produced the text it
    examined, which is how evidence gathered late revises an earlier decision.
    """
    return tool(
        _func,
        name=name,
        kind=ToolKind.CHECKER,
        reliability=reliability,
        appeal=reliability,
        deterministic=deterministic,
    )


def as_tool(obj: Any) -> Tool:
    """Coerce a function, Tool, or invoke-style object into a Tool."""
    if isinstance(obj, Tool):
        return obj
    if callable(obj):
        return tool(obj)
    if hasattr(obj, "invoke") and hasattr(obj, "name"):
        # An existing LangChain-style tool object.
        return Tool(
            func=obj.invoke,
            name=obj.name,
            description=getattr(obj, "description", ""),
            params=["inp"],
        )
    raise TypeError(
        f"Cannot interpret {type(obj).__name__} as a tool. Provide a function "
        f"decorated with @tool, or an object with .name and .invoke."
    )
