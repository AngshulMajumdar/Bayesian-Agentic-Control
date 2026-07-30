"""Deterministic baselines mirroring LangGraph and LangChain execution semantics.

WHAT THIS IS
------------
These are *semantic stand-ins*, not the real packages. They reimplement the
execution model each framework uses -- LangGraph's compiled state graph with
conditional edges, and LangChain's AgentExecutor iteration loop with parsing
retries -- against the same tool objects the BayesianRAG agent uses, so the
comparison isolates the orchestration strategy rather than differences in
tool wiring, prompt formatting, or model quality.

WHY STAND-INS
-------------
A like-for-like comparison needs both systems driving identical tools with an
identical notion of success. Running the real packages additionally requires a
language model to generate the actions, which introduces sampling noise, cost,
and version drift into what is meant to be a measurement of control flow. With
a deterministic scripted policy in place of the model, the remaining difference
is exactly the thing under study.

`adapters.py` runs the *real* libraries when they are installed, so these
numbers can be checked against them.

WHAT IS FAITHFULLY REPRODUCED
-----------------------------
LangGraph:
  - nodes as state -> state functions
  - unconditional edges and conditional edges routing on a key
  - a compiled graph executed from an entry point to END
  - recursion limit
  - a single state object threaded through, no branching

LangChain AgentExecutor:
  - the observe/decide/act loop up to max_iterations
  - a tool-calling policy that reads the scratchpad
  - `handle_parsing_errors` retry behaviour
  - early return on a final answer

WHAT IS NOT REPRODUCED
----------------------
Prompt construction, model calls, token accounting, streaming, callbacks,
checkpointing, human-in-the-loop interrupts. None of these bear on which
trajectory the orchestrator ends up taking.

FAIRNESS
--------
Two LangGraph baselines are provided. `naive` is a plain retrieve-then-answer
graph. `engineered` is what a competent engineer writes once they know this
failure mode exists: a conditional edge that routes to the authoritative source
when validation fails. The engineered graph is the honest comparison, and on
the scenario it was written for it should do well. Reporting only the naive
graph would be a strawman.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

JsonDict = Dict[str, Any]

END = "__end__"


# =============================================================================
# LangGraph semantics
# =============================================================================


class GraphRecursionError(RuntimeError):
    """Raised when a graph exceeds its recursion limit, as LangGraph does."""


@dataclass
class StateGraph:
    """A compiled control-flow graph over a single mutable state object.

    Mirrors `langgraph.graph.StateGraph`. The defining property for this
    comparison is that exactly one state is threaded through exactly one path:
    when a conditional edge picks a branch, the alternatives are gone. Nothing
    downstream can reweight a choice already made.
    """

    nodes: Dict[str, Callable[[JsonDict], JsonDict]] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)
    conditional: Dict[str, Tuple[Callable[[JsonDict], str], Dict[str, str]]] = field(
        default_factory=dict
    )
    entry: Optional[str] = None

    def add_node(self, name: str, fn: Callable[[JsonDict], JsonDict]) -> "StateGraph":
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges[src] = dst
        return self

    def add_conditional_edges(
        self, src: str, router: Callable[[JsonDict], str], mapping: Dict[str, str]
    ) -> "StateGraph":
        self.conditional[src] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def compile(self) -> "CompiledGraph":
        if self.entry is None:
            raise ValueError("No entry point set.")
        missing = [n for n in self.edges.values() if n not in self.nodes and n != END]
        if missing:
            raise ValueError(f"Edges point at undefined nodes: {missing}")
        return CompiledGraph(self)


@dataclass
class CompiledGraph:
    """An executable graph."""

    graph: StateGraph

    def invoke(self, state: JsonDict, recursion_limit: int = 25) -> JsonDict:
        """Run from the entry point to END, threading one state along one path."""
        current = self.graph.entry
        state = dict(state)
        state.setdefault("visited", [])

        for _ in range(recursion_limit):
            if current == END or current is None:
                return state

            fn = self.graph.nodes.get(current)
            if fn is None:
                raise ValueError(f"Undefined node: {current}")

            state = fn(state)
            state["visited"].append(current)

            if current in self.graph.conditional:
                router, mapping = self.graph.conditional[current]
                key = router(state)
                current = mapping.get(key, END)
            else:
                current = self.graph.edges.get(current, END)

        raise GraphRecursionError(f"Exceeded recursion limit of {recursion_limit}")


# =============================================================================
# LangChain AgentExecutor semantics
# =============================================================================


@dataclass
class AgentAction:
    """A tool invocation chosen by the policy."""

    tool: str
    tool_input: JsonDict


@dataclass
class AgentFinish:
    """A terminal answer."""

    output: str


@dataclass
class AgentExecutor:
    """The observe/decide/act loop from `langchain.agents.AgentExecutor`.

    The policy sees the full scratchpad and returns either an action or a
    finish. Retries on malformed output reproduce `handle_parsing_errors`, and
    reproduce the property that matters here: a retry redraws from the same
    policy given the same scratchpad, so it is a repeat of the prior decision
    rather than an update in light of evidence.
    """

    tools: Dict[str, Any]
    policy: Callable[[str, List[Tuple[AgentAction, JsonDict]]], Any]
    max_iterations: int = 5
    handle_parsing_errors: bool = True
    max_retries: int = 2

    def invoke(self, inputs: JsonDict) -> JsonDict:
        query = inputs.get("input", "")
        scratchpad: List[Tuple[AgentAction, JsonDict]] = []

        for _ in range(self.max_iterations):
            step = self._decide(query, scratchpad)

            if isinstance(step, AgentFinish):
                return {
                    "input": query,
                    "output": step.output,
                    "intermediate_steps": scratchpad,
                }

            tool = self.tools.get(step.tool)
            if tool is None:
                observation: JsonDict = {"error": f"tool_not_found: {step.tool}"}
            else:
                try:
                    result = tool.invoke(step.tool_input)
                    observation = result if isinstance(result, dict) else {"answer": result}
                except Exception as exc:  # noqa: BLE001
                    observation = {"error": f"{type(exc).__name__}: {exc}"}

            scratchpad.append((step, observation))

        # Iteration limit reached: return the last answer seen, as the real
        # executor does under `early_stopping_method="force"`.
        answer = _last_answer(scratchpad) or "Agent stopped due to iteration limit."
        return {"input": query, "output": answer, "intermediate_steps": scratchpad}

    def _decide(self, query: str, scratchpad):
        """Call the policy, retrying malformed output as the real executor does."""
        attempts = self.max_retries + 1 if self.handle_parsing_errors else 1
        for _ in range(attempts):
            step = self.policy(query, scratchpad)
            if isinstance(step, (AgentAction, AgentFinish)):
                return step
        return AgentFinish(output=_last_answer(scratchpad) or "Could not parse output.")


def _last_answer(scratchpad: List[Tuple[AgentAction, JsonDict]]) -> Optional[str]:
    """Most recent observation carrying an answer."""
    for _, obs in reversed(scratchpad):
        if isinstance(obs, dict) and obs.get("answer"):
            return str(obs["answer"])
    return None


# =============================================================================
# Baseline agents over a shared tool set
# =============================================================================


def build_langgraph_naive(tools: Dict[str, Any], preferred: str) -> CompiledGraph:
    """Retrieve, then answer. No validation.

    The straightforward graph someone writes before discovering that the
    convenient source can be wrong.
    """

    def retrieve(state: JsonDict) -> JsonDict:
        out = tools[preferred].invoke({"query": state["query"]})
        state["answer"] = out.get("answer", "")
        return state

    g = StateGraph()
    g.add_node("retrieve", retrieve)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", END)
    return g.compile()


def build_langgraph_engineered(
    tools: Dict[str, Any], preferred: str, fallback: str, checker: str
) -> CompiledGraph:
    """Retrieve, validate, and fall back to the authoritative source on failure.

    This is the fair comparison: it encodes exactly the recovery that this
    failure mode calls for, via a conditional edge. On the scenario it was
    written for it should perform well, and reporting only the naive graph
    would be a strawman.

    The cost is that the recovery is specified in advance. The branch exists
    because an engineer anticipated this failure; a different failure needs a
    different branch.
    """

    def retrieve(state: JsonDict) -> JsonDict:
        out = tools[preferred].invoke({"query": state["query"]})
        state["answer"] = out.get("answer", "")
        return state

    def validate(state: JsonDict) -> JsonDict:
        out = tools[checker].invoke({"text": state.get("answer", "")})
        state["valid"] = bool(out.get("ok", False))
        return state

    def escalate(state: JsonDict) -> JsonDict:
        out = tools[fallback].invoke({"query": state["query"]})
        state["answer"] = out.get("answer", "")
        return state

    g = StateGraph()
    g.add_node("retrieve", retrieve)
    g.add_node("validate", validate)
    g.add_node("escalate", escalate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "validate")
    g.add_conditional_edges(
        "validate",
        lambda s: "ok" if s.get("valid") else "retry",
        {"ok": END, "retry": "escalate"},
    )
    g.add_edge("escalate", END)
    return g.compile()


def build_langchain_agent(
    tools: Dict[str, Any], preferred: str, max_iterations: int = 5
) -> AgentExecutor:
    """A tool-calling agent that prefers the convenient source and retries on error.

    The policy is deterministic, standing in for a model that reliably picks the
    tool its description makes most attractive. It retries on tool errors, which
    is the standard remedy -- and reproduces the property under study: the retry
    consults the same scratchpad and reaches the same conclusion.
    """

    def policy(query: str, scratchpad):
        if not scratchpad:
            return AgentAction(tool=preferred, tool_input={"query": query})

        last_action, last_obs = scratchpad[-1]

        if last_obs.get("error") and len(scratchpad) <= 2:
            # Retry the same call -- a redraw, not a revision.
            return AgentAction(tool=last_action.tool, tool_input=last_action.tool_input)

        answer = _last_answer(scratchpad)
        return AgentFinish(output=answer or "No answer found.")

    return AgentExecutor(tools=tools, policy=policy, max_iterations=max_iterations)


def build_langchain_reflexive(
    tools: Dict[str, Any], preferred: str, fallback: str, checker: str
) -> AgentExecutor:
    """A reflection-style agent: answer, self-check, revise once.

    The LangChain analogue of the engineered graph, and the strongest baseline
    available without maintaining a belief over alternatives. Like the graph, the
    reflection step is written in advance for a failure the author anticipated.
    """

    def policy(query: str, scratchpad):
        used = [a.tool for a, _ in scratchpad]

        if not scratchpad:
            return AgentAction(tool=preferred, tool_input={"query": query})

        if checker not in used:
            claim = _last_answer(scratchpad) or ""
            return AgentAction(tool=checker, tool_input={"text": claim})

        verdict = next(
            (obs for a, obs in reversed(scratchpad) if a.tool == checker), {}
        )
        if not verdict.get("ok", False) and fallback not in used:
            return AgentAction(tool=fallback, tool_input={"query": query})

        return AgentFinish(output=_last_answer(scratchpad) or "No answer found.")

    return AgentExecutor(tools=tools, policy=policy, max_iterations=5)
