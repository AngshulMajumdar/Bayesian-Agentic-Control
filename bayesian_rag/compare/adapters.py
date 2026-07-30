"""Run the comparison against the real LangGraph and LangChain packages.

The baselines in `baselines.py` reimplement each framework's execution
semantics so the comparison can run without a network or a model. That is a
claim about fidelity, and a claim about fidelity should be checkable.

This module builds the same two pipelines using the actual packages, so the
stand-in numbers can be verified against them:

    pip install langgraph langchain-core
    python -c "from bayesian_rag.compare.adapters import report; report()"

The graphs are structurally identical to the stand-ins: retrieve, validate,
conditionally escalate. If the real packages produce different trajectories on
the same tools, the stand-ins are wrong and should be corrected.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

END_NODE = "__end__"


def available() -> Dict[str, Optional[str]]:
    """Report which real packages are importable, and at what version."""
    found: Dict[str, Optional[str]] = {}
    for name in ("langgraph", "langchain_core", "langchain"):
        try:
            mod = __import__(name)
            found[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            found[name] = None
    return found


def build_real_langgraph(tools: Dict[str, Any], preferred: str, fallback: str, checker: str):
    """The engineered pipeline, built with the real `langgraph`.

    Mirrors `baselines.build_langgraph_engineered` node for node and edge for
    edge, so any behavioural difference indicates a fidelity problem in the
    stand-in rather than a difference of design.
    """
    try:
        from typing import TypedDict

        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "langgraph is not installed. Install it with `pip install langgraph` "
            "to check the stand-in against the real package."
        ) from exc

    class State(TypedDict, total=False):
        query: str
        answer: str
        valid: bool

    def retrieve(state: State) -> State:
        out = tools[preferred].invoke({"query": state["query"]})
        return {**state, "answer": out.get("answer", "")}

    def validate(state: State) -> State:
        out = tools[checker].invoke({"text": state.get("answer", "")})
        return {**state, "valid": bool(out.get("ok", False))}

    def escalate(state: State) -> State:
        out = tools[fallback].invoke({"query": state["query"]})
        return {**state, "answer": out.get("answer", "")}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("validate", validate)
    graph.add_node("escalate", escalate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda s: "ok" if s.get("valid") else "retry",
        {"ok": END, "retry": "escalate"},
    )
    graph.add_edge("escalate", END)
    return graph.compile()


def compare_against_standin(trials: int = 1) -> Dict[str, Any]:
    """Run the real graph and the stand-in on identical worlds and compare.

    Returns a per-world record of both answers and whether they agree. Any
    disagreement is a defect in the stand-in.
    """
    from bayesian_rag.compare.__main__ import CORRECT, QUERY, WORLDS
    from bayesian_rag.compare.baselines import build_langgraph_engineered

    if available()["langgraph"] is None:
        return {"error": "langgraph not installed", "available": available()}

    rows = []
    for world in WORLDS:
        tools = world.build_tools()
        args = ("quick_source", "thorough_source", "verify")

        def _run(builder):
            try:
                graph = builder(tools, *args)
                return graph.invoke({"query": QUERY}).get("answer", "")
            except Exception as exc:  # noqa: BLE001
                return f"<error: {type(exc).__name__}>"

        real = _run(build_real_langgraph)
        standin = _run(build_langgraph_engineered)

        rows.append(
            {
                "world": world.name,
                "real_langgraph": real,
                "standin": standin,
                "agree": real == standin,
                "both_correct": (CORRECT in real) == (CORRECT in standin),
            }
        )

    return {
        "available": available(),
        "rows": rows,
        "all_agree": all(r["agree"] for r in rows),
    }


def report() -> None:
    """Print the fidelity check."""
    found = available()
    print("\nInstalled packages:")
    for name, version in found.items():
        print(f"  {name:16s} {version or 'not installed'}")

    if found["langgraph"] is None:
        print(
            "\nlanggraph is not installed, so the stand-in cannot be checked "
            "against it here.\nInstall with `pip install langgraph` and re-run."
        )
        return

    result = compare_against_standin()
    print("\nReal LangGraph vs stand-in, identical tools and worlds:")
    for row in result["rows"]:
        mark = "agree" if row["agree"] else "DIFFER"
        print(f"  {row['world']:22s} {mark}")
        if not row["agree"]:
            print(f"      real    : {row['real_langgraph'][:60]!r}")
            print(f"      stand-in: {row['standin'][:60]!r}")

    if result["all_agree"]:
        print("\nThe stand-in reproduces the real package on every world.")
    else:
        print("\nThe stand-in diverges. Treat its numbers as unverified.")


if __name__ == "__main__":
    report()
