"""Overriding the defaults.

The derived proposer, scorer, and selector cover the common case. Any one can
be replaced when a task needs specific behaviour -- here, a proposer that
enforces an explicit order rather than offering a flat menu.

    python examples/04_custom_components.py
"""

from bayesian_rag import Action, Agent, build_scorer, build_selector, tool


@tool(reliability=0.7)
def retrieve(query: str) -> dict:
    """Retrieve a candidate answer."""
    return {"answer": f"Draft answer for: {query}", "confidence": 0.7}


@tool(reliability=0.9)
def refine(text: str) -> dict:
    """Improve a draft answer."""
    return {"answer": f"Refined: {text}", "confidence": 0.9, "verified": True}


def staged_proposer(query, particles, t):
    """Retrieve first, refine second. A pipeline, not a menu."""
    if t == 0:
        return [Action("retrieve", {"query": query})]
    claim = ""
    if particles and particles[0].observations:
        claim = particles[0].observations[-1].output.get("answer", "")
    return [Action("refine", {"text": claim})] if claim else []


def main() -> None:
    tools = {"retrieve": retrieve, "refine": refine}

    agent = Agent(
        [retrieve, refine],
        proposer=staged_proposer,          # custom
        scorer=build_scorer(tools),        # default
        selector=build_selector(tools),    # default
        max_steps=2,
    )

    result = agent.run("How does particle filtering work?")
    print(result.explain())


if __name__ == "__main__":
    main()
