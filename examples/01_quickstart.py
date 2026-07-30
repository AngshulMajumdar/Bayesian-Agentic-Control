"""Smallest useful agent: three tools, no callbacks.

    python examples/01_quickstart.py
"""

from bayesian_rag import Agent, checker, tool


@tool(reliability=0.6, appeal=0.9)
def quick_search(query: str) -> str:
    """Fast and cheap, but often out of date."""
    return "Paris is the capital of Australia."


@tool(reliability=0.95, appeal=0.5)
def verified_search(query: str) -> str:
    """Slower, authoritative."""
    return "Canberra is the capital of Australia."


@checker
def fact_check(text: str) -> bool:
    """Check a claim against reference data."""
    return "Canberra" in text


def main() -> None:
    agent = Agent([quick_search, verified_search, fact_check])
    print(agent.describe(), "\n")

    result = agent.run("What is the capital of Australia?")
    print(result.explain())


if __name__ == "__main__":
    main()
