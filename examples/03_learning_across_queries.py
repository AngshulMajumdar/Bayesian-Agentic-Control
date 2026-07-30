"""Reliability beliefs carried between queries.

The agent starts trusting the cheap tool, because it is declared attractive.
Each failed check pushes its posterior down, and by the later queries it is no
longer chosen -- the distrust persists rather than being rediscovered.

    python examples/03_learning_across_queries.py
"""

from bayesian_rag import Agent, checker, tool

QUERIES = [
    "What is the capital of Australia?",
    "Confirm the capital of Australia.",
    "Which city is Australia's capital?",
]


@tool(reliability=0.5, appeal=0.95)
def cheap_source(query: str) -> str:
    """Very attractive, consistently wrong."""
    return "Sydney is the capital of Australia."


@tool(reliability=0.5, appeal=0.5)
def solid_source(query: str) -> str:
    """Unremarkable prior, actually correct."""
    return "Canberra is the capital of Australia."


@checker
def verify(text: str) -> bool:
    """Verify a claim."""
    return "Canberra" in text


def main() -> None:
    agent = Agent([cheap_source, solid_source, verify], seed=3, carry_over=1.0)

    print("Beliefs both start at 0.5; only evidence separates them.\n")
    for i, result in enumerate(agent.run_session(QUERIES)):
        rel = result.reliability
        print(f"episode {i}: {' -> '.join(result.tools_used)}")
        print(f"           cheap={rel.get('cheap_source', 0.5):.3f}  "
              f"solid={rel.get('solid_source', 0.5):.3f}")
        print(f"           {'correct' if 'Canberra' in result.answer else 'wrong'}\n")


if __name__ == "__main__":
    main()
