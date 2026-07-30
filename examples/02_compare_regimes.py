"""Why the inference regime matters.

`quick_search` is more attractive than `verified_search` -- it is cheap and
fast -- and it is wrong. Greedy commits to it. Only SMC folds the checker's
verdict back into the weights and recovers.

    python examples/02_compare_regimes.py
"""

from bayesian_rag import Agent, checker, tool
from bayesian_rag.utils.statistics import Comparison, Proportion

QUESTION = "What is the capital of Australia?"
TRIALS = 100


@tool(reliability=0.6, appeal=0.9)
def quick_search(query: str) -> str:
    """Fast, tempting, stale."""
    return "Paris is the capital of Australia."


@tool(reliability=0.95, appeal=0.5)
def verified_search(query: str) -> str:
    """Slow, authoritative."""
    return "Canberra is the capital of Australia."


@checker
def fact_check(text: str) -> bool:
    """Verify a claim."""
    return "Canberra" in text


def success_rate(regime: str) -> Proportion:
    wins = 0
    for seed in range(TRIALS):
        agent = Agent([quick_search, verified_search, fact_check], seed=seed)
        if "Canberra" in agent.run(QUESTION, regime=regime).answer:
            wins += 1
    return Proportion(wins, TRIALS)


def main() -> None:
    print(f"{TRIALS} seeds per regime, 95% Wilson intervals\n")

    rates = {r: success_rate(r) for r in ("greedy", "forward", "smc")}
    for regime, prop in rates.items():
        print(f"  {regime:8s} {prop.format()}")

    print()
    print(" ", Comparison("smc", "greedy", rates["smc"], rates["greedy"]))
    print(" ", Comparison("smc", "forward", rates["smc"], rates["forward"]))

    print("\nOne run under each regime:")
    agent = Agent([quick_search, verified_search, fact_check], seed=0)
    for regime, result in agent.compare(QUESTION).items():
        verdict = "correct" if "Canberra" in result.answer else "wrong  "
        print(f"  {regime:8s} {verdict}  {' -> '.join(result.tools_used)}")


if __name__ == "__main__":
    main()
