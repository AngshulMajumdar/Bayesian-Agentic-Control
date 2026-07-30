"""Prior sensitivity, and behaviour against unreliable or adversarial agents.

Two questions the three-world comparison did not answer.

1. HOW MUCH DO PRIORS MATTER?
   SMC inherits whatever reliability the user declares. World B showed a
   confidently wrong prior is costly. This sweeps the prior across its range to
   show the shape of that dependence rather than a single point on it.

2. WHAT HAPPENS WITH SHIFTY AGENTS?
   World B held a tool that was consistently wrong and merely mislabelled. That
   is not the interesting adversary. A shifty tool is one that is right
   sometimes and wrong other times, with no fixed pattern -- exactly the case a
   hard-coded branch cannot encode, because there is no single correct routing
   decision to write down. The routing has to be made per call, from evidence.

Tool-call count is not treated as a finding here. Maintaining a belief over
alternatives costs more calls by construction; the question is what the extra
calls buy.

Run:  python -m bayesian_rag.compare.priors
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from bayesian_rag import Agent, checker as checker_tool, tool as tool_decorator
from bayesian_rag.compare.baselines import (
    build_langchain_reflexive,
    build_langgraph_engineered,
    build_langgraph_naive,
)
from bayesian_rag.utils.statistics import Proportion

CORRECT = "Canberra is the capital of Australia."
WRONG_POOL = [
    "Sydney is the capital of Australia.",
    "Melbourne is the capital of Australia.",
    "Perth is the capital of Australia.",
]
QUERY = "What is the capital of Australia?"


# =============================================================================
# Tool builders
# =============================================================================


def make_shifty_tools(
    accuracies: Dict[str, float],
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Tools that are right a stated fraction of the time, wrong otherwise.

    Each call is an independent draw, so no single tool is reliably right and no
    fixed routing rule can be correct. A tool at 0.4 accuracy is not "the bad
    one" to be avoided -- it is right two calls in five, and which two is not
    knowable in advance. What can be done is to draw again and check, which is
    what a population of trajectories does for free.

    Args:
        accuracies: per-tool probability of returning the correct answer.
        priors: per-tool (reliability, appeal) as declared to the agent. Defaults
            to uninformative 0.5 / 0.5.
        seed: seeds the tools' own randomness, held fixed across orchestrators
            so every system faces an identical sequence of outcomes.
    """
    priors = priors or {}
    rng = random.Random(seed)
    tools: Dict[str, Any] = {}

    for name, accuracy in accuracies.items():
        reliability, appeal = priors.get(name, (0.5, 0.5))

        def _make(acc: float, nm: str):
            def fn(query: str) -> str:
                if rng.random() < acc:
                    return CORRECT
                return rng.choice(WRONG_POOL)

            fn.__name__ = nm
            fn.__doc__ = f"Source returning the correct answer {acc:.0%} of the time."
            return fn

        tools[name] = tool_decorator(
            _make(accuracy, name),
            name=name,
            reliability=reliability,
            appeal=appeal,
            # Each call is an independent draw. Marking these deterministic
            # would memoise the first result and hand every particle the same
            # draw, disabling the re-draw that makes the cloud worth having.
            deterministic=False,
        )

    @checker_tool
    def verify(text: str) -> bool:
        """Validate a claim against ground truth."""
        return CORRECT in (text or "")

    tools["verify"] = verify
    return tools


# =============================================================================
# Orchestrators under test
# =============================================================================


def run_smc(tools: Dict[str, Any], seed: int, max_steps: int = 4) -> bool:
    agent = Agent(list(tools.values()), seed=seed, particles=16, max_steps=max_steps)
    try:
        return CORRECT in agent.run(QUERY).answer
    except Exception:  # noqa: BLE001
        return False


def run_greedy(tools: Dict[str, Any], seed: int, max_steps: int = 4) -> bool:
    agent = Agent(list(tools.values()), seed=seed, particles=16, max_steps=max_steps)
    try:
        return CORRECT in agent.run(QUERY, regime="greedy").answer
    except Exception:  # noqa: BLE001
        return False


def run_forward(tools: Dict[str, Any], seed: int, max_steps: int = 4) -> bool:
    agent = Agent(list(tools.values()), seed=seed, particles=16, max_steps=max_steps)
    try:
        return CORRECT in agent.run(QUERY, regime="forward").answer
    except Exception:  # noqa: BLE001
        return False


def run_graph_engineered(tools: Dict[str, Any], seed: int, preferred: str,
                         fallback: str) -> bool:
    graph = build_langgraph_engineered(tools, preferred, fallback, "verify")
    try:
        return CORRECT in graph.invoke({"query": QUERY}).get("answer", "")
    except Exception:  # noqa: BLE001
        return False


def run_graph_naive(tools: Dict[str, Any], seed: int, preferred: str) -> bool:
    graph = build_langgraph_naive(tools, preferred)
    try:
        return CORRECT in graph.invoke({"query": QUERY}).get("answer", "")
    except Exception:  # noqa: BLE001
        return False


def run_reflexive(tools: Dict[str, Any], seed: int, preferred: str, fallback: str) -> bool:
    ex = build_langchain_reflexive(tools, preferred, fallback, "verify")
    try:
        return CORRECT in ex.invoke({"input": QUERY}).get("output", "")
    except Exception:  # noqa: BLE001
        return False


# =============================================================================
# Experiment 1 -- prior sensitivity
# =============================================================================


def prior_sensitivity(trials: int = 200) -> List[Dict[str, Any]]:
    """Sweep the declared prior on a tool that is actually unreliable.

    One tool is genuinely poor (0.25 accurate) and one genuinely good (0.9). The
    declared prior on the poor tool is swept from honest to badly wrong, holding
    everything else fixed, to show how far a misdeclaration moves the outcome.
    """
    rows: List[Dict[str, Any]] = []

    for declared in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        wins = 0
        for seed in range(trials):
            tools = make_shifty_tools(
                accuracies={"unreliable": 0.25, "dependable": 0.9},
                priors={
                    # The lie: an unreliable tool declared trustworthy and made
                    # attractive, so nothing but evidence can discourage it.
                    "unreliable": (declared, 0.9),
                    "dependable": (0.5, 0.5),
                },
                seed=seed,
            )
            wins += run_smc(tools, seed)
        rows.append(
            {
                "declared_prior_on_bad_tool": declared,
                "true_accuracy": 0.25,
                "smc": Proportion(wins, trials),
            }
        )

    return rows


# =============================================================================
# Experiment 2 -- shifty agents
# =============================================================================


@dataclass
class ShiftyCase:
    """A configuration of unreliable tools."""

    name: str
    accuracies: Dict[str, float]
    description: str


SHIFTY_CASES = [
    ShiftyCase(
        name="one flaky source",
        accuracies={"flaky": 0.4},
        description="A single source, right two calls in five.",
    ),
    ShiftyCase(
        name="two flaky sources",
        accuracies={"flaky_a": 0.4, "flaky_b": 0.4},
        description="Two equally unreliable sources; neither is the one to pick.",
    ),
    ShiftyCase(
        name="flaky plus dependable",
        accuracies={"flaky": 0.3, "dependable": 0.9},
        description="One poor, one good, indistinguishable before evidence.",
    ),
    ShiftyCase(
        name="mostly adversarial",
        accuracies={"adversarial": 0.1, "flaky": 0.4, "dependable": 0.85},
        description="Three sources spanning near-useless to good.",
    ),
    ShiftyCase(
        name="all unreliable",
        accuracies={"a": 0.35, "b": 0.35, "c": 0.35},
        description="No dependable source exists; only retrying finds the answer.",
    ),
]


def shifty_comparison(trials: int = 200, max_steps: int = 4) -> List[Dict[str, Any]]:
    """Compare orchestrators against stochastically unreliable tools.

    Every system faces the identical sequence of tool outcomes for a given seed,
    so differences come from the routing strategy rather than luck of the draw.
    """
    rows: List[Dict[str, Any]] = []

    for case in SHIFTY_CASES:
        names = list(case.accuracies)
        preferred = names[0]
        fallback = names[-1] if len(names) > 1 else names[0]

        counts = {k: 0 for k in
                  ("naive", "engineered", "reflexive", "greedy", "forward", "smc")}

        for seed in range(trials):
            def fresh():
                return make_shifty_tools(case.accuracies, seed=seed)

            counts["naive"] += run_graph_naive(fresh(), seed, preferred)
            counts["engineered"] += run_graph_engineered(fresh(), seed, preferred, fallback)
            counts["reflexive"] += run_reflexive(fresh(), seed, preferred, fallback)
            counts["greedy"] += run_greedy(fresh(), seed, max_steps)
            counts["forward"] += run_forward(fresh(), seed, max_steps)
            counts["smc"] += run_smc(fresh(), seed, max_steps)

        rows.append(
            {
                "case": case.name,
                "description": case.description,
                "accuracies": case.accuracies,
                "results": {k: Proportion(v, trials) for k, v in counts.items()},
            }
        )

    return rows


def budget_sweep(trials: int = 150) -> List[Dict[str, Any]]:
    """How accuracy grows with the step budget against unreliable tools.

    With a checker available, extra steps let a trajectory discard a bad draw
    and take another. This is the mechanism by which a filter over unreliable
    tools can beat any single tool's accuracy.
    """
    rows: List[Dict[str, Any]] = []
    accuracies = {"flaky_a": 0.4, "flaky_b": 0.4}

    for steps in (2, 3, 4, 5, 6, 8):
        counts = {"smc": 0, "greedy": 0, "engineered": 0}
        for seed in range(trials):
            counts["smc"] += run_smc(make_shifty_tools(accuracies, seed=seed), seed, steps)
            counts["greedy"] += run_greedy(make_shifty_tools(accuracies, seed=seed), seed, steps)
            counts["engineered"] += run_graph_engineered(
                make_shifty_tools(accuracies, seed=seed), seed, "flaky_a", "flaky_b"
            )
        rows.append(
            {
                "max_steps": steps,
                "results": {k: Proportion(v, trials) for k, v in counts.items()},
            }
        )
    return rows


# =============================================================================
# Reporting
# =============================================================================


def print_report(trials: int = 200) -> None:
    print("\n" + "=" * 78)
    print("EXPERIMENT 1 -- How much does the declared prior matter?")
    print("=" * 78)
    print("A tool with true accuracy 0.25, declared increasingly trustworthy.")
    print("Everything else held fixed. A dependable 0.9 source is also available.\n")
    print(f"  {'declared prior':>15s}   {'SMC accuracy':<24s}")
    print("  " + "-" * 45)
    for row in prior_sensitivity(trials):
        print(f"  {row['declared_prior_on_bad_tool']:>15.2f}   {row['smc'].format()}")
    print("\n  The prior costs accuracy only when it is a confident lie, and even")
    print("  then evidence recovers most of it. It is a bias, not a ceiling.\n")

    print("=" * 78)
    print("EXPERIMENT 2 -- Unreliable and adversarial agents")
    print("=" * 78)
    print("Tools return the correct answer only a fraction of the time, drawn")
    print("independently per call. No fixed routing rule can be correct, because")
    print("which source is right varies call to call.\n")

    for row in shifty_comparison(trials):
        acc = ", ".join(f"{k}={v:.0%}" for k, v in row["accuracies"].items())
        print("-" * 78)
        print(f"{row['case']}  ({acc})")
        print(f"  {row['description']}")
        print()
        for label, key in [
            ("LangGraph (naive)", "naive"),
            ("LangGraph (engineered)", "engineered"),
            ("LangChain (reflexive)", "reflexive"),
            ("BayesianRAG (greedy)", "greedy"),
            ("BayesianRAG (forward)", "forward"),
            ("BayesianRAG (SMC)", "smc"),
        ]:
            print(f"    {label:26s} {row['results'][key].format()}")
        print()

    print("=" * 78)
    print("EXPERIMENT 3 -- Accuracy against step budget (two 40% sources)")
    print("=" * 78)
    print("No source exceeds 40% alone. Extra steps let a trajectory discard a")
    print("failed draw and take another.\n")
    print(f"  {'steps':>6s}   {'SMC':<24s} {'greedy':<24s} {'engineered':<24s}")
    print("  " + "-" * 80)
    for row in budget_sweep(min(trials, 150)):
        r = row["results"]
        print(f"  {row['max_steps']:>6d}   {r['smc'].format():<24s} "
              f"{r['greedy'].format():<24s} {r['engineered'].format():<24s}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prior sensitivity and unreliable-agent experiments"
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.json:
        def _ser(p):
            return {"rate": round(p.rate, 4), "lo": round(p.interval[0], 4),
                    "hi": round(p.interval[1], 4), "n": p.trials}

        print(json.dumps({
            "prior_sensitivity": [
                {**{k: v for k, v in r.items() if k != "smc"}, "smc": _ser(r["smc"])}
                for r in prior_sensitivity(args.trials)
            ],
            "shifty": [
                {**{k: v for k, v in r.items() if k != "results"},
                 "results": {k: _ser(v) for k, v in r["results"].items()}}
                for r in shifty_comparison(args.trials)
            ],
        }, indent=2))
    else:
        print_report(args.trials)


if __name__ == "__main__":
    main()
