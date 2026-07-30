"""Comparing Bayesian orchestration against deterministic LangGraph/LangChain.

THE QUESTION WORTH ASKING
-------------------------
Whether a naive pipeline loses to a particle filter is not interesting: of
course it does, it has no recovery path at all. A competent engineer who knows
a failure mode exists writes a conditional edge for it, and that graph should
do well on the failure it was written for.

So the experiment is about generalisation. Three worlds share one tool set and
one graph:

  A. anticipated -- the convenient source is stale, the authoritative one is
     correct. Exactly what the engineer had in mind.

  B. roles reversed -- the convenient source is now correct and the
     "authoritative" one has gone stale. Nothing about the tools changed, only
     which is right today.

  C. outage -- the authoritative source errors out. The escalation target is
     unavailable.

A hard-coded branch encodes an assumption about which source to trust. When
that assumption holds the branch is optimal and cheaper than any inference.
When it stops holding, the branch keeps firing and carries the agent
confidently to the wrong answer, because nothing in the graph is watching
whether the assumption still holds.

Run:  python -m bayesian_rag.compare
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from bayesian_rag import Agent, checker as checker_tool, tool as tool_decorator
from bayesian_rag.compare.baselines import (
    build_langchain_agent,
    build_langchain_reflexive,
    build_langgraph_engineered,
    build_langgraph_naive,
)
from bayesian_rag.utils.statistics import Comparison, Proportion

CORRECT = "Canberra is the capital of Australia."
STALE = "Sydney is the capital of Australia."


# =============================================================================
# Three worlds over one tool set
# =============================================================================


@dataclass
class Outcome:
    """What a run produced, and whether the system signalled any doubt.

    `flagged` matters when no correct answer is obtainable. Returning a false
    claim while reporting it as unverified is a materially different failure
    from asserting it as fact, and a comparison that scores only accuracy
    cannot tell them apart.
    """

    answer: str
    flagged: bool
    crashed: bool = False

    @property
    def correct(self) -> bool:
        return CORRECT in self.answer


@dataclass
class World:
    """A configuration of tool behaviour plus what counts as success."""

    name: str
    description: str
    quick_answer: Optional[str]
    thorough_answer: Optional[str]
    quick_raises: bool = False
    thorough_raises: bool = False
    unwinnable: bool = False

    def build_tools(self, counter: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Tools shared verbatim by every orchestrator under test.

        `counter` records invocations, so the comparison can report what each
        strategy costs as well as what it gets right.
        """
        world = self
        calls = counter if counter is not None else {}

        def _count(name: str) -> None:
            calls[name] = calls.get(name, 0) + 1

        @tool_decorator(reliability=0.6, appeal=0.9)
        def quick_source(query: str) -> str:
            """Fast and cheap. Usually good enough."""
            _count("quick_source")
            if world.quick_raises:
                raise RuntimeError("quick_source: upstream 503")
            return world.quick_answer

        @tool_decorator(reliability=0.95, appeal=0.5)
        def thorough_source(query: str) -> str:
            """Slower, treated as authoritative."""
            _count("thorough_source")
            if world.thorough_raises:
                raise RuntimeError("thorough_source: upstream 503")
            return world.thorough_answer

        @checker_tool
        def verify(text: str) -> bool:
            """Validate a claim against ground truth."""
            _count("verify")
            return CORRECT in (text or "")

        return {
            "quick_source": quick_source,
            "thorough_source": thorough_source,
            "verify": verify,
        }


WORLDS = [
    World(
        name="A. anticipated",
        description="Convenient source stale, authoritative source correct.",
        quick_answer=STALE,
        thorough_answer=CORRECT,
    ),
    World(
        name="B. roles reversed",
        description="Convenient source correct, authoritative source now stale.",
        quick_answer=CORRECT,
        thorough_answer=STALE,
    ),
    World(
        name="C. outage",
        description="Convenient source stale; authoritative source unavailable.",
        quick_answer=STALE,
        thorough_answer=None,
        thorough_raises=True,
        unwinnable=True,
    ),
]


# =============================================================================
# Orchestrators
# =============================================================================


def run_langgraph_naive(tools: Dict[str, Any], query: str, seed: int) -> Outcome:
    graph = build_langgraph_naive(tools, preferred="quick_source")
    try:
        return Outcome(graph.invoke({"query": query}).get("answer", ""), flagged=False)
    except Exception as exc:  # noqa: BLE001 - a crash is a failed run, not an error here
        return Outcome(f"<error: {exc}>", flagged=True, crashed=True)


def run_langgraph_engineered(tools: Dict[str, Any], query: str, seed: int) -> Outcome:
    graph = build_langgraph_engineered(
        tools, preferred="quick_source", fallback="thorough_source", checker="verify"
    )
    try:
        return Outcome(graph.invoke({"query": query}).get("answer", ""), flagged=False)
    except Exception as exc:  # noqa: BLE001
        return Outcome(f"<error: {exc}>", flagged=True, crashed=True)


def run_langchain(tools: Dict[str, Any], query: str, seed: int) -> Outcome:
    executor = build_langchain_agent(tools, preferred="quick_source")
    try:
        return Outcome(executor.invoke({"input": query}).get("output", ""), flagged=False)
    except Exception as exc:  # noqa: BLE001
        return Outcome(f"<error: {exc}>", flagged=True, crashed=True)


def run_langchain_reflexive(tools: Dict[str, Any], query: str, seed: int) -> Outcome:
    executor = build_langchain_reflexive(
        tools, preferred="quick_source", fallback="thorough_source", checker="verify"
    )
    try:
        return Outcome(executor.invoke({"input": query}).get("output", ""), flagged=False)
    except Exception as exc:  # noqa: BLE001
        return Outcome(f"<error: {exc}>", flagged=True, crashed=True)


def run_bayesian(tools: Dict[str, Any], query: str, seed: int) -> Outcome:
    agent = Agent(list(tools.values()), seed=seed, particles=16, max_steps=3)
    try:
        r = agent.run(query)
        return Outcome(r.answer, flagged=r.status != "validated")
    except Exception as exc:  # noqa: BLE001
        return Outcome(f"<error: {exc}>", flagged=True, crashed=True)


def run_bayesian_neutral(tools: Dict[str, Any], query: str, seed: int) -> Outcome:
    """SMC given no prior opinion about which source to trust.

    The declared priors in these worlds encode an assumption that only holds in
    world A. Stripping them isolates what inference contributes when it is not
    handed a confidently wrong belief to start from.
    """
    neutral = [_with_neutral_prior(t) for t in tools.values()]
    agent = Agent(neutral, seed=seed, particles=16, max_steps=3)
    try:
        r = agent.run(query)
        return Outcome(r.answer, flagged=r.status != "validated")
    except Exception as exc:  # noqa: BLE001
        return Outcome(f"<error: {exc}>", flagged=True, crashed=True)


def run_bayesian_greedy(tools: Dict[str, Any], query: str, seed: int) -> Outcome:
    agent = Agent(list(tools.values()), seed=seed, particles=16, max_steps=3)
    try:
        r = agent.run(query, regime="greedy")
        return Outcome(r.answer, flagged=r.status != "validated")
    except Exception as exc:  # noqa: BLE001
        return Outcome(f"<error: {exc}>", flagged=True, crashed=True)


def _with_neutral_prior(t):
    """Copy a tool with its reliability and appeal reset to uninformative."""
    import dataclasses

    return dataclasses.replace(t, reliability=0.5, appeal=0.5)


ORCHESTRATORS: Dict[str, Callable[[Dict[str, Any], str, int], "Outcome"]] = {
    "LangGraph (naive)": run_langgraph_naive,
    "LangGraph (engineered)": run_langgraph_engineered,
    "LangChain (retry)": run_langchain,
    "LangChain (reflexive)": run_langchain_reflexive,
    "BayesianRAG (greedy)": run_bayesian_greedy,
    "BayesianRAG (SMC, declared priors)": run_bayesian,
    "BayesianRAG (SMC, neutral priors)": run_bayesian_neutral,
}

# Deterministic systems return the same answer regardless of seed, so repeating
# them is wasted work; only the sampling regimes need multiple trials.
STOCHASTIC = {"BayesianRAG (SMC, declared priors)", "BayesianRAG (SMC, neutral priors)"}

QUERY = "What is the capital of Australia?"


# =============================================================================
# Harness
# =============================================================================


@dataclass
class Score:
    """How one orchestrator fared in one world."""

    correct: Proportion
    safe: Proportion
    avg_calls: float
    crash_rate: float = 0.0

    def line(self, unwinnable: bool) -> str:
        if unwinnable:
            # Accuracy is not measurable here; what matters is whether the
            # system asserted a falsehood or admitted it could not confirm one.
            # Crashing also avoids the falsehood, but it is a different
            # behaviour from returning a result marked unverified, and the
            # report should not blur them.
            if self.crash_rate > 0.5:
                verdict = "crashes"
            elif self.safe.rate > 0.5:
                verdict = "returns, unverified"
            else:
                verdict = "asserts falsehood"
            return f"{verdict:<20s}  {self.avg_calls:4.1f} calls"
        if self.correct.trials == 1:
            verdict = "correct" if self.correct.rate == 1.0 else "WRONG"
            return f"{verdict:<20s}  {self.avg_calls:4.1f} calls"
        return f"{self.correct.format():<20s}  {self.avg_calls:4.1f} calls"


def evaluate(world: World, trials: int) -> Dict[str, Score]:
    """Run every orchestrator in one world, recording accuracy and cost."""
    results: Dict[str, Score] = {}

    for name, run in ORCHESTRATORS.items():
        n = trials if name in STOCHASTIC else 1
        wins = safe = crashes = 0
        total_calls = 0

        for seed in range(n):
            counter: Dict[str, int] = {}
            tools = world.build_tools(counter)
            outcome = run(tools, QUERY, seed)
            wins += outcome.correct
            # "Safe" means it did not assert a wrong answer unflagged.
            safe += outcome.correct or outcome.flagged
            crashes += outcome.crashed
            total_calls += sum(counter.values())

        results[name] = Score(
            correct=Proportion(wins, n),
            safe=Proportion(safe, n),
            avg_calls=total_calls / n,
            crash_rate=crashes / n,
        )

    return results


def run_all(trials: int = 200) -> Dict[str, Any]:
    return {
        "trials_stochastic": trials,
        "query": QUERY,
        "worlds": [
            {
                "name": w.name,
                "description": w.description,
                "unwinnable": w.unwinnable,
                "results": {
                    name: {
                        "success_rate": round(sc.correct.rate, 4),
                        "safe_rate": round(sc.safe.rate, 4),
                        "trials": sc.correct.trials,
                        "ci_lower": round(sc.correct.interval[0], 4),
                        "ci_upper": round(sc.correct.interval[1], 4),
                        "avg_tool_calls": round(sc.avg_calls, 2),
                    }
                    for name, sc in evaluate(w, trials).items()
                },
            }
            for w in WORLDS
        ],
    }


def print_report(trials: int = 200) -> None:
    print("\nDeterministic orchestration vs Bayesian inference")
    print(f"Query: {QUERY!r}")
    print("Deterministic systems are run once -- no seed can change them.")
    print(f"Sampling regimes are run over {trials} seeds, with 95% Wilson intervals.\n")

    per_world: List[Dict[str, Score]] = []

    for world in WORLDS:
        results = evaluate(world, trials)
        per_world.append(results)

        print("=" * 76)
        print(f"{world.name}: {world.description}")
        if world.unwinnable:
            print("  No tool can produce the correct answer here. Accuracy is not the")
            print("  measure; whether the system knows it failed is.")
        print("-" * 76)
        for name, score in results.items():
            print(f"  {name:36s} {score.line(world.unwinnable)}")
        print()

    print("=" * 76)
    print("Summary")
    print("-" * 76)
    for name in ORCHESTRATORS:
        marks = []
        for world, results in zip(WORLDS, per_world):
            sc = results[name]
            ok = sc.safe.rate > 0.5 if world.unwinnable else sc.correct.rate > 0.5
            marks.append("PASS" if ok else "fail")
        passes = marks.count("PASS")
        print(f"  {name:36s} {'  '.join(f'{m:>4s}' for m in marks)}   {passes}/3")

    print("\n  Worlds: A anticipated | B roles reversed | C outage")
    print("  A world passes on accuracy, or -- where unwinnable -- on not asserting")
    print("  a falsehood. Note that crashing also avoids asserting one.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deterministic orchestration against Bayesian inference"
    )
    parser.add_argument("--trials", type=int, default=200,
                        help="Seeds for stochastic orchestrators.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(run_all(args.trials), indent=2))
    else:
        print_report(args.trials)


if __name__ == "__main__":
    main()
