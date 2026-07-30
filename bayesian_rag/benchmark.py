"""Reproducible benchmark with interval estimates.

    python -m bayesian_rag.benchmark                  # default 200 trials
    python -m bayesian_rag.benchmark --trials 500     # tighter intervals
    python -m bayesian_rag.benchmark --json           # machine-readable

Every trial uses a distinct seed, so what is measured is behaviour across the
randomness in selection rather than one trajectory. Rates are reported with
Wilson intervals and regime comparisons with a significance verdict, because a
point estimate alone cannot tell a reader whether two numbers differ.

Re-running with the same --trials reproduces these numbers exactly: seeds are
derived deterministically from the trial index.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from bayesian_rag import __version__
from bayesian_rag.scenarios.registry import (
    SCENARIO_QUERIES,
    SCENARIO_SUCCESS_MARKERS,
    SCENARIOS,
)
from bayesian_rag.utils.statistics import Comparison, Proportion, Summary

REGIMES = ("greedy", "forward", "smc")
COMPARATIVE = ("stale_vs_verified", "web_vs_official", "ambiguous_location")


@dataclass
class Cell:
    """One scenario/regime pair evaluated across seeds."""

    scenario: str
    regime: str
    successes: int
    trials: int
    steps: List[int] = field(default_factory=list, repr=False)
    times_ms: List[float] = field(default_factory=list, repr=False)
    confidence: float = 0.95

    @property
    def proportion(self) -> Proportion:
        return Proportion(self.successes, self.trials, self.confidence)

    @property
    def step_summary(self) -> Summary:
        return Summary(self.steps, self.confidence)

    @property
    def time_summary(self) -> Summary:
        return Summary(self.times_ms, self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        lo, hi = self.proportion.interval
        return {
            "scenario": self.scenario,
            "regime": self.regime,
            "trials": self.trials,
            "successes": self.successes,
            "success_rate": round(self.proportion.rate, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "avg_steps": round(self.step_summary.mean, 3),
            "avg_time_ms": round(self.time_summary.mean, 3),
        }


def evaluate(scenario: str, regime: str, trials: int, confidence: float = 0.95) -> Cell:
    """Run one scenario/regime pair across `trials` deterministic seeds."""
    query = SCENARIO_QUERIES[scenario]
    markers = SCENARIO_SUCCESS_MARKERS[scenario]

    cell = Cell(scenario=scenario, regime=regime, successes=0, trials=trials,
                confidence=confidence)

    for seed in range(trials):
        answer, trace, posterior = SCENARIOS[scenario](seed=seed).run(query, regime=regime)
        if any(m in answer for m in markers):
            cell.successes += 1
        cell.steps.append(len(trace.steps))
        cell.times_ms.append(posterior.get("time_used_s", 0.0) * 1000.0)

    return cell


def session_effect(trials: int, confidence: float = 0.95) -> List[Dict[str, Any]]:
    """Measure what carrying reliability beliefs between episodes buys."""
    scenario = "session_learning"
    query = SCENARIO_QUERIES[scenario]
    markers = SCENARIO_SUCCESS_MARKERS[scenario]

    rows: List[Dict[str, Any]] = []
    for regime in ("forward", "smc"):
        cells = {}
        for carry in (False, True):
            wins = 0
            for seed in range(trials):
                episodes = SCENARIOS[scenario](seed=seed).run_session(
                    [query, query], regime=regime, carry_reliability=carry
                )
                if any(m in episodes[1]["answer"] for m in markers):
                    wins += 1
            cells[carry] = Proportion(wins, trials, confidence)

        cmp_ = Comparison("carried", "fresh", cells[True], cells[False], confidence)
        for carry, prop in cells.items():
            lo, hi = prop.interval
            rows.append({
                "regime": regime,
                "carry_reliability": carry,
                "episode_2_success": round(prop.rate, 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "trials": trials,
            })
        rows.append({
            "regime": regime,
            "comparison": cmp_.format(),
            "difference": round(cmp_.difference, 4),
            "significant": cmp_.significant,
            "p_value": round(cmp_.p_value, 6),
        })
    return rows


def environment() -> Dict[str, str]:
    """Context a reader needs to interpret timings."""
    return {
        "bayesian_rag": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


def run_all(trials: int, confidence: float = 0.95) -> Dict[str, Any]:
    """Evaluate every scenario under every regime, with pairwise contrasts."""
    cells = [
        evaluate(scenario, regime, trials, confidence)
        for scenario in COMPARATIVE
        for regime in REGIMES
    ]

    comparisons: List[Dict[str, Any]] = []
    by_scenario: Dict[str, Dict[str, Cell]] = {}
    for c in cells:
        by_scenario.setdefault(c.scenario, {})[c.regime] = c

    for scenario, regimes in by_scenario.items():
        for a, b in (("smc", "greedy"), ("smc", "forward"), ("forward", "greedy")):
            if a in regimes and b in regimes:
                cmp_ = Comparison(a, b, regimes[a].proportion, regimes[b].proportion, confidence)
                comparisons.append({
                    "scenario": scenario,
                    "contrast": f"{a} vs {b}",
                    "difference": round(cmp_.difference, 4),
                    "ci_lower": round(cmp_.interval[0], 4),
                    "ci_upper": round(cmp_.interval[1], 4),
                    "p_value": round(cmp_.p_value, 6),
                    "significant": cmp_.significant,
                })

    return {
        "environment": environment(),
        "trials_per_cell": trials,
        "confidence_level": confidence,
        "comparative": [c.to_dict() for c in cells],
        "comparisons": comparisons,
        "session": session_effect(trials, confidence),
    }


def print_report(data: Dict[str, Any]) -> None:
    """Render results as a table, marking contrasts whose interval excludes zero."""
    trials = data["trials_per_cell"]
    conf = int(data["confidence_level"] * 100)

    print(f"\nBayesianRAG benchmark  |  {trials} seeds/cell  |  {conf}% Wilson intervals")
    print(f"  {data['environment']['platform']}, Python {data['environment']['python']}")

    print(f"\n{'scenario':22s} {'regime':9s} {'success rate':>22s} {'steps':>7s} {'ms':>7s}")
    print("-" * 72)
    last = None
    for row in data["comparative"]:
        if last and row["scenario"] != last:
            print()
        rate = f"{row['success_rate']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        print(f"{row['scenario']:22s} {row['regime']:9s} {rate:>22s} "
              f"{row['avg_steps']:7.2f} {row['avg_time_ms']:7.2f}")
        last = row["scenario"]

    print(f"\nPairwise contrasts ({conf}% CI on the difference)")
    print("-" * 72)
    last = None
    for c in data["comparisons"]:
        if last and c["scenario"] != last:
            print()
        mark = "*" if c["significant"] else " "
        print(f"{mark} {c['scenario']:22s} {c['contrast']:16s} "
              f"{c['difference']:+.3f} [{c['ci_lower']:+.3f}, {c['ci_upper']:+.3f}]  "
              f"p={c['p_value']:.2g}")
        last = c["scenario"]

    print(f"\nCross-episode carry-over, episode 2")
    print("-" * 72)
    for row in data["session"]:
        if "comparison" in row:
            print(f"  -> {row['comparison']}\n")
        else:
            print(f"  {row['regime']:8s} carry={str(row['carry_reliability']):5s} "
                  f"{row['episode_2_success']:.3f} "
                  f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    print("* marks a difference whose interval excludes zero.\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="BayesianRAG benchmark")
    parser.add_argument("--trials", type=int, default=200,
                        help="Seeds per cell. More trials, tighter intervals.")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for intervals.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must lie in (0, 1)")

    data = run_all(args.trials, args.confidence)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_report(data)


if __name__ == "__main__":
    main()
