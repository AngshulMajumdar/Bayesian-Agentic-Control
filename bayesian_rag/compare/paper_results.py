"""Paper-level results: source accuracy sweeps and prior x accuracy ablations.

Two tracks, matching the two original orchestration patterns this library
unified:

  CHAIN track  -- a flat action-proposal loop (LangChain's AgentExecutor
                  pattern): one tool, offered every step, with a checker.
                  "Ours" is BayesianRAG's default proposer under SMC;
                  the baseline is the deterministic AgentExecutor stand-in.

  GRAPH track  -- a fixed retrieve/validate/escalate pipeline (LangGraph's
                  compiled-graph pattern): a convenient node and an
                  authoritative fallback node, wired by a conditional edge.
                  "Ours" is the same two tools under SMC, with the fixed edge
                  replaced by reweighting; the baseline is the deterministic
                  compiled-graph stand-in.

Both tracks share one manipulation: the convenient source's true accuracy is
swept, independent of what the orchestrator believes about it. This isolates
what inference buys as the environment degrades, from a system that has no
belief to revise in the first place.

The ablation grids then hold the true accuracy from that sweep, cross it with
the *declared* prior, and report both accuracy and mean posterior consensus --
so a systematic gap between them (declared 0.9, consensus should track it only
if the declaration is honest) is visible rather than assumed.

Everything here executes against the reference mock tools; see
`bayesian_rag/compare/adapters.py` to check the deterministic stand-ins
against the real LangGraph/LangChain packages.

Run:
    python -m bayesian_rag.compare.paper_results               # tables
    python -m bayesian_rag.compare.paper_results --figures      # + PNG figures
    python -m bayesian_rag.compare.paper_results --json         # machine-readable
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from bayesian_rag import Agent, checker as checker_tool, tool as tool_decorator
from bayesian_rag.compare.baselines import (
    build_langchain_agent,
    build_langchain_reflexive,
    build_langgraph_engineered,
    build_langgraph_naive,
)
from bayesian_rag.utils.statistics import Comparison, Proportion

CORRECT = "CORRECT_ANSWER"
WRONG = "WRONG_ANSWER"
QUERY = "paper-results-query"

ACCURACIES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
GRID_ACCURACIES = [0.1, 0.3, 0.5, 0.7, 0.9]
GRID_PRIORS = [0.1, 0.3, 0.5, 0.7, 0.9]

DEFAULT_PARTICLES = 16
DEFAULT_STEPS = 4

# The convenient/primary tool ("source" in the chain track, "quick" in the
# graph track) must be configured identically in both, or a difference in
# results could come from this instead of from the structural difference
# under study (one tool vs. two). Both tracks import this constant rather
# than declaring their own default, so they cannot silently drift apart.
PRIMARY_APPEAL = 0.8

# The graph track's fallback tool has no analogue in the chain track -- it IS
# the structural difference being tested -- so it is the one parameter that
# is not shared. Its accuracy is fixed at a value already present in
# ACCURACIES and GRID_ACCURACIES, rather than an unrelated number, so it is
# drawn from the same family being swept, not an arbitrary addition.
FALLBACK_ACCURACY = 0.9
FALLBACK_APPEAL = 0.5


# =============================================================================
# Tool construction
# =============================================================================


def chain_tools(
    accuracy: float, seed: int, declared_prior: Optional[float] = None,
    appeal: float = PRIMARY_APPEAL,
) -> Dict[str, Any]:
    """Single stochastic source plus a checker -- the CHAIN track's world.

    `declared_prior` is what the orchestrator is told; `accuracy` is the truth.
    They are independent by construction, which is the point of the ablation.
    `appeal` defaults to `PRIMARY_APPEAL`, shared with `graph_tools`'s `quick`,
    so the two tracks configure their primary tool identically.
    """
    rng = random.Random(seed)
    prior = accuracy if declared_prior is None else declared_prior

    @tool_decorator(
        name="source", reliability=prior, appeal=appeal, deterministic=False
    )
    def source(query: str) -> str:
        """The only available source."""
        return CORRECT if rng.random() < accuracy else WRONG

    @checker_tool
    def verify(text: str) -> bool:
        """Ground-truth check."""
        return text == CORRECT

    return {"source": source, "verify": verify}


def graph_tools(
    quick_accuracy: float,
    seed: int,
    thorough_accuracy: float = FALLBACK_ACCURACY,
    declared_prior: Optional[float] = None,
    quick_appeal: float = PRIMARY_APPEAL,
    thorough_appeal: float = FALLBACK_APPEAL,
) -> Dict[str, Any]:
    """A convenient node and an authoritative fallback -- the GRAPH track's world.

    Structurally mirrors `build_langgraph_engineered`: retrieve from `quick`,
    validate, escalate to `thorough` on failure. `declared_prior` overrides
    what the orchestrator believes about `quick`; the environment does not
    change.

    `quick_appeal` defaults to `PRIMARY_APPEAL`, identical to `chain_tools`'s
    `source` -- the only intended difference between the two tracks is the
    presence of `thorough`, not any other property of the primary tool.
    """
    rng = random.Random(seed)
    prior = quick_accuracy if declared_prior is None else declared_prior

    @tool_decorator(
        name="quick", reliability=prior, appeal=quick_appeal, deterministic=False
    )
    def quick(query: str) -> str:
        """Convenient node."""
        return CORRECT if rng.random() < quick_accuracy else WRONG

    @tool_decorator(
        name="thorough",
        reliability=thorough_accuracy,
        appeal=thorough_appeal,
        deterministic=False,
    )
    def thorough(query: str) -> str:
        """Authoritative fallback node."""
        return CORRECT if rng.random() < thorough_accuracy else WRONG

    @checker_tool
    def verify(text: str) -> bool:
        """Ground-truth check."""
        return text == CORRECT

    return {"quick": quick, "thorough": thorough, "verify": verify}


# =============================================================================
# Orchestrators
# =============================================================================


@dataclass
class Run:
    correct: bool
    consensus: float = 1.0


def run_langchain_retry(tools: Dict[str, Any]) -> Run:
    """Deterministic AgentExecutor: one source, retries on tool *error* only."""
    ex = build_langchain_agent(tools, preferred="source", max_iterations=3)
    out = ex.invoke({"input": QUERY})
    return Run(correct=(out.get("output") == CORRECT))


def run_langchain_reflexive_single(tools: Dict[str, Any]) -> Run:
    """Deterministic AgentExecutor with self-check, no second source to escalate to."""
    ex = build_langchain_reflexive(tools, preferred="source", fallback="source",
                                   checker="verify")
    out = ex.invoke({"input": QUERY})
    return Run(correct=(out.get("output") == CORRECT))


def run_bayesian_chain(tools: Dict[str, Any], seed: int, regime: str = "smc",
                       particles: int = DEFAULT_PARTICLES,
                       max_steps: int = DEFAULT_STEPS) -> Run:
    agent = Agent(list(tools.values()), seed=seed, particles=particles,
                  max_steps=max_steps)
    r = agent.run(QUERY, regime=regime)
    return Run(correct=(r.answer == CORRECT), consensus=r.consensus)


def run_langgraph_naive_run(tools: Dict[str, Any]) -> Run:
    graph = build_langgraph_naive(tools, preferred="quick")
    out = graph.invoke({"query": QUERY})
    return Run(correct=(out.get("answer") == CORRECT))


def run_langgraph_engineered_run(tools: Dict[str, Any]) -> Run:
    graph = build_langgraph_engineered(tools, preferred="quick", fallback="thorough",
                                       checker="verify")
    try:
        out = graph.invoke({"query": QUERY})
        return Run(correct=(out.get("answer") == CORRECT))
    except Exception:  # noqa: BLE001 - escalation target failing counts as a loss here
        return Run(correct=False)


def run_bayesian_graph(tools: Dict[str, Any], seed: int, regime: str = "smc",
                       particles: int = DEFAULT_PARTICLES,
                       max_steps: int = DEFAULT_STEPS) -> Run:
    agent = Agent(list(tools.values()), seed=seed, particles=particles,
                  max_steps=max_steps)
    r = agent.run(QUERY, regime=regime)
    return Run(correct=(r.answer == CORRECT), consensus=r.consensus)


# =============================================================================
# Experiment 1 & 2 -- source accuracy sweeps
# =============================================================================


def sweep_chain(trials: int = 200, accuracies: List[float] = ACCURACIES) -> List[Dict[str, Any]]:
    """CHAIN track: our orchestrator vs deterministic LangChain, by source accuracy.

    The declared prior tracks the true accuracy (an honest orchestrator), so
    this isolates the effect of the inference regime, not of a miscalibrated
    belief -- that question is answered separately by the ablation grids.
    """
    rows = []
    for acc in accuracies:
        retry_wins = reflexive_wins = 0
        smc_wins = 0

        for seed in range(trials):
            tools = chain_tools(acc, seed)
            retry_wins += run_langchain_retry(tools).correct
            tools2 = chain_tools(acc, seed)
            reflexive_wins += run_langchain_reflexive_single(tools2).correct
            tools3 = chain_tools(acc, seed)
            smc_wins += run_bayesian_chain(tools3, seed).correct

        rows.append({
            "source_accuracy": acc,
            "LangChain (retry)": Proportion(retry_wins, trials),
            "LangChain (reflexive)": Proportion(reflexive_wins, trials),
            "BayesianRAG-Chain (SMC)": Proportion(smc_wins, trials),
        })
    return rows


def sweep_graph(trials: int = 200, accuracies: List[float] = ACCURACIES,
                thorough_accuracy: float = FALLBACK_ACCURACY) -> List[Dict[str, Any]]:
    """GRAPH track: our orchestrator vs deterministic LangGraph, by source accuracy.

    `thorough` is fixed at `thorough_accuracy`, itself stochastic rather than
    perfect, so the engineered graph's escalation is not a guaranteed correct
    answer -- it is a better bet, exactly as in a real fallback source.
    """
    rows = []
    for acc in accuracies:
        naive_wins = engineered_wins = smc_wins = 0

        for seed in range(trials):
            tools = graph_tools(acc, seed, thorough_accuracy)
            naive_wins += run_langgraph_naive_run(tools).correct
            tools2 = graph_tools(acc, seed, thorough_accuracy)
            engineered_wins += run_langgraph_engineered_run(tools2).correct
            tools3 = graph_tools(acc, seed, thorough_accuracy)
            smc_wins += run_bayesian_graph(tools3, seed).correct

        rows.append({
            "source_accuracy": acc,
            "LangGraph (naive)": Proportion(naive_wins, trials),
            "LangGraph (engineered)": Proportion(engineered_wins, trials),
            "BayesianRAG-Graph (SMC)": Proportion(smc_wins, trials),
        })
    return rows


# =============================================================================
# Experiment 3 & 4 -- prior x accuracy ablation grids
# =============================================================================


def ablation_chain(
    trials: int = 120,
    accuracies: List[float] = GRID_ACCURACIES,
    priors: List[float] = GRID_PRIORS,
) -> List[Dict[str, Any]]:
    """CHAIN orchestrator: accuracy achieved for every (true accuracy, prior) pair."""
    rows = []
    for acc in accuracies:
        for prior in priors:
            wins = 0
            consensus_sum = 0.0
            for seed in range(trials):
                tools = chain_tools(acc, seed, declared_prior=prior)
                r = run_bayesian_chain(tools, seed)
                wins += r.correct
                consensus_sum += r.consensus
            rows.append({
                "true_accuracy": acc,
                "declared_prior": prior,
                "accuracy": Proportion(wins, trials),
                "mean_consensus": round(consensus_sum / trials, 4),
            })
    return rows


def ablation_graph(
    trials: int = 120,
    accuracies: List[float] = GRID_ACCURACIES,
    priors: List[float] = GRID_PRIORS,
    thorough_accuracy: float = FALLBACK_ACCURACY,
) -> List[Dict[str, Any]]:
    """GRAPH orchestrator: accuracy achieved for every (true accuracy, prior) pair.

    The prior is declared on `quick` only; `thorough` keeps an honest,
    uninformative-appeal prior throughout, so the grid isolates what
    misjudging the *convenient* source costs -- the failure mode the graph
    track is built around.
    """
    rows = []
    for acc in accuracies:
        for prior in priors:
            wins = 0
            consensus_sum = 0.0
            for seed in range(trials):
                tools = graph_tools(acc, seed, thorough_accuracy, declared_prior=prior)
                r = run_bayesian_graph(tools, seed)
                wins += r.correct
                consensus_sum += r.consensus
            rows.append({
                "true_accuracy": acc,
                "declared_prior": prior,
                "accuracy": Proportion(wins, trials),
                "mean_consensus": round(consensus_sum / trials, 4),
            })
    return rows


# =============================================================================
# Reporting
# =============================================================================


def _fmt_sweep_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    header = f"  {'accuracy':>9s}" + "".join(f"  {c:>28s}" for c in cols)
    lines = [header, "  " + "-" * (11 + 30 * len(cols))]
    for row in rows:
        line = f"  {row['source_accuracy']:>9.2f}"
        for c in cols:
            line += f"  {row[c].format():>28s}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_grid(rows: List[Dict[str, Any]], accuracies: List[float],
             priors: List[float], field_name: str = "accuracy") -> str:
    by_key = {(r["true_accuracy"], r["declared_prior"]): r for r in rows}
    header = f"  {'true acc \\\\ prior':>18s}" + "".join(f"  {p:>9.2f}" for p in priors)
    lines = [header, "  " + "-" * (20 + 11 * len(priors))]
    for acc in accuracies:
        line = f"  {acc:>18.2f}"
        for p in priors:
            cell = by_key[(acc, p)]
            if field_name == "accuracy":
                line += f"  {cell['accuracy'].rate:>9.3f}"
            else:
                line += f"  {cell['mean_consensus']:>9.3f}"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class Results:
    """Everything computed for one report, so text/figures/JSON share one run."""

    chain_rows: List[Dict[str, Any]]
    graph_rows: List[Dict[str, Any]]
    chain_grid: List[Dict[str, Any]]
    graph_grid: List[Dict[str, Any]]
    trials_sweep: int
    trials_grid: int


def compute(trials_sweep: int, trials_grid: int) -> Results:
    """Run every experiment exactly once."""
    return Results(
        chain_rows=sweep_chain(trials_sweep),
        graph_rows=sweep_graph(trials_sweep),
        chain_grid=ablation_chain(trials_grid),
        graph_grid=ablation_graph(trials_grid),
        trials_sweep=trials_sweep,
        trials_grid=trials_grid,
    )


def print_report(results: "Results") -> None:
    trials_sweep, trials_grid = results.trials_sweep, results.trials_grid

    print("=" * 80)
    print("EXPERIMENT 1 -- Source accuracy: BayesianRAG-Chain vs deterministic LangChain")
    print("=" * 80)
    print("Single source, checker available, no second tool to escalate to.")
    print(f"Declared prior tracks true accuracy (honest orchestrator). {trials_sweep} seeds/row.\n")
    print(_fmt_sweep_table(
        results.chain_rows,
        ["LangChain (retry)", "LangChain (reflexive)", "BayesianRAG-Chain (SMC)"],
    ))

    print("\n\n" + "=" * 80)
    print("EXPERIMENT 2 -- Source accuracy: BayesianRAG-Graph vs deterministic LangGraph")
    print("=" * 80)
    print("Convenient source swept; authoritative fallback fixed at 90% accurate.")
    print(f"Declared prior tracks true accuracy. {trials_sweep} seeds/row.\n")
    print(_fmt_sweep_table(
        results.graph_rows,
        ["LangGraph (naive)", "LangGraph (engineered)", "BayesianRAG-Graph (SMC)"],
    ))

    print("\n\n" + "=" * 80)
    print("EXPERIMENT 3 -- Ablation: BayesianRAG-Chain, true accuracy x declared prior")
    print("=" * 80)
    print(f"Accuracy achieved. {trials_grid} seeds/cell.\n")
    print(_fmt_grid(results.chain_grid, GRID_ACCURACIES, GRID_PRIORS, "accuracy"))
    print("\nMean posterior consensus (same grid):\n")
    print(_fmt_grid(results.chain_grid, GRID_ACCURACIES, GRID_PRIORS, "consensus"))

    print("\n\n" + "=" * 80)
    print("EXPERIMENT 4 -- Ablation: BayesianRAG-Graph, true accuracy x declared prior")
    print("=" * 80)
    print(f"Prior declared on the convenient source only. {trials_grid} seeds/cell.\n")
    print(_fmt_grid(results.graph_grid, GRID_ACCURACIES, GRID_PRIORS, "accuracy"))
    print("\nMean posterior consensus (same grid):\n")
    print(_fmt_grid(results.graph_grid, GRID_ACCURACIES, GRID_PRIORS, "consensus"))

    print("\n\nReading the grids: the diagonal-ish region where declared prior")
    print("roughly matches true accuracy is where the orchestrator is honest.")
    print("Off-diagonal cells show what a wrong prior costs, holding the")
    print("environment fixed -- compare a row across columns, not the whole grid.\n")


def make_figures(results: "Results", outdir: str) -> List[str]:
    """Render the sweeps and grids as PNG figures. Requires matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(outdir, exist_ok=True)
    paths = []

    chain_rows, graph_rows = results.chain_rows, results.graph_rows

    for rows, cols, title, fname in [
        (chain_rows, ["LangChain (retry)", "LangChain (reflexive)", "BayesianRAG-Chain (SMC)"],
         "Chain track: source accuracy vs task accuracy", "sweep_chain.png"),
        (graph_rows, ["LangGraph (naive)", "LangGraph (engineered)", "BayesianRAG-Graph (SMC)"],
         "Graph track: source accuracy vs task accuracy", "sweep_graph.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x = [r["source_accuracy"] for r in rows]
        for c in cols:
            y = [r[c].rate for r in rows]
            lo = [r[c].interval[0] for r in rows]
            hi = [r[c].interval[1] for r in rows]
            ax.plot(x, y, marker="o", label=c)
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1,
                label="y = x (matches source accuracy)")
        ax.set_xlabel("True source accuracy")
        ax.set_ylabel("Task accuracy (95% CI band)")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # --- ablation heatmaps ----------------------------------------------
    chain_grid, graph_grid = results.chain_grid, results.graph_grid

    for grid, title, fname in [
        (chain_grid, "Chain orchestrator: accuracy by true accuracy x declared prior",
         "ablation_chain.png"),
        (graph_grid, "Graph orchestrator: accuracy by true accuracy x declared prior",
         "ablation_graph.png"),
    ]:
        by_key = {(r["true_accuracy"], r["declared_prior"]): r for r in grid}
        mat = np.array([
            [by_key[(a, p)]["accuracy"].rate for p in GRID_PRIORS]
            for a in GRID_ACCURACIES
        ])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xticks(range(len(GRID_PRIORS)))
        ax.set_xticklabels([f"{p:.1f}" for p in GRID_PRIORS])
        ax.set_yticks(range(len(GRID_ACCURACIES)))
        ax.set_yticklabels([f"{a:.1f}" for a in GRID_ACCURACIES])
        ax.set_xlabel("Declared prior")
        ax.set_ylabel("True source accuracy")
        ax.set_title(title, fontsize=9)
        for i in range(len(GRID_ACCURACIES)):
            for j in range(len(GRID_PRIORS)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="task accuracy")
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def to_json(results: "Results") -> Dict[str, Any]:
    """Serialize a computed Results for storage or transmission."""

    def _ser(p: Proportion) -> Dict[str, Any]:
        return {"rate": round(p.rate, 4), "ci_lower": round(p.interval[0], 4),
                "ci_upper": round(p.interval[1], 4), "trials": p.trials}

    return {
        "trials_sweep": results.trials_sweep,
        "trials_grid": results.trials_grid,
        "sweep_chain": [
            {**{k: v for k, v in r.items() if not isinstance(v, Proportion)},
             **{k: _ser(v) for k, v in r.items() if isinstance(v, Proportion)}}
            for r in results.chain_rows
        ],
        "sweep_graph": [
            {**{k: v for k, v in r.items() if not isinstance(v, Proportion)},
             **{k: _ser(v) for k, v in r.items() if isinstance(v, Proportion)}}
            for r in results.graph_rows
        ],
        "ablation_chain": [
            {**{k: v for k, v in r.items() if k != "accuracy"}, "accuracy": _ser(r["accuracy"])}
            for r in results.chain_grid
        ],
        "ablation_graph": [
            {**{k: v for k, v in r.items() if k != "accuracy"}, "accuracy": _ser(r["accuracy"])}
            for r in results.graph_grid
        ],
    }


def run_all(trials_sweep: int, trials_grid: int) -> Dict[str, Any]:
    """Convenience wrapper: compute once, return the serialized form."""
    return to_json(compute(trials_sweep, trials_grid))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-level results")
    parser.add_argument("--trials-sweep", type=int, default=200)
    parser.add_argument("--trials-grid", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--outdir", type=str, default="paper_figures")
    args = parser.parse_args()

    results = compute(args.trials_sweep, args.trials_grid)

    if args.json:
        print(json.dumps(to_json(results), indent=2))
    else:
        print_report(results)

    if args.figures:
        paths = make_figures(results, args.outdir)
        print("\nFigures written:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
