"""FastAPI surface over the orchestration runtime.

Endpoints are regime-parameterized rather than regime-specific: the same
scenario can be run greedily or under a particle filter by changing one field,
which is what makes the comparison in `/benchmarks/run` meaningful.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

from bayesian_rag import __version__
from bayesian_rag.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ContrastOut,
    RegimeStats,
    RunRequest,
    RunResponse,
    SessionRequest,
    SessionResponse,
    StepOut,
)
from bayesian_rag.core.inference_regimes import REGIME_SUMMARY, Regime
from bayesian_rag.scenarios.registry import (
    SCENARIO_QUERIES,
    SCENARIO_SUCCESS_MARKERS,
    SCENARIOS,
)
from bayesian_rag.tools.mock_tools import default_toolset
from bayesian_rag.utils.statistics import Comparison, Proportion

app = FastAPI(
    title="BayesianRAG Orchestration API",
    version=__version__,
    description=(
        "Probabilistic and Bayesian orchestration for tool-augmented agents. "
        "Every scenario runs under any of three inference regimes over one "
        "shared generative model."
    ),
)


def _resolve(scenario: str) -> None:
    if scenario not in SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIOS)}",
        )


def _query_for(scenario: str, given: str | None) -> str:
    return given or SCENARIO_QUERIES.get(scenario, "")


def _is_success(scenario: str, answer: str) -> bool:
    markers = SCENARIO_SUCCESS_MARKERS.get(scenario, [])
    return any(m in answer for m in markers) if markers else False


@app.get("/api/v1/health")
def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/info")
def info() -> Dict[str, Any]:
    """Package version and a summary of the available inference regimes."""
    return {
        "name": "bayesian-rag",
        "version": __version__,
        "description": "Bayesian orchestration for tool-augmented agents",
        "regimes": {r.value: REGIME_SUMMARY[r] for r in Regime},
    }


@app.get("/api/v1/tools")
def tools() -> Dict[str, List[str]]:
    """Names of the built-in reference tools."""
    return {"tools": sorted(default_toolset().keys())}


@app.get("/api/v1/scenarios")
def scenarios() -> Dict[str, Any]:
    """Registered scenarios with their default queries and success markers."""
    return {
        "scenarios": [
            {
                "name": name,
                "default_query": SCENARIO_QUERIES.get(name, ""),
                "success_markers": SCENARIO_SUCCESS_MARKERS.get(name, []),
            }
            for name in sorted(SCENARIOS)
        ]
    }


@app.post("/api/v1/agents/run", response_model=RunResponse)
def run_agent(req: RunRequest) -> RunResponse:
    """Run one scenario once and return the answer plus the posterior."""
    _resolve(req.scenario)
    query = _query_for(req.scenario, req.query)

    agent = SCENARIOS[req.scenario]()
    answer, trace, posterior = agent.run(
        query, regime=req.regime, n_particles=req.n_particles
    )

    return RunResponse(
        scenario=req.scenario,
        regime=req.regime,
        query=query,
        answer=answer,
        steps=[
            StepOut(t=s.t, tool=s.tool, ok=s.ok, output=s.output) for s in trace.steps
        ],
        posterior=posterior,
    )


@app.post("/api/v1/benchmarks/run", response_model=BenchmarkResponse)
def run_benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """Repeat a scenario across seeds for each regime and compare outcomes.

    Seeds vary per trial so the comparison reflects behaviour across the
    randomness in proposal and selection, not one lucky trajectory.
    """
    _resolve(req.scenario)
    query = _query_for(req.scenario, req.query)

    results: List[RegimeStats] = []
    proportions: Dict[str, Proportion] = {}
    for regime in req.regimes:
        wins = 0
        steps: List[int] = []
        times: List[float] = []
        usage: Counter = Counter()

        for seed in range(req.trials):
            agent = SCENARIOS[req.scenario](seed=seed)
            answer, trace, posterior = agent.run(query, regime=regime)
            if _is_success(req.scenario, answer):
                wins += 1
            steps.append(len(trace.steps))
            times.append(posterior.get("time_used_s", 0.0) * 1000.0)
            usage.update(s.tool for s in trace.steps)

        prop = Proportion(wins, req.trials, req.confidence)
        proportions[regime] = prop
        lo, hi = prop.interval
        results.append(
            RegimeStats(
                regime=regime,
                trials=req.trials,
                successes=wins,
                success_rate=round(prop.rate, 4),
                ci_lower=round(lo, 4),
                ci_upper=round(hi, 4),
                avg_steps=round(statistics.mean(steps), 3) if steps else 0.0,
                avg_time_ms=round(statistics.mean(times), 3) if times else 0.0,
                tool_usage=dict(usage),
            )
        )

    # Point estimates alone cannot tell a caller whether two regimes differ.
    contrasts: List[ContrastOut] = []
    for a, b in (("smc", "greedy"), ("smc", "forward"), ("forward", "greedy")):
        if a in proportions and b in proportions:
            cmp_ = Comparison(a, b, proportions[a], proportions[b], req.confidence)
            lo, hi = cmp_.interval
            contrasts.append(
                ContrastOut(
                    contrast=f"{a} vs {b}",
                    difference=round(cmp_.difference, 4),
                    ci_lower=round(lo, 4),
                    ci_upper=round(hi, 4),
                    p_value=round(cmp_.p_value, 6),
                    significant=cmp_.significant,
                )
            )

    return BenchmarkResponse(
        scenario=req.scenario,
        query=query,
        confidence_level=req.confidence,
        results=results,
        contrasts=contrasts,
    )


@app.post("/api/v1/agents/session", response_model=SessionResponse)
def run_session(req: SessionRequest) -> SessionResponse:
    """Run repeated episodes, optionally carrying reliability beliefs forward."""
    _resolve(req.scenario)
    query = _query_for(req.scenario, req.query)

    agent = SCENARIOS[req.scenario]()
    episodes = agent.run_session(
        [query] * req.episodes,
        regime=req.regime,
        carry_reliability=req.carry_reliability,
    )

    return SessionResponse(
        scenario=req.scenario,
        regime=req.regime,
        carry_reliability=req.carry_reliability,
        episodes=episodes,
    )
