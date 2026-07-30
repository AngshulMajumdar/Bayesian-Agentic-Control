"""The user-facing agent: run a query under any regime, get an answer and a posterior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bayesian_rag.bayesian.reliability_model import ToolReliabilityState
from bayesian_rag.core.inference_regimes import Regime, plan_for
from bayesian_rag.core.particle import EpisodeTrace, Particle, StepRecord, Tool
from bayesian_rag.core.smc_runner import (
    Proposer,
    Scorer,
    Selector,
    SMCConfig,
    SMCResult,
    SMCRunner,
)


@dataclass
class AgentConfig:
    """Inference budget plus the default regime for this agent."""

    smc: SMCConfig = field(default_factory=SMCConfig)
    regime: str = "smc"
    top_k_particles: int = 3
    # How much of a previous episode's evidence to carry forward. 1.0 keeps it
    # in full; lower values discount stale evidence against fresh observations.
    carry_over_factor: float = 1.0


class BayesianAgent:
    """Backend-agnostic orchestrator over a tool set.

    The agent owns the tools and the three pluggable components (proposer,
    scorer, selector). Which regime runs is decided per call, so the same agent
    can be benchmarked across the ladder without rebuilding anything.
    """

    def __init__(
        self,
        tools: Dict[str, Tool],
        proposer: Proposer,
        scorer: Scorer,
        selector: Selector,
        cfg: Optional[AgentConfig] = None,
    ):
        self.tools = tools
        self.proposer = proposer
        self.scorer = scorer
        self.selector = selector
        self.cfg = cfg or AgentConfig()

    def run(
        self,
        query: str,
        regime: Optional[str] = None,
        n_particles: Optional[int] = None,
        init_reliability: Optional[ToolReliabilityState] = None,
        _return_beliefs: bool = False,
    ) -> Tuple[str, EpisodeTrace, Dict[str, Any]]:
        """Answer a query.

        Returns:
            (answer, trace, posterior) -- the MAP answer, the winning
            trajectory step by step, and a summary of the particle posterior
            including learned tool reliabilities.
        """
        plan = plan_for(regime or self.cfg.regime, self.cfg.smc, n_particles)
        runner = SMCRunner(tools=self.tools, cfg=self.cfg.smc)

        result = runner.run(
            query=query,
            proposer=self.proposer,
            scorer=self.scorer,
            selector=self.selector,
            n_particles=plan.n_particles,
            reweight=plan.reweight,
            explore=plan.explore,
            init_reliability=init_reliability,
        )

        trace = EpisodeTrace(query=query)
        posterior = summarize_posterior(result, plan.regime, self.cfg.top_k_particles)

        best = result.best_particle()
        if best is None:
            trace.final_answer = "No trajectory survived inference."
            trace.meta = {"regime": plan.regime.value, "steps_run": result.steps_run}
            return trace.final_answer, trace, posterior

        for t, (action, obs) in enumerate(zip(best.actions, best.observations)):
            trace.add_step(
                StepRecord(
                    t=t,
                    tool=action.tool_name,
                    ok=obs.ok,
                    output=obs.output,
                    tool_reliability_means=best.reliability.snapshot_means(),
                )
            )

        answer = best.final_answer()
        if answer is None:
            last = best.last_observation()
            answer = last.short() if last else "No observations produced."

        trace.final_answer = answer
        trace.meta = {
            "regime": plan.regime.value,
            "n_particles": plan.n_particles,
            "steps_run": result.steps_run,
            "time_used_s": round(result.time_used_s, 4),
            "resamples": result.resamples,
        }
        if _return_beliefs:
            posterior["_beliefs"] = best.reliability
        return answer, trace, posterior

    def run_session(
        self,
        queries: List[str],
        regime: Optional[str] = None,
        carry_reliability: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run several episodes, optionally carrying beliefs between them.

        Carrying reliability forward is what turns repeated use into learning:
        episode two starts already knowing which tools earned trust in episode one.
        """
        results: List[Dict[str, Any]] = []
        beliefs: Optional[ToolReliabilityState] = None

        for i, q in enumerate(queries):
            answer, trace, posterior = self.run(
                q, regime=regime, init_reliability=beliefs, _return_beliefs=True
            )
            learned: Optional[ToolReliabilityState] = posterior.pop("_beliefs", None)
            results.append(
                {
                    "episode": i,
                    "query": q,
                    "answer": answer,
                    "tools_used": [s.tool for s in trace.steps],
                    "n_steps": len(trace.steps),
                    "tool_reliability_means": posterior.get("tool_reliability_means", {}),
                    "evidence_carried_in": round(beliefs.total_evidence(), 2) if beliefs else 0.0,
                }
            )
            if carry_reliability and learned is not None:
                # Transport the posterior itself. Rebuilding it from means would
                # preserve the estimate while throwing away its confidence.
                beliefs = learned.tempered(self.cfg.carry_over_factor)
        return results


def marginalize_answers(result: SMCResult) -> List[Dict[str, Any]]:
    """Pool posterior mass by answer rather than by trajectory.

    Trajectory mass understates agreement: particles that reach the same
    conclusion by different routes are in complete agreement, yet each holds
    only a small share. Pooling by answer gives the quantity that actually
    matters -- how much of the posterior backs each distinct claim -- and makes
    genuine disagreement distinguishable from mere path diversity.
    """
    if not result.particles:
        return []

    weights = result.weights()
    pooled: Dict[str, Dict[str, Any]] = {}

    for particle, w in zip(result.particles, weights):
        answer = particle.final_answer()
        if not answer:
            continue
        entry = pooled.setdefault(
            answer, {"answer": answer, "mass": 0.0, "status": particle.answer_status()}
        )
        entry["mass"] += w
        # A claim that was validated anywhere is reported as validated.
        if particle.answer_status() == "validated":
            entry["status"] = "validated"

    ranked = sorted(pooled.values(), key=lambda e: e["mass"], reverse=True)
    for e in ranked:
        e["mass"] = round(e["mass"], 6)
    return ranked


def summarize_posterior(
    result: SMCResult, regime: Regime, top_k: int = 3
) -> Dict[str, Any]:
    """Condense the particle set into a reportable posterior summary."""
    if not result.particles:
        return {
            "regime": regime.value,
            "n_particles": 0,
            "steps_run": result.steps_run,
            "time_used_s": round(result.time_used_s, 4),
            "ess_history": [],
            "top_particles": [],
            "tool_reliability_means": {},
        }

    weights = result.weights()
    order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:top_k]

    top: List[Dict[str, Any]] = []
    for i in order:
        p = result.particles[i]
        top.append(
            {
                "mass": round(weights[i], 6),
                "tools": p.tools_used(),
                "answer": p.final_answer(),
                "status": p.answer_status(),
                "tool_reliability_means": {
                    k: round(v, 4) for k, v in p.reliability.snapshot_means().items()
                },
            }
        )

    best = result.best_particle()
    answers = marginalize_answers(result)
    return {
        "regime": regime.value,
        "answers": answers,
        "consensus": answers[0]["mass"] if answers else 0.0,
        "answer_status": best.answer_status() if best else "unverified",
        "n_particles": len(result.particles),
        "steps_run": result.steps_run,
        "time_used_s": round(result.time_used_s, 4),
        "resamples": result.resamples,
        "ess_history": [round(x, 4) for x in result.ess_history],
        "cache": result.cache_stats,
        "top_particles": top,
        "tool_reliability_means": {
            k: round(v, 4) for k, v in (best.reliability.snapshot_means() if best else {}).items()
        },
    }
