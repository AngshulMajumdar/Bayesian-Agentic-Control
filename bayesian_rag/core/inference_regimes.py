"""The inference ladder: three regimes over one generative model.

All three share the model

    p(a_{1:T}, o_{1:T}) = prod_t p(a_t | h_t) p(o_t | a_t, h_t)

and differ only in how they infer over it:

    greedy   -- one hypothesis, argmax proposal, likelihoods ignored
    forward  -- many hypotheses, sampled proposals, likelihoods still ignored
    smc      -- many hypotheses, reweighted by p(o_t | a_t, h_t), resampled on
                ESS collapse

Because the regime is a parameter rather than a code path, the same tools,
proposer, and scorer run unchanged under all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from bayesian_rag.core.smc_runner import SMCConfig


class Regime(str, Enum):
    """Identifier for an inference regime."""

    GREEDY = "greedy"
    FORWARD = "forward"
    SMC = "smc"

    @classmethod
    def parse(cls, value: "str | Regime") -> "Regime":
        """Accept aliases used across the earlier prototypes."""
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower().replace("-", "_")
        aliases = {
            "greedy": cls.GREEDY,
            "dg": cls.GREEDY,
            "deterministic": cls.GREEDY,
            "deterministic_greedy": cls.GREEDY,
            "forward": cls.FORWARD,
            "pf": cls.FORWARD,
            "probabilistic": cls.FORWARD,
            "probabilistic_forward": cls.FORWARD,
            "smc": cls.SMC,
            "bayesian": cls.SMC,
            "bayesian_smc": cls.SMC,
        }
        if key not in aliases:
            raise ValueError(
                f"Unknown regime {value!r}. Expected one of: greedy, forward, smc."
            )
        return aliases[key]


@dataclass(frozen=True)
class RegimePlan:
    """How a regime parameterizes the shared SMC loop."""

    regime: Regime
    n_particles: int
    reweight: bool
    explore: bool
    description: str

    @property
    def maintains_posterior(self) -> bool:
        """True when the regime tracks p(a_{1:t} | o_{1:t}) rather than a prior."""
        return self.reweight and self.n_particles > 1


def plan_for(
    regime: "str | Regime",
    cfg: Optional[SMCConfig] = None,
    n_particles: Optional[int] = None,
) -> RegimePlan:
    """Resolve a regime into concrete runner settings.

    Greedy collapses to a single hypothesis and discards likelihood information.
    Forward sampling keeps the hypotheses but still never reweights them, so its
    particle cloud stays a sample from the prior. Only SMC folds the observation
    likelihood back into the weights, which is what lets evidence revive a branch
    that an early step had disfavoured.
    """
    r = Regime.parse(regime)
    base = cfg.n_particles if cfg is not None else 16
    requested = n_particles if n_particles is not None else base

    if r is Regime.GREEDY:
        return RegimePlan(
            regime=r,
            n_particles=1,
            reweight=False,
            explore=False,
            description="Single trajectory via argmax proposal; no recovery once committed.",
        )
    if r is Regime.FORWARD:
        return RegimePlan(
            regime=r,
            n_particles=max(2, requested),
            reweight=False,
            explore=True,
            description="Sampled trajectories from the prior; retries are prior draws, not posterior updates.",
        )
    return RegimePlan(
        regime=r,
        n_particles=max(2, requested),
        reweight=True,
        explore=True,
        description="Particle filter over trajectories; observation likelihood reweights competing branches.",
    )


REGIME_SUMMARY = {
    Regime.GREEDY: {
        "name": "Deterministic Greedy",
        "cost": "lowest",
        "robustness": "low",
        "recovers_from_early_error": False,
        "use_when": "Prototyping, cheap tools, unambiguous routing.",
    },
    Regime.FORWARD: {
        "name": "Probabilistic Forward",
        "cost": "moderate",
        "robustness": "medium",
        "recovers_from_early_error": False,
        "use_when": "Exploration under a retry budget, no reliable likelihood signal.",
    },
    Regime.SMC: {
        "name": "Bayesian SMC",
        "cost": "highest",
        "robustness": "high",
        "recovers_from_early_error": True,
        "use_when": "Conflicting sources, expensive mistakes, evidence arrives mid-trajectory.",
    },
}
