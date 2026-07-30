"""Beta-Bernoulli conjugate model for online tool reliability learning.

Each tool j has a latent success probability theta_j with a Beta(alpha, beta)
posterior. After n trials with k successes:

    posterior      = Beta(alpha_0 + k, beta_0 + n - k)
    posterior mean = alpha / (alpha + beta)

Posterior means drive tool selection; posterior samples enable Thompson sampling.
Pure stdlib -- no scipy/numpy dependency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BetaBelief:
    """Beta posterior over a single tool's success probability."""

    alpha: float = 1.0
    beta: float = 1.0

    def mean(self) -> float:
        """Posterior mean E[theta]."""
        denom = self.alpha + self.beta
        return self.alpha / denom if denom > 0 else 0.5

    def variance(self) -> float:
        """Posterior variance Var[theta]."""
        s = self.alpha + self.beta
        if s <= 0:
            return 0.0
        return (self.alpha * self.beta) / (s * s * (s + 1.0))

    def n_observations(self) -> float:
        """Pseudo-count of absorbed observations (excludes the prior)."""
        return max(0.0, (self.alpha - 1.0) + (self.beta - 1.0))

    def update(self, success: bool, weight: float = 1.0) -> None:
        """Conjugate update after one (possibly weighted) observation."""
        if weight <= 0:
            return
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    def sample(self, rng: Optional[random.Random] = None) -> float:
        """Draw theta ~ Beta(alpha, beta) for Thompson sampling."""
        r = rng or random
        return r.betavariate(max(self.alpha, 1e-6), max(self.beta, 1e-6))

    def copy(self) -> "BetaBelief":
        """An independent copy of this belief."""
        return BetaBelief(alpha=self.alpha, beta=self.beta)


@dataclass
class ToolReliabilityState:
    """Per-particle reliability beliefs across a tool set."""

    beliefs: Dict[str, BetaBelief] = field(default_factory=dict)

    def get(self, tool_name: str) -> BetaBelief:
        """Return the belief for a tool, creating an uninformative prior if new."""
        if tool_name not in self.beliefs:
            self.beliefs[tool_name] = BetaBelief()
        return self.beliefs[tool_name]

    def update(self, tool_name: str, success: bool, weight: float = 1.0) -> None:
        """Absorb one observation for a tool."""
        self.get(tool_name).update(success=success, weight=weight)

    def update_graded(self, tool_name: str, score: float, weight: float = 1.0) -> None:
        """Absorb a graded score in [0, 1] as fractional success/failure mass."""
        score = min(max(float(score), 0.0), 1.0)
        belief = self.get(tool_name)
        belief.alpha += weight * score
        belief.beta += weight * (1.0 - score)

    def snapshot_means(self) -> Dict[str, float]:
        """Posterior means for every tool seen so far."""
        return {name: b.mean() for name, b in self.beliefs.items()}

    def routing_distribution(self, tools: Optional[List[str]] = None) -> Dict[str, float]:
        """Normalized distribution over tools proportional to posterior means."""
        names = tools if tools is not None else list(self.beliefs.keys())
        if not names:
            return {}
        means = {n: self.get(n).mean() for n in names}
        total = sum(means.values())
        if total <= 0:
            uniform = 1.0 / len(names)
            return {n: uniform for n in names}
        return {n: m / total for n, m in means.items()}

    def sample_tool(self, tools: List[str], rng: Optional[random.Random] = None) -> str:
        """Thompson sampling: draw from each posterior, take the argmax."""
        r = rng or random
        samples = {t: self.get(t).sample(r) for t in tools}
        return max(samples, key=samples.get)

    def best_tool(self, tools: Optional[List[str]] = None) -> Optional[str]:
        """Tool with the highest posterior mean."""
        names = tools if tools is not None else list(self.beliefs.keys())
        if not names:
            return None
        return max(names, key=lambda t: self.get(t).mean())

    def rank_tools(self) -> List[Tuple[str, float]]:
        """Tools sorted by posterior mean, descending."""
        return sorted(self.snapshot_means().items(), key=lambda kv: kv[1], reverse=True)

    def copy(self) -> "ToolReliabilityState":
        """Deep-enough copy for particle resampling."""
        return ToolReliabilityState(
            beliefs={name: b.copy() for name, b in self.beliefs.items()}
        )

    def tempered(self, factor: float = 1.0, prior_strength: float = 1.0) -> "ToolReliabilityState":
        """Carry these beliefs into a new episode, optionally discounted.

        Evidence from a previous episode is usually worth less than evidence
        gathered now -- the query differs, and tools drift. `factor` scales the
        accumulated pseudo-counts: `factor=1.0` carries the posterior unchanged,
        `factor=0.5` halves the evidence behind it, `factor=0.0` discards it.

        Note that discounting also pulls each mean back toward the prior, which
        is the correct behaviour rather than a side effect: with less evidence
        the estimate should be less confident *and* less far from where it
        started. Only `factor=1.0` leaves the mean untouched.

        This deliberately transports (alpha, beta) rather than reconstructing
        them from the mean. A rebuild preserves the mean but discards the
        sample size, silently converting fifty confident observations into a
        near-uninformative prior.
        """
        if factor < 0:
            raise ValueError(f"factor must be >= 0, got {factor}")
        out = ToolReliabilityState()
        for name, b in self.beliefs.items():
            evidence_a = max(0.0, b.alpha - prior_strength)
            evidence_b = max(0.0, b.beta - prior_strength)
            out.beliefs[name] = BetaBelief(
                alpha=prior_strength + evidence_a * factor,
                beta=prior_strength + evidence_b * factor,
            )
        return out

    def total_evidence(self) -> float:
        """Pseudo-observations absorbed across all tools."""
        return sum(b.n_observations() for b in self.beliefs.values())
