"""Interval estimation and hypothesis tests for benchmark results.

A success rate quoted from 30 trials without an interval is not a reproducible
result: a reader cannot tell whether 0.93 and 0.87 differ. Everything here is
stdlib-only, so reporting intervals never depends on scipy being installed.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Two-sided normal quantiles for common confidence levels.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}


def z_for(confidence: float) -> float:
    """Normal quantile for a two-sided interval at `confidence`."""
    if confidence in _Z:
        return _Z[confidence]
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    return _inverse_normal_cdf(0.5 + confidence / 2.0)


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation to the normal quantile function.

    Accurate to roughly 1e-9 over the range that matters here, which is far
    beyond what a benchmark interval needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    plow, phigh = 0.02425, 1 - 0.02425

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass(frozen=True)
class Proportion:
    """A success rate with an interval, and enough context to reproduce it."""

    successes: int
    trials: int
    confidence: float = 0.95

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def interval(self) -> Tuple[float, float]:
        """Wilson score interval.

        Preferred over the normal approximation because it stays inside [0, 1]
        and remains sensible at rates near 0 or 1 -- exactly where these
        benchmarks live, since greedy scores 0.00 and SMC scores above 0.90.
        The normal interval would report impossible bounds at both ends.
        """
        n = self.trials
        if n == 0:
            return (0.0, 0.0)
        z = z_for(self.confidence)
        p = self.rate
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    @property
    def margin(self) -> float:
        lo, hi = self.interval
        return (hi - lo) / 2.0

    def format(self, places: int = 3) -> str:
        lo, hi = self.interval
        return f"{self.rate:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True)
class Comparison:
    """Difference between two proportions, with a significance verdict."""

    label_a: str
    label_b: str
    a: Proportion
    b: Proportion
    confidence: float = 0.95

    @property
    def difference(self) -> float:
        return self.a.rate - self.b.rate

    @property
    def interval(self) -> Tuple[float, float]:
        """Newcombe's interval for a difference of proportions.

        Built from the two Wilson intervals, so it inherits their good
        behaviour when either rate sits at 0 or 1.
        """
        l1, u1 = self.a.interval
        l2, u2 = self.b.interval
        d = self.difference
        lower = d - math.sqrt((self.a.rate - l1) ** 2 + (u2 - self.b.rate) ** 2)
        upper = d + math.sqrt((u1 - self.a.rate) ** 2 + (self.b.rate - l2) ** 2)
        return (max(-1.0, lower), min(1.0, upper))

    @property
    def significant(self) -> bool:
        """True when the interval for the difference excludes zero."""
        lo, hi = self.interval
        return lo > 0 or hi < 0

    @property
    def p_value(self) -> float:
        """Two-sided p-value from a pooled two-proportion z-test."""
        n1, n2 = self.a.trials, self.b.trials
        if n1 == 0 or n2 == 0:
            return 1.0
        pooled = (self.a.successes + self.b.successes) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        if se == 0:
            return 0.0 if self.difference != 0 else 1.0
        z = self.difference / se
        return 2.0 * (1.0 - _normal_cdf(abs(z)))

    def format(self) -> str:
        lo, hi = self.interval
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.label_a} - {self.label_b} = {self.difference:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}], p={self.p_value:.2g} ({verdict})"
        )

    def __str__(self) -> str:
        return self.format()


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class Summary:
    """Mean of a continuous measurement with an interval."""

    values: Sequence[float]
    confidence: float = 0.95

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def interval(self) -> Tuple[float, float]:
        n = len(self.values)
        if n < 2:
            return (self.mean, self.mean)
        margin = z_for(self.confidence) * self.stdev / math.sqrt(n)
        return (self.mean - margin, self.mean + margin)

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    def format(self, places: int = 2) -> str:
        lo, hi = self.interval
        return f"{self.mean:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"

    def __str__(self) -> str:
        return self.format()


def required_trials(
    expected_rate: float, margin: float = 0.05, confidence: float = 0.95
) -> int:
    """Trials needed to estimate a rate to within +/- `margin`.

    Useful for choosing a benchmark size deliberately rather than defaulting to
    a round number and hoping it is enough.
    """
    if not 0.0 <= expected_rate <= 1.0:
        raise ValueError("expected_rate must lie in [0, 1]")
    if margin <= 0:
        raise ValueError("margin must be > 0")
    z = z_for(confidence)
    p = min(max(expected_rate, 0.01), 0.99)
    return math.ceil(z * z * p * (1 - p) / (margin * margin))
