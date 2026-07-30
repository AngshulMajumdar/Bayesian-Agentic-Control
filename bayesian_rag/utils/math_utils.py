"""Numerically stable weight arithmetic and resampling for the particle filter."""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from typing import Any, List, Optional

NEG_INF = -1e30


def normalize_logweights(logw: List[float]) -> List[float]:
    """Normalize log-weights so that exp() of the result sums to one."""
    if not logw:
        return []
    m = max(logw)
    if not math.isfinite(m):
        uniform = math.log(1.0 / len(logw))
        return [uniform] * len(logw)
    exps = [math.exp(w - m) for w in logw]
    z = sum(exps)
    if z <= 0 or not math.isfinite(z):
        uniform = math.log(1.0 / len(logw))
        return [uniform] * len(logw)
    return [math.log(x / z) if x > 0 else NEG_INF for x in exps]


def weights_from_logweights(logw: List[float]) -> List[float]:
    """Normalized linear-space weights."""
    return [math.exp(w) for w in normalize_logweights(logw)]


def ess_from_logweights(logw: List[float]) -> float:
    """Effective sample size: 1 / sum(w_i^2) for normalized w."""
    ws = weights_from_logweights(logw)
    denom = sum(w * w for w in ws)
    return 1.0 / denom if denom > 0 else 0.0


def systematic_resample(logw: List[float], rng: random.Random) -> List[int]:
    """Low-variance systematic resampling; returns parent indices."""
    ws = weights_from_logweights(logw)
    n = len(ws)
    if n == 0:
        return []
    start = rng.random() / n
    positions = [start + i / n for i in range(n)]
    indexes: List[int] = []
    cumulative = 0.0
    j = 0
    for i, w in enumerate(ws):
        cumulative += w
        while j < n and positions[j] <= cumulative:
            indexes.append(i)
            j += 1
    while len(indexes) < n:
        indexes.append(n - 1)
    return indexes


def safe_log(x: float, floor: float = 1e-12) -> float:
    """log with a floor, so a zero likelihood kills a particle without crashing."""
    return math.log(max(float(x), floor))


class LRUCache:
    """Bounded cache so repeated identical tool calls cost one execution."""

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Return a cached value and mark it most-recently used, or None."""
        if key not in self._data:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        """Store a value, evicting the least-recently used entry if full."""
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def stats(self) -> dict:
        """Hit/miss counts, useful for confirming memoisation is working."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "size": len(self._data),
        }
