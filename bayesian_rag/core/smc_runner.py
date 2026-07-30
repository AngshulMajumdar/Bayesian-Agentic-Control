"""Sequential Monte Carlo runner: the single inference engine.

One loop implements the whole inference ladder. The regime supplied by the
caller decides how many hypotheses survive each step:

    propose -> select -> observe -> weight -> (resample if ESS collapses)

Greedy keeps one hypothesis and ignores the weights; forward sampling keeps
several but never reweights them; SMC keeps several and reweights by the
observation likelihood.
"""

from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bayesian_rag.bayesian.reliability_model import ToolReliabilityState
from bayesian_rag.core.particle import (
    _DETERMINISTIC_KEY,
    _RNG_KEY,
    Action,
    Observation,
    Particle,
    Tool,
)
from bayesian_rag.utils.math_utils import (
    LRUCache,
    ess_from_logweights,
    normalize_logweights,
    safe_log,
    systematic_resample,
)

# --- Component contracts -----------------------------------------------------
# Proposer: (query, particles, t) -> candidate actions for this step
Proposer = Callable[[str, List[Particle], int], List[Action]]
# Selector: (query, particle, candidates, t) -> the action this particle takes
Selector = Callable[[str, Particle, List[Action], int], Action]
# Scorer: (query, particle, action, obs, t) -> (likelihood, success_label, extra_updates)
#   likelihood     : p(o_t | a_t, h_t), in (0, 1]
#   success_label  : True/False to update this tool's belief, or None to skip
#   extra_updates  : optional [(tool_name, success, weight), ...] for retroactive
#                    credit assignment (e.g. a checker validating an earlier tool)
Scorer = Callable[
    [str, Particle, Action, Observation, int],
    Tuple[float, Optional[bool], Optional[Sequence[Tuple[str, bool, float]]]],
]


@dataclass
class SMCConfig:
    """Inference budget and resampling policy."""

    n_particles: int = 16
    max_steps: int = 4
    max_candidates: int = 3
    ess_ratio_threshold: float = 0.40
    time_budget_s: float = 10.0
    cache_maxsize: int = 256
    seed: Optional[int] = 0
    resample_on_final_step: bool = False

    def __post_init__(self) -> None:
        """Reject configurations that would fail silently rather than loudly.

        n_particles=0 previously produced an empty particle set and an agent
        that answered nothing, with no indication why.
        """
        if self.n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {self.n_particles}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if not 0.0 <= self.ess_ratio_threshold <= 1.0:
            raise ValueError(
                f"ess_ratio_threshold must lie in [0, 1], got {self.ess_ratio_threshold}"
            )
        if self.time_budget_s <= 0:
            raise ValueError(f"time_budget_s must be > 0, got {self.time_budget_s}")
        if self.cache_maxsize < 0:
            raise ValueError(f"cache_maxsize must be >= 0, got {self.cache_maxsize}")


@dataclass
class SMCResult:
    """Posterior particle set plus diagnostics."""

    particles: List[Particle]
    norm_logw: List[float]
    steps_run: int
    time_used_s: float
    ess_history: List[float] = field(default_factory=list)
    resamples: int = 0
    cache_stats: Dict[str, float] = field(default_factory=dict)

    def weights(self) -> List[float]:
        """Normalized particle weights in linear space."""
        return [math.exp(w) for w in self.norm_logw]

    def best_particle(self) -> Optional[Particle]:
        """Maximum a posteriori particle."""
        if not self.particles:
            return None
        ws = self.weights()
        return self.particles[max(range(len(ws)), key=lambda i: ws[i])]


class SMCRunner:
    """Executes the propose/observe/weight/resample loop over a tool set."""

    def __init__(self, tools: Dict[str, Tool], cfg: Optional[SMCConfig] = None):
        self.tools = tools
        self.cfg = cfg or SMCConfig()
        self.cache = LRUCache(maxsize=self.cfg.cache_maxsize)
        self.rng = random.Random(self.cfg.seed)

    # -- environment ---------------------------------------------------------

    def execute(self, action: Action) -> Observation:
        """Run a tool, memoising it only when it declares itself deterministic.

        Caching a stochastic tool would hand every particle the same draw,
        collapsing the cloud to a single sample and disabling the re-draw that
        lets a filter beat an unreliable source. Tools opt out via
        `@tool(deterministic=False)`.
        """
        tool = self.tools.get(action.tool_name)
        cacheable = getattr(tool, "deterministic", True)

        cache_key = "tool:" + action.key()
        if cacheable:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        started = time.time()
        if tool is None:
            obs = Observation(
                action=action,
                output={},
                ok=False,
                error=f"tool_not_found: {action.tool_name}",
                latency_s=time.time() - started,
            )
        else:
            try:
                out = tool.invoke(action.args)
                obs = Observation(
                    action=action,
                    output=out if isinstance(out, dict) else {"result": out},
                    ok=True,
                    latency_s=time.time() - started,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as a failed observation
                obs = Observation(
                    action=action,
                    output={},
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_s=time.time() - started,
                )

        if cacheable:
            self.cache.set(cache_key, obs)
        return obs

    # -- inference loop ------------------------------------------------------

    def run(
        self,
        query: str,
        proposer: Proposer,
        scorer: Scorer,
        selector: Selector,
        n_particles: Optional[int] = None,
        reweight: bool = True,
        explore: bool = True,
        init_reliability: Optional[ToolReliabilityState] = None,
    ) -> SMCResult:
        """Run inference.

        Args:
            query: the user request driving proposal.
            proposer: emits candidate actions for a step.
            scorer: turns an observation into a likelihood and belief updates.
            selector: picks one candidate per particle.
            n_particles: overrides the configured particle count (1 for greedy).
            reweight: when False, likelihoods are recorded but never accumulated
                into the weights -- this is what makes forward sampling forward
                sampling rather than SMC.
            explore: when False, selectors take their argmax branch, collapsing
                the cloud to one trajectory. A particle filter over identical
                particles is just a slow greedy run, so this must stay True for
                forward and SMC.
            init_reliability: carry beliefs in from a previous episode.

        Returns:
            SMCResult holding the final particle set and diagnostics.
        """
        n = n_particles if n_particles is not None else self.cfg.n_particles
        start = time.time()

        particles = [
            Particle(reliability=init_reliability.copy() if init_reliability else ToolReliabilityState())
            for _ in range(n)
        ]

        ess_history: List[float] = []
        resamples = 0
        steps_run = 0

        for t in range(self.cfg.max_steps):
            if time.time() - start >= self.cfg.time_budget_s:
                break

            proposed = _validate_candidates(proposer(query, particles, t), t)
            if not proposed:
                break
            if len(proposed) > self.cfg.max_candidates:
                warnings.warn(
                    f"Step {t}: proposer offered {len(proposed)} candidates but "
                    f"max_candidates={self.cfg.max_candidates}; dropping "
                    f"{len(proposed) - self.cfg.max_candidates}. Raise "
                    f"SMCConfig.max_candidates to consider them all.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            candidates = proposed[: self.cfg.max_candidates]

            advanced: List[Particle] = []
            for p in particles:
                nxt = p.copy_shallow()
                # Selectors read these to decide whether to diversify. A greedy
                # run wants argmax; a particle cloud is only useful if its
                # members actually explore different branches.
                nxt.state[_DETERMINISTIC_KEY] = not explore
                nxt.state[_RNG_KEY] = self.rng
                try:
                    action = _validate_action(selector(query, nxt, candidates, t))
                finally:
                    # Scratch keys are the runner's business, not the user's.
                    nxt.state.pop(_RNG_KEY, None)
                    nxt.state.pop(_DETERMINISTIC_KEY, None)
                # Resolve per-particle placeholders before touching the tool,
                # so a checker always inspects this trajectory's own claim.
                action = nxt.resolve(action)
                obs = self.execute(action).detached()

                likelihood, success_label, extra = _normalize_score(
                    scorer(query, nxt, action, obs, t)
                )

                if reweight:
                    nxt.logw += safe_log(likelihood)

                if success_label is not None:
                    nxt.reliability.update(action.tool_name, bool(success_label))
                if extra:
                    for tool_name, succ, weight in extra:
                        nxt.reliability.update(tool_name, bool(succ), float(weight))

                nxt.actions.append(action)
                nxt.observations.append(obs)
                nxt.trace.append(
                    {
                        "t": t,
                        "tool": action.tool_name,
                        "ok": obs.ok,
                        "likelihood": round(likelihood, 6),
                        "logw": round(nxt.logw, 6),
                    }
                )
                advanced.append(nxt)

            particles = advanced
            steps_run = t + 1

            logw = [p.logw for p in particles]
            ess = ess_from_logweights(logw)
            ess_history.append(ess)

            is_final = t == self.cfg.max_steps - 1
            may_resample = reweight and len(particles) > 1 and (
                self.cfg.resample_on_final_step or not is_final
            )
            if may_resample and ess / max(1, len(particles)) < self.cfg.ess_ratio_threshold:
                idxs = systematic_resample(logw, self.rng)
                particles = [particles[i].copy_shallow() for i in idxs]
                for p in particles:
                    p.logw = 0.0
                resamples += 1

        final_logw = [p.logw for p in particles] if particles else []
        return SMCResult(
            particles=particles,
            norm_logw=normalize_logweights(final_logw) if final_logw else [],
            steps_run=steps_run,
            time_used_s=time.time() - start,
            ess_history=ess_history,
            resamples=resamples,
            cache_stats=self.cache.stats(),
        )


def _validate_candidates(proposed, t: int) -> List[Action]:
    """Fail informatively on a malformed proposal instead of deep in the loop."""
    if proposed is None:
        return []
    if not isinstance(proposed, (list, tuple)):
        raise TypeError(
            f"Proposer must return a list of Action at step {t}, got "
            f"{type(proposed).__name__}."
        )
    for i, a in enumerate(proposed):
        if not isinstance(a, Action):
            raise TypeError(
                f"Proposer returned {type(a).__name__} at index {i} (step {t}); "
                f"every element must be an Action."
            )
    return list(proposed)


def _validate_action(action) -> Action:
    if not isinstance(action, Action):
        raise TypeError(
            f"Selector must return an Action, got {type(action).__name__}."
        )
    return action


def _normalize_score(ret) -> Tuple[float, Optional[bool], Optional[Sequence]]:
    """Coerce a scorer's return into (likelihood, label, extra).

    Accepts a bare number or a 1- to 3-tuple. A non-finite or out-of-range
    likelihood is a bug in the caller's scorer, and one that would otherwise
    corrupt every weight downstream while still returning plausible-looking
    numbers -- so it is rejected here rather than propagated.
    """
    if ret is None:
        raise TypeError(
            "Scorer returned None; expected a likelihood or a "
            "(likelihood, success_label, extra_updates) tuple."
        )

    if isinstance(ret, tuple):
        raw = ret[0] if len(ret) > 0 else 1.0
        label = ret[1] if len(ret) > 1 else None
        extra = ret[2] if len(ret) > 2 else None
    else:
        raw, label, extra = ret, None, None

    try:
        likelihood = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Scorer likelihood must be numeric, got {type(raw).__name__!s} ({raw!r})."
        ) from exc

    if math.isnan(likelihood):
        raise ValueError("Scorer returned NaN likelihood; check the scoring function.")
    if likelihood < 0:
        raise ValueError(
            f"Scorer returned a negative likelihood ({likelihood}); "
            "likelihoods must be non-negative."
        )
    if math.isinf(likelihood):
        raise ValueError("Scorer returned an infinite likelihood.")

    if label is not None and not isinstance(label, bool):
        label = bool(label)

    if extra is not None:
        extra = list(extra)
        for item in extra:
            if not (isinstance(item, (tuple, list)) and len(item) == 3):
                raise ValueError(
                    "Each extra update must be a (tool_name, success, weight) "
                    f"triple, got {item!r}."
                )

    return min(max(likelihood, 1e-12), 1.0), label, extra
