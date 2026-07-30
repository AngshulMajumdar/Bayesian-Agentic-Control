"""Default proposer, scorer, and selector derived from tool metadata.

Supplying these by hand is the main friction in using a particle filter for
orchestration, and getting the selector subtly wrong disables the method
silently. These defaults are derived from what `@tool` already knows, so the
common case needs no callbacks at all. Each can still be overridden
individually when a task needs something specific.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from bayesian_rag.core.particle import Action, Observation, Particle, Pending
from bayesian_rag.tools.decorator import Tool, ToolKind


def build_proposer(tools: Dict[str, Tool], allow_repeat: bool = True):
    """Offer every applicable tool at each step.

    Deliberately flat: the menu is not narrowed by hand-written conditionals,
    because scripting the recovery into the control flow would mean the control
    flow, not the inference, was doing the work.

    The only structural rule is that a checker needs something to check, so it
    is withheld until a claim exists.
    """

    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        # Only used to decide whether *any* claim exists yet; the claim each
        # checker actually inspects is resolved per particle at execution time.
        any_claim = _pending_claim(particles)
        candidates: List[Action] = []

        for name, tl in tools.items():
            if tl.is_checker:
                if not any_claim:
                    continue
                candidates.append(Action(name, _args_for(tl, query, Pending.CLAIM)))
                continue

            if not allow_repeat and _already_used(particles, name):
                continue

            candidates.append(Action(name, _args_for(tl, query, Pending.CLAIM)))

        return candidates

    return proposer


def build_scorer(
    tools: Dict[str, Tool],
    checker_weight: float = 2.0,
    validated: float = 0.95,
    recovered: float = 0.60,
    stranded: float = 0.15,
):
    """Turn observations into likelihoods and reliability updates.

    Action tools are scored by the confidence they report.

    Checkers are scored by the epistemic state they leave the trajectory in,
    not by their verdict alone. This distinction matters. Scoring a failed
    check as a near-zero likelihood punishes the particle for having done the
    right thing: it gathered the decisive evidence and was then eliminated
    before it could act on it, taking that evidence with it.

    So a trajectory is scored by what it can now assert:

        validated -- its answer passed a check
        recovered -- its answer failed, but it still holds another it has not
                     disproved, so it can still answer
        stranded  -- every answer it holds has been disproved

    A failed check also emits a retroactive update against whichever tool
    produced the rejected text, which is what redirects later selection: the
    source is penalised when it is caught, not when it answers.
    """

    def scorer(
        query: str, p: Particle, a: Action, obs: Observation, t: int
    ) -> Tuple[float, Optional[bool], Optional[Sequence[Tuple[str, bool, float]]]]:
        if not obs.ok:
            return 1e-6, False, None

        tl = tools.get(a.tool_name)

        if tl is not None and tl.is_checker:
            passed = bool(obs.output.get("ok", False))
            blamed = _last_non_checker(p, tools)
            extra = [(blamed, passed, checker_weight)] if blamed else None

            if passed:
                return validated, None, extra

            # The check has just failed. Whether this trajectory is still
            # usable depends on what else it holds.
            checked = _checked_text(a)
            has_alternative = any(
                (ans := o.answer()) and ans != checked for o in p.observations
            )
            return (recovered if has_alternative else stranded), None, extra

        confidence = float(obs.output.get("confidence", 0.5))
        verified = obs.output.get("verified", obs.output.get("is_verified"))
        label = True if verified else None
        return max(min(confidence, 1.0), 1e-6), label, None

    return scorer


def build_selector(tools: Dict[str, Tool], temperature: float = 1.0):
    """Choose among candidates by (learned reliability x prior appeal).

    Under greedy this takes the argmax. Under the exploring regimes it samples
    proportionally, which is essential rather than cosmetic: if every particle
    made the same choice the cloud would carry no information and reweighting
    would have nothing to act on, quietly reducing an N-particle filter to a
    slow greedy run.
    """

    def selector(query: str, p: Particle, actions: List[Action], t: int) -> Action:
        if len(actions) == 1:
            return actions[0]

        scored: List[Tuple[Action, float]] = []
        for act in actions:
            tl = tools.get(act.tool_name)
            appeal = tl.appeal if tl is not None else 0.5
            learned = p.reliability.get(act.tool_name).mean()
            scored.append((act, max(1e-9, learned * appeal)))

        if p.is_deterministic():
            return max(scored, key=lambda pair: pair[1])[0]

        weights = [s ** (1.0 / max(temperature, 1e-6)) for _, s in scored]
        total = sum(weights)
        if total <= 0:
            return scored[0][0]

        draw = p.rng().random() * total
        cumulative = 0.0
        for (act, _), w in zip(scored, weights):
            cumulative += w
            if draw <= cumulative:
                return act
        return scored[-1][0]

    return selector


def seed_priors(tools: Dict[str, Tool], prior_strength: float = 2.0):
    """Express each tool's declared reliability as an initial Beta posterior.

    Without this every tool starts at an identical 0.5, and a declared
    `reliability` would have no effect until evidence arrived. `prior_strength`
    is kept low so a couple of real observations can overturn a declared prior.
    """
    from bayesian_rag.bayesian.reliability_model import BetaBelief, ToolReliabilityState

    state = ToolReliabilityState()
    for name, tl in tools.items():
        state.beliefs[name] = BetaBelief(
            alpha=1.0 + prior_strength * tl.reliability,
            beta=1.0 + prior_strength * (1.0 - tl.reliability),
        )
    return state


# --- helpers -----------------------------------------------------------------


def _args_for(tl: Tool, query: str, claim: Any) -> Dict[str, Any]:
    """Fill a tool's parameters from what the episode currently knows."""
    args: Dict[str, Any] = {}
    qp = tl.query_param()
    cp = tl.claim_param()
    if qp:
        args[qp] = query
    if cp and cp != qp:
        args[cp] = claim
    if not args and tl.params:
        args[tl.params[0]] = query
    return args


def _pending_claim(particles: List[Particle]) -> str:
    """The most recent answer awaiting validation, from the leading particle."""
    if not particles:
        return ""
    for obs in reversed(particles[0].observations):
        ans = obs.answer()
        if ans:
            return ans
    return ""


def _already_used(particles: List[Particle], name: str) -> bool:
    return bool(particles) and name in particles[0].tools_used()


def _checked_text(a: Action) -> str:
    """The claim a checker was asked to examine."""
    for value in a.args.values():
        if isinstance(value, str) and value:
            return value
    return ""


def _last_non_checker(p: Particle, tools: Dict[str, Tool]) -> Optional[str]:
    """The most recent tool in this trajectory that produced a claim."""
    for act in reversed(p.actions):
        tl = tools.get(act.tool_name)
        if tl is None or not tl.is_checker:
            return act.tool_name
    return None
