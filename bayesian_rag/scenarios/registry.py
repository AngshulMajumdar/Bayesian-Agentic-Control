"""Reference scenarios exercising the inference ladder on conflicting evidence.

Design note. It is easy to write a scenario in which every regime succeeds, by
scripting the recovery into the proposer with hand-written conditionals ("if the
check failed, call the verified source"). That measures the control flow, not
the inference. These scenarios deliberately avoid it: the proposer offers the
same tool menu at every step, and which trajectory wins is decided entirely by
selection and reweighting.

The failure mode being reproduced is the realistic one. The cheap source is more
attractive a priori -- that is why anyone reaches for it -- and it is confidently
wrong. Only a checker, one step later, reveals the conflict. A greedy agent has
already committed and keeps re-picking its favourite; a particle filter kept
alternative branches alive and can promote one when the evidence lands.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bayesian_rag.agents.bayesian_agent import AgentConfig, BayesianAgent
from bayesian_rag.core.particle import Action, Observation, Particle, Tool
from bayesian_rag.core.smc_runner import SMCConfig
from bayesian_rag.tools.mock_tools import (
    AskClarificationTool,
    ConsistencyCheckTool,
    FastSearchTool,
    GeoDisambiguateTool,
    NoisyWebSearchTool,
    NoticeCheckTool,
    OfficialNoticeTool,
    SchoolSearchIN,
    SchoolSearchUS,
    VerifiedSearchTool,
)


# --- selectors ---------------------------------------------------------------


def selector_by_reliability(
    prior_scores: Dict[str, float], temperature: float = 1.0
) -> Callable:
    """Score candidates by (learned reliability x prior appeal), then choose.

    The learned factor is what makes this adaptive: a tool that keeps failing
    validation loses posterior mass and stops being chosen, however attractive
    its prior appeal.

    How the score becomes a choice depends on the regime. Greedy takes the
    argmax. Forward and SMC sample proportionally, because a particle cloud
    whose members all pick the same branch carries no information -- there would
    be nothing for the likelihood to reweight.
    """

    def selector(query: str, p: Particle, actions: List[Action], t: int) -> Action:
        scored = [
            (
                act,
                max(
                    1e-9,
                    p.reliability.get(act.tool_name).mean()
                    * prior_scores.get(act.tool_name, 0.5),
                ),
            )
            for act in actions
        ]

        if p.is_deterministic() or len(scored) == 1:
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


def selector_first(query: str, p: Particle, actions: List[Action], t: int) -> Action:
    """Take the proposer's only/top candidate; used when routing is forced."""
    return actions[0]


# --- scenario 1: stale vs verified -------------------------------------------


def build_stale_vs_verified(n_particles: int = 24, seed: int = 7) -> BayesianAgent:
    """A cheap, confident, stale source competes with a slow authoritative one.

    The menu is identical at every step: both sources plus a checker. Nothing in
    the control flow steers the agent toward the right answer, so any recovery
    that happens is produced by inference.
    """
    tools: Dict[str, Tool] = {
        "fast_search": FastSearchTool(),
        "verified_search": VerifiedSearchTool(),
        "consistency_check": ConsistencyCheckTool(),
    }

    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        if t == 0:
            # No checker yet -- there is nothing to check.
            return [
                Action("fast_search", {"query": query}),
                Action("verified_search", {"query": query}),
            ]
        return [
            Action("fast_search", {"query": query}),
            Action("verified_search", {"query": query}),
            Action("consistency_check", {"text": _pending_claim(particles)}),
        ]

    def scorer(query, p, a, obs, t):
        return _source_and_check_scorer(p, a, obs, check_tool="consistency_check")

    return BayesianAgent(
        tools=tools,
        proposer=proposer,
        scorer=scorer,
        # fast_search is the most attractive a priori: it is cheap and quick.
        # That is precisely why a greedy agent walks into the stale answer.
        selector=selector_by_reliability(
            {"fast_search": 0.85, "verified_search": 0.55, "consistency_check": 0.60}
        ),
        cfg=AgentConfig(
            smc=SMCConfig(n_particles=n_particles, max_steps=3, max_candidates=3, seed=seed)
        ),
    )


# --- scenario 2: ambiguous location ------------------------------------------


def build_ambiguous_location(n_particles: int = 16, seed: int = 11) -> BayesianAgent:
    """An ambiguous place name must be resolved before downstream tools fire.

    Here the routing genuinely depends on evidence gathered mid-trajectory, so
    the branch taken at the last step cannot be planned in advance. All regimes
    should succeed; the scenario is a control, confirming that the added
    machinery does not break straightforward sequential work.
    """
    tools: Dict[str, Tool] = {
        "geo_disambiguate": GeoDisambiguateTool(),
        "ask_clarification": AskClarificationTool(),
        "school_search_us": SchoolSearchUS(),
        "school_search_india": SchoolSearchIN(),
    }

    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        if t == 0:
            return [Action("geo_disambiguate", {"query": query})]
        if t == 1:
            return [Action("ask_clarification", {"hint": query})]
        if t == 2:
            choice = _scan_outputs(particles, "user_choice")
            if choice == "saltlake_kolkata":
                return [Action("school_search_india", {})]
            return [Action("school_search_us", {})]
        return []

    def scorer(query, p, a, obs, t):
        if not obs.ok:
            return 1e-6, False, None
        return float(obs.output.get("confidence", 0.8)), True, None

    return BayesianAgent(
        tools=tools,
        proposer=proposer,
        scorer=scorer,
        selector=selector_first,
        cfg=AgentConfig(
            smc=SMCConfig(n_particles=n_particles, max_steps=3, max_candidates=1, seed=seed)
        ),
    )


# --- scenario 3: noisy web vs official notice --------------------------------


def build_web_vs_official(n_particles: int = 20, seed: int = 13) -> BayesianAgent:
    """A widely repeated wrong deadline competes with the authoritative notice.

    Same shape as scenario 1 with a sharper cost asymmetry: acting on the wrong
    deadline is expensive, and the noisy source is the one everybody reaches for.
    """
    tools: Dict[str, Tool] = {
        "noisy_web_search": NoisyWebSearchTool(),
        "official_notice": OfficialNoticeTool(),
        "notice_check": NoticeCheckTool(),
    }

    def proposer(query: str, particles: List[Particle], t: int) -> List[Action]:
        if t == 0:
            return [
                Action("noisy_web_search", {"query": query}),
                Action("official_notice", {"query": query}),
            ]
        return [
            Action("noisy_web_search", {"query": query}),
            Action("official_notice", {"query": query}),
            Action("notice_check", {"text": _pending_claim(particles)}),
        ]

    def scorer(query, p, a, obs, t):
        return _source_and_check_scorer(
            p, a, obs, check_tool="notice_check", pass_score=0.97, fail_score=0.03
        )

    return BayesianAgent(
        tools=tools,
        proposer=proposer,
        scorer=scorer,
        selector=selector_by_reliability(
            {"noisy_web_search": 0.85, "official_notice": 0.55, "notice_check": 0.60}
        ),
        cfg=AgentConfig(
            smc=SMCConfig(n_particles=n_particles, max_steps=3, max_candidates=3, seed=seed)
        ),
    )


# --- scenario 4: session learning --------------------------------------------


def build_session_learning(n_particles: int = 24, seed: int = 7) -> BayesianAgent:
    """Scenario 1's agent, intended for `run_session` over repeated queries.

    Episode one pays to discover which source survives validation. Episode two
    inherits those beliefs, so the learned reliability term now outweighs the
    cheap source's prior appeal and the agent reaches the answer sooner.
    """
    return build_stale_vs_verified(n_particles=n_particles, seed=seed)


# --- scoring helpers ---------------------------------------------------------


def _source_and_check_scorer(
    p: Particle,
    a: Action,
    obs: Observation,
    check_tool: str,
    pass_score: float = 0.95,
    fail_score: float = 0.05,
) -> Tuple[float, Optional[bool], Optional[Sequence[Tuple[str, bool, float]]]]:
    """Score retrieval by self-reported confidence, checkers by their verdict.

    When a checker fires it also emits a retroactive update against whichever
    tool produced the text it examined. That is the mechanism by which late
    evidence reaches back and changes the standing of an earlier decision: the
    stale source is not punished when it answers, but when it is caught.
    """
    if not obs.ok:
        return 1e-6, False, None

    if a.tool_name == check_tool:
        passed = bool(obs.output.get("ok", False))
        blamed = _last_source_tool(p, check_tool)
        extra = [(blamed, passed, 2.0)] if blamed else None
        return (pass_score if passed else fail_score), None, extra

    confidence = float(obs.output.get("confidence", 0.1))
    verified = True if obs.output.get("is_verified") else None
    return confidence, verified, None


def _last_source_tool(p: Particle, check_tool: str) -> Optional[str]:
    """Most recent non-checker tool in this particle's trajectory."""
    for act in reversed(p.actions):
        if act.tool_name != check_tool:
            return act.tool_name
    return None


def _pending_claim(particles: List[Particle]) -> str:
    """Claim text awaiting validation, taken from the leading particle."""
    if particles and particles[0].observations:
        for obs in reversed(particles[0].observations):
            ans = obs.answer()
            if ans:
                return ans
    return ""


def _scan_outputs(particles: List[Particle], key: str):
    """First value found for `key` across the leading particle's observations."""
    if not particles:
        return None
    for obs in reversed(particles[0].observations):
        if isinstance(obs.output, dict) and key in obs.output:
            return obs.output[key]
    return None


SCENARIOS: Dict[str, Callable[..., BayesianAgent]] = {
    "stale_vs_verified": build_stale_vs_verified,
    "ambiguous_location": build_ambiguous_location,
    "web_vs_official": build_web_vs_official,
    "session_learning": build_session_learning,
}

SCENARIO_QUERIES: Dict[str, str] = {
    "stale_vs_verified": "What is the best time to visit Serbia? Consider weather and budget.",
    "ambiguous_location": "What are good schools near Salt Lake?",
    "web_vs_official": "What is the scholarship deadline?",
    "session_learning": "What is the best time to visit Serbia? Consider weather and budget.",
}

SCENARIO_SUCCESS_MARKERS: Dict[str, List[str]] = {
    "stale_vs_verified": ["Late April to June"],
    "ambiguous_location": ["Kolkata"],
    "web_vs_official": ["April 30"],
    "session_learning": ["Late April to June"],
}
