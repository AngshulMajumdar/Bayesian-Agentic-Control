"""The high-level entry point.

    from bayesian_rag import Agent, tool, checker

    @tool(reliability=0.6, appeal=0.9)
    def quick_search(query: str) -> str:
        \"\"\"Fast but sometimes stale.\"\"\"
        ...

    @tool(reliability=0.95, appeal=0.5)
    def verified_search(query: str) -> str:
        \"\"\"Slow and authoritative.\"\"\"
        ...

    @checker
    def fact_check(text: str) -> bool:
        ...

    agent = Agent([quick_search, verified_search, fact_check])
    result = agent.run("What is the capital of Australia?")
    print(result.answer)

Everything below that is optional. Supply a `proposer`, `scorer`, or `selector`
to override any default; supply none and the agent derives all three from the
tool metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from bayesian_rag.agents.bayesian_agent import AgentConfig, BayesianAgent
from bayesian_rag.agents.defaults import (
    build_proposer,
    build_scorer,
    build_selector,
    seed_priors,
)
from bayesian_rag.bayesian.reliability_model import ToolReliabilityState
from bayesian_rag.core.particle import EpisodeTrace
from bayesian_rag.core.smc_runner import SMCConfig
from bayesian_rag.tools.decorator import Tool, as_tool

ToolLike = Union[Tool, Any]


@dataclass
class Step:
    """One tool call in the winning trajectory."""

    index: int
    tool: str
    ok: bool
    output: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        mark = "ok" if self.ok else "failed"
        return f"Step({self.index}, {self.tool!r}, {mark})"


@dataclass
class Result:
    """What an agent run produced, plus enough to see why.

    `answer` is the maximum-a-posteriori answer. `reliability` reports what the
    run learned about each tool, and `alternatives` shows which competing
    trajectories retained mass -- useful when the answer is close.
    """

    answer: str
    query: str
    regime: str
    steps: List[Step]
    reliability: Dict[str, float]
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    answers: List[Dict[str, Any]] = field(default_factory=list)
    consensus: float = 1.0
    clarify_threshold: float = 0.6
    status: str = "unverified"
    n_particles: int = 0
    resamples: int = 0
    elapsed_s: float = 0.0
    ess_history: List[float] = field(default_factory=list)
    trace: Optional[EpisodeTrace] = None

    @property
    def tools_used(self) -> List[str]:
        """Tool names along the winning trajectory, in order."""
        return [s.tool for s in self.steps]

    @property
    def validated(self) -> bool:
        """True when a checker confirmed the answer being returned."""
        return self.status == "validated"

    @property
    def refuted(self) -> bool:
        """True when every answer found was rejected by a checker.

        The answer is still returned, since it is what the agent found, but a
        caller should treat it as unsupported. Silently asserting a claim the
        agent itself disproved is worse than reporting that nothing held up.
        """
        return self.status == "refuted"

    @property
    def confidence(self) -> float:
        """Posterior mass backing the returned answer, in [0, 1].

        Pooled across every trajectory that reached this answer, so reaching one
        conclusion by several routes counts as agreement rather than as doubt.
        """
        return self.consensus

    @property
    def competing_answers(self) -> List[Dict[str, Any]]:
        """Distinct answers that retained posterior mass, most-backed first."""
        return self.answers

    @property
    def needs_clarification(self) -> bool:
        """True when no answer commands enough of the posterior to be asserted.

        This is the state a single-path pipeline cannot represent. Having
        committed to one trajectory it has exactly one answer, and no way to
        notice that the evidence supports several. Here the split is visible, so
        the caller can ask a follow-up question instead of picking arbitrarily.
        """
        return len(self.answers) > 1 and self.consensus < self.clarify_threshold

    def clarification_request(self) -> Optional[str]:
        """A question to put to the user when the evidence does not settle it.

        Returns None when one answer is well supported.
        """
        if not self.needs_clarification:
            return None
        options = "\n".join(
            f"  {i + 1}. {a['answer']}  ({a['mass']:.0%} of the evidence)"
            for i, a in enumerate(self.answers[:4])
        )
        return (
            f"I could not settle this from the sources available. "
            f"{len(self.answers)} answers retained support:\n{options}\n"
            f"Which should I take, or can you narrow the question?"
        )

    def explain(self) -> str:
        """A human-readable account of the run."""
        lines = [
            f"Query   : {self.query}",
            f"Answer  : {self.answer}",
            f"Regime  : {self.regime} ({self.n_particles} particle"
            f"{'s' if self.n_particles != 1 else ''})",
            f"Path    : {' -> '.join(self.tools_used) or '(none)'}",
            f"Consensus: {self.consensus:.2f} of posterior mass",
            f"Status  : {self.status}",
        ]
        if self.needs_clarification:
            lines.append("  the evidence is split; a follow-up question is warranted:")
            for a in self.answers[:4]:
                lines.append(f"    {a['mass']:.0%}  {a['answer'][:60]}")
        if self.refuted:
            lines.append(
                "  warning: every answer found was rejected by a checker; "
                "the result above is unsupported."
            )
        if self.reliability:
            learned = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.reliability.items()))
            lines.append(f"Learned : {learned}")
        if len(self.alternatives) > 1:
            lines.append("Alternatives considered:")
            for alt in self.alternatives[1:]:
                path = " -> ".join(alt["tools"])
                lines.append(f"  {alt['mass']:.3f}  {path}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.answer

    def __repr__(self) -> str:
        return f"Result(answer={self.answer[:48]!r}, tools={self.tools_used})"


class Agent:
    """A tool-using agent that reasons over trajectories instead of committing.

    Args:
        tools: functions decorated with `@tool`/`@checker`, or objects exposing
            `.name` and `.invoke`.
        regime: "smc" (default), "forward", or "greedy". SMC is the only regime
            that revises earlier decisions in light of later evidence.
        particles: hypotheses maintained per step. Ignored under greedy.
        max_steps: tool calls per episode.
        seed: fixes the run. Identical inputs and seed give identical output.
        proposer / scorer / selector: override the derived defaults.
        carry_over: fraction of a previous episode's evidence retained by
            `run_session`. 1.0 keeps it all; lower discounts stale evidence.
        clarify_threshold: consensus below which `Result.needs_clarification`
            is set, prompting a follow-up question rather than an arbitrary pick.
        cache: set False to disable memoisation for every tool at once,
            regardless of what each declared. Useful when tools are stochastic
            and were not marked as such.
    """

    def __init__(
        self,
        tools: Sequence[ToolLike],
        *,
        regime: str = "smc",
        particles: int = 16,
        max_steps: int = 3,
        seed: Optional[int] = 0,
        temperature: float = 1.0,
        allow_repeat: bool = True,
        prior_strength: float = 2.0,
        carry_over: float = 1.0,
        proposer=None,
        scorer=None,
        selector=None,
        cache: bool = True,
        clarify_threshold: float = 0.6,
        time_budget_s: float = 30.0,
    ):
        if not tools:
            raise ValueError("An agent needs at least one tool.")

        self.tools: Dict[str, Tool] = {}
        for t in tools:
            wrapped = as_tool(t)
            if wrapped.name in self.tools:
                raise ValueError(
                    f"Duplicate tool name {wrapped.name!r}. Pass name= to @tool "
                    f"to disambiguate."
                )
            self.tools[wrapped.name] = wrapped

        if not cache:
            import dataclasses

            self.tools = {
                n: dataclasses.replace(t, deterministic=False)
                for n, t in self.tools.items()
            }

        self.regime = regime
        self.clarify_threshold = clarify_threshold
        self._priors = seed_priors(self.tools, prior_strength)

        self._agent = BayesianAgent(
            tools=self.tools,
            proposer=proposer or build_proposer(self.tools, allow_repeat=allow_repeat),
            scorer=scorer or build_scorer(self.tools),
            selector=selector or build_selector(self.tools, temperature=temperature),
            cfg=AgentConfig(
                smc=SMCConfig(
                    n_particles=particles,
                    max_steps=max_steps,
                    max_candidates=max(len(self.tools), 1),
                    seed=seed,
                    time_budget_s=time_budget_s,
                ),
                regime=regime,
                carry_over_factor=carry_over,
            ),
        )

    # -- running -------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        regime: Optional[str] = None,
        particles: Optional[int] = None,
        beliefs: Optional[ToolReliabilityState] = None,
    ) -> Result:
        """Answer a query and report how the answer was reached."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        answer, trace, posterior = self._agent.run(
            query,
            regime=regime or self.regime,
            n_particles=particles,
            init_reliability=beliefs if beliefs is not None else self._priors,
        )
        return _to_result(query, answer, trace, posterior, self.clarify_threshold)

    def run_session(
        self, queries: Sequence[str], *, regime: Optional[str] = None, carry: bool = True
    ) -> List[Result]:
        """Run several queries in order, carrying learned reliabilities forward.

        Later episodes begin already knowing which tools survived scrutiny, so a
        tool that was attractive but kept failing validation is distrusted from
        the outset rather than re-discovered each time.
        """
        results: List[Result] = []
        beliefs = self._priors

        for q in queries:
            answer, trace, posterior = self._agent.run(
                q,
                regime=regime or self.regime,
                init_reliability=beliefs,
                _return_beliefs=True,
            )
            learned = posterior.pop("_beliefs", None)
            results.append(_to_result(q, answer, trace, posterior, self.clarify_threshold))
            if carry and learned is not None:
                beliefs = learned.tempered(self._agent.cfg.carry_over_factor)
        return results

    def compare(
        self, query: str, regimes: Iterable[str] = ("greedy", "forward", "smc")
    ) -> Dict[str, Result]:
        """Run the same query under several regimes.

        Nothing but the inference differs, so any divergence in the answers is
        attributable to it.
        """
        return {r: self.run(query, regime=r) for r in regimes}

    # -- introspection -------------------------------------------------------

    @property
    def tool_names(self) -> List[str]:
        """Registered tool names, sorted."""
        return sorted(self.tools)

    def describe(self) -> str:
        """Summarise the configured tools."""
        lines = [f"Agent with {len(self.tools)} tools (regime={self.regime}):"]
        for name, t in sorted(self.tools.items()):
            lines.append(
                f"  {name:24s} {t.kind:8s} reliability={t.reliability:.2f} "
                f"appeal={t.appeal:.2f}  {t.description}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Agent(tools={self.tool_names}, regime={self.regime!r})"


def _to_result(
    query: str,
    answer: str,
    trace: EpisodeTrace,
    posterior: Dict[str, Any],
    clarify_threshold: float = 0.6,
) -> Result:
    return Result(
        answer=answer,
        query=query,
        regime=posterior.get("regime", "smc"),
        steps=[
            Step(index=s.t, tool=s.tool, ok=s.ok, output=s.output) for s in trace.steps
        ],
        reliability=posterior.get("tool_reliability_means", {}),
        status=posterior.get("answer_status", "unverified"),
        answers=posterior.get("answers", []),
        consensus=posterior.get("consensus", 1.0),
        clarify_threshold=clarify_threshold,
        alternatives=posterior.get("top_particles", []),
        n_particles=posterior.get("n_particles", 0),
        resamples=posterior.get("resamples", 0),
        elapsed_s=posterior.get("time_used_s", 0.0),
        ess_history=posterior.get("ess_history", []),
        trace=trace,
    )
