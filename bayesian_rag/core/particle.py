"""Unified state types shared by every backend and inference regime."""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from bayesian_rag.bayesian.reliability_model import ToolReliabilityState

JsonDict = Dict[str, Any]

# Private scratch keys the runner attaches to particle state during selection.
_DETERMINISTIC_KEY = "_br_deterministic"
_RNG_KEY = "_br_rng"


class Pending(str):
    """Placeholder for a value only the executing particle can supply.

    A proposer runs once per step and emits one menu shared by every particle,
    but a checker must inspect the claim held by the trajectory that is
    checking -- not whichever claim the first particle happened to hold.
    Passing `Pending.CLAIM` defers that lookup to execution time, when the
    particle is known.

    Without this, a particle that used a good source could be blamed for a bad
    claim produced by a different particle, driving a reliable tool's posterior
    steadily downwards.
    """

    CLAIM: "Pending"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Pending({str.__str__(self)})"


Pending.CLAIM = Pending("<pending:claim>")


@runtime_checkable
class Tool(Protocol):
    """Minimal tool contract, compatible with LangChain's `invoke` interface."""

    name: str

    def invoke(self, inp: JsonDict) -> JsonDict:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class Action:
    """A single tool invocation proposed by the agent."""

    tool_name: str
    args: JsonDict = field(default_factory=dict)

    def key(self) -> str:
        """Stable cache key for this exact invocation."""
        return f"{self.tool_name}::{json.dumps(self.args, sort_keys=True, ensure_ascii=False, default=str)}"


@dataclass(frozen=True)
class Observation:
    """The outcome of executing an action against the environment."""

    action: Action
    output: JsonDict = field(default_factory=dict)
    ok: bool = True
    error: Optional[str] = None
    latency_s: float = 0.0
    ts: float = field(default_factory=time.time)

    def answer(self) -> Optional[str]:
        """Convenience accessor for the conventional `answer` field."""
        if isinstance(self.output, dict) and self.output.get("answer"):
            return str(self.output["answer"])
        return None

    def short(self, max_chars: int = 240) -> str:
        """Truncated JSON rendering for logs and traces."""
        s = json.dumps(self.output, ensure_ascii=False, sort_keys=True, default=str)
        return s if len(s) <= max_chars else s[: max_chars - 3] + "..."

    def detached(self) -> "Observation":
        """A copy whose output cannot be mutated through a shared reference.

        The tool cache hands one Observation to every particle that issued the
        same call. Without this, a particle mutating its own observation would
        silently rewrite its siblings' history.
        """
        return Observation(
            action=self.action,
            output=copy.deepcopy(self.output),
            ok=self.ok,
            error=self.error,
            latency_s=self.latency_s,
            ts=self.ts,
        )


@dataclass
class Particle:
    """One hypothesis about the agent's action trajectory.

    Carries the trajectory itself, this particle's private reliability beliefs,
    an open-ended state dict (for LangChain-style transition models), and the
    unnormalized log-weight used by the particle filter.
    """

    actions: List[Action] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    reliability: ToolReliabilityState = field(default_factory=ToolReliabilityState)
    state: JsonDict = field(default_factory=dict)
    logw: float = 0.0
    trace: List[JsonDict] = field(default_factory=list)

    def copy_shallow(self) -> "Particle":
        """Copy for propagation/resampling; reliability beliefs are cloned."""
        return Particle(
            actions=list(self.actions),
            observations=list(self.observations),
            reliability=self.reliability.copy(),
            state=dict(self.state),
            logw=self.logw,
            trace=list(self.trace),
        )

    # -- selection context ---------------------------------------------------
    # The runner attaches these before calling a selector and strips them
    # afterwards. Selectors should use the accessors below rather than reading
    # the scratch keys, which are private and may be renamed.

    def is_deterministic(self) -> bool:
        """True when the active regime wants an argmax rather than a sample.

        Greedy runs a single particle, so sampling would only add noise to a
        trajectory that has no siblings to be compared against. Exploring
        regimes need the opposite: identical particles carry no information.
        """
        return bool(self.state.get(_DETERMINISTIC_KEY, False))

    def rng(self) -> "random.Random":
        """The run's seeded generator. Use this so results stay reproducible.

        Falling back to the global `random` module would make a run
        irreproducible even with a seed set, which is the kind of bug that only
        shows up when someone tries to replicate a result.
        """
        got = self.state.get(_RNG_KEY)
        return got if got is not None else random

    def last_observation(self) -> Optional[Observation]:
        """Most recent observation, or None if the trajectory is empty."""
        return self.observations[-1] if self.observations else None

    def last_action(self) -> Optional[Action]:
        """Most recent action, or None if the trajectory is empty."""
        return self.actions[-1] if self.actions else None

    def pending_claim(self) -> str:
        """This trajectory's most recent answer, awaiting validation."""
        for obs in reversed(self.observations):
            ans = obs.answer()
            if ans:
                return ans
        return ""

    def resolve(self, action: "Action") -> "Action":
        """Substitute any Pending placeholders against this particle's history."""
        if not any(isinstance(v, Pending) for v in action.args.values()):
            return action
        resolved = {
            k: (self.pending_claim() if isinstance(v, Pending) else v)
            for k, v in action.args.items()
        }
        return Action(tool_name=action.tool_name, args=resolved)

    def tools_used(self) -> List[str]:
        """Tool names along this trajectory, in order."""
        return [a.tool_name for a in self.actions]

    def invalidated_claims(self) -> set:
        """Claim texts that a checker in this trajectory explicitly rejected.

        A checker records the text it examined in its action arguments, so a
        failed verdict can be tied back to the exact claim that failed.
        """
        rejected = set()
        for action, obs in zip(self.actions, self.observations):
            if not obs.ok or "ok" not in obs.output:
                continue
            if obs.output.get("ok"):
                continue
            for value in action.args.values():
                if isinstance(value, str) and value:
                    rejected.add(value)
        return rejected

    def answer_status(self) -> str:
        """Whether this trajectory's answer survived its own checks.

        Returns "validated" if a checker passed on the answer being returned,
        "refuted" if every answer it holds was rejected, and "unverified" if it
        was never checked.
        """
        rejected = self.invalidated_claims()
        answers = [a for o in self.observations if (a := o.answer())]
        if not answers:
            return "unverified"

        surviving = [a for a in answers if a not in rejected]
        if not surviving:
            return "refuted"

        returned = surviving[-1]
        for action, obs in zip(self.actions, self.observations):
            if obs.ok and obs.output.get("ok") is True:
                if returned in [v for v in action.args.values() if isinstance(v, str)]:
                    return "validated"
        return "unverified"

    def validated_claims(self) -> set:
        """Claim texts that a checker in this trajectory explicitly confirmed."""
        confirmed = set()
        for action, obs in zip(self.actions, self.observations):
            if not obs.ok or "ok" not in obs.output:
                continue
            if not obs.output.get("ok"):
                continue
            for value in action.args.values():
                if isinstance(value, str) and value:
                    confirmed.add(value)
        return confirmed

    def final_answer(self) -> Optional[str]:
        """The best-supported answer this trajectory holds.

        Preference order, strongest first:

          1. an answer a checker in this trajectory confirmed
          2. the most recent answer no checker rejected
          3. the most recent answer at all

        Recency alone is the wrong rule. A trajectory that found the right
        passage, validated it, and then went on to look at something else
        would otherwise discard the confirmed answer in favour of whatever it
        happened to touch last -- throwing away the evidence it just paid to
        gather. Evidence outranks recency.
        """
        rejected = self.invalidated_claims()
        validated = self.validated_claims()

        latest = None
        latest_unrejected = None

        for obs in reversed(self.observations):
            ans = obs.answer()
            if not ans:
                continue
            if latest is None:
                latest = ans
            if ans in validated:
                return ans
            if ans not in rejected and latest_unrejected is None:
                latest_unrejected = ans

        return latest_unrejected if latest_unrejected is not None else latest

    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class StepRecord:
    """One step of the winning trajectory, for user-facing traces."""

    t: int
    tool: str
    ok: bool
    output: JsonDict = field(default_factory=dict)
    tool_reliability_means: Dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeTrace:
    """Full record of one agent run."""

    query: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    meta: JsonDict = field(default_factory=dict)

    def add_step(self, rec: StepRecord) -> None:
        """Append one step to the trace."""
        self.steps.append(rec)
