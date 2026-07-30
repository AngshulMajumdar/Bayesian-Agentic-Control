"""Request and response models for the HTTP surface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

REGIMES = {"greedy", "forward", "smc"}


class RunRequest(BaseModel):
    """Run one scenario under one inference regime."""

    scenario: str = Field(..., description="Registered scenario name.")
    query: Optional[str] = Field(
        None, description="Query text. Defaults to the scenario's canonical query."
    )
    regime: str = Field("smc", description="One of: greedy, forward, smc.")
    n_particles: Optional[int] = Field(
        None, ge=1, le=256, description="Override the particle count."
    )

    @field_validator("regime")
    @classmethod
    def _check_regime(cls, v: str) -> str:
        if v.lower() not in REGIMES:
            raise ValueError(f"regime must be one of {sorted(REGIMES)}")
        return v.lower()


class StepOut(BaseModel):
    t: int
    tool: str
    ok: bool
    output: Dict[str, Any] = {}


class RunResponse(BaseModel):
    scenario: str
    regime: str
    query: str
    answer: str
    steps: List[StepOut]
    posterior: Dict[str, Any]


class BenchmarkRequest(BaseModel):
    """Repeat a scenario across seeds and report aggregate behaviour."""

    scenario: str
    query: Optional[str] = None
    regimes: List[str] = Field(default_factory=lambda: ["greedy", "forward", "smc"])
    trials: int = Field(30, ge=1, le=1000)
    confidence: float = Field(0.95, gt=0.0, lt=1.0)

    @field_validator("regimes")
    @classmethod
    def _check_regimes(cls, v: List[str]) -> List[str]:
        bad = [r for r in v if r.lower() not in REGIMES]
        if bad:
            raise ValueError(f"unknown regimes {bad}; expected {sorted(REGIMES)}")
        return [r.lower() for r in v]


class RegimeStats(BaseModel):
    regime: str
    trials: int
    successes: int
    success_rate: float
    ci_lower: float = Field(..., description="Wilson interval lower bound.")
    ci_upper: float = Field(..., description="Wilson interval upper bound.")
    avg_steps: float
    avg_time_ms: float
    tool_usage: Dict[str, int]


class ContrastOut(BaseModel):
    """Difference between two regimes, with a significance verdict."""

    contrast: str
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool


class BenchmarkResponse(BaseModel):
    scenario: str
    query: str
    confidence_level: float
    results: List[RegimeStats]
    contrasts: List[ContrastOut] = []


class SessionRequest(BaseModel):
    """Run repeated episodes to exercise cross-episode belief carry-over."""

    scenario: str = "session_learning"
    query: Optional[str] = None
    episodes: int = Field(2, ge=1, le=10)
    regime: str = "smc"
    carry_reliability: bool = True

    @field_validator("regime")
    @classmethod
    def _check_regime(cls, v: str) -> str:
        if v.lower() not in REGIMES:
            raise ValueError(f"regime must be one of {sorted(REGIMES)}")
        return v.lower()


class SessionResponse(BaseModel):
    scenario: str
    regime: str
    carry_reliability: bool
    episodes: List[Dict[str, Any]]
