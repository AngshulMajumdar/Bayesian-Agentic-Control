"""BayesianRAG: probabilistic orchestration for tool-augmented agents.

Quick start:

    from bayesian_rag import Agent, tool, checker

    @tool(reliability=0.6, appeal=0.9)
    def quick_search(query: str) -> str:
        \"\"\"Fast, sometimes stale.\"\"\"
        return "..."

    @tool(reliability=0.95, appeal=0.5)
    def verified_search(query: str) -> str:
        \"\"\"Slow, authoritative.\"\"\"
        return "..."

    @checker
    def fact_check(text: str) -> bool:
        return "..." in text

    agent = Agent([quick_search, verified_search, fact_check])
    print(agent.run("What is X?").answer)
"""

# High-level API -- what most users need.
from bayesian_rag.agents.agent import Agent, Result, Step
from bayesian_rag.tools.decorator import Tool, ToolKind, as_tool, checker, tool

# Lower-level API -- for custom proposers, scorers, and selectors.
from bayesian_rag.agents.bayesian_agent import (
    AgentConfig,
    BayesianAgent,
    summarize_posterior,
)
from bayesian_rag.agents.defaults import (
    build_proposer,
    build_scorer,
    build_selector,
    seed_priors,
)
from bayesian_rag.bayesian.reliability_model import BetaBelief, ToolReliabilityState
from bayesian_rag.core.inference_regimes import REGIME_SUMMARY, Regime, plan_for
from bayesian_rag.core.particle import (
    Action,
    EpisodeTrace,
    Observation,
    Particle,
    StepRecord,
)
from bayesian_rag.core.smc_runner import SMCConfig, SMCResult, SMCRunner

__version__ = "1.7.0"

__all__ = [
    # high level
    "Agent",
    "Result",
    "Step",
    "tool",
    "checker",
    "Tool",
    "ToolKind",
    "as_tool",
    # configuration
    "AgentConfig",
    "SMCConfig",
    "Regime",
    "REGIME_SUMMARY",
    # lower level
    "BayesianAgent",
    "SMCRunner",
    "SMCResult",
    "summarize_posterior",
    "plan_for",
    "build_proposer",
    "build_scorer",
    "build_selector",
    "seed_priors",
    # state types
    "Action",
    "Observation",
    "Particle",
    "StepRecord",
    "EpisodeTrace",
    "BetaBelief",
    "ToolReliabilityState",
    "__version__",
]
