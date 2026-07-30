"""Wrap a retriever as tools the `Agent` API already knows how to use.

Two shapes, matching the two failure modes a basic retriever actually has:

  `rank_tools`   -- one tool per rank position (top-1, top-2, top-3, ...) of a
                    single retrieval call. Useful when the retriever is
                    deterministic but its ranking can be *wrong*: the correct
                    document exists in the pool, just not at rank 1. Diversity
                    across particles comes from trying different ranks.

  `single_tool`  -- one tool that re-runs retrieval and returns its top-1
                    result on every call. Useful when the retriever itself is
                    *noisy*: repeated calls to the same query can surface
                    different documents. Diversity across particles comes from
                    independent re-draws.

Both return plain `bayesian_rag.tool`-decorated objects, so they compose with
`Agent` exactly like any other tool -- nothing about the agent or the default
proposer/scorer/selector needs to know retrieval is involved.
"""

from __future__ import annotations

from typing import Dict, Sequence

from bayesian_rag.rag.basic import NoisyRetriever, TermOverlapRetriever
from bayesian_rag.tools.decorator import Tool, tool as tool_decorator


def rank_tools(
    retriever: TermOverlapRetriever,
    query: str,
    top_k: int = 3,
    appeal_schedule: Sequence[float] = (0.9, 0.6, 0.4),
    reliability: float = 0.5,
) -> Dict[str, Tool]:
    """One tool per rank of a single (deterministic) retrieval call.

    `reliability` is left uninformative (0.5) for every rank by default, so
    the agent's advantage -- if any -- comes from the checker and selection
    process, not from a prior that already encodes which rank is correct.
    `appeal` decreases with rank, reflecting a real, non-cheating property: a
    system naturally trusts whatever its retriever ranked first slightly more
    before it has any other evidence.
    """
    results = retriever.retrieve(query, top_k=top_k)
    tools: Dict[str, Tool] = {}

    for i, scored in enumerate(results):
        name = f"retrieve_rank_{i + 1}"
        appeal = appeal_schedule[min(i, len(appeal_schedule) - 1)]
        text = scored.document.text

        def _make(passage: str):
            def _retrieve(q: str) -> str:
                return passage

            return _retrieve

        fn = _make(text)
        fn.__doc__ = f"Rank-{i + 1} retrieval result for the query."
        tools[name] = tool_decorator(
            fn, name=name, reliability=reliability, appeal=appeal, deterministic=True
        )

    return tools


def single_tool(
    retriever: NoisyRetriever,
    query: str,
    name: str = "retrieve",
    appeal: float = 0.8,
    reliability: float = 0.5,
) -> Dict[str, Tool]:
    """One tool that re-queries a noisy retriever and returns its top-1 result.

    `deterministic=False` is not optional here: memoising a stochastic
    retrieval call would hand every particle the same cached draw, silently
    disabling the mechanism this tool exists to demonstrate.
    """

    def _retrieve(q: str) -> str:
        results = retriever.retrieve(query, top_k=1)
        return results[0].document.text if results else ""

    _retrieve.__doc__ = "Retrieve the top result for the query."
    return {
        name: tool_decorator(
            _retrieve, name=name, reliability=reliability, appeal=appeal,
            deterministic=False,
        )
    }
