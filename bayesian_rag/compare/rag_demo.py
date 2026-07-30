"""Normal vs. Bayesian RAG on three very basic, self-contained examples.

No embeddings, no GPU, no network, no LLM call for generation -- everything
here is standard-library term-frequency retrieval plus extractive answers, so
the whole pipeline runs on CPU in milliseconds. See `bayesian_rag/rag/basic.py`
for the retriever and `bayesian_rag/rag/tools.py` for how it becomes tools.

All entities and facts in the three corpora are fictional, invented for this
demo. That is deliberate, not decorative: it forces every answer to come from
the retrieved text rather than from anything a language model might already
"know," which is the actual property a RAG pipeline is supposed to have.

Normal RAG here means the simplest possible pipeline: one retrieval call, take
the top-1 result, return it. No checking, no retry -- this is what "basic RAG"
usually means in a tutorial, and it is what all three instances are built to
break.

Bayesian RAG means handing the exact same tools to `Agent` under the SMC
regime, with no custom proposer, scorer, or selector -- the point being that
nothing beyond declaring the tools is needed to get the improvement:

    agent = Agent(list(tools.values()))
    answer = agent.run(query).answer

Three instances, three distinct failure modes:

  1. LEXICAL TRAP     -- the retriever's own top-1 choice is wrong because a
                         long, keyword-repeating document outranks a short,
                         correct one. Deterministic corpus; the fix is trying
                         other ranks and checking them.

  2. STALE DOCUMENT    -- two documents agree on topic but disagree on the
                         current fact; a corroborating third document (a
                         "changelog") is needed to tell which one is current.

  3. NOISY INDEX       -- the retriever itself is unreliable: the same query
                         surfaces different top-1 results on different calls.
                         Deterministic corpora do not apply; the fix is
                         re-querying and checking, not re-ranking.

Run:
    python -m bayesian_rag.compare.rag_demo
    python -m bayesian_rag.compare.rag_demo --trials 200 --json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from bayesian_rag import Agent, checker as checker_tool
from bayesian_rag.rag.basic import (
    Document,
    NoisyRetriever,
    TermOverlapRetriever,
    contains_fact,
    contains_year,
)
from bayesian_rag.rag.tools import rank_tools, single_tool
from bayesian_rag.utils.statistics import Comparison, Proportion

PARTICLES = 16
MAX_STEPS = 4


# =============================================================================
# Instance 1 -- lexical trap
# =============================================================================


def instance_1_tools_and_query():
    """A long, keyword-stuffed distractor outranks a short, correct passage."""
    docs = [
        Document(
            "distractor",
            "The Lumenwell Bridge is one of the region's most photographed "
            "landmarks. Visitors flock to the Lumenwell Bridge every summer "
            "for the view, and the cafes near the Lumenwell Bridge are a "
            "popular stop on the Lumenwell Bridge walking tour.",
        ),
        Document(
            "correct",
            "The Lumenwell Bridge was completed in 1927 after four years "
            "of construction.",
        ),
    ]
    query = "What year was the Lumenwell Bridge completed?"
    retriever = TermOverlapRetriever(docs)

    tools = rank_tools(retriever, query, top_k=2)

    @checker_tool
    def has_year(text: str) -> bool:
        """Does the retrieved passage actually contain a year?"""
        return contains_year(text)

    tools["has_year"] = has_year
    return tools, query, "1927"


# =============================================================================
# Instance 2 -- stale vs. current document
# =============================================================================


def instance_2_tools_and_query():
    """Two documents on the same topic disagree; a changelog says which is current."""
    stale = Document(
        "stale",
        "As reported in 2015, the Kestrel Solar Array has a total "
        "generating capacity of the Kestrel Solar Array at 40 MW.",
    )
    current = Document(
        "current",
        "Following the 2020 expansion, the Kestrel Solar Array's capacity "
        "was upgraded to 95 MW.",
    )
    changelog_text = (
        "Changelog: the previously reported Kestrel Solar Array figure of "
        "40 MW is outdated. As of the 2020 expansion, the correct current "
        "capacity is 95 MW."
    )
    docs = [stale, current]
    query = "What is the current capacity of the Kestrel Solar Array?"
    retriever = TermOverlapRetriever(docs)

    tools = rank_tools(retriever, query, top_k=2)

    def _extract_mw(text: str) -> Optional[str]:
        import re

        m = re.search(r"(\d+)\s*MW", text or "")
        return m.group(1) if m else None

    def _current_value_from_changelog(text: str) -> Optional[str]:
        """Extract specifically the *current* figure, not just the first number.

        A naive "first number in the text" extraction is wrong here: the
        changelog mentions the outdated figure before the current one
        ("...40 MW is outdated... current capacity is 95 MW"), so grabbing
        the first match would silently validate the stale answer. The
        checker has to look for the figure attached to "current", the same
        way a real citation-freshness check would target the relevant span
        rather than the nearest number.
        """
        import re

        m = re.search(r"current[^.]*?(\d+)\s*MW", text, flags=re.IGNORECASE)
        return m.group(1) if m else _extract_mw(text)

    @checker_tool
    def matches_changelog(text: str) -> bool:
        """Cross-reference a candidate figure against the changelog's current value.

        The checker does not know the answer directly -- it extracts the
        current figure from a separate corroborating document at runtime and
        compares against that, the same indirection a real citation-freshness
        check would use.
        """
        current_value = _current_value_from_changelog(changelog_text)
        candidate_value = _extract_mw(text)
        return candidate_value is not None and candidate_value == current_value

    tools["matches_changelog"] = matches_changelog
    return tools, query, "95 MW"


# =============================================================================
# Instance 3 -- noisy retrieval index
# =============================================================================


def instance_3_tools_and_query(seed: int, corruption: float = 0.9):
    """The retriever itself is unreliable; repeated queries disagree."""
    docs = [
        Document("correct", "The Thornbury Observatory was founded in 1963."),
        Document("filler_a", "The Alderbrook Lighthouse was decommissioned in 1988."),
        Document("filler_b", "The Marrow Canal was widened in 2004."),
    ]
    query = "In what year was the Thornbury Observatory founded?"
    base = TermOverlapRetriever(docs)
    noisy = NoisyRetriever(base, corruption=corruption, rng=random.Random(seed))

    tools = single_tool(noisy, query, name="retrieve")

    @checker_tool
    def about_thornbury(text: str) -> bool:
        """Does the passage actually mention Thornbury Observatory and a year?"""
        return "thornbury" in (text or "").lower() and contains_year(text)

    tools["about_thornbury"] = about_thornbury
    return tools, query, "1963"


# =============================================================================
# Runners
# =============================================================================


def run_normal_rag(tools: Dict[str, Any], query: str) -> str:
    """The simplest possible RAG pipeline: one retrieval call, top-1, no checking.

    Only the rank-1 (or the single) retrieval tool is invoked -- checkers are
    not consulted, and no other rank is tried.
    """
    rank_1_name = "retrieve_rank_1" if "retrieve_rank_1" in tools else "retrieve"
    return tools[rank_1_name].invoke({"q": query})["answer"]


def run_bayesian_rag(tools: Dict[str, Any], query: str, seed: int = 0):
    """Identical tools, handed to `Agent` under SMC. No custom wiring."""
    agent = Agent(list(tools.values()), seed=seed, particles=PARTICLES, max_steps=MAX_STEPS)
    return agent.run(query)


# =============================================================================
# Experiments
# =============================================================================


def run_instance_1_and_2(builder: Callable, trials: int):
    """Deterministic corpora: normal RAG is a single fixed pass; Bayesian RAG's
    own selection sampling is stochastic, so its accuracy is measured over seeds.
    """
    tools0, query, expected = builder()
    normal_answer = run_normal_rag(tools0, query)
    normal_correct = contains_fact(normal_answer, expected)

    wins = 0
    for seed in range(trials):
        tools, _, _ = builder()
        result = run_bayesian_rag(tools, query, seed=seed)
        wins += contains_fact(result.answer, expected)

    return normal_answer, expected, normal_correct, Proportion(wins, trials)


def run_instance_3(trials: int):
    """Both pipelines see the same noisy retriever; each trial is a fresh draw."""
    normal_wins = 0
    bayesian_wins = 0

    for seed in range(trials):
        tools_n, query, expected = instance_3_tools_and_query(seed=seed)
        normal_wins += contains_fact(run_normal_rag(tools_n, query), expected)

        tools_b, _, _ = instance_3_tools_and_query(seed=seed)
        result = run_bayesian_rag(tools_b, query, seed=seed)
        bayesian_wins += contains_fact(result.answer, expected)

    return Proportion(normal_wins, trials), Proportion(bayesian_wins, trials)


def run_all(trials: int = 150) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for name, desc, builder in [
        ("1. Lexical trap", "Long distractor outranks short correct passage.",
         instance_1_tools_and_query),
        ("2. Stale document", "Two docs disagree; a changelog says which is current.",
         instance_2_tools_and_query),
    ]:
        normal_answer, expected, normal_correct, bayes_prop = run_instance_1_and_2(
            builder, trials
        )
        rows.append({
            "instance": name,
            "description": desc,
            "expected": expected,
            "normal_answer": normal_answer,
            "normal_correct": normal_correct,
            "bayesian_accuracy": bayes_prop,
        })

    normal_prop, bayes_prop = run_instance_3(trials)
    rows.append({
        "instance": "3. Noisy index",
        "description": "Retriever itself unreliable; repeated queries disagree.",
        "expected": "1963",
        "normal_accuracy": normal_prop,
        "bayesian_accuracy": bayes_prop,
    })

    return {"trials": trials, "rows": rows}


def print_report(trials: int = 150) -> None:
    data = run_all(trials)

    print("=" * 78)
    print("Normal RAG (single retrieval, top-1, no checking) vs. Bayesian RAG")
    print("(same tools, Agent under SMC, no custom code)")
    print("=" * 78)
    print("No embeddings, no GPU, no network -- term-frequency retrieval only.\n")

    for row in data["rows"]:
        print("-" * 78)
        print(f"{row['instance']}: {row['description']}")
        print(f"  expected fact: {row['expected']!r}")

        if "normal_answer" in row:
            print(f"  Normal RAG answer : {row['normal_answer'][:70]!r}")
            print(f"  Normal RAG correct: {row['normal_correct']}  (deterministic, single pass)")
            print(f"  Bayesian RAG      : {row['bayesian_accuracy'].format()}  "
                  f"({trials} seeds)")
        else:
            print(f"  Normal RAG        : {row['normal_accuracy'].format()}  "
                  f"({trials} seeds, noisy retriever)")
            print(f"  Bayesian RAG      : {row['bayesian_accuracy'].format()}  "
                  f"({trials} seeds)")

            cmp_ = Comparison("Bayesian", "Normal", row["bayesian_accuracy"],
                              row["normal_accuracy"])
            print(f"  {cmp_}")
        print()

    print("=" * 78)
    print("How to read instances 1-2: Normal RAG's answer is fixed (no")
    print("randomness in a single deterministic pass); Bayesian RAG's own")
    print("selection sampling is stochastic, so its accuracy is reported over")
    print(f"{trials} seeds rather than a single run.\n")


def to_json(trials: int = 150) -> Dict[str, Any]:
    data = run_all(trials)
    out = {"trials": trials, "rows": []}
    for row in data["rows"]:
        r = dict(row)
        if "bayesian_accuracy" in r:
            p = r["bayesian_accuracy"]
            r["bayesian_accuracy"] = {
                "rate": round(p.rate, 4), "ci_lower": round(p.interval[0], 4),
                "ci_upper": round(p.interval[1], 4), "trials": p.trials,
            }
        if "normal_accuracy" in r:
            p = r["normal_accuracy"]
            r["normal_accuracy"] = {
                "rate": round(p.rate, 4), "ci_lower": round(p.interval[0], 4),
                "ci_upper": round(p.interval[1], 4), "trials": p.trials,
            }
        out["rows"].append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normal vs Bayesian RAG, 3 instances")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(to_json(args.trials), indent=2))
    else:
        print_report(args.trials)


if __name__ == "__main__":
    main()
