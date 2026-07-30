"""Normal vs. Bayesian RAG using retrievers people actually deploy.

WHY THIS REPLACES THE EARLIER DEMO
----------------------------------
`rag_demo.py` scored retrieval by raw term-frequency overlap with no length
normalization. No production pipeline does that, and its specific weakness --
a long document that repeats a keyword outranks a short correct one -- is what
made that demo's "lexical trap" instance work. Both TF-IDF/cosine and BM25
solve that instance immediately:

    term-overlap  top-1 = distractor   (wrong)
    tfidf         top-1 = correct
    bm25          top-1 = correct

So that instance measured a bad retriever, not a good orchestrator. It is
dropped here and replaced with a failure mode real retrievers actually have.

WHAT THIS USES INSTEAD
----------------------
BM25 (Okapi, k1=1.5, b=0.75 -- Elasticsearch/Lucene defaults) as the default,
with TF-IDF/cosine (scikit-learn `TfidfVectorizer`, as in LangChain's own
TFIDFRetriever) available via `--retriever tfidf`. Both are standard lexical
baselines, both CPU-only, no GPU and no network.

THE THREE INSTANCES
-------------------
Each is a failure mode BM25 and TF-IDF genuinely exhibit, verified below
rather than asserted:

  1. COREFERENCE     -- the answer-bearing passage refers to the entity as
                        "the company" instead of repeating its name, so it
                        loses on lexical overlap to a passage that names the
                        entity repeatedly but does not answer the question.
                        The answer sits at rank 2. Neither retriever does
                        coreference resolution; this is the single most common
                        real RAG failure ("answer is in top-k, not top-1").

  2. STALE DOCUMENT   -- two passages are equally on-topic but disagree,
                        because one is outdated. Lexical retrievers have no
                        notion of recency, so nothing in the scoring function
                        can prefer the current one.

  3. NOISY INDEX      -- the retriever's ranking is degraded by replication
                        lag or an eventually-consistent shard. Orthogonal to
                        which scoring function is used: it wraps whichever
                        retriever you chose.

Normal RAG = one retrieval call, top-1, no checking.
Bayesian RAG = the same retriever's top-k as tools, handed to `Agent`, no
custom proposer/scorer/selector.

Run:
    python -m bayesian_rag.compare.rag_real
    python -m bayesian_rag.compare.rag_real --retriever tfidf
    python -m bayesian_rag.compare.rag_real --trials 300 --json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from bayesian_rag import Agent, checker as checker_tool
from bayesian_rag.rag.basic import Document, contains_fact, contains_year
from bayesian_rag.rag.retrievers import (
    BM25Retriever,
    LSARetriever,
    NoisyWrapper,
    TfidfRetriever,
)
from bayesian_rag.rag.tools import rank_tools
from bayesian_rag.tools.decorator import tool as tool_decorator
from bayesian_rag.utils.statistics import Comparison, Proportion

PARTICLES = 16
MAX_STEPS = 4

RETRIEVER_CLASSES = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
    "lsa": LSARetriever,
}


# =============================================================================
# Instance 1 -- coreference: the answer says "the company", not the entity name
# =============================================================================


def instance_coreference(retriever_name: str = "bm25"):
    docs = [
        Document(
            "no_answer",
            "Veridian Labs operates three research sites. Veridian Labs "
            "focuses on materials science, and Veridian Labs publishes "
            "its findings annually.",
        ),
        Document(
            "has_answer",
            "The company was founded by Dr. Ana Sorel in 1998 after she "
            "left her academic post.",
        ),
        Document(
            "filler",
            "Quarterly logistics report for the northern distribution corridor.",
        ),
    ]
    query = "Who founded Veridian Labs?"
    retriever = RETRIEVER_CLASSES[retriever_name](docs)

    tools = rank_tools(retriever, query, top_k=3)

    @checker_tool
    def names_a_founder(text: str) -> bool:
        """Does this passage actually name a person as founder?

        A generic answer-shape check: looks for founding language plus a
        capitalised personal name. It does not encode "Ana Sorel" anywhere.
        """
        t = text or ""
        has_founding_verb = bool(re.search(r"\bfound(ed|er)\b", t, re.IGNORECASE))
        has_person = bool(re.search(r"\b(Dr|Mr|Ms|Mrs|Prof)\.?\s+[A-Z][a-z]+", t)) or bool(
            re.search(r"\bby\s+[A-Z][a-z]+\s+[A-Z][a-z]+", t)
        )
        return has_founding_verb and has_person

    tools["names_a_founder"] = names_a_founder
    return tools, query, "Ana Sorel"


# =============================================================================
# Instance 2 -- staleness: equally on-topic, one outdated
# =============================================================================


def instance_staleness(retriever_name: str = "bm25"):
    """Vocabulary drift plus staleness: the outdated passage matches the query's
    wording better than the current one does.

    This is the realistic shape of the problem. An original spec uses the
    canonical phrasing a user's question naturally mirrors ("generating
    capacity"); a later amendment uses internal shorthand ("output rating").
    The retriever scores the outdated passage far higher -- BM25 5.40 vs 0.22 --
    because lexical scoring has no notion of recency and no way to know the
    two passages describe the same quantity.

    An earlier version of this instance had both passages phrased identically,
    and BM25 then ranked the *current* one first: Normal RAG already succeeded,
    so there was no failure to fix. The `retriever_diagnostics` function exists
    to catch exactly that.
    """
    docs = [
        Document(
            "stale",
            "The Kestrel Solar Array current generating capacity is rated at "
            "40 MW under the standard facilities survey methodology.",
        ),
        Document(
            "current",
            "Post-expansion output rating for Kestrel: 95 MW.",
        ),
        # Padding, not decoration. With only two documents an LSA projection is
        # rank-1: every document collapses onto one axis, all cosine scores
        # become 1.0, and the "ranking" is decided by an arbitrary tiebreak.
        # That produced a spurious apparent win for dense retrieval here. These
        # off-topic facility notes give the SVD enough rank for its scores to
        # mean anything, and they do not mention capacity or either figure, so
        # they cannot affect which of the two real passages is correct.
        Document(
            "pad_a",
            "Facility note: routine inspection of auxiliary systems and the "
            "grid interconnect was completed on schedule.",
        ),
        Document(
            "pad_b",
            "Facility note: perimeter fencing and access road maintenance "
            "were carried out by the contracted vendor.",
        ),
    ]
    changelog = (
        "Facilities changelog: the 2015 figure for the Kestrel Solar Array is "
        "superseded. Following the 2020 expansion the correct current "
        "capacity is 95 MW."
    )
    query = "What is the current generating capacity of the Kestrel Solar Array?"
    retriever = RETRIEVER_CLASSES[retriever_name](docs)

    # Depth 4, not 2: after padding the corpus so LSA's projection is
    # non-degenerate, the correct passage sits at rank 3 (BM25/TF-IDF) or 4
    # (LSA). Retrieving only the top 2 would put it out of the agent's reach
    # entirely, which would test nothing.
    tools = rank_tools(retriever, query, top_k=4)

    def _extract_mw(text: str) -> Optional[str]:
        m = re.search(r"(\d+)\s*MW", text or "")
        return m.group(1) if m else None

    def _current_from_changelog(text: str) -> Optional[str]:
        # Target the figure attached to "current", not merely the first number:
        # the changelog mentions the superseded year before the current value.
        m = re.search(r"current[^.]*?(\d+)\s*MW", text, flags=re.IGNORECASE)
        return m.group(1) if m else _extract_mw(text)

    @checker_tool
    def matches_changelog(text: str) -> bool:
        """Cross-reference a candidate figure against a separate changelog."""
        current_value = _current_from_changelog(changelog)
        candidate = _extract_mw(text)
        return candidate is not None and candidate == current_value

    tools["matches_changelog"] = matches_changelog
    return tools, query, "95 MW"


# =============================================================================
# Instance 3 -- noisy index (wraps whichever retriever is in use)
# =============================================================================


def instance_noisy(seed: int, retriever_name: str = "bm25", corruption: float = 0.9):
    docs = [
        Document("correct", "The Thornbury Observatory was founded in 1963."),
        Document("filler_a", "The Alderbrook Lighthouse was decommissioned in 1988."),
        Document("filler_b", "The Marrow Canal was widened in 2004."),
    ]
    query = "In what year was the Thornbury Observatory founded?"
    base = RETRIEVER_CLASSES[retriever_name](docs)
    noisy = NoisyWrapper(base, corruption=corruption, rng=random.Random(seed))

    def _retrieve(q: str) -> str:
        results = noisy.retrieve(query, top_k=1)
        return results[0].document.text if results else ""

    _retrieve.__doc__ = "Retrieve the top result for the query."
    tools = {
        "retrieve": tool_decorator(
            _retrieve, name="retrieve", reliability=0.5, appeal=0.8,
            deterministic=False,
        )
    }

    @checker_tool
    def about_thornbury(text: str) -> bool:
        """Does the passage mention Thornbury Observatory and contain a year?"""
        return "thornbury" in (text or "").lower() and contains_year(text)

    tools["about_thornbury"] = about_thornbury
    return tools, query, "1963"


# =============================================================================
# Pipelines
# =============================================================================


def run_normal(tools: Dict[str, Any], query: str) -> str:
    """One retrieval call, top-1, no checking."""
    name = "retrieve_rank_1" if "retrieve_rank_1" in tools else "retrieve"
    return tools[name].invoke({"q": query})["answer"]


def run_bayesian(tools: Dict[str, Any], query: str, seed: int = 0):
    """Same tools, `Agent` under SMC, no custom wiring."""
    agent = Agent(list(tools.values()), seed=seed, particles=PARTICLES, max_steps=MAX_STEPS)
    return agent.run(query)


# =============================================================================
# Diagnostics: verify each instance actually defeats the real retriever
# =============================================================================


def retriever_diagnostics(retriever_name: str) -> List[Dict[str, Any]]:
    """Report where the correct passage ranks under the real retriever.

    An instance is only worth reporting if the retriever genuinely fails it.
    This makes that checkable rather than asserted.
    """
    rows = []

    docs_1 = [
        Document("no_answer",
                 "Veridian Labs operates three research sites. Veridian Labs "
                 "focuses on materials science, and Veridian Labs publishes "
                 "its findings annually."),
        Document("has_answer",
                 "The company was founded by Dr. Ana Sorel in 1998 after she "
                 "left her academic post."),
        Document("filler",
                 "Quarterly logistics report for the northern distribution corridor."),
    ]
    r1 = RETRIEVER_CLASSES[retriever_name](docs_1)
    ranked = r1.retrieve("Who founded Veridian Labs?", top_k=len(docs_1))
    rank_of_answer = next(
        (i for i, s in enumerate(ranked, 1) if s.document.doc_id == "has_answer"), None
    )
    rows.append({
        "instance": "1. Coreference",
        "top_1": ranked[0].document.doc_id,
        "correct_rank": rank_of_answer,
        "retriever_fails": ranked[0].document.doc_id != "has_answer",
    })

    docs_2 = [
        Document("stale",
                 "The Kestrel Solar Array current generating capacity is rated at "
                 "40 MW under the standard facilities survey methodology."),
        Document("current",
                 "Post-expansion output rating for Kestrel: 95 MW."),
        Document("pad_a",
                 "Facility note: routine inspection of auxiliary systems and the "
                 "grid interconnect was completed on schedule."),
        Document("pad_b",
                 "Facility note: perimeter fencing and access road maintenance "
                 "were carried out by the contracted vendor."),
    ]
    r2 = RETRIEVER_CLASSES[retriever_name](docs_2)
    ranked2 = r2.retrieve(
        "What is the current generating capacity of the Kestrel Solar Array?",
        top_k=len(docs_2),
    )
    rank_of_current = next(
        (i for i, s in enumerate(ranked2, 1) if s.document.doc_id == "current"), None
    )
    rows.append({
        "instance": "2. Staleness",
        "top_1": ranked2[0].document.doc_id,
        "correct_rank": rank_of_current,
        "retriever_fails": ranked2[0].document.doc_id != "current",
    })

    rows.append({
        "instance": "3. Noisy index",
        "top_1": "(varies per call)",
        "correct_rank": None,
        "retriever_fails": True,
    })
    return rows


# =============================================================================
# Experiments
# =============================================================================


def run_deterministic_instance(builder: Callable, retriever_name: str, trials: int):
    tools0, query, expected = builder(retriever_name)
    normal_answer = run_normal(tools0, query)
    normal_correct = contains_fact(normal_answer, expected)

    wins = 0
    for seed in range(trials):
        tools, _, _ = builder(retriever_name)
        wins += contains_fact(run_bayesian(tools, query, seed=seed).answer, expected)

    return normal_answer, expected, normal_correct, Proportion(wins, trials)


def run_noisy_instance(retriever_name: str, trials: int):
    normal_wins = bayes_wins = 0
    for seed in range(trials):
        tools_n, query, expected = instance_noisy(seed, retriever_name)
        normal_wins += contains_fact(run_normal(tools_n, query), expected)

        tools_b, _, _ = instance_noisy(seed, retriever_name)
        bayes_wins += contains_fact(run_bayesian(tools_b, query, seed=seed).answer, expected)

    return Proportion(normal_wins, trials), Proportion(bayes_wins, trials)


def run_all(retriever_name: str = "bm25", trials: int = 300) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for label, builder in [
        ("1. Coreference", instance_coreference),
        ("2. Staleness", instance_staleness),
    ]:
        normal_answer, expected, normal_correct, bayes = run_deterministic_instance(
            builder, retriever_name, trials
        )
        rows.append({
            "instance": label,
            "expected": expected,
            "normal_answer": normal_answer,
            "normal_correct": normal_correct,
            "bayesian": bayes,
        })

    normal_p, bayes_p = run_noisy_instance(retriever_name, trials)
    rows.append({
        "instance": "3. Noisy index",
        "expected": "1963",
        "normal_accuracy": normal_p,
        "bayesian": bayes_p,
    })

    return {
        "retriever": retriever_name,
        "trials": trials,
        "diagnostics": retriever_diagnostics(retriever_name),
        "rows": rows,
    }


def print_report(retriever_name: str = "bm25", trials: int = 300) -> None:
    data = run_all(retriever_name, trials)

    print("=" * 78)
    print(f"Normal vs. Bayesian RAG  |  retriever: {retriever_name.upper()}")
    print("=" * 78)
    print("Normal RAG = one retrieval call, top-1, no checking.")
    print("Bayesian RAG = same retriever's top-k as tools, Agent under SMC.")
    print("CPU-only: BM25 / TF-IDF, no embeddings, no GPU, no network.\n")

    print("Retriever diagnostics -- does the real retriever actually fail here?")
    print("-" * 78)
    for d in data["diagnostics"]:
        rank = d["correct_rank"]
        rank_str = f"rank {rank}" if rank else "n/a"
        print(f"  {d['instance']:18s} top-1={str(d['top_1']):18s} "
              f"correct at {rank_str:8s} fails={d['retriever_fails']}")
    print()

    for row in data["rows"]:
        print("-" * 78)
        print(f"{row['instance']}   (expected: {row['expected']!r})")
        if "normal_answer" in row:
            print(f"  Normal RAG   : {'CORRECT' if row['normal_correct'] else 'WRONG'}"
                  f"   {row['normal_answer'][:52]!r}")
            print(f"  Bayesian RAG : {row['bayesian'].format()}   ({trials} seeds)")
        else:
            print(f"  Normal RAG   : {row['normal_accuracy'].format()}   ({trials} seeds)")
            print(f"  Bayesian RAG : {row['bayesian'].format()}   ({trials} seeds)")
            cmp_ = Comparison("Bayesian", "Normal", row["bayesian"], row["normal_accuracy"])
            print(f"  {cmp_}")
        print()


def to_json(retriever_name: str = "bm25", trials: int = 300) -> Dict[str, Any]:
    data = run_all(retriever_name, trials)

    def _ser(p: Proportion):
        return {"rate": round(p.rate, 4), "ci_lower": round(p.interval[0], 4),
                "ci_upper": round(p.interval[1], 4), "trials": p.trials}

    out = {"retriever": data["retriever"], "trials": data["trials"],
           "diagnostics": data["diagnostics"], "rows": []}
    for row in data["rows"]:
        r = dict(row)
        for key in ("bayesian", "normal_accuracy"):
            if key in r:
                r[key] = _ser(r[key])
        out["rows"].append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normal vs Bayesian RAG with production-standard retrievers"
    )
    parser.add_argument("--retriever", choices=["bm25", "tfidf", "lsa"], default="bm25")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--both", action="store_true",
                        help="Run against BM25, TF-IDF, and LSA in turn.")
    args = parser.parse_args()

    names = ["bm25", "tfidf", "lsa"] if args.both else [args.retriever]

    if args.json:
        print(json.dumps({n: to_json(n, args.trials) for n in names}, indent=2))
    else:
        for n in names:
            print_report(n, args.trials)


if __name__ == "__main__":
    main()
