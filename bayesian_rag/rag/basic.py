"""A minimal, dependency-free RAG retriever.

No embeddings, no GPU, no network: scoring is raw term-frequency overlap over
a small in-memory corpus, using nothing beyond the standard library. This is
deliberately basic -- see `bayesian_rag/compare/rag_demo.py` for the specific
failure modes that follow from that -- but the interface (query in, ranked
documents out) is the same shape a production retriever exposes, so nothing
here would need to change to plug in a real embedding index later.

Every step of a RAG pipeline (retrieve, verify, generate) is CPU-only. A GPU
only becomes relevant if you choose to self-host an embedding model or a
generation model at scale; calling either as an API, or using a retriever
this simple, needs no GPU at any point.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_WORD_RE = re.compile(r"[a-zA-Z]+")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; nothing fancier than that."""
    return [w.lower() for w in _WORD_RE.findall(text)]


@dataclass(frozen=True)
class Document:
    """One corpus entry.

    `tags` is bookkeeping for building a scenario (e.g. marking which document
    is the "correct" one) -- the retriever and checkers never read it; only
    the demo's accuracy-scoring code does, after the fact.
    """

    doc_id: str
    text: str
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class TermOverlapRetriever:
    """Ranks documents by raw term-frequency overlap with the query.

    Deliberately not length-normalized -- no TF-IDF, no BM25. This is what
    makes a real, common failure mode reproducible: a long document that
    repeats the query's words many times can outrank a short document that
    states the answer exactly once. A production retriever (TF-IDF, BM25, or
    an embedding index) would largely correct for this; the point of using
    something this basic is to make the failure visible rather than assume it
    away.
    """

    def __init__(self, documents: List[Document]):
        self.documents = list(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[ScoredDocument]:
        q_terms = Counter(tokenize(query))
        scored = []
        for doc in self.documents:
            d_terms = Counter(tokenize(doc.text))
            score = float(sum(d_terms[t] * n for t, n in q_terms.items()))
            scored.append(ScoredDocument(doc, score))
        scored.sort(key=lambda s: (-s.score, s.document.doc_id))
        return scored[:top_k]


class NoisyRetriever:
    """Wraps a retriever and randomly perturbs its ranking.

    Models a degraded or eventually-consistent index: with probability
    `corruption`, a call returns a random shuffle of the underlying top-N
    instead of the true score order. Each call is an independent draw, seeded
    from the generator passed in -- repeated queries against the same index
    must sometimes surface the right document and sometimes not, or there is
    nothing for a re-drawing strategy to exploit.
    """

    def __init__(self, inner: TermOverlapRetriever, corruption: float, rng: random.Random):
        if not 0.0 <= corruption <= 1.0:
            raise ValueError(f"corruption must lie in [0, 1], got {corruption}")
        self.inner = inner
        self.corruption = corruption
        self.rng = rng

    def retrieve(self, query: str, top_k: int = 3) -> List[ScoredDocument]:
        pool_size = max(top_k, min(len(self.inner.documents), top_k + 2))
        ranked = self.inner.retrieve(query, top_k=pool_size)
        if self.rng.random() < self.corruption:
            shuffled = list(ranked)
            self.rng.shuffle(shuffled)
            ranked = shuffled
        return ranked[:top_k]


def contains_fact(text: str, expected_substring: str) -> bool:
    """Case-insensitive substring check, used to score final-answer accuracy."""
    return expected_substring.lower() in (text or "").lower()


def contains_year(text: str) -> bool:
    """Heuristic answer-shape check: does this passage contain a 4-digit year?

    A generic, non-cheating verifier -- it checks that the retrieved passage
    is complete enough to plausibly answer a "what year" question, without
    looking up or encoding the specific correct year anywhere.
    """
    return bool(_YEAR_RE.search(text or ""))
