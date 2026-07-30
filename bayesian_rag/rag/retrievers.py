"""Retrievers people actually deploy: TF-IDF/cosine and BM25.

The earlier `TermOverlapRetriever` in `basic.py` scored by raw term frequency
with no length normalization. No production pipeline does that, and its
weakness is exactly what made a "lexical trap" demo easy to construct -- a
long keyword-repeating document outranks a short correct one purely because
it is long. Beating that retriever is not evidence of much.

These are the standard baselines instead:

  TfidfRetriever -- scikit-learn's TfidfVectorizer with cosine similarity.
                    This is the default lexical retriever in most tutorials
                    and in LangChain's own TFIDFRetriever. Length
                    normalization is built into cosine similarity, and IDF
                    down-weights terms that appear in every document.

  BM25Retriever  -- Okapi BM25 (k1=1.5, b=0.75, the standard defaults used by
                    Elasticsearch, Lucene, and rank_bm25). Length
                    normalization via `b`, term-frequency saturation via `k1`.
                    This is the strongest widely-used lexical retriever and
                    the honest baseline to beat.

Both expose the same `retrieve(query, top_k) -> List[ScoredDocument]`
interface as the basic retriever, so they drop into the existing tool wrappers
and demo harness unchanged.

Still no GPU and no network: TF-IDF and BM25 are CPU-only by construction, and
scikit-learn is a pure CPU dependency here.
"""

from __future__ import annotations

import math
import warnings
import random
from collections import Counter
from typing import List, Optional

from bayesian_rag.rag.basic import Document, ScoredDocument, tokenize


class TfidfRetriever:
    """TF-IDF vectors with cosine similarity, via scikit-learn.

    Uses the same configuration a standard pipeline would: sublinear TF
    scaling off, L2 normalization on (scikit-learn's default), English
    stop-words left in place so behaviour stays comparable to BM25.
    """

    def __init__(self, documents: List[Document]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.documents = list(documents)
        self._vectorizer = TfidfVectorizer(lowercase=True, norm="l2")
        self._matrix = self._vectorizer.fit_transform(
            [d.text for d in self.documents]
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[ScoredDocument]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        scored = [
            ScoredDocument(doc, float(score))
            for doc, score in zip(self.documents, sims)
        ]
        scored.sort(key=lambda s: (-s.score, s.document.doc_id))
        return scored[:top_k]


class BM25Retriever:
    """Okapi BM25 with the standard k1=1.5, b=0.75 defaults.

        score(q, d) = sum_t IDF(t) * (f(t,d) * (k1 + 1))
                                   / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

    `b` controls length normalization and `k1` controls how quickly repeated
    occurrences of a term stop adding score. Together they are precisely what
    defeats the "long document repeats the keyword" failure mode that raw term
    overlap falls for.
    """

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must lie in [0, 1], got {b}")

        self.documents = list(documents)
        self.k1 = k1
        self.b = b

        self._doc_terms = [Counter(tokenize(d.text)) for d in self.documents]
        self._doc_lens = [sum(c.values()) for c in self._doc_terms]
        n_docs = len(self.documents)
        self._avgdl = (sum(self._doc_lens) / n_docs) if n_docs else 0.0

        df = Counter()
        for terms in self._doc_terms:
            for term in terms:
                df[term] += 1

        # Okapi IDF with the +0.5 smoothing, floored at a small positive value
        # so a term appearing in every document contributes ~0 rather than a
        # negative score.
        self._idf = {}
        for term, count in df.items():
            self._idf[term] = max(
                1e-9, math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0)
            )

    def retrieve(self, query: str, top_k: int = 3) -> List[ScoredDocument]:
        q_terms = tokenize(query)
        scored: List[ScoredDocument] = []

        for doc, terms, length in zip(self.documents, self._doc_terms, self._doc_lens):
            score = 0.0
            for term in q_terms:
                if term not in terms:
                    continue
                freq = terms[term]
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * (length / self._avgdl if self._avgdl else 1.0)
                )
                score += self._idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / denom
            scored.append(ScoredDocument(doc, float(score)))

        scored.sort(key=lambda s: (-s.score, s.document.doc_id))
        return scored[:top_k]


class NoisyWrapper:
    """Degrade any retriever's ranking, modelling a flaky/replicated index.

    Identical in behaviour to `basic.NoisyRetriever` but accepts any object
    exposing `retrieve(query, top_k)`, so it composes with TF-IDF and BM25.
    """

    def __init__(self, inner, corruption: float, rng: random.Random):
        if not 0.0 <= corruption <= 1.0:
            raise ValueError(f"corruption must lie in [0, 1], got {corruption}")
        self.inner = inner
        self.corruption = corruption
        self.rng = rng

    @property
    def documents(self):
        return self.inner.documents

    def retrieve(self, query: str, top_k: int = 3) -> List[ScoredDocument]:
        pool_size = max(top_k, min(len(self.inner.documents), top_k + 2))
        ranked = self.inner.retrieve(query, top_k=pool_size)
        if self.rng.random() < self.corruption:
            shuffled = list(ranked)
            self.rng.shuffle(shuffled)
            ranked = shuffled
        return ranked[:top_k]





class LSARetriever:
    """Dense retrieval via Latent Semantic Analysis (TF-IDF + truncated SVD).

    This is a genuine *dense* retriever: documents and queries are projected
    into a low-dimensional continuous space and compared by cosine similarity,
    rather than matched on surface terms. It is the classical dense method,
    and unlike transformer embeddings it needs no pretrained weights, so it
    runs offline and on CPU.

    It is included to test a specific claim honestly. The lexical results in
    `RAG_REAL_RESULTS.md` note that a dense retriever "would likely solve"
    the coreference instance, since dense methods capture co-occurrence
    structure that surface matching misses. That claim should be checked
    rather than asserted, and LSA is the strongest dense retriever available
    without a network.

    IMPORTANT SCOPE LIMIT. LSA is not a transformer embedding model. It learns
    co-occurrence structure *from the indexed corpus only*, so on a corpus of
    two or three documents there is almost no co-occurrence signal to learn --
    the SVD has nothing to generalize from. Results here therefore bound what
    dense retrieval does on *tiny* corpora, and do not settle what a
    pretrained sentence-transformer would do, which has semantic knowledge
    from its training data independent of corpus size.
    """

    def __init__(self, documents: List[Document], n_components: Optional[int] = None):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.documents = list(documents)
        n_docs = len(self.documents)

        self._vectorizer = TfidfVectorizer(lowercase=True, norm="l2")
        tfidf = self._vectorizer.fit_transform([d.text for d in self.documents])

        # SVD rank cannot exceed min(n_docs, n_features) - 1.
        max_rank = max(1, min(n_docs, tfidf.shape[1]) - 1)
        self.n_components = min(n_components or max_rank, max_rank)

        if self.n_components < 2:
            # A rank-1 projection puts every document on one axis, so all
            # cosine similarities collapse to 1.0 and ranking is decided by
            # whatever the tiebreak happens to be. That is not a retrieval
            # result, and it silently looks like one -- so say so.
            warnings.warn(
                f"LSARetriever built with n_components={self.n_components} over "
                f"{n_docs} documents. A rank-1 projection makes all documents "
                f"collinear, so similarity scores are degenerate (all 1.0) and "
                f"ranking is arbitrary. Use a corpus of at least 3 documents "
                f"before interpreting these scores.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._svd = TruncatedSVD(n_components=self.n_components, random_state=0)
        self._doc_vectors = self._svd.fit_transform(tfidf)

    @property
    def is_degenerate(self) -> bool:
        """True when the projection rank is too low for scores to be meaningful."""
        return self.n_components < 2

    def retrieve(self, query: str, top_k: int = 3) -> List[ScoredDocument]:
        import numpy as np

        q_tfidf = self._vectorizer.transform([query])
        q_vec = self._svd.transform(q_tfidf)[0]

        q_norm = np.linalg.norm(q_vec)
        scored: List[ScoredDocument] = []
        for doc, vec in zip(self.documents, self._doc_vectors):
            d_norm = np.linalg.norm(vec)
            sim = float(np.dot(q_vec, vec) / (q_norm * d_norm)) if q_norm and d_norm else 0.0
            scored.append(ScoredDocument(doc, sim))

        scored.sort(key=lambda s: (-s.score, s.document.doc_id))
        return scored[:top_k]


RETRIEVERS = {
    "tfidf": TfidfRetriever,
    "bm25": BM25Retriever,
    "lsa": LSARetriever,
}
