# Normal vs. Bayesian RAG with Production-Standard Retrievers

```bash
python -m bayesian_rag.compare.rag_real --both --trials 300     # BM25, TF-IDF, LSA
```

---

## Two retractions

**1. The original demo's "lexical trap" was a strawman.** An earlier version of
this comparison scored retrieval by raw term-frequency overlap with no length
normalization — a retriever nobody deploys. Its headline instance worked only
because of that weakness. Both standard retrievers solve it outright:

| Retriever | top-1 on the old Instance 1 |
|---|---|
| term-overlap (old demo) | `distractor` — **wrong** |
| TF-IDF / cosine | `correct` |
| BM25 | `correct` |

That instance measured a bad retriever, not a good orchestrator. Dropped, with
a regression test pinning the finding.

**2. "A dense retriever would likely solve coreference" was wrong.** The
previous version of this document speculated that dense retrieval would handle
Instance 1. It does not. LSA (TF-IDF + truncated SVD, a genuine dense method)
ranks the answer at **rank 3 — worse than BM25's rank 2**.

*Scope:* this falsifies the claim for LSA-style dense retrieval, which learns
co-occurrence structure from the indexed corpus only. It does not settle what a
pretrained sentence-transformer would do, since those carry semantic knowledge
independent of corpus size. `sentence-transformers` is not installable in this
offline environment, so that remains untested.

---

## A framework bug this work uncovered

While diagnosing an unexpectedly low score, inspection of actual winning
trajectories showed paths like:

```
r3 > CHK > CHK > r1 > r2        # found the answer, validated it, then discarded it
```

`Particle.final_answer()` preferred **recency over validation status**. A
trajectory that found the correct passage, had a checker confirm it, and then
looked at something else would return whatever it touched last — throwing away
the confirmation it had just paid to obtain.

Fixed: preference order is now (1) an answer a checker confirmed, (2) the most
recent answer no checker rejected, (3) the most recent answer. Four regression
tests in `tests/test_regressions.py`.

**Impact — this was not a demo tweak.** It moved results across the whole project:

| | before | after |
|---|---|---|
| RAG coreference | 0.737 | **1.000** |
| RAG staleness | 0.557 | **0.903** |
| RAG noisy index | 0.843 | **1.000** |
| Paper Exp-1 chain @ acc 0.50 | 0.887 | **1.000** |
| Paper Exp-1 chain @ acc 0.60 | 0.900 | **1.000** |

It also **corrects an earlier explanation**. `PAPER_RESULTS.md` attributed the
mid-range dip (accuracy 0.4–0.6) to the step/particle budget, and showed more
particles removing it. That explanation was wrong: the dip was this bug. At
default budget it is now gone entirely. The low-end drop at accuracy ≤ 0.10
*is* a genuine budget effect and remains.

---

## What is used now

- **BM25** (Okapi, k1=1.5, b=0.75 — Elasticsearch/Lucene defaults)
- **TF-IDF / cosine** (scikit-learn `TfidfVectorizer`, as in LangChain's `TFIDFRetriever`)
- **LSA** (TF-IDF + truncated SVD) — dense, included to test the claim above

All CPU-only. No embeddings, no GPU, no network.

Every instance is verified to genuinely defeat every retriever before its result
is reported — asserted in `test_all_three_retrievers_fail_every_instance`:

| Instance | BM25 | TF-IDF | LSA |
|---|---|---|---|
| 1. Coreference | answer at rank 2 | rank 2 | rank 3 |
| 2. Staleness | rank 3 | rank 3 | rank 4 |
| 3. Noisy index | varies per call | varies | varies |

---

## Results

300 seeds, 95% Wilson intervals.

| Instance | Normal RAG | Bayesian (BM25) | Bayesian (TF-IDF) | Bayesian (LSA) |
|---|---|---|---|---|
| 1. Coreference | **wrong**, always | **1.000 [0.987, 1.000]** | **1.000** | 0.967 [0.940, 0.982] |
| 2. Staleness | **wrong**, always | **0.903 [0.865, 0.932]** | **0.903** | **0.903** |
| 3. Noisy index | 0.420 [0.366, 0.477] | **1.000 [0.987, 1.000]** | **1.000** | **1.000** |

Instance 3's gap: **+0.580 [+0.522, +0.634], p < 1e-15**. Instances 1–2 have no
baseline variance — Normal RAG is a single deterministic pass, wrong every time.

---

## Instance 1 — Coreference

```
no_answer:  "Veridian Labs operates three research sites. Veridian Labs focuses
             on materials science, and Veridian Labs publishes annually."
has_answer: "The company was founded by Dr. Ana Sorel in 1998..."
Query:      "Who founded Veridian Labs?"
```

The answer-bearing passage says *"The company"* and never repeats the entity
name; the non-answering passage names it three times. BM25 scores them 3.06 vs
0.93. No lexical retriever does coreference resolution, so the answer lands at
rank 2 — *the single most common real RAG failure*.

**Verified not to be a small-corpus artifact.** The failure holds from 3 to 52
documents, with the score gap widening (test:
`test_coreference_failure_survives_corpus_growth`).

**One genuine mitigation found:** adding documents that link the entity name to
the anaphor in context ("Veridian Labs expanded... The company added...") moves
the answer to rank 1 for all retrievers. Corpus structure fixes this, not
retriever choice.

---

## Instance 2 — Staleness with vocabulary drift

```
stale:   "The Kestrel Solar Array current generating capacity is rated at 40 MW
          under the standard facilities survey methodology."
current: "Post-expansion output rating for Kestrel: 95 MW."
Query:   "What is the current generating capacity of the Kestrel Solar Array?"
```

The outdated passage uses the canonical phrasing the query mirrors; the
amendment uses shorthand. BM25 scores them **5.40 vs 0.22**. Lexical scoring has
no notion of recency and cannot tell the two describe the same quantity.

Two off-topic facility notes pad the corpus. That padding is not decoration: with
only two documents, LSA's projection is **rank-1**, every document becomes
collinear, all cosine scores collapse to exactly 1.0, and ranking is decided by
an arbitrary tiebreak — which produced a spurious apparent win for dense
retrieval. `LSARetriever` now warns on that degenerate case
(`test_lsa_warns_on_a_degenerate_rank_one_projection`).

Retrieval depth is 4, since after padding the correct passage sits at rank 3–4;
retrieving only the top 2 would put it out of the agent's reach entirely.

**A checker bug found earlier and retained here:** an initial version took the
*first* number in the changelog, which was the superseded one, and validated the
stale answer — scoring 5%, worse than chance. Fixed by targeting the figure
attached to the word "current".

---

## Instance 3 — Noisy index

Three documents, retriever wrapped so 90% of calls return a shuffled ranking
(replication lag / eventually-consistent shard). Orthogonal to scoring function.

Normal RAG's 0.420 matches the closed form `0.9 × (1/3) + 0.1 × 1.0 = 0.40`.

Bayesian RAG declares the tool `deterministic=False`, so it is **not memoized** —
each particle draws independently, as when re-querying a replicated cluster.
Marking a stochastic tool `deterministic=True` silently caps accuracy at the
first cached draw; that was a real defect here, found and fixed earlier.

---

## Limitations

- **Lexical and LSA retrievers only.** A pretrained sentence-transformer is
  untested (not installable offline) and might solve Instance 1, where LSA
  fails. Instances 2–3 are recency and index-noise problems, which no retriever
  architecture addresses.
- **All facts fictional**, forcing answers to come from retrieved text. Says
  nothing about real-corpus performance.
- **Generation is extractive** — the retrieved passage *is* the answer.
- **Checkers are hand-written heuristics.** Two real bugs in them were found
  during this work; both are documented above rather than quietly fixed.
- **Small corpora** (3–4 documents; coreference tested to 52). BM25's IDF
  statistics behave differently at scale.
- **Uniform-shuffle noise model** is harsher and less structured than real
  replication lag.
