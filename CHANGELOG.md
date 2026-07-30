# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [1.7.0] — 2026-07-29

### Added
- Whitepaper (`whitepaper/`) with complete theory and all measured results.
- `LSARetriever` — dense retrieval via TF-IDF + truncated SVD, with a warning
  on degenerate rank-1 projections.
- `HANDOFF.md` — verified vs. unverified claims, known traps, open work.
- Zenodo/GitHub packaging: `CITATION.cff`, `.zenodo.json`, `CONTRIBUTING.md`.

### Fixed
- **`Particle.final_answer()` preferred recency over validation status.** A
  trajectory that found the correct answer, had a checker confirm it, then
  examined something else would return whatever it touched last, discarding the
  confirmation. Preference is now validated → unrejected → latest. This moved
  results project-wide (RAG coreference 0.737 → 1.000; paper Exp-1 at source
  accuracy 0.50, 0.887 → 1.000).

### Changed
- **Corrected an earlier explanation.** The mid-range accuracy dip previously
  attributed to step/particle budget was caused by the `final_answer()` bug
  above. Adding particles masked it rather than fixing it.

### Removed
- **Retracted the original RAG "lexical trap" instance.** It relied on raw
  term-frequency overlap with no length normalisation; BM25 and TF-IDF solve it
  outright. It measured a weak retriever, not a strong orchestrator.
- **Retracted the claim that dense retrieval would resolve coreference.** LSA
  ranks the answer *worse* than BM25. Scoped: untested for pretrained
  transformer embeddings.

## [1.6.0] — 2026-07-29

### Added
- `bayesian_rag/rag/retrievers.py` — BM25 (Okapi) and TF-IDF/cosine.
- `bayesian_rag/compare/rag_real.py` — RAG comparison on production-standard
  retrievers, with diagnostics verifying each instance genuinely defeats them.

## [1.5.0] — 2026-07-29

### Added
- `bayesian_rag/rag/` — retrieval primitives and tool wrappers.
- Three-instance normal-vs-Bayesian RAG comparison.

### Fixed
- Changelog checker extracted the first number in the reference text, which was
  the superseded figure, validating the stale answer (scored 5%, worse than
  chance).

## [1.4.0] — 2026-07-29

### Fixed
- **Chain and Graph tracks used different `appeal` values** (0.8 vs 0.9) for
  what is meant to be the same primary tool, confounding the comparison. Both
  now derive from `PRIMARY_APPEAL`, enforced by a parity test.

## [1.3.0] — 2026-07-29

### Added
- `bayesian_rag/compare/paper_results.py` — source-accuracy sweeps and
  prior × accuracy ablation grids for both orchestrator styles, with figures.

## [1.2.0] — 2026-07-29

### Added
- `Result.consensus` / `needs_clarification` / `clarification_request()` —
  posterior mass marginalised over answers rather than trajectories.
- `@tool(deterministic=False)` and `Agent(cache=False)`.

### Fixed
- **Memoisation cached stochastic tools**, handing every particle the same draw
  and collapsing the cloud to a single sample — silently disabling the
  re-drawing that lets a filter exceed an unreliable tool's accuracy.

## [1.1.0] — 2026-07-29

### Added
- `bayesian_rag/compare/` — deterministic LangGraph/LangChain stand-ins,
  three-world generalisation study, adapters to verify against the real
  packages.
- `Result.status` — validated / unverified / refuted.

## [1.0.0] — 2026-07-29

### Added
- High-level API: `Agent`, `@tool`, `@checker`, `Result`.
- `bayesian_rag/utils/statistics.py` — Wilson intervals, Newcombe differences,
  two-proportion tests, all standard-library.

### Fixed
- Cross-particle claim misattribution via `Pending.CLAIM` per-particle
  resolution; a reliable tool's posterior was being driven down by claims
  produced elsewhere in the cloud.
- Belief carry-over rebuilt a Beta from its mean, destroying sample size
  (50 observations became ~2). Now transports (α, β) via `tempered()`.
- A failed check near-zeroed the particle that performed it, eliminating the
  one holding decisive evidence. Scoring is now by epistemic state.
- Argmax selectors produced zero particle diversity, silently reducing an
  N-particle filter to a slow greedy run.
- Config validation, NaN/negative likelihood rejection, observation detachment,
  scratch-key leakage into user-visible particle state.
