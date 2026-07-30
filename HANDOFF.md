# Handoff

State of the project, what is verified vs. not, and the traps that will
otherwise be rediscovered. Read this before continuing in a fresh session —
it is faster and more reliable than replaying the conversation.

**Version 1.7.0 · 218 tests passing, 1 skipped (HTTP tests, FastAPI absent)**

---

## What this is

Agent orchestration as sequential inference over action trajectories. One
generative model, three inference regimes selected at runtime:

| Regime | Particles | Explores | Reweights | Revises earlier decisions |
|---|---|---|---|---|
| `greedy` | 1 | no | no | no |
| `forward` | N | yes | no | no |
| `smc` (default) | N | yes | yes | **yes** |

Because the regime is an argument, identical tools/proposer/scorer run under
all three — that is what makes the comparisons controlled.

```python
from bayesian_rag import Agent, tool, checker

@tool(reliability=0.6, appeal=0.9)      # appeal ≠ reliability is the failure mode
def quick(query: str) -> str: ...

@checker
def verify(text: str) -> bool: ...

Agent([quick, verify]).run(q).answer
```

---

## Run everything

```bash
pytest -q                                              # 218 + 12 HTTP
python -m bayesian_rag.benchmark --trials 300          # main scenarios
python -m bayesian_rag.compare                         # 3 worlds vs LangGraph/LangChain
python -m bayesian_rag.compare.priors                  # unreliable agents, prior sweep
python -m bayesian_rag.compare.paper_results --figures # sweeps + ablations
python -m bayesian_rag.compare.rag_real --both         # RAG, 3 retrievers
```

---

## Traps — read before changing anything

Each of these was a real bug that cost significant debugging time. All have
regression tests; none are obvious from reading the code.

1. **Stochastic tools must be `@tool(deterministic=False)`.** Memoization
   otherwise hands every particle the same cached draw, collapsing the cloud
   to one sample. Silent: the run completes and returns something plausible.

2. **Selectors must sample, not argmax, under exploring regimes.** Identical
   particles carry no information, so reweighting has nothing to act on and an
   N-particle filter becomes a slow greedy run.

3. **Checkers resolve `Pending.CLAIM` per particle.** The proposer emits one
   menu shared across particles; baking in particle-0's claim caused a particle
   to be blamed for a claim produced elsewhere, driving a *reliable* tool's
   posterior steadily down while it kept being correct.

4. **`final_answer()` prefers validated > unrejected > latest.** Recency alone
   discards a confirmed answer if the trajectory later wandered. Fixing this
   moved results project-wide (RAG coreference 0.737 → 1.000; paper Exp-1 at
   accuracy 0.5, 0.887 → 1.000) and **invalidated an earlier explanation** —
   see the correction in `results/PAPER_RESULTS.md`.

5. **Belief carry-over uses `tempered()`,** which transports (α, β). Rebuilding
   a Beta from its mean preserved the estimate and destroyed the sample size —
   50 observations became ~2.

6. **A failed check must not near-zero the checking particle.** That eliminates
   the particle holding the decisive evidence. Scoring is by epistemic state:
   validated / recovered / stranded.

7. **LSA on <3 documents is rank-1 and meaningless** — all cosine scores
   collapse to 1.0 and ranking is an arbitrary tiebreak. It now warns. This
   produced a spurious "dense retrieval wins" result before it was caught.

---

## Verified vs. not

**Verified here:** all inference behaviour; statistics (Wilson intervals,
Newcombe differences, two-proportion tests); the LangGraph-engineered baseline
matches its closed form `quick + (1-quick)×thorough`; BM25/TF-IDF/LSA against
the instances; every regression above.

**NOT verified — do not cite as measured:**
- **The LangChain/LangGraph baselines are semantic stand-ins**, not the real
  packages. No network here. Run
  `python -m bayesian_rag.compare.adapters` with `langgraph` installed.
- **The 12 HTTP tests have never executed.** FastAPI not installable offline.
  Run `pip install -e ".[dev]" && pytest -q` before trusting the API.
- **Pretrained embedding retrieval is untested.** `sentence-transformers`
  unavailable. LSA falsified the "dense solves coreference" claim, but LSA is
  not a transformer; that question is open.
- **All tools are mocks / i.i.d. Bernoulli, all facts fictional, all corpora
  tiny.** Nothing here speaks to real corpora or real model proposers.

---

## Results, in one place

**Main scenarios** (300 seeds): greedy 0.000 · forward ~0.51 · SMC ~0.95 on
both adversarial scenarios; all three 1.000 on the control.

**Unreliable agents** — the strongest result. A source right 1 call in 10
yields a system right 9 in 10; 64 particles turn a 35% source into 99%.

**Priors matter** — a source with true accuracy 20% declared at 0.99 drops SMC
from 0.993 to 0.380. A false high prior also makes the agent *stop checking its
own work* (verified against seeded selection scores, not inferred).

**vs. deterministic orchestration** — an engineered LangGraph graph is
competitive and cheaper *where the failure mode is known*. SMC's advantage is
not needing to name the failure in advance. Call count is structural, not a
finding.

**RAG** (300 seeds, BM25): coreference 1.000, staleness 0.903, noisy index
1.000, against Normal RAG's wrong/wrong/0.420.

---

## Retracted claims

Kept visible deliberately; do not reintroduce.

1. **Original RAG "lexical trap"** relied on raw term-overlap with no length
   normalization. BM25 and TF-IDF solve it outright. Retracted.
2. **"A dense retriever would likely solve coreference"** — LSA does not; it
   ranks the answer *worse* than BM25. Scoped: untested for transformers.
3. **"The mid-range accuracy dip is a budget effect"** — it was the
   `final_answer()` bug (trap 4). More particles masked it rather than fixing it.
4. **Early drafts contained invented benchmark numbers** (62%/88%/+26%).
   Everything now reported is measured and regenerable.

---

## Open work

- Validate stand-ins against real LangGraph/LangChain (needs network).
- Run the HTTP suite (needs FastAPI).
- Embedding retrieval via sentence-transformers — closes the one open scoping
  question.
- Real corpora and real model proposers; everything here is synthetic.
- Extend to 3+ competing sources; only 2 tools per track are tested.
- Sweep `PRIMARY_APPEAL` (fixed at 0.8, shared across tracks by a parity test).

---

## Layout

```
bayesian_rag/
├── agents/      agent.py (Agent, Result) · defaults.py · bayesian_agent.py
├── core/        particle.py · smc_runner.py · inference_regimes.py
├── bayesian/    reliability_model.py
├── tools/       decorator.py (@tool, @checker) · mock_tools.py
├── rag/         basic.py · retrievers.py (BM25/TF-IDF/LSA) · tools.py
├── compare/     baselines.py · adapters.py · priors.py · paper_results.py · rag_real.py
├── utils/       math_utils.py · statistics.py
└── api/         FastAPI (untested here)

results/  PAPER_RESULTS.md · RAG_REAL_RESULTS.md · *.png · raw JSON
```
