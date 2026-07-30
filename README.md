<div align="center">

# BayesianRAG

**Tool-using agents that reason over alternatives instead of committing to the first one.**

[![Tests](https://img.shields.io/badge/tests-218%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Dependencies](https://img.shields.io/badge/core%20dependencies-none-brightgreen)](pyproject.toml)

**📄 Whitepaper:** [`whitepaper/WHITEPAPER.pdf`](whitepaper/WHITEPAPER.pdf) — architecture, algorithm, experimental protocol, and complete results (11 figures): main benchmark, unreliable-agent amplification, prior-sensitivity ablations, the LangGraph/LangChain comparison, all three RAG case studies, and the nine engineering defects found and fixed. Markdown source alongside.


[Whitepaper](whitepaper/whitepaper.pdf) · [Results](results/) · [Handoff](HANDOFF.md) · [Changelog](CHANGELOG.md)

</div>

---

```python
from bayesian_rag import Agent, tool, checker

@tool(reliability=0.6, appeal=0.9)      # attractive, unreliable
def quick_search(query: str) -> str:
    """Fast and cheap, often out of date."""
    return "Paris is the capital of Australia."

@tool(reliability=0.95, appeal=0.5)     # unglamorous, correct
def verified_search(query: str) -> str:
    """Slower, authoritative."""
    return "Canberra is the capital of Australia."

@checker
def fact_check(text: str) -> bool:
    """Validate a claim."""
    return "Canberra" in text

agent = Agent([quick_search, verified_search, fact_check])
print(agent.run("What is the capital of Australia?").answer)
# Canberra is the capital of Australia.
```

That is the whole setup. No graph, no chain, no callbacks.

---

## The problem

Agent frameworks select actions greedily and handle failure by retrying. Both
follow from one design decision: **keep a single trajectory, discard the
alternatives.** Evidence arriving later then has no branch left to promote, and
a retry redraws from the same proposal — a draw from the prior, not a posterior
update.

BayesianRAG treats an episode as inference over trajectories

$$p(a_{1:T}, o_{1:T}) = \prod_{t=1}^{T} p(a_t \mid h_t)\, p(o_t \mid a_t, h_t)$$

and makes the inference regime a **runtime argument**, so identical tools run
under all three without modification.

| Regime | Particles | Explores | Reweights | Revises earlier decisions |
|---|---|---|---|---|
| `greedy` | 1 | no | no | no |
| `forward` | N | yes | no | no |
| `smc` *(default)* | N | yes | **yes** | **yes** |

Maintaining several hypotheses is not the same as performing inference over
them. `forward` explores but never reweights, so its cloud stays a sample from
the prior. Only `smc` targets the posterior — and the gap is large.

---

## Results

All numbers reproduce exactly from a documented command. Seeds are indexed by
trial; 95% Wilson intervals throughout.

### Unreliable tools — the strongest result

Tools that are right *sometimes*, with no stable pattern. No fixed routing rule
can help, because which source is correct varies call to call.

| Source accuracy | 5% | 10% | 20% | 35% |
|---|---|---|---|---|
| **System accuracy** | 0.800 | **0.935** | 0.970 | 0.985 |

*A source right one call in ten yields a system right nine times in ten.*
64 particles turn a 35% source into 99%.

### Retrieval-augmented generation

Against **BM25**, **TF-IDF/cosine**, and **LSA** — retrievers people actually
deploy. Every instance is verified to genuinely defeat every retriever before
its result is reported.

| Instance | Normal RAG | Bayesian RAG |
|---|---|---|
| Coreference — answer says *"the company"*, lands at rank 2 | wrong, always | **1.000** |
| Staleness under vocabulary drift | wrong, always | **0.903** |
| Noisy index (replication lag) | 0.420 | **1.000** |

### The inference ladder

| Scenario | Greedy | Forward | SMC |
|---|---|---|---|
| `stale_vs_verified` | 0.000 | 0.510 | **0.957** |
| `web_vs_official` | 0.000 | 0.520 | **0.947** |
| `ambiguous_location` *(control)* | 1.000 | 1.000 | 1.000 |

Greedy fails **completely**, not merely often. The control confirms the
machinery costs nothing where routing is already unambiguous.

### Where it does *not* help

Reported as measured:

- **An engineered conditional graph is competitive and cheaper** when the
  failure mode is known in advance. Use a conditional edge when you can name
  the failure; reach for inference when you cannot.
- **A confidently wrong prior is expensive.** A source with true accuracy 20%
  declared at 0.99 drops accuracy from 0.993 to 0.380 — worse than a system
  holding no prior at all. It also makes the agent *stop checking its own work*.
- **Call count is structural, not a finding.** Maintaining a belief over
  alternatives costs more calls by construction.

```bash
python -m bayesian_rag.benchmark --trials 300           # inference ladder
python -m bayesian_rag.compare                          # vs deterministic orchestration
python -m bayesian_rag.compare.priors                   # unreliable tools, prior sweep
python -m bayesian_rag.compare.paper_results --figures  # sweeps + ablation grids
python -m bayesian_rag.compare.rag_real --both          # RAG, three retrievers
```

---

## Install

```bash
pip install bayesian-rag            # core — zero dependencies
pip install "bayesian-rag[api]"     # + FastAPI service
pip install -e ".[dev]"             # + pytest, scikit-learn, matplotlib
```

Python ≥ 3.10. No numpy, no scipy in the core — including the statistics.

---

## Guide

**Declare tools.** `reliability` is how often it is right; `appeal` is how
attractive it looks beforehand. Keeping them separate matters — a tool that is
*attractive and wrong* is the entire failure mode.

```python
@tool(reliability=0.6, appeal=0.9)              # tempting, unreliable
@tool(reliability=0.95, appeal=0.5)             # unglamorous, correct
@tool(reliability=0.5, deterministic=False)     # stochastic — see below
@checker                                         # verdict, never an answer
```

> ⚠️ **Stochastic tools must be marked `deterministic=False`.** Memoisation
> otherwise hands every particle the same cached draw, collapsing the cloud to
> one sample and silently disabling the mechanism that lets a filter exceed an
> unreliable tool's accuracy.

**Inspect a run.**

```python
r = agent.run(query)
r.answer                 # maximum a posteriori answer
r.consensus              # posterior mass backing it, pooled by answer
r.status                 # validated | unverified | refuted
r.needs_clarification    # evidence insufficient to assert
r.reliability            # learned posterior mean per tool
print(r.explain())
```

**Ask instead of guessing.** When sources irreconcilably disagree, a single-path
pipeline holds one answer and cannot represent the split. Here it is visible:

```python
if r.needs_clarification:
    print(r.clarification_request())
```
```
I could not settle this from the sources available. 2 answers retained support:
  1. The deadline is April 30.  (50% of the evidence)
  2. The deadline is May 30.    (50% of the evidence)
Which should I take, or can you narrow the question?
```

**Compare regimes** — nothing but the inference differs:

```python
for regime, result in agent.compare(query).items():
    print(regime, result.answer)
```

**Learn across queries:**

```python
for r in agent.run_session([q1, q2, q3]):
    print(r.reliability)   # distrust persists rather than being rediscovered
```

**HTTP service:**

```bash
uvicorn bayesian_rag.api.main:app --port 8000
```

---

## Design notes

Each of these was a real defect found during development; all have regression
tests. See [`HANDOFF.md`](HANDOFF.md) for the full list.

**Diversity is load-bearing.** An argmax selector makes every particle choose
identically — the cloud carries no information and reweighting has nothing to
act on, silently reducing an N-particle filter to a slow greedy run.

**Checkers inspect their own particle's claim.** A shared proposal menu with a
claim bound to particle zero causes cross-particle misattribution, driving a
*reliable* tool's posterior down while it keeps being correct. Resolved via a
`Pending.CLAIM` placeholder per particle.

**Evidence outranks recency.** `final_answer()` prefers validated → unrejected →
latest. Taking the latest answer discards a confirmation the trajectory already
paid to obtain — worth up to 35 points, and it masquerades as a budget limit.

**Finding a problem is not punished.** Scoring a failed check near zero
eliminates the particle holding the decisive evidence. Checkers are scored by
the epistemic state they leave the trajectory in.

**Carry-over transports the posterior.** Rebuilding a Beta from its mean
preserves the estimate and destroys the sample size — 50 observations become ~2.

---

## Limitations

- **Synthetic tools**, i.i.d. Bernoulli. Real failures correlate with query
  type, load, and upstream incidents.
- **Perfect checkers.** A real verifier is unreliable, which would compress
  every gap above. Two genuine checker bugs were found during this work; both
  are documented rather than quietly fixed.
- **Comparison baselines are semantic stand-ins**, not the real LangChain and
  LangGraph packages — no network in the evaluation environment. Verify with
  `python -m bayesian_rag.compare.adapters`.
- **Scenarios were built to separate the regimes.** They establish the
  mechanism works where it matters, not how often that arises.
- **Dense retrieval only partially tested.** LSA falsified our own conjecture
  that it would resolve coreference. Pretrained transformers remain untested.
- **Short horizons, small corpora.**

Two claims have been **retracted** and three explanations **corrected**; both
are recorded in [`CHANGELOG.md`](CHANGELOG.md) and kept visible deliberately.

---

## Citation

```bibtex
@software{majumdar2026bayesianrag,
  author  = {Majumdar, Angshul},
  title   = {BayesianRAG: Bayesian Orchestration of Tool-Augmented Agents},
  year    = {2026},
  version = {1.7.0},
  url     = {https://github.com/AngshulMajumdar/BayesianRAG}
}
```

## License

MIT — see [LICENSE](LICENSE).
