# Contributing

## Setup

```bash
git clone https://github.com/AngshulMajumdar/BayesianRAG
cd BayesianRAG
pip install -e ".[dev]"
pytest -q
```

The core package has no required runtime dependencies. `[dev]` adds FastAPI,
httpx, pytest, scikit-learn, and matplotlib for the API, tests, and figures.

## Before opening a pull request

```bash
pytest -q                                       # 218 tests, 12 HTTP
python -m bayesian_rag.benchmark --trials 300   # results must reproduce
```

Results are seeded by trial index and reproduce exactly. If a benchmark number
changes, that is either a bug or a finding — say which in the PR description.

## Standards

**Claims must be measured.** Any number in a README, docstring, or results file
must be regenerable by a documented command. Do not write a figure you have not
run.

**Report results that go against the method.** Several findings here do — an
engineered conditional graph beats SMC where the failure mode is known; a
confidently wrong prior costs more than holding no prior. These are kept
visible deliberately.

**Retractions stay visible.** `CHANGELOG.md` and `results/` record two
retracted claims and three corrected explanations. Do not remove them.

**Every bug fix gets a regression test** in `tests/test_regressions.py`, with a
docstring explaining what the defect was and why it was hard to see. Seven of
these encode traps that each cost significant debugging time — see
`HANDOFF.md`.

**Tests must be deterministic.** Seed everything. A test asserting that
reseeding the global RNG cannot perturb a seeded run already exists; do not
break it.

## Traps

`HANDOFF.md` lists seven non-obvious failure modes in this codebase. Read it
before changing the runner, the selector, or answer selection.

## Scope

In scope: inference regimes, reliability modelling, calibration, retrieval
integration, evaluation.

Out of scope: prompt engineering, model-specific adapters, hosted-service
integrations.
