# Release checklist

## Before tagging

- [ ] Add ORCID to `CITATION.cff` and `.zenodo.json` (currently blank, not guessed)
- [ ] Confirm the repository URL in `CITATION.cff`, `.zenodo.json`,
      `pyproject.toml`, `README.md`, and the whitepaper
- [ ] `pytest -q` — 218 pass, 12 HTTP skip without FastAPI
- [ ] `pip install -e ".[dev]" && pytest -q` — **run the 12 HTTP tests**;
      they have never executed in the development environment
- [ ] `python -m bayesian_rag.compare.adapters` with `langgraph` installed —
      verifies the deterministic stand-ins against the real package
- [ ] Regenerate results and confirm they match what is committed:
      ```
      python -m bayesian_rag.benchmark --trials 300
      python -m bayesian_rag.compare.paper_results --trials-sweep 300 --trials-grid 150 --figures
      python -m bayesian_rag.compare.rag_real --both --trials 300
      ```
- [ ] Rebuild the whitepaper: `cd whitepaper && pdflatex WHITEPAPER.md` (twice)

## GitHub

- [ ] Push, then create a release tagged `v1.7.0`
- [ ] Attach `whitepaper/WHITEPAPER.pdf` to the release
- [ ] Set repository topics: `bayesian-inference`, `sequential-monte-carlo`,
      `llm-agents`, `rag`, `particle-filter`, `agent-orchestration`
- [ ] Set the description to the one-line summary from `README.md`

## Zenodo

- [ ] Link the GitHub repository at https://zenodo.org/account/settings/github/
- [ ] Enable the toggle for this repository **before** creating the release —
      Zenodo only captures releases made after the toggle is on
- [ ] Create the GitHub release; Zenodo mints the DOI automatically from
      `.zenodo.json`
- [ ] Add the DOI badge to `README.md`:
      `[![DOI](https://zenodo.org/badge/DOI/<doi>.svg)](https://doi.org/<doi>)`
- [ ] Update `CITATION.cff` with the DOI under `identifiers:`

## LinkedIn

- [ ] Insert the DOI into the chosen post from `LINKEDIN_POST.md`
- [ ] Attach `WHITEPAPER.pdf` directly (better reach than a link)
- [ ] Do not claim the baselines were validated against the real LangChain and
      LangGraph packages — they are semantic stand-ins
