# LinkedIn post

Three versions. The first is the recommended one.

---

## Version A — technical, leads with the counter-intuitive result (recommended)

> **A retrieval source that is correct 10% of the time can drive a system that is correct 93% of the time.**
>
> That result surprised me enough that I spent a while trying to find the bug. There wasn't one — but there were four others, and finding them is most of what this work turned out to be.
>
> The setup: tool-augmented LLM agents select actions greedily and recover from failure by retrying. Both follow from one design decision — the agent keeps a single execution trajectory and discards the alternatives. Evidence arriving later has no surviving branch to promote. A retry redraws from the same proposal distribution, so it is a draw from the prior, not a posterior update.
>
> Treat the episode as sequential inference over trajectories instead:
>
> p(a₁:T, o₁:T) = ∏ₜ p(aₜ | hₜ) · p(oₜ | aₜ, hₜ)
>
> Greedy selection, forward sampling, and sequential Monte Carlo then fall out as three inference strategies over ONE generative model, not three architectures. Making the regime a runtime parameter means identical tools and likelihoods run under all three — so any difference in outcome is attributable to inference alone.
>
> Measured, 300 seeds, 95% Wilson intervals:
>
> • Greedy 0.000 · Forward 0.510 · SMC 0.957 on adversarial routing
> • RAG coreference, staleness, index noise vs BM25/TF-IDF/LSA: 1.000 / 0.903 / 1.000, against a conventional pipeline's two deterministic failures and 0.420
> • Forward vs SMC is the substantive gap. Both explore identically; they differ only in whether the observation likelihood reaches the weights. That alone converts a coin flip into recovery.
>
> Some findings went against the method, and those are in the paper too:
>
> • A correctly engineered conditional graph is competitive AND cheaper when you can name the failure mode in advance. Use a conditional edge then. Reach for inference when you cannot.
> • A confidently wrong prior is worse than no prior — 0.993 → 0.380. It also makes the agent stop checking its own work, which I only found by inspecting the agent's actual selection scores.
>
> The bugs were the interesting part. Memoisation was handing every particle the same cached draw, collapsing the cloud to a single sample. A checker was blaming the wrong particle for another's claim, driving a reliable tool's posterior down while it kept being correct. And answer selection preferred recency over validation — the agent found the right answer, confirmed it, then threw it away. Fixing that last one moved results across the entire project and invalidated an explanation I had already written down.
>
> I also retracted two claims. An early RAG demo used a retriever with no length normalisation, which nobody deploys — BM25 solves that instance outright, so it measured a weak retriever rather than a strong orchestrator. And I had speculated that dense retrieval would resolve coreference; LSA ranks the answer worse than BM25 does. Both retractions are in the changelog, kept visible.
>
> Whitepaper with the full derivation, MIT-licensed code, 218 tests, and every number regenerable by a documented command:
>
> 🔗 GitHub: github.com/AngshulMajumdar/BayesianRAG
> 📄 Zenodo: [DOI]
>
> #MachineLearning #LLM #RAG #BayesianInference #AIEngineering #OpenSource

---

## Version B — shorter, for wider reach

> **Your RAG pipeline returns the top-ranked passage. The answer is at rank 2.**
>
> This is the most common failure in retrieval-augmented generation, and it has a specific cause: the answer-bearing passage often refers to the entity as "the company" rather than repeating its name, so it loses on lexical overlap to a passage that names it repeatedly and answers nothing.
>
> BM25 scores them 3.06 vs 0.93. TF-IDF agrees. No lexical retriever does coreference resolution.
>
> Conventional pipeline: wrong, every time.
> Treating retrieval as Bayesian inference over trajectories: 1.000.
>
> The idea is that agent execution is sequential inference, not control flow. Instead of committing to one trajectory and retrying on failure, maintain a weighted population of them and let the observation likelihood reweight branches. A checker's verdict at step t revises the standing of the tool that produced the claim at step t−1.
>
> The most striking consequence: a source correct 10% of the time yields a system correct 93% of the time, because independent re-draws plus validation beat any single draw.
>
> Where it does NOT help, measured and reported: if you can name the failure mode in advance, a conditional edge is cheaper and deterministic. And a confidently wrong prior is worse than no prior at all.
>
> Whitepaper, MIT code, 218 tests, everything reproducible:
> 🔗 github.com/AngshulMajumdar/BayesianRAG
>
> #RAG #LLM #MachineLearning #AIEngineering

---

## Version C — the engineering-culture angle

> I spent this project mostly finding my own bugs. Sharing them, because the debugging was more instructive than the result.
>
> The work: formulating tool-augmented agent execution as sequential Monte Carlo over action trajectories, so that greedy selection, forward sampling, and Bayesian filtering become three inference strategies over one model rather than three architectures.
>
> **Bug 1 — memoisation defeated the entire method.** Caching identical tool calls is an obvious optimisation. It is also silently wrong for stochastic tools: every particle received the same cached draw, so 32 logical invocations became 1 execution and the particle cloud collapsed to a single sample. The runs completed and returned plausible answers the whole time.
>
> **Bug 2 — a checker blamed the wrong particle.** The proposer emits one candidate menu shared across particles, but each particle must check its own claim. Binding the claim to particle zero meant a particle using a reliable tool was penalised for a claim produced elsewhere. I watched a correct tool's posterior fall 0.50 → 0.25 → 0.17 while it kept being right.
>
> **Bug 3 — the agent discarded answers it had just confirmed.** Answer selection took the most recent claim. A trajectory that found the right passage, validated it, then looked elsewhere returned whatever it touched last. Fixing this moved results across the entire project — and invalidated an explanation I had already written into the results document, where I had attributed the effect to compute budget. More particles had been masking the bug, not fixing it.
>
> **Two retractions.** An early demo used a retriever with no length normalisation — BM25 solves that case outright, so the demo measured a weak retriever rather than a strong orchestrator. And I had claimed dense retrieval would resolve coreference; testing it showed LSA ranks the answer worse than BM25.
>
> All of this is in the changelog and the paper. The retractions stay visible on purpose. A result you cannot reproduce is not a result, and a caveat you quietly delete is worse than one you never wrote.
>
> 218 tests, MIT licensed, every number regenerable by a documented command:
> 🔗 github.com/AngshulMajumdar/BayesianRAG
>
> #SoftwareEngineering #MachineLearning #OpenSource #ResearchSoftware

---

## Notes before posting

- **Replace `[DOI]`** with the Zenodo DOI once minted.
- **Confirm the GitHub URL** — `AngshulMajumdar/BayesianRAG` is assumed
  throughout the repo metadata; change it in `CITATION.cff`, `.zenodo.json`,
  `pyproject.toml`, and `README.md` if it differs.
- **Add your ORCID** to `CITATION.cff` and `.zenodo.json` (currently commented
  out — deliberately left blank rather than guessed).
- Attach `WHITEPAPER.pdf` directly to the LinkedIn post — PDF documents get
  substantially better reach than external links.
- **Do not describe the LangChain/LangGraph baselines as measured against the
  real packages.** They are semantic stand-ins; the repo says so and the post
  should not overstate it. If someone asks in the comments, that is the honest
  answer and a good thread.
