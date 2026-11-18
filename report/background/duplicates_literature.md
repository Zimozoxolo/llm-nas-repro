# Duplicate / wasted LLM proposals: literature check (2026-11-18)

Question: is it new to (1) measure and break down duplicate LLM proposals in LLM-NAS, (2) count the budget in
evaluated architectures rather than LLM calls, (3) compare LLM-NAS and baselines at equal evaluations?
Abstract-level checks only unless stated; full texts not read.

## Already done by others (cite)
- **Duplicate filtering in LLM-driven architecture generation (code-level, open-ended spaces).**
  - Khalid, Ignatov, Timofte, "From Memorization to Creativity: LLM as a Designer of Novel Neural Architectures",
    arXiv 2601.02997, CVPRW 2026. MinHash-Jaccard novelty filter; 455 unique admitted architectures; secondary
    summary: without the filter the admitted set halves (455 -> 220). Search result also mentions they track
    rejections before a non-duplicate is accepted (not verified in full text).
  - Adhikari, Timofte, Ignatov, "Delta-Based Neural Architecture Search: LLM Fine-Tuning via Code Diffs",
    arXiv 2605.04903. MinHash-Jaccard novelty filter; reports valid rates per model (DeepSeek-Coder-7B 75.3%,
    Qwen2.5-Coder-7B 72.1%, Mistral-7B 66.6%).
  - arXiv 2512.24120 (LLM-based NN generation): hash-based dedup, ~5% formatting-only duplicates (search summary).
  - arXiv 2609.40258 (LLM-guided evolutionary discovery of spiking sequence models): novelty as an objective;
    search summary says ~30% of raw proposals removed as near-duplicates (not in abstract; unverified).
- **Dedup / novelty gates in LLM evolutionary search generally.** Lyria (arXiv 2507.04034, dedup with re-request
  up to a cap = our "filter and re-prompt"); ShinkaEvolve (arXiv 2509.19349, embedding novelty rejection);
  VendiEvolve (ICML 2026 workshop, notes duplicates pile up and narrow the search); FunSearch (dedup by
  behaviour, secondary source only).
- **LLMs narrow the search / repeat near their prior.** Zhang, Chen, Portet, Peyrard, "What Makes an LLM a Good
  Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search", arXiv 2604.19440, Findings ACL 2026 (15
  LLMs, 8 tasks; strong optimisers are local refiners that narrow the search; novelty does not predict final
  performance). Redko et al., "Prior Knowledge or Search?", arXiv 2605.19782 (LLMs converge to the same kernel
  parameters regardless of input size or temperature; greedy optimisers).
- **Budget-matched LLM vs classical search.** Rodrigues et al., "When Is an LLM Worth It for Hyperparameter
  Optimization? A Budget-Matched Study...", arXiv 2606.21641 (LLM advisor overtaken by random search / TPE / GP-BO
  within a handful of evaluations; early lead came from a default configuration, not the LLM). HPO, not NAS.
  Li & Talwalkar 2019 (random search as NAS baseline) - already cited.
- **Classical dedup in GAs.** Mauldin, "Maintaining Diversity in Genetic Search", AAAI 1984, pp. 247-250 (verified).
  Ronald 1998 "Duplicate genotypes..." NOT found - do not cite until verified. Raidl 1999 (genotypic vs
  phenotypic duplicate elimination) found as context.

## Not found (still looks new, hedge as "to our knowledge")
- An exact-match count of duplicate proposals **in a fixed, enumerable cell space** (NAS-Bench-201) for LLM-NAS,
  **broken down by source** (copied shown parent / copied shown last-cycle child / sibling / re-invented unseen
  history), across several LLMs, and its growth over generations.
- The point that LLM-NAS's budget of 120 LLM calls yields only ~145 (GPT-4.1) or 76-86 (Haiku, DeepSeek)
  evaluated architectures vs 270 for budget-matched baselines, and an equal-evaluations comparison
  (GPT-4.1 0.928 vs random sampling 0.930 at 145 evaluations).
- GENIUS (2304.10970) and LLMatic (2306.01102) on NAS-Bench-201: no duplicate analysis surfaced; full texts not
  checked.

## To do before claiming
- Read full texts of 2601.02997 (and its CVPRW supplement), 2605.04903, 2609.40258 for any source breakdown.
- Check GENIUS and LLMatic full texts for duplicate handling.
