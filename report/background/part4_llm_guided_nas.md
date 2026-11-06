# Background, Part 4: LLM-guided NAS and the gap it claims to fill

Status: draft for your review. Parts 1-3 are in this folder. Labels: **author-stated** (the paper says it), **my reading** (my inference). Nothing here has been put into the reports yet.

---

## 0. How the selection was made (please check this first)

This field is young, so citation counts are small and are a weak ranking signal. I searched OpenAlex (queried 2026-11-06) by title and topic and read the papers below. Counts run lower than Google Scholar.

| Role | Paper | Year | OpenAlex citations | How much I read |
|---|---|---|---|---|
| Precursor (LLM as mutation operator) | Evolution through Large Models (ELM, Lehman et al.) | 2022 | 89 | abstract and introduction only |
| Precursor (LLM as optimiser) | Large Language Models as Optimizers (OPRO, Yang et al.) | 2023 | 386 | abstract, limitations, prompt-optimisation set-up |
| LLM-NAS | EvoPrompting (Chen et al.; NeurIPS 2023) | 2023 | 56 | method, experiments, NATS-Bench appendix |
| LLM-NAS | Can GPT-4 Perform NAS? (GENIUS, Zheng et al.) | 2023 | 52 | method, experiments, limitations |
| LLM-NAS | LLMatic (Nasir et al.; GECCO 2024) | 2024 | 60 | whole main text |
| LLM-NAS | Design Principle Transfer in NAS via LLMs (LAPT, Zhou et al.; AAAI 2025) | 2025 | 21 | abstract, introduction, NAS-Bench-201 experiments |
| LLM-NAS | CoLLM-NAS (Li, Lin, Wang) | 2025 | not returned | abstract, introduction, experiments |
| LLM-NAS | LLM as a Tool, Not an Agent (LLMasTool, Yoshimura et al.) | 2026 | 0 | abstract, benchmark table, limitations |
| The paper reproduced | LLM-NAS (Zhu, Zhang, Huang) | 2025 | 1-2 | read earlier in this project |

**Not read:** the other LLM-based search papers named in these tables (LeMo-NADe, NADER, LM-Searcher, RZ-NAS), "Large Language Models as Evolutionary Optimizers" (153) and the survey "Evolutionary Computation in the Era of Large Language Model" (176). The search was keyword based and may have missed papers.

---

## 1. What the "gap" is, in the authors' own words

Four claimed problems with pre-LLM search recur (author-stated):

1. **Hand-crafted search spaces limit and bias the search** (EvoPrompting, LLMatic, LLMasTool). They propose letting the LLM write code or edit architectures so the space is not fixed in advance.
2. **Mutation in evolution or RL is uninformed and wasteful.** LLMatic: search by evolution or RL "is quite mindless and wasteful ... how uninformed the changes are", so it asks whether stored design knowledge can inform the search.
3. **Design knowledge is not reused.** GENIUS asks whether GPT-4 has relevant knowledge and elicits "design principles"; LAPT learns written principles and uses them to shrink the space; LLM-NAS co-evolves a knowledge base and claims earlier LLM methods use static prompts.
4. **Cost.** LLM-NAS: supernet methods need GPU-days, so an LLM plus a zero-cost predictor reduces search to minutes; CoLLM-NAS reports 4-10x lower search cost against evolutionary search over a supernet.

How these connect to Parts 1-3 (my reading):
- (2) and (3) are claims about the **search strategy** only. Part 2 found that prior knowledge in search is not new (NSGA-Net initialises from hand-crafted architectures; PNAS searches simple to complex) and that random search is a documented strong baseline.
- (4) is largely a claim about the **evaluator**. Part 3 found that cost has fallen through tables, predictors and zero-cost proxies. LLM-NAS uses a tabulated benchmark and a cheap predictor, so its cost saving does not depend on the LLM.
- (1) is a different gap: it applies to spaces with no natural mutation operator, which is not the fixed NAS-Bench-201 cell LLM-NAS searches. See hook 6.

---

## 2. The papers

### 2.1 Precursors
- **ELM (Lehman et al., 2022).** A code LLM replaces the mutation operator in genetic programming; with MAP-Elites it generates many working Python programs for Sodarace walkers the LLM had not seen in pre-training. *Why it matters:* it is the origin of "LLM as mutation operator", cited by EvoPrompting and LLMatic. Not NAS. (Abstract only.)
- **OPRO (Yang et al., 2023).** The LLM is the optimiser: previous solutions and scores go in the prompt, new solutions come out. Shown on linear regression, travelling salesman and prompt optimisation. *Author-stated limits:* context length limits problem size; on bumpy landscapes the LLM can get stuck; on larger travelling-salesman instances a classical heuristic (farthest insertion) beats the LLMs. Not designed to beat specialised solvers. *Use for us:* a published statement that LLM optimisers are not claimed to beat specialised search.

### 2.2 EvoPrompting (Chen, Dohan, So; NeurIPS 2023; arXiv 2302.14838)
- **Method (author-stated).** A code LLM (62B PaLM) is the crossover and mutation operator of an evolutionary loop over **Python code** for architectures. Few-shot prompts are built from the fittest models, with the target accuracy and size in the prompt; the LLM is prompt-tuned between rounds. Fitness is validation error times model size. Search is seeded with hand-designed models. 10 rounds, 1,600 models generated; every child model is **trained** (MNIST-1D: 8,000 steps on a P100).
- **Results (author-stated).** On MNIST-1D, the best 20 models give a Pareto front closer to the origin than seed models and baselines. On the CLRS reasoning benchmark it finds graph-network variants beating the Triplet-GMPNN baseline on 21 of 30 tasks (abstract), with similar size.
- **Baselines and ablations (author-stated).** Naive few-shot prompting; EvoPrompting without prompt-tuning; random parents. Random parents plus prompt-tuning is "no more effective than naive prompting alone".
- **Pitfalls (author-stated).** "Lack of an extensive comparison against other standard NAS techniques" because they were not designed for open-ended search. The one comparison is in an appendix on the NATS-Bench **size** space, with other methods' numbers copied from the NATS-Bench paper.
- **My reading of that appendix.** CIFAR-10 test accuracy: EvoPrompting 93.11 ± 0.90, random search 93.03 ± 0.25, REA 93.22 ± 0.16. The EvoPrompting-random gap (0.08) is far inside EvoPrompting's own standard deviation. The authors describe it as "competitive"; it is also what random search would give.
- **No conventional-operator control.** The ablations remove parts of the LLM method; none replaces the LLM operator with a standard evolutionary operator on the same evaluator.

### 2.3 GENIUS (Zheng et al.; arXiv 2304.10970)
- **Method (author-stated).** GPT-4 is a black-box optimiser: a text description of the problem goes in, a configuration comes out, it is trained or looked up, and the accuracy goes back in the next prompt (10 iterations, "rationale" requested).
- **Results (author-stated).** NAS-Bench-Macro (6,561 architectures): best result ranked 8 of 6,561, worst of the runs 61 (3 runs, 10 iterations). Channel-Bench-Macro (16,384): rank 33 (ResNet) and 16 (MobileNet), one trial each. NAS-Bench-201 (5 trials, temperature 0): 93.79 ± 0.09 / 70.91 ± 0.72 / 44.96 ± 1.02, "slightly behind" the best differentiable methods. ImageNet, MobileNetV2-style space: 77.8% top-1 at 329M FLOPs, search cost 5.6 GPU-days (20-epoch trainings). The paper states the space has "approximately 730" candidates, which I could not reconcile (possibly a typographical error).
- **Baseline (author-stated).** Random sampling: best of 10 random draws, averaged over 10,000 repeats. I did not extract the numbers.
- **Pitfalls (author-stated, Sec. 6).** (a) Cannot control or see GPT-4's pre/post-processing; results vary even at temperature 0. (b) **Benchmark contamination**: GPT-4 may have seen the benchmarks, so it may search "from memory", "a significant caveat". (c) On NAS-Bench-201 later iterations sometimes did worse than earlier ones, unexplained. (d) FLOPs constraints were often violated until an extra feedback loop was added. (e) The authors say amortised supernet methods can reuse a search across constraints; GENIUS cannot.
- **My reading.** The paper says it is "preliminary" and not aimed at state of the art. On NAS-Bench-201 it matches the random-search value that other papers report (below).

### 2.4 LLMatic (Nasir, Earle, Cleghorn, James, Togelius; GECCO 2024; arXiv 2306.01102)
- **Method (author-stated).** A 6.1B code LLM (CodeGen-Mono) mutates or crosses over network code, inside quality-diversity search (CVT-MAP-Elites) with two archives: networks (width-to-depth ratio, FLOPs) and prompts plus temperature. 20 generations of 100 offspring = **2,000 evaluations**; each CIFAR-10 network trained 50 epochs.
- **Results (author-stated).** CIFAR-10: competitive networks "while evaluating just 2,000 candidates", compared with the 8,000 models of the original EfficientNet search (note: a different search and dataset). NAS-Bench-201 test accuracy over 10 runs: 94.26 ± 0.13 / 71.62 ± 1.73 / 45.87 ± 0.96, against random search 93.70 / 71.04 / 44.57, GENIUS, and Λ-DARTS 94.36 / 73.51 / 46.34 (their "optimal" row reads 94.47 / 74.17 / 47.33).
- **Ablations (author-stated).** 30 seeds. Network-archive-only, prompt-archive-only, mutation-only, crossover-only, and "Random-NN-Generation" (LLM without evolution) are all worse; "each component is necessary".
- **Pitfalls.** *Author-stated:* "more variation than the competing methods"; they note a small LLM and a small budget. *My reading:* the ablations never replace the LLM with a conventional mutation operator, the NAS-Bench-201 table has no evolutionary baseline, and the random-search budget is not stated there.

### 2.5 LAPT: Design Principle Transfer (Zhou, Wu, Feng, Lu, Tan; AAAI 2025)
- **Method (author-stated).** GPT-4 reads a set of known architectures, writes "design principles" in text, and these prune the search space for a new task; principles are refined from the new search results. REA searches the reduced space.
- **Results (author-stated; NAS-Bench-201, Table 2).** Number of architectures **trained**: LAPT-REA 4.9 (CIFAR-10, 94.36%) and 8.6 (CIFAR-100, 73.45%); REA over 500; HEBO 100 (94.34); MetaD2A 100 (94.37); TNAS-BO 29 and 59 (94.37, 73.51). LAPT ran 20 seeds. On Transfer101 with 50 trained architectures it beats REA on 5 of 6 tasks and is level with the best transfer method.
- **My reading.** This is a transfer setting: knowledge comes from earlier tasks, and the counted cost is trained architectures only, not LLM calls or the source search. The gain comes with REA unchanged; the LLM supplies a smaller space, not the search.

### 2.6 CoLLM-NAS (Li, Lin, Wang; arXiv 2509.26037, v2 May 2026)
- **Method (author-stated).** On top of two-stage supernet NAS (OFA, SPOS, AutoFormer), a stateful Navigator LLM writes a strategy and a stateless Generator LLM produces candidates; a coordinator validates and evaluates them. Qwen3-30B-A3B run locally. Subnets are evaluated on the validation set with the baseline's supernet.
- **Their statement of the problem (author-stated):** existing LLM-based NAS methods "fail to surpass established NAS baselines on standard benchmarks" and are computationally heavy.
- **Results (author-stated).** ImageNet: up to 0.7 points better than the baseline's evolutionary search, with 250 instead of 1,000 evaluated architectures and 4-10x lower search cost (cost includes LLM inference; supernet training excluded). On NAS-Bench-201 (up to 100 architectures, 10 runs, true accuracies): 94.37 / 73.44 / 46.79 against random search 93.83 / 71.52 / 45.37, RL 93.94 / 71.78 / 45.95, evolution 94.23 / 72.82 / 46.49 (their evolution: population 10 over 20 iterations). GENIUS and LLMatic values are copied from another paper.
- **My reading.** Evolution here is within 0.14 (CIFAR-10) of CoLLM-NAS on a table where random search is 0.54 lower; the budgets given for the controls are not matched to 100 and are not all stated.

### 2.7 LLMasTool (Yoshimura et al., Sony; arXiv 2604.16555)
- **Method (author-stated).** Mines reusable modules from PyTorch, MMCV, MMPretrain and timm (about 1,400 in total), represents architectures as trees, and evolves them with tree transformations; a diversity-guided Bayesian algorithm plans coarse moves and **the LLM (Qwen3-8B) only fills the remaining degrees of freedom**. "LLM as a tool, not an agent."
- **Their statement of the problem (author-stated):** LLMs "struggle to reliably generate complex, valid architectures", and "proposals are often biased toward a narrow set of patterns observed in their training data".
- **Results (author-stated).** NAS-Bench-201-style setting but an **open-ended** space started from ResNet-32: 95.08 / 76.52 / 50.43 after 100 architectures, above the tabulated optimum of the fixed cell space (94.37 / 73.51 / 47.31), so these numbers are not comparable with cell-space tables. Their own random search and REA rows (500 architectures, cell space): 93.70 / 71.04 / 44.57 and 93.92 / 71.84 / 45.54.
- **Limitations (author-stated).** LLM generation takes about one minute; training each candidate takes 60 to 150 minutes (200 epochs on CIFAR-100 or ImageNet16-120). They say future work needs reliable zero-shot evaluation of structurally different architectures.
- **My reading.** Even the paper arguing the LLM should be used sparingly keeps conventional search doing the planning.

### 2.8 LLM-NAS (the paper being reproduced; read earlier)
- **Claimed gap (author-stated):** supernet methods cost GPU-days; applying an LLM directly to a HW-NAS space shows **exploration bias**; earlier LLM methods "rely on static prompts". Remedy: complexity partitioning, a co-evolved knowledge base, and a zero-cost predictor.
- **Claimed result (author-stated):** higher hypervolume, lower IGD and up to 54% lower latency at similar accuracy than baselines, in 120 API calls and about 3 minutes against days for supernet methods.
- **Our study (earlier work):** with the same evaluator and the same niche structure, none of 24 LLM runs (four LLMs, six devices, CIFAR-10, one seed each) exceeded random-operator evolution, and one exceeded random sampling; this is a statement about a small tabulated space (see Part 3, hook 7).

---

## 3. Comparison across the LLM-NAS papers

| Paper | LLM | Space | Evaluator | Evaluated per search | Compared with a conventional operator on the same evaluator? |
|---|---|---|---|---|---|
| EvoPrompting | PaLM 62B | Python code (MNIST-1D, CLRS) | train each child | 1,600 generated | No (ablations are LLM variants; NATS-Bench appendix copies numbers) |
| GENIUS | GPT-4 | NAS-Bench-Macro, Channel-Bench-Macro, NB201, MobileNetV2 space | lookup / 20-epoch training | 10 iterations | Random sampling only |
| LLMatic | CodeGen 6.1B | CIFAR-10 code; NB201 | train 50 epochs / lookup | 2,000 | No evolutionary control; random search row, budget unstated |
| LAPT | GPT-4 | NB201, Trans101 (transfer) | lookup / training | 5-9 trained (counted) | REA, HEBO etc. (different knowledge access) |
| CoLLM-NAS | Qwen3-30B | OFA, SPOS, AutoFormer supernets; NB201 | supernet validation / lookup | 250 (ImageNet), 100 (NB201) | Yes on ImageNet (baseline evolution); NB201 rows have unstated budgets |
| LLMasTool | Qwen3-8B | open-ended module trees | train each candidate | 100-500 | Yes in NB201-style table (cell space) |
| LLM-NAS | GPT-4.1 and others | NB201 cell + HW-NAS-Bench | ZC predictor + lookup | 120 calls | Paper: some baselines; our study: yes, same evaluator |

---

## 4. Hooks for the "why LLMs" argument (hooks, not conclusions)

1. **The literature mostly does not test LLM against a conventional operator on the same evaluator.** EvoPrompting, LLMatic and GENIUS ablate the LLM method or compare with random sampling; none swaps the operator while holding the evaluator and budget fixed. That is what our controls do.
2. **Where tables allow a comparison, earlier LLM methods sit near random search on NAS-Bench-201** (my reading): GENIUS 93.79 vs random 93.83 (CoLLM-NAS's table) or 93.70 (LLMasTool's); LLMatic 71.62 on CIFAR-100 against evolution 72.82. CoLLM-NAS itself states earlier LLM methods "fail to surpass established NAS baselines". This is independent support for our result, with the caveat that budgets and repeat counts differ across the tables.
3. **Cost savings come from the evaluator.** LLMasTool reports about one minute of LLM time against 60-150 minutes of training per candidate, so with trained evaluation the LLM is negligible, and the benefit must come from needing fewer evaluations. With LLM-NAS's roughly one-millisecond evaluation the ratio reverses and the LLM is the cost. This states precisely when an LLM operator could pay off (expensive evaluation) and fits the "untested" note in our discussion.
4. **Open-ended spaces are where the argument is strongest.** EvoPrompting (new graph networks on CLRS) and LLMasTool work where no natural mutation operator exists. The LLM-NAS paper searches a fixed six-edge cell where a conventional operator is trivial (my reading). The two uses should not be conflated in the introduction.
5. **Contamination is a stated, unresolved risk** (GENIUS): NAS-Bench-201 and HW-NAS-Bench are public and may be in the training data. Our six-device results cannot rule this out either; this belongs in threats to validity.
6. **Authors' own warnings** (non-reproducibility at temperature 0, LLM bias to narrow patterns, invalid architectures, FLOPs violations) support running several seeds, which we did for controls but not for LLMs.

---

## 5. What is unverified or not covered
- ELM, OPRO: abstracts and limitation sections only. LAPT, CoLLM-NAS, LLMasTool: not read end to end. LLM-NAS: read earlier; its baseline table was not re-read for this draft.
- GENIUS random-search numbers were not extracted; the "730" candidates statement is unexplained.
- Cross-paper numbers come from tables in the cited papers; several copy each other's values and use different budgets, so my "near random search" statements are about those tables, not about re-runs.
- Citation counts: CoLLM-NAS not returned by OpenAlex; LLM-NAS has two records (1 and 2).
- Not read: LeMo-NADe, NADER, LM-Searcher, RZ-NAS, the survey on evolutionary computation with LLMs, and "LLMs as Evolutionary Optimizers".

## 6. Questions for your review
1. Is this paper set right (8 LLM-related papers including the one reproduced), or should I read and add NADER, LM-Searcher, or the LLM-evolutionary-optimiser papers?
2. Hook 4 separates open-ended spaces from fixed cells. Do you want that distinction in the introduction?
3. Hook 2 relies on cross-paper tables with unmatched budgets. Do you want me to state it as support with that caveat, or leave it to the discussion?
4. After your review, shall I move to the actual rewrite of the Introduction (why HW-NAS, how it was done, pitfalls and advantages, then the gap), drafted as one piece for your review before it goes into the reports?

## Sources
- EvoPrompting: https://arxiv.org/abs/2302.14838
- GENIUS: https://arxiv.org/abs/2304.10970
- LLMatic: https://arxiv.org/abs/2306.01102
- LAPT: https://doi.org/10.1609/aaai.v39i21.34463 (arXiv 2408.11330)
- CoLLM-NAS: https://arxiv.org/abs/2509.26037
- LLMasTool: https://arxiv.org/abs/2604.16555
- ELM: https://arxiv.org/abs/2206.08896
- OPRO: https://arxiv.org/abs/2309.03409
- LLM-NAS: https://arxiv.org/abs/2510.01472
- Citation counts: OpenAlex API, queried 2026-11-06
