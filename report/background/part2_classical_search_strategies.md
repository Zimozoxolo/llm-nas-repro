# Background, Part 2: how architectures were searched before LLM guidance

Status: DRAFT FOR REVIEW. Nothing here has been put in the LaTeX reports yet.
Every number was read from the paper text (arXiv PDFs), unless marked *(computed)* or *(not verified)*.
"Author-stated" means the paper says it. "My reading" means I inferred it and you should challenge it.
Part 1 covered hardware-aware work (MnasNet, ProxylessNAS, FBNet, Once-for-All, NetAdapt, MobileNetV3, ChamNet). This part covers the **search strategies** underneath them.

---

## 0. How the selection was made (please check this first)

**Organising frame.** Elsken, Metzen and Hutter (JMLR 2019, 1,839 citations on OpenAlex) split every NAS method into three parts: the **search space** (which architectures can be represented), the **search strategy** (how the space is explored) and the **performance estimation strategy** (how a candidate's quality is obtained, since full training is expensive). I use this frame because it lets us say exactly what an LLM changes.

**Papers.** One or two of the most-cited representatives per strategy family (OpenAlex counts, queried 2026-11-05; OpenAlex counts lower than Google Scholar, so use the ranking, not the values):

| Family | Paper | Year | OpenAlex citations |
|---|---|---|---|
| Reinforcement learning | Learning Transferable Architectures for Scalable Image Recognition (NASNet), Zoph et al. | 2018 | 6,630 |
| Reinforcement learning | Neural Architecture Search with Reinforcement Learning, Zoph and Le | 2016/17 | 5,090 |
| Differentiable | DARTS, Liu et al. | 2018/19 | 3,650 |
| Evolution | Regularized Evolution for Image Classifier Architecture Search, Real et al. | 2019 | 2,674 (+432 on a duplicate record) |
| Surrogate-guided | Progressive Neural Architecture Search (PNAS), Liu et al. | 2018 | 1,945 |
| Weight sharing | Efficient NAS via Parameter Sharing (ENAS), Pham et al. | 2018 | about 1,551 and 1,196 on two records |
| Weight sharing | Single Path One-Shot NAS with Uniform Sampling, Guo et al. | 2020 | 882 |
| Evolution, multi-objective | NSGA-Net, Lu et al. | 2019 | 574 |
| Random search and evaluation | Random Search and Reproducibility for NAS, Li and Talwalkar | 2019 | 562 |
| Surrogate-guided | BANANAS, White et al. | 2021 | 287 |
| Evaluation | Evaluating the Search Phase of NAS, Yu, Sciuto et al. | 2019/20 | 219 and 136 |

**Choices to check.**
1. I did not include Large-Scale Evolution of Image Classifiers (Real et al. 2017, 1,372) or Neural Architecture Optimization (533); say if you want them.
2. Zero-cost proxies and benchmark tables (NAS-Bench-201, NASWOT, Abdelfattah et al.) are **performance estimation**, so I left them for Part 3.
3. Bayesian optimisation with Gaussian processes (NASBOT) did not return in my citation query, so it is not covered; BANANAS stands for the BO family.

---

## 1. Where the LLM sits in this frame (my reading, not a finding)

| NAS component | Classical choices (this part) | LLM-NAS |
|---|---|---|
| Search space | NASNet-style cells, chain/layer spaces | NAS-Bench-201 cell, 15,625 architectures, partitioned into six complexity niches |
| Search strategy | RL controller, evolution, gradient descent, surrogate-guided search, random | **An LLM** proposing children by crossover/mutation, with a text knowledge base of design rules |
| Performance estimation | Train to convergence, shared weights, learned predictor | Zero-cost-proxy XGBoost predictor and a latency table |

So the LLM changes **one** of the three components, the search strategy. The classical papers below show what that component was doing before, what it cost, and how it compared with a random baseline.

---

## 2. The families

### 2.1 Reinforcement learning

**Zoph and Le, "Neural Architecture Search with Reinforcement Learning" (arXiv 1611.01578; ICLR 2017).**
- *Idea.* A recurrent-network **controller** writes an architecture as a variable-length string; the child network is built and trained, and its validation accuracy is the reward for the **REINFORCE** policy gradient.
- *Evaluation.* **Every sampled child is trained to convergence.** Cost: 800 GPUs trained concurrently, 12,800 architectures sampled for CIFAR-10.
- *Results.* 3.65% test error on CIFAR-10 (author-stated "0.09 percent better and 1.05x faster" than the previous state of the art with a similar scheme); Penn Treebank perplexity 62.4.
- *Cost in later papers.* NASNet's footnote quotes this search as 800 GPUs for 28 days; 800 × 28 = **22,400 GPU-days** *(computed; the footnote writes "GPU-hours", which looks like a slip)*, on K40 GPUs.
- *Advantages (author-stated).* Flexible: it can compose novel variable-length structures, which fixed-length hyper-parameter optimisers could not; learns from the reward alone with no bootstrapping.
- *Pitfalls.* Author-stated: related neuro-evolution work was "slow or require many heuristics"; here, training each child "can take hours", so they needed distributed training. *My reading:* the cost is dominated by training candidates, and the controller RNN adds a policy to train on top.

**Zoph, Vasudevan, Shlens, Le, "Learning Transferable Architectures for Scalable Image Recognition" (NASNet; arXiv 1707.07012; CVPR 2018).**
- *Idea.* Search a **cell** on CIFAR-10 and stack copies on ImageNet; the contribution is the *search space* that makes this transfer ("NASNet search space"), plus ScheduledDropPath regularisation. Controller trained with PPO.
- *Cost.* 500 GPUs for 4 days (author-stated); the paper calls this almost 7× faster than the original NAS (28 days) and says 2,000 GPU-hours *(sic; 500 × 4 = 2,000 GPU-days by my computation)*.
- *Results.* 2.4% CIFAR-10 error; 82.7% top-1 / 96.2% top-5 on ImageNet, 1.2% better than the best human design with 9 billion fewer FLOPs; a small version reached 74.0% top-1, 3.1% above equal-size mobile models.
- *Random baseline (important).* The authors **ran random search in the same space**: "random search is slightly worse than reinforcement learning on the CIFAR-10 dataset", the gap being "smaller than what is found in the original work", which they read as showing that the search space is well constructed and that "random search is a difficult baseline to beat". In Section 4.4 the best RL model beat the best random-search model by over 1% on CIFAR-10 at equal compute, and RL found an entire range of better models.
- *Advantages.* Strong, transferable results; search-space design that reduces cost.
- *Pitfalls.* Still thousands of GPU-days; searched on a **proxy dataset** (CIFAR-10) and transferred; ProxylessNAS later argues proxy-optimised blocks are not guaranteed to be optimal on the target task or hardware (Part 1). Both Zoph papers use no hardware metric; MnasNet later added measured latency to this exact controller setup.

### 2.2 Evolution

**Real, Aggarwal, Huang, Le, "Regularized Evolution for Image Classifier Architecture Search" (arXiv 1802.01548; AAAI 2019).**
- *Idea.* Tournament-selection evolution with an **age** property: the oldest member is removed each cycle, not the worst ("aging evolution"). Mutations only reconnect an edge's origin or relabel an edge, in the NASNet space.
- *Evaluation.* Every individual is **trained**. Each of the controlled experiments used 450 K40 GPUs for 20,000 models (about 7 days; 450 × 7 = 3,150 GPU-days, matching DARTS's quote for AmoebaNet).
- *Controlled comparison.* Five repeats each of evolution, RL and random search (RS), same code and hardware. Evolution produced more accurate models at the **earlier stages**, which matters when experiments must be stopped early; at the full 20,000 models evolution and RL had **similar accuracy**; both were better than RS (author-stated).
- *Results.* AmoebaNet-A reached 83.9% top-1 / 96.6% top-5 on ImageNet at the larger size; the authors say it is the first evolved classifier to surpass hand designs.
- *Advantages (author-stated).* Simple method, faster than RL early in the search, useful with limited compute; only a few hyper-parameters.
- *Pitfalls.* *My reading:* the evolution-over-RL advantage is concentrated early and disappears by the end of the budget; the absolute cost (3,150 GPU-days) is the highest in this part; the operator is a random mutation, so all prior knowledge is in the search space and the tournament, not in the proposals.
- *Relevance.* **A random-mutation operator inside an evolutionary loop is exactly what the LLM replaces.** Regularized evolution is the standard strong version of that operator.

**Lu, Whalen, Boddeti, Dhebar, Deb, Goodman, Banzhaf, "NSGA-Net" (arXiv 1810.03522; GECCO 2019).**
- *Idea.* Multi-objective NAS with **NSGA-II**; objectives are classification error and **FLOPs**; the whole network structure is searched, not only a repeated cell. Three stages: initialisation from hand-crafted architectures (prior knowledge), exploration by crossover and mutation, and exploitation by a **Bayesian network** built from the search history.
- *Results.* 3.85% error on CIFAR-10 with 3.34M parameters and 1,290 MFLOPs, at **8 GPU-days** (author-stated), versus the 3,150 and 2,000 GPU-days the authors quote for two earlier methods (they call both RL; 3,150 is AmoebaNet's figure in other papers); they note it is less efficient than DARTS.
- *Advantages.* A whole Pareto set in one run, flexible space, orders of magnitude cheaper than the RL and evolution runs above.
- *Pitfalls.* Author-stated: hypervolume could not be compared with other multi-objective NAS papers because they do not report their fronts and use different complexity objectives, so there is **no common multi-objective yardstick**. *My reading:* FLOPs is a proxy for hardware cost, the thing MnasNet, FBNet and NetAdapt (Part 1) show can disagree with measured latency.
- *Relevance.* Multi-objective, Pareto-based search, population-based operators and prior-knowledge initialisation all appear here, before LLMs.

### 2.3 Gradient-based (differentiable) search

**Liu, Simonyan, Yang, "DARTS" (arXiv 1806.09055; ICLR 2019).**
- *Idea.* Relax the discrete choice among operations to a **softmax over all candidates**, so architecture weights α can be trained by gradient descent jointly with the network weights, with a bi-level formulation and a first/second-order approximation.
- *Cost and results.* **1.5 GPU-days** (first order, 3.00 ± 0.14% error) or **4 GPU-days** (second order, **2.76 ± 0.09%**) on CIFAR-10 with 3.3M parameters, against 2,000 GPU-days for NASNet and 3,150 for AmoebaNet; Penn Treebank perplexity 55.7 in about 1 GPU-day.
- *Random baseline.* The authors write that random search "is competitive for both convolutional and recurrent models", reflecting the importance of search-space design, and report DARTS 2.76 ± 0.09 vs random 3.29 ± 0.15 on CIFAR-10 and 55.7 vs 59.4 on PTB.
- *Ablation.* Optimising α and w together over the training and validation data (no bi-level) gave 4.16 ± 0.16%, which is **worse than random search** (author-stated).
- *Advantages.* Three orders of magnitude cheaper; easy to implement; the latency term can be added to the loss (ProxylessNAS, FBNet in Part 1).
- *Pitfalls.* Author-stated: the search is repeated four times and the best cell chosen; results transfer less well from a small source task (PTB to WikiText-2); the supernet holds all candidate operations in memory; in an appendix experiment with a deeper supernet (20 cells instead of 8) the initial channels had to be cut from 16 to 6 to fit one GPU, the search cost doubled and the error was slightly worse (2.88 ± 0.09% vs 2.76 ± 0.09%). *My reading, from Part 1 and titles only:* ProxylessNAS builds binarised paths for exactly this memory problem; follow-up work with titles such as "DARTS+: early stopping" and "Fair DARTS" points to instabilities in the method, but I did not read those papers.
- *Fit for non-differentiable objectives.* Latency is not differentiable, so ProxylessNAS and FBNet model it (predictor, lookup table). Authors of BANANAS list "optimization of other/multiple non-differentiable objectives" among the advantages of zeroth-order search.

### 2.4 Weight sharing and one-shot supernets

**Pham, Guan, Zoph, Le, Dean, "ENAS" (arXiv 1802.03268; ICML 2018).**
- *Idea.* All child models are sub-graphs of one large graph and **share weights**, so a child is not trained from scratch; an RNN controller selects subgraphs by policy gradient while the shared weights train on cross-entropy.
- *Cost and results.* About 10 hours for the Penn Treebank cell (perplexity 55.8) and 11.5 hours on one GPU for the CIFAR-10 cell (**0.45 GPU-days**), 3.54% test error, 2.89% with CutOut versus 2.65% for NASNet-A; the abstract claims "1000x less expensive than standard NAS".
- *Pitfalls.* Author-stated: the discovered models appear to be local minima (changing a few connections raised error by 2.1%). *From the later study by Yu et al. (Section 2.6):* ENAS's CIFAR-10 accuracy beats random sampling but its PTB result does not (Table 1 there).

**Guo, Zhang, Mu, Heng, Liu, Wei, Sun, "Single Path One-Shot NAS with Uniform Sampling" (arXiv 1904.00420; ECCV 2020).**
- *Idea.* Train a supernet where each path is a single architecture, sampling paths **uniformly**, to reduce weight coupling; then search the trained supernet with an **evolutionary algorithm** that accepts constraints (FLOPs, latency).
- *Cost.* Supernet training 12 GPU-days on 8 GPUs, search under 1 GPU-day, retraining 16: total 29 GPU-days versus 31 for ProxylessNAS and 36 for FBNet under the same table; the supernet is trained once and reused for different constraints.
- *Hardware.* Handles FLOPs and latency constraints in the search step; latency measured on a Titan XP.
- *Pitfalls (author-stated).* On NAS-Bench-201 (and reduced spaces) the Kendall tau between supernet ranking and true ranking is only **0.42-0.64**; the authors state the ranking is "partially correlated, but not perfectly correlated", so the method "can not guarantee to find the real best architecture". They also state that "several state-of-the-art NAS methods perform similarly to the random search".
- *Advantages.* Decouples supernet training from search, so many constraints can be searched cheaply; one supernet supports channel, mixed-precision and block searches.

### 2.5 Surrogate-guided and Bayesian search

**Liu, Zoph, Neumann, Shlens, Hua, Li, Fei-Fei, Yuille, Huang, Murphy, "Progressive NAS" (PNAS; arXiv 1712.00559; ECCV 2018).**
- *Idea.* Sequential model-based optimisation: search cells in order of **increasing complexity** while training a **surrogate** (MLP or RNN ensemble) that predicts accuracy, so only the surrogate's top-K children are actually trained.
- *Evaluation.* Only the surrogate-selected candidates are trained; the surrogate is retrained as data arrives.
- *Results.* In the same search space PNAS is up to **5× more efficient than the RL method in models evaluated and 8× in total compute** (author-stated); the paper also compares with random search over 6,000 cells and reports that PNAS "significantly outperforms" it; it reports 21× less compute than the original NAS and 36× less than hierarchical evolution, with the caveat that those comparisons are across different setups.
- *Advantages.* Sample-efficient; the learned surrogate is a precursor of the accuracy predictors in ChamNet and once-for-all (Part 1).
- *Pitfalls.* *My reading:* the surrogate is trained on a few thousand trained cells and is only asked to extrapolate one block further; its ranking quality bounds the search; the paper reports rank correlation by search stage.

**White, Neiswanger, Savani, "BANANAS: Bayesian Optimization with Neural Architectures for NAS" (arXiv 1910.11858; AAAI 2021).**
- *Idea.* "BO + neural predictor": five components (architecture encoding, neural predictor, uncertainty calibration, acquisition function, acquisition optimisation); a novel **path encoding** of the architecture.
- *Evidence.* On NAS-Bench-101 (test error after 150 architecture evaluations, Table 2, averaged over 200 trials): random search 6.341, REINFORCE 6.436, regularised evolution 6.109, local search 5.932, **BANANAS 5.923**. So REINFORCE is **worse than random** in this table and the best method is 0.42 points below random.
- *Authors' stated advantages of zeroth-order search:* simple parallelism, easy implementation, portability to diverse spaces, optimisation of multiple non-differentiable objectives.
- *Pitfalls.* *My reading:* gains over random search are small in absolute terms on tabular benchmarks; the authors themselves cite that "multiple papers have concluded that random search is a competitive baseline".
- *Relevance.* BANANAS evaluates on **NAS-Bench-101/201 tables** (precomputed accuracies), as LLM-NAS and our reproduction do with NAS-Bench-201 and HW-NAS-Bench, and follows the NAS best-practices checklist (author-stated).

### 2.6 Random search and the evaluation of search itself

**Li and Talwalkar, "Random Search and Reproducibility for NAS" (arXiv 1902.07638).**
- *Claim (author-stated).* NAS is a specialised hyper-parameter optimisation problem and random search is a competitive baseline; random search with early stopping performs **at least as well as ENAS** on PTB and CIFAR-10; random search with weight sharing reaches a state-of-the-art PTB result and a highly competitive CIFAR-10 result.
- *Reproducibility (author-stated).* Of 12 NAS papers published since 2018 at NeurIPS, ICML and ICLR, **none were exactly reproducible** (missing search or evaluation code, random seeds or undocumented tuning).
- *Three problems named:* inadequate baselines, complex methods without ablations, lack of reproducibility.

**Yu, Sciuto, Jaggi, Musat, Salzmann, "Evaluating the Search Phase of NAS" (arXiv 1902.08142; ICLR 2020).**
- *Claim (author-stated).* (i) On average state-of-the-art NAS algorithms perform **similarly to a random policy**; (ii) weight sharing degrades the ranking of candidates to the point of not reflecting their true performance.
- *Table 1.* CIFAR-10 mean top-1 accuracy: ENAS 96.79 ± 0.11 (t-test p = 0.01 vs random), DARTS 96.62 ± 0.23 (p = 0.20), NAO 96.86 ± 0.17 (p = 0.00), random 96.44 ± 0.19. PTB perplexity (lower is better): ENAS 59.88 ± 1.92 (p = 0.73), DARTS 60.61 ± 2.54 (p = 0.62), NAO 61.99 ± 1.95 (p = 0.02, worse than random), random 60.13 ± 0.65.
- *Reduced-space findings.* All search policies failed to find the best architecture; DARTS never found a top-5 architecture; NAO's ranking was not correctly updated.
- *Pitfalls of this evidence.* *My reading:* both papers are about weight-sharing-era methods and image/language benchmarks, not hardware-aware multi-objective search; they establish the **method** (compare with random sampling, report seeds and tests) more than a verdict.
- *Relevance.* This is the closest published precedent to the control experiments in our justification section.

---

## 3. Comparison across families

Costs are as reported by each paper, on different hardware and tasks, so **they are not directly comparable** (author-stated in several places, e.g. PNAS and NASNet).

| Family | How a candidate is evaluated | Reported search cost | Main advantage | Main pitfall | Hardware objectives |
|---|---|---|---|---|---|
| RL (Zoph and Le; NASNet) | Train each child | 22,400 GPU-days (original); 500 GPUs × 4 days (NASNet) | Flexible, composes new structures | Cost; proxy datasets; policy to train | Added later (MnasNet latency reward) |
| Evolution (Regularized Evolution) | Train each individual | 450 GPUs × about 7 days | Simple; strong early | Gain over RL vanishes late; cost | Possible (NSGA-II, ChamNet fitness) |
| Multi-objective evolution (NSGA-Net) | Train each individual | 8 GPU-days | Pareto set in one run | FLOPs proxy; no common yardstick | FLOPs; latency in other work |
| Gradient (DARTS) | Shared weights, bi-level | 1.5-4 GPU-days | Cheap, easy | Memory; instability; discretisation; search repeated | Needs a differentiable cost model |
| Weight sharing (ENAS) | Shared weights | 0.45 GPU-day | Very cheap | Rank degradation; near random in some tests | Not native |
| One-shot (SPOS) | Trained supernet, uniform paths | 29 GPU-days total | Many constraints from one supernet | Kendall tau 0.42-0.64 | FLOPs/latency in the evolutionary step |
| Surrogate (PNAS, BANANAS) | Learned predictor plus partial training | 5×-8× below RL (PNAS) | Sample efficiency | Predictor error bounds quality | HyperPower, Reagen, ChamNet (Part 1) |
| Random (Li and Talwalkar) | Early stopping or shared weights | Lowest | Strong baseline | None as a method; it is the yardstick | Trivially compatible |

---

## 4. What this part establishes for the "why LLMs" argument (hooks, not conclusions)

1. **Random search is a strong, repeatedly documented baseline.** NASNet ("a difficult baseline to beat"), DARTS ("competitive"), SPOS (quoting others), BANANAS ("competitive"), Li and Talwalkar (ENAS no better), Yu et al. (similar to random on average). Our own result, that LLM operators do not beat random-operator evolution on HW-NAS-Bench, fits this literature; it is not an outlier.
2. **The same two gaps keep appearing:** the benefit is attributed to the search strategy, but the search space and the performance estimator often account for it (NASNet's authors on search-space design; DARTS on search-space design; our finding on the zero-cost predictor).
3. **Cost reductions came from performance estimation, not from smarter search:** NAS to ENAS/DARTS/PNAS cut cost by sharing weights or by predicting accuracy. LLM-NAS's cost cut follows the same route (training-free predictor) and then adds a per-search API cost.
4. **Prior knowledge is not new.** NSGA-Net initialises from hand-crafted architectures; PNAS searches from simple to complex; ChamNet and once-for-all use predictors. The LLM's claim is a more general, language-expressed prior. That claim is what the justification section tests.
5. **Evaluation practice is weak in the field:** none of 12 recent NAS papers exactly reproducible (Li and Talwalkar), single seeds (Yu et al.), no shared multi-objective yardstick (NSGA-Net). This supports the reproducibility and multi-seed emphasis of this study, and it is a limit on every "state of the art" claim cited here, including claims about LLM-NAS.

## 5. What is unverified or not covered
- Costs are the papers' own numbers on different hardware; two papers' footnotes use "GPU-hours" where the arithmetic gives GPU-days (flagged above).
- I read abstracts, introductions, method and results passages, not every appendix; BANANAS's component analysis and PNAS's predictor study are not summarised in detail.
- Not covered: NASBOT and Gaussian-process BO for NAS, Hyperband/BOHB, Monte Carlo tree search, hardware-specific latency predictors (BRP-NAS, HELP), and Large-Scale Evolution.
- Follow-up papers about DARTS instability are referred to by title only.
- Zero-cost proxies and benchmark tables are deferred to Part 3.

## 6. Questions for your review
1. Is the three-part frame (search space / strategy / performance estimation) the right way to organise Parts 2 and 3, and the statement that the LLM changes only the strategy?
2. Are these eleven papers enough, or should I add Large-Scale Evolution, Neural Architecture Optimization (also a learned-encoder search) or a Gaussian-process BO paper?
3. Hook 1 (random search is a strong baseline) is the strongest link to your results. Do you want it in the introduction or kept for the justification section?
4. Shall Part 3 be hardware cost models and performance estimators (measured, lookup table, predictor, benchmark tables, zero-cost proxies), as planned?

## Sources
- Zoph and Le: https://arxiv.org/abs/1611.01578
- NASNet: https://arxiv.org/abs/1707.07012
- Regularized Evolution: https://arxiv.org/abs/1802.01548
- NSGA-Net: https://arxiv.org/abs/1810.03522
- DARTS: https://arxiv.org/abs/1806.09055
- ENAS: https://arxiv.org/abs/1802.03268
- Single Path One-Shot: https://arxiv.org/abs/1904.00420
- PNAS: https://arxiv.org/abs/1712.00559
- BANANAS: https://arxiv.org/abs/1910.11858
- Random Search and Reproducibility: https://arxiv.org/abs/1902.07638
- Evaluating the Search Phase of NAS: https://arxiv.org/abs/1902.08142
- Elsken et al. survey: https://arxiv.org/abs/1808.05377
- Citation counts: OpenAlex API, queried 2026-11-05
