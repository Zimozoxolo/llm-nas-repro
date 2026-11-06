# Introduction rewrite: draft for your review

Status: draft only. Nothing here is in the LaTeX reports yet. Citations are written as (Author, year) and will be converted to the reports' numbered references after you approve. Every number comes from Parts 1-4 in this folder, where it was read from the paper itself; the audit table at the end shows which part each claim comes from.

This draft replaces the current Introduction, including the ChamNet, MobileNetV3, search-space-limit, published-table and open-ended-versus-fixed-cell paragraphs added earlier, which are folded in below.

---

## 1. Introduction

### 1.1 Why hardware-aware search

A network that is accurate is not necessarily one that runs well on a given device. Hardware-aware neural architecture search (HW-NAS) looks for architectures that trade task accuracy against a cost measured on the target hardware, such as latency or energy. The reason to measure, not to count operations, is that operation counts predict the cost poorly. MobileNet and NASNet have similar FLOPs (575M and 564M) but run in 113 ms and 183 ms on the same phone (Tan et al., 2019); a network with 19% fewer multiply-accumulates was 29% slower in practice (Yang et al., 2018). On the six devices of HW-NAS-Bench, the Kendall rank correlation between FLOPs and measured latency ranges from 0.83 on an FPGA down to 0.18 on an Edge TPU (Li et al., 2021). The best network also differs by device: an architecture searched for an iPhone X was 1.4 times faster there than one searched for a Samsung phone (Wu et al., 2019).

### 1.2 How it was done before LLMs

Early work on hardware-aware network design includes multi-objective structure search with a modelled implementation cost (Smithson et al., 2016), Bayesian optimisation of accelerator and network settings (Hernández-Lobato et al., 2016; Reagen et al., 2017), power- and memory-constrained hyper-parameter optimisation from GPU measurements (Stamoulis et al., 2017), and search with latency measured on the target phone (Yang et al., 2018; Tan et al., 2019).

MnasNet put measured latency into a reinforcement-learning reward and needed about 4.5 days on 64 TPUv2 devices for roughly 8,000 sampled models (Tan et al., 2019). Later work reduced that cost mainly by removing per-candidate training. ProxylessNAS made latency differentiable through a prediction model and searched on the target task in 200 GPU-hours (Cai et al., 2019). FBNet used an additive lookup table of operator latencies and 216 GPU-hours (Wu et al., 2019). Once-for-All trained one large network once (about 1,200 GPU-hours) and then searched sub-networks for each device with an accuracy predictor and a latency table (Cai et al., 2020). ChamNet adapted an existing network with a genetic algorithm whose fitness came from a Gaussian-process accuracy predictor, an energy predictor and a lookup table of about 350,000 operator latencies, which its authors report takes minutes for a new platform once the predictors exist (Dai et al., 2019). In this recipe the saving comes from the evaluator; the search operator is a conventional genetic algorithm. MobileNetV3 combined a MnasNet-style search with layer-wise adaptation (Yang et al., 2018) and then improved the result by hand; its authors state that redesigning the first and last layers and adding the h-swish nonlinearity lay outside the search space, and call the blend of search and human intuition an open question (Howard et al., 2019).

### 1.3 Search strategies, with their advantages and pitfalls

A search needs a search space, a search strategy and a way to estimate performance (Elsken et al., 2019). Reinforcement learning composes novel structures but trained every child: 12,800 architectures on 800 GPUs in the original work (Zoph and Le, 2017). Regularised evolution is simple and ahead of reinforcement learning early in a search, but the two were similar at 20,000 models, at about 450 GPUs for 7 days (Real et al., 2019). Gradient-based search (DARTS) cut the cost to 1.5-4 GPU-days (Liu et al., 2019) but its authors note the search must be repeated and memory grows with the candidate set, and later benchmark studies report it collapsing to a cell of skip connections (Dong and Yang, 2020; Dong et al., 2021). Weight sharing made evaluation cheaper still (0.45 GPU-days for ENAS; Pham et al., 2018), at the price of weaker ranking: the supernet's Kendall correlation with true ranking was only 0.42-0.64 in one study (Guo et al., 2020). Surrogate-guided search trains only the candidates a predictor selects, using up to five times fewer evaluations than reinforcement learning (Liu et al., 2018).

Random search has repeatedly been reported as a strong baseline in this literature (Zoph et al., 2018; Liu et al., 2019; Li and Talwalkar, 2020), and one study found that several state-of-the-art algorithms perform similarly to a random policy on average (Yu et al., 2020). The same literature documents weak evaluation practice: of 12 recent NAS papers, none were exactly reproducible (Li and Talwalkar, 2020).

### 1.4 How performance is estimated today

Most of the cost reduction above came from cheaper performance estimation, not from smarter search. Tabulated benchmarks store the trained accuracy of every architecture in a space: NAS-Bench-201 holds 15,625 cells (Dong and Yang, 2020), of which NATS-Bench lists about 6.5 thousand as distinct topologies (Dong et al., 2021). HW-NAS-Bench adds latency and energy for each cell on six devices, four measured on hardware (Edge GPU, Raspberry Pi 4, Edge TPU, Pixel 3) and two estimated by simulators (Eyeriss, FPGA) (Li et al., 2021). Zero-cost proxies score untrained networks in about a second, and combining a proxy with random search, reinforcement learning or evolution sped them up by up to four times on NAS-Bench-101 (Abdelfattah et al., 2021). A boosted-tree regressor fitted on sampled architectures is the standard way to make spaces of 10^18 to 10^21 architectures cheap to query (Zela et al., 2022).

These tools have pitfalls. Tables exist only for small spaces; proxies are biased toward larger architectures (Krishnakumar et al., 2022); a latency predictor's errors can change which models are Pareto optimal (Dudziak et al., 2020); and conclusions from small tabulated spaces need not carry over to large ones. In NATS-Bench, random search trails regularised evolution on CIFAR-100 and ImageNet-16-120 (Dong et al., 2021), and in the 10^18-10^21 spaces of NAS-Bench-301 random search needs many times more evaluations than regularised evolution or HEBO to reach a comparable architecture (Zela et al., 2022).

### 1.5 Where language models enter, and the gap they claim to fill

LLM-guided search papers argue four gaps in the earlier approaches: hand-crafted search spaces restrict and bias the search (Chen et al., 2023; Nasir et al., 2024); mutation in evolution or reinforcement learning is uninformed and wasteful (Nasir et al., 2024); design knowledge is not reused (Zheng et al., 2023; Zhou et al., 2025); and supernet search is costly (Zhu et al., 2025; Li et al., 2025). LLM-NAS (Zhu et al., 2025) addresses hardware-aware search by replacing both the supernet and the evolutionary operator with a language model that is confined to complexity niches to counter exploration bias, co-evolves a textual knowledge base of design rules, and scores candidates with a training-free zero-cost predictor. It reports near-perfect Pareto fronts (hypervolume about 0.99) at 120 API calls per search.

Seen against the history above, three of these claims concern the search operator and its space, and the fourth, cost, concerns how candidates are evaluated. Using prior knowledge in search is not new (NSGA-Net starts from hand-crafted architectures, Lu et al., 2019; PNAS searches from simple to complex, Liu et al., 2018), and the cost saving of LLM-NAS comes from a tabulated benchmark and a cheap predictor, as in ChamNet, not from the language model. The change LLM-NAS makes is the operator. Whether that change improves on a conventional operator that uses the same evaluator is the first question this study tests. Whether the language model supplies design knowledge that search alone lacked, as MobileNetV3's authors did by hand, is the second.

LLM operators are most plausible where no natural mutation operator exists, as when searching over program code (Chen et al., 2023; Yoshimura et al., 2026); the NAS-Bench-201 cell is a fixed set of six edges with five operators each, where a conventional operator is simple, so our result concerns that setting. Published tables on NAS-Bench-201 place earlier LLM-based methods close to random search (GENIUS, Zheng et al., 2023; EvoPrompting on the NATS-Bench size space, Chen et al., 2023), and the authors of CoLLM-NAS state that such methods fail to surpass established baselines (Li et al., 2025); because the budgets and repeat counts in those tables are not matched, we treat this as motivation, not as evidence. Our finding that random and evolutionary controls match the LLM operator is therefore a statement about this small space with a cheap evaluator, and we do not extend it to large search spaces or to expensive evaluators.

### 1.6 This study

This study asks two questions of LLM-NAS and builds a framework to answer them: (1) with the same evaluator and budget, does the LLM operator improve on a conventional operator; (2) does it supply design knowledge that search lacked. Its three goals are:

1. Build an organised, re-runnable reproduction: one command per paper table or figure, exact replay without API access, and an explicit record of every assumption.
2. Validate that the benchmark data we use is the data the paper used.
3. Run the method with the paper's LLM (GPT-4.1), several other LLMs and an open 7B model, and compare with the paper's reported numbers and with non-LLM controls that use the same evaluator and the same niche structure, at matched numbers of evaluated architectures.

Section 2 describes the method and framework, Section 3 the results, Section 4 what was discovered, and Section 5 next steps. Appendix A defines all technical terms and formulas.

---

## 2. Audit table (claim → where it was read)

| Claim in the draft | Source and part |
|---|---|
| 575M/564M FLOPs, 113/183 ms | MnasNet, Part 1 §2.1 |
| 19% fewer MACs, 29% longer latency | NetAdapt, Part 1 §2.5 |
| Kendall FLOPs-latency 0.18 (Edge TPU) to 0.83 (FPGA) | HW-NAS-Bench, Part 3 §2.5 |
| iPhone X search 1.4× faster than Samsung-searched | FBNet, Part 1 §2.3 |
| Early-work wording | Part 1 §1.4 |
| MnasNet 4.5 days, 64 TPUv2, ~8K models | Part 1 §2.1 |
| ProxylessNAS 200 GPU-hours; FBNet 216; OFA ~1,200; ChamNet ~350K records | Part 1 §2.2-2.4, 2.7 |
| MobileNetV3 manual changes and "open question" | Part 1 §2.6 |
| RL 12,800 archs / 800 GPUs; evolution 450 GPUs × ~7 days, similar at 20,000 | Part 2 §2.1-2.2 |
| DARTS 1.5-4 GPU-days; memory and repeat caveats; skip-connection collapse | Part 2 §2.3; Part 3 §3.1, 3.1b |
| ENAS 0.45 GPU-days; SPOS Kendall 0.42-0.64 | Part 2 §2.4 |
| PNAS up to 5× fewer evaluations than RL | Part 2 §2.5 |
| Random search strong; 12 papers not reproducible | Part 2 §2.1, 2.3, 2.6 |
| NAS-Bench-201 15,625; 6.5k unique; HW-NAS-Bench 4 measured / 2 simulated | Part 3 §3.1, 3.1b, 2.5 |
| Zero-cost proxies up to 4× | Abdelfattah et al., Part 3 §3.3 |
| Boosted trees for 10^18-10^21 spaces; random search many times slower | NAS-Bench-301, Part 3 §3.1c |
| Proxy size bias; predictor error changes Pareto set | Part 3 §3.3, 2.4 |
| NATS-Bench: random trails REA on CIFAR-100 / ImageNet-16-120 | Part 3 §3.1b |
| Four claimed gaps | Part 4 §1 |
| NSGA-Net and PNAS prior knowledge | Part 2 §2.2, 2.5 |
| Open-ended vs fixed cell; published-table hook | Part 4 §2-4 |

## 3. Things I want you to decide

1. **Length.** This is about three times the current Introduction. Sections 1.2 and 1.3 could be cut to a short paragraph each if you want the Introduction brief and the detail moved to a Background section. Which do you prefer?
2. **Tone of 1.5.** I wrote that three of the four claimed gaps concern the operator and its space, and the cost claim concerns the evaluator, so the saving is not the LLM's. Is that framing fair, or too pointed for an Introduction?
3. **Goals list.** I rewrote goal 3 to say "matched numbers of evaluated architectures" and added the two questions. Is that how you want the goals stated?
4. **Section numbering.** I used Section 2 to 5 and Appendix A to match the standalone report; the full report may need different numbers. Shall I check each?
5. **Citations.** About 30 references are new. After you approve the text I will add them to both bibliographies and verify each key against the papers read.
