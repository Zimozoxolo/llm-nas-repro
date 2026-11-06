# Background, Part 3: how hardware cost and architecture quality were estimated

Status: DRAFT FOR REVIEW. Nothing here has been put in the LaTeX reports yet.
Every number was read from the paper text (arXiv PDFs), unless marked *(computed)* or *(not verified)*.
"Author-stated" means the paper says it. "My reading" means I inferred it and you should challenge it.
Part 1: hardware-aware methods. Part 2: search strategies. This part covers the third component of the frame, **performance estimation**, in its two halves: **hardware cost** (latency, energy) and **quality** (accuracy).

---

## 0. How the selection was made (please check this first)

OpenAlex citation counts (queried 2026-11-05; they run lower than Google Scholar, so use the ranking):

| Half | Paper | Year | OpenAlex citations |
|---|---|---|---|
| Quality | Speeding Up Hyperparameter Optimization by Extrapolation of Learning Curves (Domhan et al.) | 2015 | 431 |
| Quality (zero-cost) | Pruning neural networks without any data by iteratively conserving synaptic flow (SynFlow) | 2020 | 372 |
| Benchmark | NAS-Bench-201 (Dong and Yang) | 2020 | 392 |
| Benchmark | NAS-Bench-101 (Ying et al.) | 2019 | 318 |
| Quality (zero-cost) | Neural Architecture Search without Training (NASWOT, Mellor et al.) | 2020/21 | 277 |
| Hardware | nn-Meter (Zhang et al.) | 2021 | 181 |
| Quality (zero-cost) | Zero-Cost Proxies for Lightweight NAS (Abdelfattah et al.) | 2021 | 177 |
| Hardware | BRP-NAS (Dudziak et al.) | 2020 | 150 |
| Hardware | NeuralPower (Cai et al.) | 2017 | 120 |
| Quality (predictors) | How Powerful are Performance Predictors in NAS? (White et al.) | 2021 | 70 |
| Benchmark (zero-cost) | NAS-Bench-Suite-Zero (Krishnakumar et al.) | 2022 | 30 |
| Hardware benchmark | HW-NAS-Bench (Li et al.) | 2021 | **not returned** by OpenAlex (records indexed with 0 citations); included because your reproduction uses it |

**Gaps to know about.**
1. **Paleo** (an analytic performance model, 154 citations) could not be downloaded (the server returned a web page, not a PDF), so it is not summarised.
2. NATS-Bench (173 citations) and NAS-Bench-301 were added after your review and are read in sections 3.1b and 3.1c (NATS-Bench: introduction, benchmark description, analysis and search results read; NAS-Bench-301: abstract through the case study and limitations read, appendices only searched).
3. HELP (latency prediction by meta-learning, 14-17 citations) is not covered.
4. For SynFlow and Domhan I read the abstract and opening only; for NAS-Bench-Suite-Zero I read the abstract, introduction and tables of contents, not the analysis sections.

---

## 1. Where this sits in the frame, and what LLM-NAS uses (my reading)

| Half | Options | What LLM-NAS uses |
|---|---|---|
| Hardware cost | FLOPs/params proxy → analytic or regression models → operator lookup table → kernel-level predictor → end-to-end GNN predictor → direct measurement → **precomputed table of all architectures** | **HW-NAS-Bench**: a table lookup of latency per architecture per device |
| Quality | Train to convergence → **tabulated benchmark** → learning-curve extrapolation → learned predictor → weight sharing → **zero-cost proxies** | **NAS-Bench-201** tabulated accuracy for scoring the final front, and an **XGBoost regressor over 13 zero-cost proxies** (from NAS-Bench-Suite-Zero) as the search-time quality estimate |

So the whole estimation side of LLM-NAS is made of tables and a cheap predictor. That is why one evaluation costs about a millisecond in our runs, and why the LLM call dominates cost.

---

## 2. Hardware cost: from proxies to tables

### 2.1 Why proxies fail (the motivation, from papers read in Part 1 and here)
FLOPs and parameter counts are the cheapest hardware cost. HW-NAS-Bench (below) quantifies how poorly they track measured latency on the devices LLM-NAS uses; NetAdapt, MnasNet and FBNet (Part 1) gave the same warning.

### 2.2 NeuralPower (Cai, Juan, Stamoulis, Marculescu; arXiv 1710.05420; ACML 2017)
- *Idea.* A **layer-wise predictive framework based on sparse polynomial regression** that predicts runtime, power and energy of a CNN on a GPU **before training**, plus an "energy-precision ratio" metric for choosing an architecture.
- *Results (author-stated).* Up to 68.5% better prediction accuracy than the best published model; at network level, an average accuracy of 88.24% (runtime), 88.34% (power) and 97.21% (energy).
- *Advantages.* Gives a per-layer breakdown (identifies bottlenecks); needs no training of the network; is the model used by HyperPower (Part 1) for constraints.
- *Pitfalls.* *My reading:* summing layer-wise predictions assumes layers run independently (the same assumption nn-Meter later attacks); GPU platforms only; runtime accuracy of 88% leaves a visible error that can reorder close candidates.

### 2.3 nn-Meter (Zhang, Han, Wei, Zheng, Cao, Yang, Liu; MobiSys 2021)
- *Problem (author-stated).* Operator-level lookup tables, as used by many NAS papers, "do not consider the model latency differences caused by runtime optimizations", such as operator **fusion**; FLOPs are not a direct latency measure; measuring on devices is laborious (ProxylessNAS explores about 0.3 million models in one search).
- *Idea.* Split a model into **kernels** (the device's actual execution units, found by "kernel detection" using designed test cases) and predict each kernel's latency, with **adaptive sampling** to build the predictors efficiently; sum the kernel predictions.
- *Results.* On a dataset of 26,000 models: **99.0% (mobile CPU) and 99.1% (mobile GPU) of models predicted within ±10% error, and 83.4% on an Intel VPU** (author-stated).
- *Advantages.* Accounts for fusion; works across different devices and frameworks; reusable predictors.
- *Pitfalls.* *My reading:* needs a per-device kernel-detection and profiling step; the VPU result (83.4%) shows accuracy drops on harder devices; results are for CNNs on three platforms.
- *Relevance.* Independent confirmation of the weakness of additive lookup tables (the FBNet assumption noted in Part 1).

### 2.4 BRP-NAS (Dudziak, Chau, Abdelfattah, Lee, Kim, Lane; arXiv 2007.08668, v4 January 2021)
- *Idea.* An **end-to-end graph-convolutional-network (GCN) latency predictor**, which the authors say is the first end-to-end latency predictor, plus an accuracy predictor trained to predict which of two architectures is better (a **binary relation**) and an iterative data-selection strategy; also releases **LatBench**, a latency dataset of NAS-Bench-201 models on a broad range of devices.
- *Evidence (author-stated).* With a layer-wise predicted latency, the Pareto-optimal models in the predicted space "do not always lie at the Pareto frontier" of the measured space, so even a perfect search on predicted latency would often return models that are not Pareto optimal in practice. The GCN predictor outperforms the layer-wise one.
- *Relevance.* **Direct support for a point in our reports:** the quality of a Pareto front depends on the quality of the latency estimate, not only on the search algorithm.
- *Pitfalls.* *My reading:* needs measured latency data to train each device's predictor; accuracy predictor still needs trained architectures; I did not extract the numeric error values.

### 2.5 HW-NAS-Bench (Li, Yu, Fu, Zhang, Zhao, You, Yu, Wang, Lin; arXiv 2103.10584; ICLR 2021)
This is the table your reproduction queries, so it is covered in more detail.
- *What it is (author-stated).* "The first public dataset for HW-NAS research": hardware cost for **all** networks in the **NAS-Bench-201** space (15,625 cells, 46,875 counting the three datasets) and the **FBNet** space (9^22 ≈ 10^21), on six devices.
- *Devices and method (Table 1).*

| Device | Category | How the cost was obtained | Metrics |
|---|---|---|---|
| Edge GPU (Jetson TX2) | commercial edge | **Measured**, TensorRT, 50-run average | latency, energy |
| Raspberry Pi 4 | commercial edge | **Measured**, TensorFlow Lite | latency |
| Edge TPU dev board | ASIC | **Measured**, TFLite + Edge TPU compiler | latency |
| Pixel 3 | commercial edge | **Measured**, TFLite benchmark binary, big cores only | latency |
| ASIC-Eyeriss | ASIC | **Estimated** by simulators (Accelergy + Timeloop; DNN-Chip Predictor) | latency, energy |
| FPGA (Xilinx ZC706) | FPGA | **Estimated** with a Vivado HLS pipeline | latency, energy |

- **Implication for our reports (check this).** Two of your six devices, Eyeriss and the FPGA, are **simulator estimates**, not on-device measurements. If our reports call all six "measured latency", that wording needs correcting.
- *FBNet space.* For the commercial devices, the latency of an FBNet network is **approximated by summing the latency of its unique blocks**. Validation on 100 random networks (Table 2, CIFAR-100 / ImageNet): Pearson 0.92 / 0.86 (Edge GPU latency), 0.92 / 0.99 (Raspberry Pi 4), 0.93 / 0.92 (Pixel 3), but only **0.49 / 0.72 for the Edge TPU** (Kendall 0.36 / 0.58), which the authors attribute to the "in-house Edge TPU compiler" (conjecture).
- *FLOPs are a poor proxy (author-stated, Table 3, NAS-Bench-201, Kendall rank correlation of FLOPs with measured latency, CIFAR-10):* Edge GPU 0.36, Edge TPU **0.18**, Eyeriss 0.42, Pixel 3 0.68, Raspberry Pi 4 0.74, FPGA 0.83. (Parameter counts give identical values in NAS-Bench-201.)
- *Devices disagree (author-stated).* Correlations between devices' costs can be as low as −0.00 (Edge GPU latency vs Eyeriss energy, FBNet space), so the best architecture on one device can be poor on another. Table 5 (ProxylessNAS on FBNet space, CIFAR-100): the Edge-GPU-targeted network has 9.96 ms on the Edge GPU; the Raspberry-Pi-targeted one 13.88 ms (39% higher *(computed; the text says "about 50%")*); the FPGA-targeted one 20.65 ms (107% higher *(computed; the text says "over 100%")*), at similar accuracy (74.1%, 73.5%, 73.5%).
- *Advantages.* Lets non-hardware experts run HW-NAS by lookup at negligible cost; reproducible; shows device-specific search is needed.
- *Pitfalls.* Author-stated: the FBNet-space numbers are approximations; the Edge TPU approximation correlates weakly. *My reading:* the benchmark fixes a compiler and runtime per device, so it measures one software stack, not the device in general; a table cannot exceed the search spaces it covers (NAS-Bench-201 is 15,625 cells). In our own work we found 4,699 Edge TPU architectures with failed compilation recorded as −1 (issue \#11 on the benchmark repository), which is a data-quality caveat of the table.

---

## 3. Quality: from training to training-free

### 3.1 Tabulated benchmarks

**NAS-Bench-101 (Ying, Klein, Real, Christiansen, Murphy, Hutter; arXiv 1902.09635; ICML 2019).**
- 423,000 unique convolutional cells (after removing graph isomorphisms; up to 7 vertices), each trained and evaluated on CIFAR-10 **three times** at several epoch lengths (108 epochs is the longest), more than 5 million trained models in total (author-stated). Quality can be queried "in milliseconds".
- *Pitfall (author-stated):* a restricted, small cell space; accuracy is for one training recipe.

**NAS-Bench-201 (Dong and Yang; arXiv 2001.00326; ICLR 2020).**
- Fixed space of 4 nodes and 5 operations, **15,625 cells**, trained for **200 epochs** (SGD with Nesterov momentum, cosine learning rate 0.1 to 0, weight decay 0.0005) with fixed hyper-parameters on CIFAR-10, CIFAR-100 and ImageNet-16-120, with train/validation/test loss and accuracy per epoch, and multiple seeds.
- *Benchmarked algorithms:* 10 NAS algorithms. Author-stated: DARTS (first and second order) "quickly converge to find the architecture whose edges are all skip connection"; random search, regularised evolution and REINFORCE can be completed "in seconds" with the pre-computed information.
- *Isomorphism (author-stated, appendix).* Only **12,751** of the 15,625 cells are unique topologies if skip-connect isomorphism is considered, and only **6,466** if zeroize is also considered, yet every architecture was trained separately, so numerically the duplicates have slightly different accuracies.
- *Relevance.* The 15,625-architecture space is small and contains many functionally equivalent cells; random methods can sample a large fraction of it (our controls evaluate about 270, 1.7%). The all-skip collapse of DARTS is a related kind of degenerate cell to the roughly 10%-accuracy architecture we found on the true fronts of three devices (I have not checked that they are the same architecture).

### 3.1b NATS-Bench (Dong, Liu, Musial, Gabrys; arXiv 2009.00437; IEEE TPAMI 2021)
- **What it is.** An extension of NAS-Bench-201 to two search spaces: a **topology** space St (the same 15,625 cells; the paper's Table 1 lists 6.5k unique) and a new **size** space Ss (8 channel choices for each of 5 channel settings = 8^5 = 32,768 architectures). Each is trained on CIFAR-10, CIFAR-100 and ImageNet-16-120. Topology space: 12 epochs (H0) and 200 epochs (H1); size space: 12 epochs and 90 epochs (H2). Per-epoch loss and accuracy, parameters, FLOPs, **latency**, and training/evaluation times are released (author-stated). I did not read which hardware or method the latency refers to.
- **Why it matters here.** The LLM-NAS search space is NATS-Bench's topology space (NB201), so this is the benchmark paper for the space we use.
- **Findings useful to us (author-stated).**
  - Accuracy varies "even when the number of parameters or FLOPs stays the same" (Fig. 3), so cell connectivity and operators matter beyond size.
  - Rankings are fairly consistent across the three datasets for topology but quite different for size; Kendall rank correlation between datasets drops when only top architectures are compared, so "the best architecture on one dataset" does not transfer reliably.
  - **Seed noise:** ranking by one random trial versus the average of trials gives Kendall tau about 0.77, 0.70 and 0.77 on CIFAR-10, CIFAR-100 and ImageNet-16-120 for the top 20% of topologies.
  - 13 algorithms benchmarked (500 runs each for multi-trial methods). Topology space: on CIFAR-10 "most methods have similar performance"; on CIFAR-100 the order at the longest budget is REA ≥ BOHB ≥ REINFORCE ≥ RANDOM. **Size space:** REA "significantly outperforms" the others and RANDOM is worst on CIFAR-100 and ImageNet-16-120. The authors recommend REA when compute allows.
  - DARTS (first and second order) converge on CIFAR-10 to an architecture with many skip connections that "performs poorly", while doing "relatively well" on CIFAR-100 and ImageNet-16-120 (author explanation: more search data).
- **Pitfalls (author-stated or my reading, marked).** The authors say their re-implemented size-space methods may differ from the originals and that whether the observations generalise is "under investigation". *My reading:* the topology space has only 6.5k unique cells, so a method that samples a few thousand architectures covers a large share of it.
- **Tension with Part 2's random-search hook.** NATS-Bench finds random search *behind* REA and BOHB on CIFAR-100 and ImageNet-16-120, so "random search is a strong baseline" is not uniform in this benchmark. In our six-device runs the controls were scored on a different setup (zero-cost predictor, HW-NAS-Bench latency), so the two results do not contradict, but the hook should be worded with this in mind.

### 3.1c NAS-Bench-301 and surrogate benchmarks (Zela, Siems, Zimmer, Lukasik, Keuper, Hutter; arXiv 2008.09777; ICLR 2022)
- **Problem stated by the authors.** Tabular benchmarks need exhaustive evaluation, so they are limited to small spaces (6k to 423k architectures), while standard spaces exceed 10^18. They warn that this "can cause results gained on tabular NAS benchmarks to not generalize to realistic search spaces", citing local search, which looked good on tabular benchmarks but not in a realistic space (White et al. 2020).
- **Method.** Fit a regression model (a *surrogate*) on a sampled subset of architectures and query it like a table. Built for the DARTS space (>10^18 architectures, about 60k trained for the surrogate, CIFAR-10) and the FBNet space (9^22 ≈ 10^21, 25k random architectures, CIFAR-100).
- **Evidence (author-stated).**
  - On NAS-Bench-101, a GIN surrogate trained on one seed predicts the mean of two other seeds with mean absolute error 3.45e-3, against 4.53e-3 for a table lookup; the surrogate overtakes the table once trained on more than about 21,500 architectures.
  - Best surrogates on DARTS: LGBoost (R2 0.892, sparse Kendall tau 0.816), XGBoost (0.832, 0.817), GIN (0.832, 0.778). Runtime is predicted well (sKT 0.936).
  - Optimiser trajectories on the surrogate "closely resemble" those on the real benchmark, so runs that cost thousands of GPU hours can be simulated in seconds.
  - FBNet space: random search "cannot identify a good architecture even when evaluating 2500 architectures"; regularised evolution and HEBO are at least 8 and 16 times faster to find a better one, with a 1-2% absolute error gap over random search. The DARTS space shows the same pattern with a smaller gap.
  - Case study: a conclusion from a 11.8 GPU-day run (local search is poor on DARTS) was revised with the surrogate; with a longer budget local search becomes among the best.
- **Pitfalls (author-stated).** Algorithms may overfit the surrogate; the surrogate should be a black box and methods using the same model internally should not be benchmarked on it; versions must be stated; surrogates fitted on random data predict well-performing regions less well.
- **Why it matters here.** (1) The same recipe LLM-NAS uses, a gradient-boosted regressor as the estimate of accuracy, is the standard way to make large spaces cheap, so it is prior art. (2) *My reading:* random search looking strong may be a property of a small, tabular space like NB201; in a 10^18 to 10^21 space the benchmark paper finds it far behind. This is a limit on how far our "controls match the LLM" result can be extended and belongs in the discussion, not as a retraction. (3) HW-NAS-Bench is cited as one of the tabular benchmarks with this size limit.

### 3.2 Learning-curve extrapolation (Domhan, Springenberg, Hutter; IJCAI 2015)
- *Idea (abstract only).* Mimic a human who stops a bad run early: fit a probabilistic model to the first part of the learning curve and terminate runs predicted to be poor.
- *Result (author-stated):* roughly **twofold** speed-up of hyper-parameter optimisation methods.
- *Pitfall.* *My reading:* each query still trains the network for part of its schedule, so the cost per candidate is orders of magnitude above a zero-cost proxy.

### 3.3 Zero-cost proxies

**SynFlow (Tanaka, Kunin, Yamins, Ganguli; arXiv 2006.05467).** A *pruning* paper (abstract only read): a conservation law explains "layer collapse" in pruning at initialisation, and the **data-free** Iterative Synaptic Flow algorithm avoids it. Zero-cost NAS papers adopt its saliency score as a proxy.

**NASWOT, "Neural Architecture Search without Training" (Mellor, Turner, Storkey, Crowley; ICML 2021).**
- *Idea.* Score an **untrained** network by how distinct the activation patterns of different inputs are (overlap of binary activation codes), then pick the highest-scoring of N random networks.
- *Results on NAS-Bench-201 (CIFAR-10 test accuracy, Table 2):* NASWOT with N = 10 / 100 / 1,000 takes 3 / 30 / 306 seconds and gives 92.44 / 92.81 / 92.96%; **random search with 12,000 seconds of training gives 93.70%**, REA 93.92%, REINFORCE 93.85%; **picking one architecture at random gives 86.61 ± 13.46%**. The authors state that NASWOT outperforms all the weight-sharing methods; in their table this holds for most, but GDAS is higher on CIFAR-10 (93.51%) and CIFAR-100 (70.61%) than NASWOT with N = 1,000 (92.96%, 69.98%) and lower on ImageNet-16-120 (41.84% vs 44.44%). DARTS (both orders) and ENAS collapse to 54.30% on CIFAR-10.
- *Authors' caveat.* The non-weight-sharing methods "do outperform NASWOT"; NAS-Bench-201 hyper-parameters are fixed and may explain DARTS's failure.
- *Relevance.* A training-free score plus random sampling reaches within about 1 point of trained random search in seconds, which is the evaluator-only benefit.

**Zero-Cost Proxies for Lightweight NAS (Abdelfattah, Mehrotra, Dudziak, Lane; arXiv 2101.08134; ICLR 2021).**
- *Idea.* Adapt six pruning-at-initialisation metrics (grad_norm, snip, grasp, fisher, synflow, jacob_cov) to score a whole network from **one minibatch**.
- *Results (author-stated).* Spearman correlation with final validation accuracy on NAS-Bench-201 of **0.82 for the best proxy (synflow)** against 0.61 for the reduced-training proxy EcoNAS, at three orders of magnitude less compute; **only synflow was robust across all five benchmarks**. Used to "warm up" and to propose moves inside **random search, reinforcement learning, aging evolution and predictor-based search**, they gave sample-efficiency gains on all of them, up to **4× faster on NAS-Bench-101**.
- *Open question (author-stated):* why synflow works is unexplained.
- *Relevance (important).* The combination "zero-cost proxy plus classical search algorithm" **predates LLM-NAS** and includes random search and evolution, which is exactly what our controls do.

**NAS-Bench-Suite-Zero (Krishnakumar, White, Zela, Tu, Safari, Hutter; arXiv 2210.03230; NeurIPS 2022 Datasets and Benchmarks).**
- *What it is.* 13 zero-cost proxies pre-computed over **28 tasks**, in one codebase.
- *Findings (author-stated).* ZC proxies "capture substantial complementary information" (an information-theoretic analysis); they show **biases** such as a tendency to favour larger architectures or more convolutions, and a procedure to reduce the cell-size bias; feeding all 13 into the surrogate models of BANANAS and NPENAS improves the surrogate's Spearman rank correlation **by up to 42%**.
- *Critical context (author-stated in the introduction, citing other work).* "Simple baselines such as 'number of parameters' and 'FLOPS' are competitive with all existing ZC proxies across most settings", and "most ZC proxies do not generalize well across different benchmarks".
- *Relevance.* **LLM-NAS's quality estimator is this design**: an XGBoost model over the 13 proxies. The size-bias and parameter-count-baseline findings apply to it directly.

### 3.4 Comparing predictor families

**White, Zela, Ru, Liu, Hutter, "How Powerful are Performance Predictors in NAS?" (arXiv 2104.01177; NeurIPS 2021).**
- *Scope.* 31 predictors across learning-curve extrapolation, weight sharing, supervised (model-based) learning and zero-cost proxies, compared by initialisation time and query time.
- *Findings (author-stated).* **SynFlow and Jacobian covariance (about 3 seconds each) outperform all model-based predictors even after 30 hours of initialisation** on NAS-Bench-101/201; boosted-tree methods win when initialisation time is long; predictor families carry **complementary** information, and a simple combination ("OMNI") improves the best individual predictors.
- *Pitfalls.* *My reading:* results are on tabular cell benchmarks; they say nothing about hardware metrics.

---

## 4. Comparison

| Component | Method | Cost per candidate | Main advantage | Main pitfall |
|---|---|---|---|---|
| Hardware | FLOPs/params | negligible | trivial | poor rank correlation with latency (Kendall 0.18-0.36 on Edge TPU, Edge GPU) |
| Hardware | Layer-wise regression (NeuralPower) | milliseconds | per-layer breakdown | ignores fusion; GPU only |
| Hardware | Operator lookup table (FBNet, ChamNet; Part 1) | milliseconds | simple, differentiable | additive assumption; ignores fusion (nn-Meter) |
| Hardware | Kernel predictor (nn-Meter) | milliseconds | handles fusion | per-device profiling; 83.4% within ±10% on VPU |
| Hardware | End-to-end GCN (BRP-NAS) | milliseconds | fewer Pareto errors than layer-wise | needs measured data per device |
| Hardware | Measured on device (MnasNet, NetAdapt) | seconds-minutes, device in the loop | exact | does not scale to many devices |
| Hardware | Precomputed table (HW-NAS-Bench) | lookup | reproducible, no hardware needed | 2 of 6 devices simulated; one software stack; fixed spaces |
| Quality | Train to convergence | hours-days | exact | cost |
| Quality | Tabulated benchmark (NAS-Bench-101/201) | lookup | reproducible, cheap | small spaces; one training recipe |
| Quality | Tabulated benchmark with size space (NATS-Bench) | lookup | adds channel-size space, 3 datasets, 13 algorithms | topology space has 6.5k unique cells; seed noise (tau 0.70-0.77 on top 20%) |
| Quality | Surrogate benchmark (NAS-Bench-301) | milliseconds | works for spaces of 10^18-10^21; beats table lookup on noise | surrogate bias; risk of overfitting algorithms to it |
| Quality | Learning-curve extrapolation (Domhan) | partial training | about 2× faster | still trains |
| Quality | Learned predictor (BANANAS, BRP-NAS) | milliseconds after data | sample-efficient | needs trained architectures |
| Quality | Zero-cost proxy (SynFlow, NASWOT, etc.) | seconds, one minibatch | needs no training | size bias; not consistent across benchmarks |

---

## 5. What this part establishes for the "why LLMs" argument and for our reports (hooks, not conclusions)

1. **The expensive part of NAS has been removed by estimation, not by search.** Tables, predictors and proxies make each evaluation cost about a millisecond. LLM-NAS inherits this; it therefore explains why random and evolutionary controls finish in a second and the LLM operator costs dollars.
2. **Zero-cost proxy plus classical search is established prior art** (Abdelfattah et al.: random search, RL, aging evolution and predictor search all sped up; NASWOT; White et al.). This is the strongest literature support for treating the controls as the fair comparison.
3. **The estimator may carry much of the result.** Proxies are biased toward larger architectures and parameter count is competitive with them (NAS-Bench-Suite-Zero); predicted-vs-true latency can change which models are Pareto optimal (BRP-NAS). A natural, untested check for our work: compare the 13-proxy predictor with a parameter-count-only predictor. *This is a suggestion, not a finding.*
4. **Report wording to check:** Eyeriss and FPGA latencies are simulator estimates in HW-NAS-Bench; the Edge TPU data have the compile-failure −1 records and the weakest approximation; the benchmark uses one compiler/runtime per device.
5. **Why hardware-awareness is needed at all** has independent numerical support: FLOPs correlate weakly with latency on some devices (0.18 on Edge TPU), and the best architecture differs across devices.
6. **Search-space caveats** (NAS-Bench-201 has only 6,466 distinct topologies after isomorphism; DARTS collapses to an all-skip cell) fit the degenerate-cell finding already in the reports. NATS-Bench independently reports DARTS converging to a skip-heavy cell on CIFAR-10.
7. **Scope limit on our random-search result (added with NATS-Bench and NAS-Bench-301).** Random search is behind REA/BOHB on CIFAR-100 and ImageNet-16-120 in NATS-Bench, and far behind in the 10^18-10^21 spaces of NAS-Bench-301. Our result (LLM operators do not beat random-operator evolution, 0 of 24) holds for the small NB201 space with a cheap predictor; the reports should not suggest it applies to large spaces.
8. **The estimator recipe is standard.** A boosted-tree regressor as the quality estimate is how NAS-Bench-301 makes large spaces cheap, so LLM-NAS's XGBoost predictor is a recognised technique, not a novel one.

## 6. What is unverified or not covered
- Paleo and HELP are not covered; Paleo could not be downloaded.
- NATS-Bench: I did not read its appendices (hyper-parameter details, per-algorithm tables) or which device its latency field comes from; "6.5k unique" is from its Table 1. NAS-Bench-301: appendices were only searched; the numbers above come from the main text.
- NATS-Bench and NAS-Bench-301 citation counts: NATS-Bench 173 (OpenAlex, queried earlier); NAS-Bench-301 not queried.
- SynFlow and Domhan: abstract and opening only. NAS-Bench-Suite-Zero: abstract and introduction only; the bias and information-theory analyses are described from the authors' summary.
- BRP-NAS numeric prediction errors were not extracted; nn-Meter's per-model results beyond the headline percentages were not read.
- HW-NAS-Bench's appendix (per-device pipeline details) was not read; the Edge TPU compile-failure issue comes from the benchmark's GitHub issue \#11 used earlier in this project.
- Citation counts are OpenAlex values; HW-NAS-Bench has none.

## 7. Questions for your review
1. NATS-Bench and NAS-Bench-301 are now added (fourteen sources). Is HELP still wanted, or is this enough?
2. Hook 4 (Eyeriss and FPGA simulated): done, both reports corrected.
3. Hook 3 (parameter-count-only predictor): recommended as future work, awaiting your decision.
4. Hook 7 changes how far we can generalise the random-search result. Do you want that limit stated in the introduction as well as the discussion?
5. After this, shall I go to Part 4: the gap that LLM guidance claims to fill (LLM-NAS and the related LLM-NAS papers) and how it connects to Parts 1-3?

## Sources
- HW-NAS-Bench: https://arxiv.org/abs/2103.10584
- nn-Meter: https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf
- BRP-NAS: https://arxiv.org/abs/2007.08668
- NeuralPower: https://arxiv.org/abs/1710.05420
- NAS-Bench-101: https://arxiv.org/abs/1902.09635
- NAS-Bench-201: https://arxiv.org/abs/2001.00326
- NATS-Bench: https://arxiv.org/abs/2009.00437
- NAS-Bench-301: https://arxiv.org/abs/2008.09777
- Domhan et al.: https://www.ijcai.org/Proceedings/15/Papers/487.pdf
- SynFlow: https://arxiv.org/abs/2006.05467
- NASWOT: https://arxiv.org/abs/2006.04647
- Zero-Cost Proxies for Lightweight NAS: https://arxiv.org/abs/2101.08134
- NAS-Bench-Suite-Zero: https://arxiv.org/abs/2210.03230
- White et al.: https://arxiv.org/abs/2104.01177
- Citation counts: OpenAlex API, queried 2026-11-05
