# Background, Part 1: early HW-NAS work and the key methods (five most-cited, plus MobileNetV3 and ChamNet)

Status: DRAFT FOR REVIEW. Nothing here has been put in the LaTeX reports yet.
Every number below was read from the paper text itself (arXiv PDFs), unless marked *(computed)* or *(not verified)*.
"Author-stated" means the paper says it. "My reading" means I inferred it and you should challenge it.

---

## 0. How the selection was made (please check this first)

**Citation counts** come from OpenAlex (queried 2026-11-05). OpenAlex generally counts fewer citations than Google Scholar, so use the *ranking*, not the absolute values. Semantic Scholar was rate-limited. Counts for HyperPower and HW-NAS-Bench were not returned.

| Rank | Paper | Year | OpenAlex citations |
|---|---|---|---|
| 1 | MnasNet: Platform-Aware NAS for Mobile (Tan et al.) | 2018/19 | 3,458 |
| 2 | ProxylessNAS (Cai, Zhu, Han) | 2018/19 | 1,652 |
| 3 | FBNet (Wu et al.) | 2018/19 | 1,302 |
| 4 | Once-for-All (Cai et al.) | 2019/20 | 1,024 |
| 5 | NetAdapt (Yang et al.) | 2018 | 573 |
| — | ChamNet | 2019 | 272 |
| — | DPP-Net | 2018 | 193 |
| — | MONAS | 2018 | 174 |
| — | HW-NAS survey (Benmeziane et al.) | 2021 | 137 |

**Decisions made.**
1. *Added at your request:* Searching for MobileNetV3 (10,077 citations) and ChamNet (272). The set is therefore **seven papers**: the five above plus these two. MobileNetV3 would rank 2nd by citations; EfficientNet (18,422) is still excluded as a model paper rather than a HW-NAS method (say if you want it too).
2. *NetAdapt is not strictly NAS.* It prunes a pretrained network and does not search topologies. It stays because it is the clearest early statement of "measure the real device, not a proxy", and MobileNetV3 uses it.

---

## 1. Early work on hardware-aware network design

**Early work includes** Smithson et al. (2016), Hernández-Lobato et al. (2016), HyperPower (Stamoulis et al., 2017), Reagen et al. (2017), NetAdapt (Yang et al., 2018) and MnasNet (Tan et al., 2018). Which of them counts as hardware-aware network design depends on how strictly the term is read:

| Reading of "hardware-aware" | Early work | Date | Why |
|---|---|---|---|
| Hardware enters a **search over network structure**, even through an analytic cost model | **Smithson et al., ICCAD 2016** (arXiv 1611.02120) | submitted 7 Nov 2016 | Multi-objective search over layers, nodes/filters, kernel sizes (not just training settings), trading accuracy against a modelled implementation cost. The cost is **not measured on hardware**. |
| Hardware metrics **predicted from real GPU measurements** and used as a constraint | **HyperPower** (Stamoulis et al.), arXiv 1712.02446 | 6 Dec 2017 | Author-stated first to model power and memory of NNs on GPUs from measurements. Searches structure within a fixed skeleton (see below). |
| **Latency measured on the target device** inside the loop | **NetAdapt** (Apr 2018), then **MnasNet** (Jul 2018) | 2018 | Direct on-phone measurement of latency inside the search loop. |
| Hardware as the thing being **designed** (not a constraint on the network) | Hernández-Lobato et al., NIPS BayesOpt workshop 2016; Reagen et al., ISLPED 2017 | 2016-17 | Accelerator design-space exploration with Bayesian optimisation. **Both now read** (see 1.2). They tune accelerator parameters plus a few training settings of a small MNIST network; the network topology is essentially fixed. |

### 1.1 Settled: does HyperPower search architecture? (read in full)

**Partly. It searches structural hyper-parameters of a fixed AlexNet-style skeleton, not topology.**
- Search space (author-stated, Sec. 4): AlexNet variants for MNIST (6 hyper-parameters) and CIFAR-10 (13). Per convolution layer: number of features 20-80 and kernel size 2-5; pooling kernel 1-3; fully-connected units 200-700; plus learning rate (0.001-0.1), momentum (0.8-0.95) and weight decay (0.0001-0.01).
- So width and kernel size are searched, **depth, operator types and connectivity are not**. The authors call it hyper-parameter optimisation, not NAS, and I agree. It is the same *kind* of choice MnasNet and FBNet later search, in a much smaller space.
- Hardware: GTX 1070 and Tegra TX1. Power and memory are **linear models** `P(z) = Σ w_j z_j`, trained on offline random samples measured at inference (RMSPE under 7%). They enter the acquisition function as indicator terms (HW-IECI), so violating configurations are never proposed.
- Evaluation: every proposed network is **trained to completion** (with early termination of diverging runs). Runs were 2 h (MNIST) and 5 h (CIFAR-10) with 3-5 repeats.
- Headline claims (verified in Tables 3-5): up to 112.99× faster to reach the number of samples a constraint-unaware method queried; up to 57.20× more evaluations in the same time; up to 30.12× faster to reach its best error; up to 67.6% better accuracy.
- **Relevant to our thesis (my reading):** the paper includes a constraint-aware *random* baseline (Rand) next to the Bayesian optimisers. On CIFAR-10/GTX 1070 the best error was 24.39% (Rand) vs 21.81% (HW-IECI); on MNIST/GTX 1070 1.01% vs 0.81%. The big speed-ups are against **constraint-unaware** baselines, i.e. much of the benefit comes from the cheap hardware predictor, not the search algorithm. This is the same structure of argument as our matched-evaluation finding.
- **Limitations (author-stated):** only AlexNet-size networks on two small datasets; ImageNet left as future work; non-linear and probabilistic-constraint variants left as future work. **My reading:** two GPUs only; the linear model only captures structural parameters that affect cost additively.

### 1.2 The two Bayesian-optimisation accelerator papers (both read)

Both co-design a DNN accelerator with its network settings on **MNIST**, with **energy from the Aladdin simulator** (not real hardware) and error from training with DeepNet, using Spearmint. Neither searches network topology, so neither is HW-NAS; they matter as the origin of multi-objective, Pareto-based, budget-limited search where each evaluation is expensive, and both include non-Bayesian baselines.

**Hernández-Lobato, Gelbart, Reagen, Adolf, Hernandez-Lobato, Whatmough, Brooks, Wei, Adams, "Designing neural network hardware accelerators with decoupled objective evaluations", NIPS 2016 BayesOpt workshop (workshop paper).**
- *Idea (author-stated).* Energy does not depend on the trained weights, so the two objectives (energy, error) can be evaluated **separately**; choose at each step which objective to evaluate and where, using predictive-entropy search for multi-objective BO (PESMO). Error needs training; energy needs simulation.
- *Setup.* 3-hidden-layer MLP on MNIST; 8 design parameters (neurons per layer 50-250, learning rate, dropout, L2, memory bandwidth, loop parallelism, total bits, fraction integer bits); 600 evaluations; 20 repeats; space discretised to 10,000 Halton points.
- *Baselines.* Coupled BO, **NSGA-II** and **random search**.
- *Results (Table 2, hypervolume, mean ± s.e.).* Random 258.90 ± 0.55; NSGA-II 259.55 ± 0.57; coupled BO 262.73 ± 0.45; **decoupled BO 264.96 ± 0.31**. Average evaluation time 374,369 s (decoupled) vs 554,098 s (coupled). Decoupled BO evaluated the error objective more often than energy.
- *Pitfalls.* Author-stated: assumes both objectives cost the same time (they say this is easy to correct). *My reading:* tiny task (one MLP, MNIST); simulator energy; hypervolume is in arbitrary units and the best method is only about 2.3% above random (264.96 vs 258.90), so the ranking is consistent but the margins are small; decoupling relies on one objective being computable without training. The same holds for latency, which depends on structure and not on trained weights, so the idea transfers to HW-NAS (my reading).

**Reagen, Hernández-Lobato, Adolf, Gelbart, Whatmough, Wei, Brooks, "A Case for Efficient Accelerator Design Space Exploration via Bayesian Optimization", ISLPED 2017 (6 pages).** First page did not extract, so the abstract is taken from the Cambridge repository record; sections II-VI were read.
- *Idea.* Adapt multi-objective BO (predictive entropy search, discrete parameters handled as continuous then rounded, bits re-parameterised hierarchically) to **14 free parameters** of accelerator + DNN.
- *Baselines.* Stochastic grid search (SGS, i.e. uniform random sampling, Bergstra and Bengio) and a genetic algorithm (NSGA-II).
- *Results (author-stated).* Mean over sampled points: SGS 61.4% error at 14.2 µJ; GA 28.4% at 4.29 µJ; BO 14.4% at 3.95 µJ. At a 1 µJ budget the best error is 3.4% (GA), 2.7% (SGS), 2.25% (BO). For a 2.2% error target BO uses 58% less energy than GA and SGS cannot meet it. BO reaches a hypervolume of 0.12 after 32 samples, SGS needs 97 (3×).
- *Landscape.* Almost 60% of SGS samples do no better than random guessing (about 90% error); the authors call the landscape "peaky". GA actually beats BO in hypervolume between iterations 2 and 29 (author-stated), because GA starts with random sampling and got lucky.
- *Pitfalls.* Author-stated: MNIST only, chosen for ease of comparison; energy from a simulator, which they say does not detract from the conclusions. *My reading:* comparing **mean error of sampled points** favours adaptive methods, since BO concentrates its samples near good designs; the Pareto-front and hypervolume comparisons are the fairer ones.

**What both give the report:** they are early examples of (i) Pareto/hypervolume as the quality measure, which LLM-NAS also uses, (ii) comparing against NSGA-II and random baselines with the same evaluator, and (iii) an honest finding that random sampling is a weak baseline on a **peaky** landscape. That last point should be set against our finding that random sampling is a strong baseline on NAS-Bench-201, where most architectures are decent.

### 1.3 Pre-2017 antecedents (non-NAS; short survey, what I verified)

These show that "hardware-aware" model design predates NAS, and set up why NAS papers stress **direct** metrics.

| Work | Date | What it did | Verified how |
|---|---|---|---|
| Han, Pool, Tran, Dally, *Learning both weights and connections* | NIPS 2015 | Magnitude pruning to shrink networks for efficiency | Cited by HyperPower for per-operation energy figures; method described from its title only, paper not read |
| Rouhani, Mirhoseini, Koushanfar, *DeLight* | ISLPED 2016 | Uses energy characterisation to bound network size to available physical resources | Abstract text via search; not read |
| Yang, Chen, Sze, *Energy-aware pruning* | arXiv Nov 2016; CVPR 2017 | Prunes layers guided by an energy model built from hardware measurements; AlexNet energy reduced 3.7×, GoogLeNet 1.6×, under 1% top-5 loss | Abstract via search; not read |
| Gelbart, Snoek, Adams, *Bayesian optimization with unknown constraints* | arXiv 2014 | General constrained Bayesian optimisation, not hardware-specific | Cited by HyperPower; not read |

Pattern: these papers **optimise for an indirect or modelled cost** (weights, MACs, energy tables), which is exactly what NetAdapt, MnasNet and FBNet later criticise (e.g. NetAdapt's example of 19% fewer MACs but 29% longer latency).

### 1.4 Suggested wording for the report

"Early work on hardware-aware network design includes energy-aware pruning (Han et al., 2015; Yang et al., 2017), multi-objective search over network structure with an analytic cost model (Smithson et al., 2016), Bayesian optimisation of accelerator and network settings (Hernández-Lobato et al., 2016; Reagen et al., 2017), power- and memory-constrained hyper-parameter optimisation with predictors fitted to GPU measurements (Stamoulis et al., 2018), and search with latency measured on the target device (Yang et al., 2018; Tan et al., 2019)."

Citation years follow the published versions (HyperPower: DATE 2018, arXiv Dec 2017; NetAdapt: ECCV 2018; MnasNet: CVPR 2019, arXiv Jul 2018). The wording avoids "first" and "earliest" because the list is not exhaustive.

### 1.5 What is still unverified
- Pre-2016 antecedents beyond those listed; the survey is deliberately short and not systematic.
- This list is not exhaustive, so the report should say "early work includes", not "the first".

---

## 2. The seven papers

### 2.1 MnasNet (Tan et al., CVPR 2019; arXiv 1807.11626)

- **Problem.** Mobile CNNs must be small, fast and accurate; hand-balancing is hard. Earlier NAS used FLOPs as a latency proxy.
- **Idea.** Put **measured latency on real phones** into the NAS reward. Author-stated motivation: MobileNet and NASNet have similar FLOPs (575M vs 564M) but latencies of 113 ms vs 183 ms.
- **Reward.** `maximize ACC(m) × [LAT(m)/T]^w`, with `w = α` if `LAT ≤ T`, else `β`. They set `α = β = -0.07`, from the empirical observation that doubling latency buys about 5% relative accuracy. The alternative `α = 0, β = -1` is a hard constraint. This gives a soft constraint and an approximate Pareto front in one run.
- **Search space.** A *factorised hierarchical* space: the network is split into blocks and each block's ops, kernel, SE ratio, skip type, filters and layer count are searched separately, so layers can differ. Authors argue earlier cell-stacking spaces preclude layer diversity.
- **Search algorithm.** RL controller (the NASNet RNN) trained with PPO. Each sampled model is trained for 5 epochs on ImageNet and run on a Pixel 1 big CPU core.
- **Cost.** 4.5 days on 64 TPUv2 devices, about 8K models sampled, 15 transferred to full ImageNet *(author-stated)*. That is roughly 6,900 TPU-hours *(computed)*. Other papers (ProxylessNAS, FBNet, OFA) quote 40,000 GPU-hours for MnasNet; that is **their estimate, not MnasNet's own figure**.
- **Results.** 75.2% ImageNet top-1 at 78 ms on a Pixel: 1.8× faster than MobileNetV2 with +0.5% accuracy, 2.3× faster than NASNet with +1.2%.
- **Advantages.** Directly optimises the metric that matters; no proxy for latency; one search yields several trade-offs; the reward is simple and reusable.
- **Pitfalls.** *Author-stated:* the controller is deliberately the inefficient NASNet one. *My reading:* each candidate is trained (5 epochs on ImageNet), so cost is in the thousands of accelerator-hours; a **phone farm** is needed for latency measurement; the result is tied to one phone and one CPU core; the weights `α, β` are tuned by a heuristic. Authors say approximating latency was hard in early experiments, which is why they measure.
- **Relevance to LLM-NAS.** Establishes the accuracy-versus-measured-latency trade-off that later HW-NAS work, including LLM-NAS, inherits; the training-per-candidate cost it carries is what later methods try to remove.

### 2.2 ProxylessNAS (Cai, Zhu, Han, ICLR 2019; arXiv 1812.00332)

- **Problem.** Conventional NAS needs about 10^4 GPU-hours; differentiable NAS cuts GPU-hours but its memory grows linearly with the number of candidate ops. So both rely on **proxy tasks** (CIFAR-10, few blocks, few epochs) whose results are "not guaranteed to be optimal on the target task", especially for latency.
- **Idea.** Search **directly on the target task and target hardware**.
- **Method.** An over-parameterised net with all candidate paths; architecture parameters are **binarised** and only one path is active at a time (BinaryConnect-style), so memory is the same as training one compact net. Redundant paths are pruned at the end ("path-level pruning").
- **Making latency differentiable.** Expected latency of a mixed op `E[latency_i] = Σ_j p_j × F(o_j)`, where `F` is a **latency prediction model** (RMSE 0.75 ms against measured on the phone). Expected latency is added to the loss; a REINFORCE variant is the alternative.
- **Cost.** **200 GPU-hours** on ImageNet *(author-stated)*; the paper's Table 2 lists 40,000 for MnasNet and 48,000 for NASNet-A.
- **Results.** ImageNet 74.6% top-1 at 78 ms (Proxyless-R) on a Pixel 1, 0.6% above MnasNet at slightly lower latency, 200× fewer GPU-hours. GPU model: 75.1% top-1 at 5.1 ms on a V100 against 6.1 ms for MobileNetV2/MnasNet. CIFAR-10: 2.08% test error with 5.7M parameters.
- **Specialisation claim.** Author-stated "first work to study specialised architectures for different hardware" (GPU, CPU, mobile) and shows they differ.
- **Advantages.** Large-scale target task with no proxy; memory like standard training; works for several platforms; also gives design insights per device.
- **Pitfalls.** *My reading:* a large latency-prediction model must be built per device; the gradient route needs the latency model to be differentiable and additive across ops; a supernet's shared weights make ranking noisy (a general supernet issue, not tested here); the search is still repeated for each device.

### 2.3 FBNet (Wu et al., CVPR 2019; arXiv 1812.03443)

- **Problem.** NAS is expensive; optimal architectures depend on resolution and device ("non-transferable optimality"); FLOPs are inconsistent with latency (author example: NASNet-A has about MobileNetV1's FLOPs but is slower).
- **Idea.** **Differentiable NAS (DNAS)** with a layer-wise space and a **latency lookup table**.
- **Search space.** 22 searchable layers × 9 candidate blocks (kernel 3/5, expansion 1/3/6, group conv, plus skip) ≈ 9^22 ≈ 10^21 architectures.
- **Loss.** `L(a, w_a) = CE(a, w_a) · α · log(LAT(a))^β`, with `LAT(a) = Σ_l LAT(b_l)` the sum of per-operator latencies from a lookup table measured on the target device. Architecture distribution trained by SGD using Gumbel-Softmax on a stochastic super net.
- **Cost.** 216 GPU-hours for FBNet-B *(author-stated)*; the authors "estimate" this as about 420× less than MnasNet (an estimate, not a measurement).
- **Results.** FBNet-B: 74.1% top-1, 295M FLOPs, 23.1 ms on a Samsung S8: 2.4× smaller and 1.5× faster than MobileNetV2-1.3 at similar accuracy. Smallest: 50.2% at 2.9 ms. A model searched for iPhone X was 1.4× faster there than a Samsung-searched one.
- **Advantages.** Cheap enough to redesign per device and per resolution; lookup tables need only benchmarking a few hundred operators; differentiable latency term.
- **Pitfalls (author-stated).** The additivity assumption: "the runtime of each operator is independent of other operators", valid "for many mobile CPUs and DSPs" where operators run sequentially. *My reading:* this is the assumption that breaks on accelerators with pipelining, fusion or memory contention (e.g. GPUs, FPGAs), which matters directly for HW-NAS-Bench devices; proxy-dataset search was still used; the filter counts per layer are hand-picked.
- **Relevance to LLM-NAS.** Lookup-table latency is the idea closest to a benchmark's per-architecture latency table. I have **not** verified how HW-NAS-Bench builds its latencies, so do not state a link in the report until that is checked.

### 2.4 Once-for-All, OFA (Cai et al., ICLR 2020; arXiv 1908.09791)

- **Problem.** NAS-per-device costs grow **linearly** in the number of deployment scenarios, O(N), with large energy and CO2 cost.
- **Idea.** **Decouple training from search**: train one big network once, then pick specialised sub-networks without retraining. Cost goes from O(N) to O(1) in training.
- **Method.** Progressive shrinking (PS): train the largest net, then fine-tune to support smaller depth, width, kernel size and resolution, sharing weights and distilling. More than 10^19 sub-networks. For each device: sample sub-networks to train an **accuracy predictor** and a **latency lookup table**, then run **evolutionary search** guided by the predictors (cost described as negligible).
- **Cost.** About 1,200 V100 GPU-hours one-time, plus 40 GPU-hours to collect predictor data. Optional per-scenario fine-tuning adds "25N" or "75N" GPU-hours in the paper's table (rows "#25" and "#75"); I did not read what those variants mean in detail, so treat that part as unchecked. Paper's table lists per-N costs for MnasNet (40,000 N), ProxylessNAS (200 N + 300 N), FBNet-C (216 N + 360 N). Total CO2: 16× fewer than ProxylessNAS, 19× fewer than FBNet, up to 1,300× fewer than MnasNet for N = 40.
- **Results.** 80.0% ImageNet top-1 under the mobile setting (<600M MACs; 595M MACs). Up to +4.0% over MobileNetV3, or 1.5× faster at equal accuracy; 2.6× faster than EfficientNet. Winner of 3rd and 4th LPCVC.
- **Advantages.** Amortised cost across many devices; very large space; predictor + evolution is the same pattern LLM-NAS uses (predictor + search), so it is the closest classical analogue.
- **Pitfalls.** *My reading:* high up-front cost (about 1,200 GPU-hours) that only pays off when N is large; quality of every sub-network depends on the shared-weight training working (which is why PS was needed: plain sampling "leads to significant accuracy drops", author-stated); predictor errors propagate; the sub-network space is a fixed MobileNetV3-style family, so it cannot discover new topologies.

### 2.5 NetAdapt (Yang et al., ECCV 2018; arXiv 1804.03230)

- **Problem.** Network-simplification methods target indirect metrics (MACs, weights) that need not reduce latency/energy. Author evidence: a network with 19% fewer MACs had 29% longer latency in practice.
- **Idea.** Put **empirical measurements of direct metrics** (latency, energy) from the actual device inside the optimisation loop, so no platform-specific model or knowledge is needed.
- **Method.** Starting from a pretrained network, repeatedly (i) impose a resource reduction ΔR, (ii) generate one proposal per layer by pruning filters until the budget is met, (iii) short-term fine-tune each, (iv) keep the most accurate; stop at the budget, then long-term fine-tune. Constraint: `Res_j(Net) ≤ Bud_j`.
- **Results.** Up to 1.7× faster measured latency than the MobileNet multipliers on a Pixel 1 CPU at equal or higher accuracy; 1.6× faster and +0.3% accuracy against MorphNet; also evaluated on a Samsung Galaxy S8 mobile GPU and against ADC.
- **Advantages.** Platform-agnostic (any device you can time); produces a family of networks along the way; interpretable (you can see why a proposal won); complementary to hand-designed efficient nets.
- **Pitfalls.** *My reading:* it only shrinks an existing network, so it cannot find new topologies and inherits the starting design; each iteration evaluates many proposals with fine-tuning, and on-device measurement means a device in the loop; the paper reports no GPU-hours figure in the parts I read, so I cannot give a cost comparison.
- **Relevance.** Origin of the "measure, don't estimate" principle that MnasNet, MobileNetV3 and later work use.

---

### 2.6 Searching for MobileNetV3 (Howard et al., ICCV 2019; arXiv 1905.02244)

- **Problem.** Next-generation mobile backbones for CPUs, combining search with hand design.
- **Idea (author-stated).** Use two complementary searches: **platform-aware NAS** (the MnasNet recipe) for global block structure, then **NetAdapt** for layer-wise filter counts; then add manual improvements.
- **How hardware enters.** Latency on a Pixel phone in the MnasNet reward `ACC(m) × [LAT(m)/TAR]^w`. For the Large model they **reuse MnasNet-A1** as the seed (same RNN controller and same factorised hierarchical search space, so no new search). For Small models they re-ran the search with a **smaller weight, w = -0.15 instead of -0.07**, because accuracy changes more steeply with latency for small models.
- **NetAdapt modification.** Proposals are chosen by maximising `ΔAcc/|ΔLatency|` (the trade-off slope) instead of minimising accuracy change; proposals reduce an expansion layer or bottlenecks; fine-tune T = 10,000 steps; step size δ = 0.01·L; the final network is **re-trained from scratch**. The authors found longer fine-tuning raised the proposals' accuracy but did not change the final accuracy.
- **Manual changes (author-stated "outside the scope of the current search space").** Redesigned the expensive first and last layers (moving the final 1×1 expansion past average pooling), and introduced the h-swish nonlinearity and a new segmentation head (LR-ASPP).
- **Results.** Large: +3.2% ImageNet top-1 at 20% lower latency than MobileNetV2; Small: +6.6% accuracy at comparable latency; Large detection over 25% faster at about equal accuracy on COCO; LR-ASPP 34% faster on Cityscapes.
- **Search cost.** The text I extracted gives no GPU/TPU-hour figure for the search.
- **Advantages.** Combines coarse global search and fine local adaptation; shows search and human design are complementary; strong, widely used models.
- **Pitfalls.** *Author-stated:* "it remains an open question of how best to blend automatic search techniques with human intuition". *My reading:* a large share of the final gain comes from changes outside the search space (layer redesign, h-swish), so the search alone did not produce the final model; the Large model's search was inherited from MnasNet, so it is not an independent test of NAS; the reward weight needed manual re-tuning for a different latency regime.
- **Relevance.** Evidence that even the headline NAS-derived model needed human insight, which is the gap an LLM prior claims to fill; it is also an example of the reward weight being a hand-set hyper-parameter.

### 2.7 ChamNet (Dai et al., CVPR 2019; arXiv 1812.08934, 21 Dec 2018)

- **Problem.** Platform-aware NAS and sequential model-based optimisation must be rerun for every network, platform and constraint, so the cost is O(m·n·k) for m networks, n platforms and k constraints. Author example: latency-driven MnasNet needs "hundreds of GPU hours".
- **Idea.** "Chameleon": adapt an existing efficient network (existing building blocks, no new cells, no RL) to a target platform and budget by searching its hyper-parameters (filter counts, bottleneck counts), guided by **predictors**.
- **Method.** Efficient evolutionary search: an adaptive genetic algorithm over a hyper-parameter vector `x`, fitness `R = A(x) - [α·H(F(x, plat) - thres)]^w` (accuracy minus a Heaviside penalty on violating the resource threshold). Predictors: an **accuracy predictor** (Gaussian process, with Bayesian optimisation and quasi-Monte-Carlo sampling choosing which networks to train), an **energy predictor** (also GP), and a **latency lookup table of operators** (about 350K records for a device, built once). Predictions take under one CPU second.
- **Settings.** Initial pool k = 2,048 architectures; predictors built from 240 selected samples; population 96, top 12 kept, 100 iterations; α = 10/ms or 10/mJ, w = 2.
- **Cost claim (author-stated).** Training cost O(m + n) instead of O(m·n·k): m accuracy predictors and n resource predictors; "search cost is negligible once the predictors are built", about minutes per new platform/constraint.
- **Results.** 73.8% (CPU) and 75.3% (DSP) top-1 at 20 ms on ImageNet. Against MnasNet on a Samsung S8 (Snapdragon 835): +1.7% top-1 and 1.75× speed-up. Platforms: Snapdragon 835 CPU, Hexagon v62 DSP, Intel Xeon CPU, Nvidia GPU. Energy-constrained: 60.0% at 14 mJ vs 58.8% at 19 mJ (a 26% energy cut and +1.2% accuracy).
- **Advantages.** Cheap per new device or constraint once predictors exist; handles latency and energy; shows that distribution of computation across layers matters; simple evolutionary search suffices.
- **Pitfalls.** *Author-stated:* a one-time cost to build predictors per network and per platform; it adapts a given base network family rather than discovering new blocks. *My reading:* the accuracy predictor still needs trained networks (240 samples here) and its accuracy limits what the search can find; results are on ImageNet-scale mobile models and hand-picked platforms; no comparison with a non-learned (random) search over the same predictors is reported in the sections I read.
- **Relevance (important for your thesis).** ChamNet is the closest classical analogue to LLM-NAS: **accuracy predictor + latency lookup table + a population search with a hardware-aware fitness**, producing a good architecture in minutes (Dec 2018). The LLM-NAS paper's cost saving also comes from predictor-based evaluation; the difference is that its search operator is an LLM rather than a genetic algorithm. This supports the position in the justification section that the saving comes from the evaluator, not from the LLM.

## 3. Cross-cutting picture so far (for you to confirm before Part 2)

| Method | Search strategy | Candidate evaluation | How hardware enters | Reported search cost |
|---|---|---|---|---|
| MnasNet | RL (PPO) | Train 5 epochs on ImageNet | Measured latency on Pixel in the reward | 4.5 days on 64 TPUv2 (~8K models) |
| ProxylessNAS | Gradient, binarised paths | Shared supernet weights | Differentiable expected latency from a predictor | 200 GPU-hours |
| FBNet | Gradient (Gumbel-Softmax) | Shared supernet weights | Additive lookup-table latency in loss | 216 GPU-hours |
| OFA | Evolution over a trained supernet | Accuracy predictor | Latency lookup table | ~1,200 GPU-hours once + ~25/model |
| NetAdapt | Greedy iterative pruning | Short fine-tune | Measured on device per proposal | Not stated in sections read |
| MobileNetV3 | MnasNet-style RL search + NetAdapt + manual redesign | Training (search inherited from MnasNet for Large) | Measured Pixel latency in reward; NetAdapt slope rule | Not stated in sections read |
| ChamNet | Evolutionary (GA) | GP accuracy predictor, GP energy predictor | Latency lookup table (~350K ops) and energy predictor | One-time predictors; minutes per new target (author-stated) |

Themes visible already (these are the hooks for the "why LLMs" argument in a later part, **not conclusions yet**):
1. Cost fell from thousands of GPU-hours (MnasNet) to hundreds (ProxylessNAS, FBNet) mainly by **removing per-candidate training** (weight sharing, predictors), which is the same move LLM-NAS makes with its zero-cost predictor.
2. Latency modelling moved from **measured** (MnasNet, NetAdapt) to **predicted/lookup** (ProxylessNAS, FBNet, OFA) because device farms do not scale; the additivity assumption is the weak point.
3. Every method here targets **one family of devices (mobile CPU/GPU/DSP)** and a **fixed backbone family** (MobileNetV2/V3 blocks).
4. **ChamNet shows the full recipe LLM-NAS uses (predictors + latency table + population search) in 2018 without any LLM.** The question for the later "why LLMs" part is therefore what the LLM adds over the genetic algorithm in that recipe.
5. **MobileNetV3 shows the limits of search alone:** its authors improved the searched network by hand and call the blend of search and human intuition an open question.
6. **The accelerator papers show a baseline lesson:** random search lost to Bayesian optimisation on the peaky accelerator landscape (most random designs were useless). On NAS-Bench-201 most architectures are decent, which is a possible, untested reason why our random controls were strong.

## 4. Questions for your review
1. Is "Early work includes Smithson et al. 2016 (structure search, modelled cost), HyperPower 2017 (predictors from GPU measurements) and NetAdapt and MnasNet 2018 (measured latency)" the right way to open the introduction?
2. Do the seven summaries (MnasNet, ProxylessNAS, FBNet, Once-for-All, NetAdapt, MobileNetV3, ChamNet) capture what you want from each? In particular, should each be cut to a 3-4 line entry for the introduction, with the full notes kept here?
3. The ChamNet point (predictor + latency table + population search without an LLM) and the MobileNetV3 point (search plus human insight) are my proposed hooks for the "why LLMs" argument. Do you want them in the introduction, or kept for the later part?
4. Plan for the next parts: (2) classical search strategies (RL / evolution / differentiable / one-shot / predictor-guided BO) with pros and cons, (3) hardware cost models (measured / lookup / predictor / benchmark tables), (4) the gap that LLM guidance claims to fill.

## Sources
- MobileNetV3 (read): https://arxiv.org/abs/1905.02244
- ChamNet (read): https://arxiv.org/abs/1812.08934
- Hernández-Lobato et al. 2016 (read in full): https://bayesopt.github.io/papers/2016/Lobato.pdf
- Reagen et al. 2017 (read from page 2 on; abstract from the repository record): https://doi.org/10.17863/CAM.18915
- MnasNet: https://arxiv.org/abs/1807.11626
- ProxylessNAS: https://arxiv.org/abs/1812.00332
- FBNet: https://arxiv.org/abs/1812.03443
- Once-for-All: https://arxiv.org/abs/1908.09791
- NetAdapt: https://arxiv.org/abs/1804.03230
- HyperPower (read in full): https://arxiv.org/abs/1712.02446
- Smithson et al., ICCAD 2016: https://arxiv.org/abs/1611.02120
- Yang, Chen, Sze, energy-aware pruning (abstract only): https://arxiv.org/abs/1611.05128
- DeLight (abstract only): https://aceslab.org/mw/images/4/47/Rouhani2016delight.pdf
- Hernández-Lobato et al. 2016 (abstract only): https://vlsiarch.eecs.harvard.edu/publications/designing-neural-network-hardware-accelerators-decoupled-objective-evaluations
- Benmeziane et al., HW-NAS survey: https://ijcai.org/proceedings/2021/0592.pdf
- FNAS: https://arxiv.org/pdf/1901.11211
- Citation counts: OpenAlex API, queried 2026-11-05
