# Reproduction notes

Paper: *LLM-NAS: LLM-Driven Hardware-Aware Neural Architecture Search*, arXiv:2510.01472v4 (4 Dec 2025).
No official code was available when this repository was written, so the implementation follows the
paper text, Algorithm 1 (App. A) and the prompts in Appendix E.

## 1. Paper item → command

| Paper | Command | Output (`results/reports/…`) | Status |
|---|---|---|---|
| Fig. 1 (Edge GPU) / Fig. 9 (6 devices) | `run fig1_bias` | `fig1_bias/bias_<device>.png` | reproduced |
| Fig. 3 | `fig3` | `fig3_params.png` | reproduced (deterministic) |
| Table 1 | `check-data` (niche sizes) | console | reproduced (deterministic) |
| Table 2 | `run table3_main` → `report table3_main` | `table3_main/report.md` | reproduced (LLM-NAS rows); baseline rows quoted |
| Table 3 | `run table3_main` → `report table3_main` | `table3_main/report.md`, `fronts_*.png` | LLM-NAS reproduced; LLMatic/FairNAS/PRP-NAS quoted |
| Table 4 | any `report` | "Table 4" section | LLM-NAS cost measured; baselines quoted |
| Table 5, Figs. 6–8 | `run table5_ablation` → `report table5_ablation` | `table5_ablation/report.md`, `fronts_*.png` | reproduced |
| Fig. 4 / App. E prompts | `src/llmnas/prompts.py` | every prompt is saved in `llm_calls.jsonl` | implemented |
| Table 6, Fig. 5 (ViT) | – | – | **not reproduced** (see §5) |

**Why the baselines are quoted.** LLMatic trains every candidate (17 GPU-days per dataset and device), and FairNAS, PRP-NAS
and DARTS need supernet training (2–10 GPU-days each). Their numbers come from `paper_results.py`.
Two cheap extra baselines run alongside the ablations as floors: `random_sampling` and `no_knowledge_base`.

## 2. What the paper specifies (used as stated)

| Item | Value |
|---|---|
| Benchmark | HW-NAS-Bench (NB201 cell space, 15,625 archs), 6 devices, CIFAR-10/100, ImageNet16-120 |
| Niches | Table 1 (by number of `nor_conv_3x3`; S0/S1 split by `nor_conv_1x1`) |
| LLM | GPT-4.1 |
| Generations | 10 |
| Crossover probability | 0.5 |
| Predictor | XGBoost on the 13 NAS-Bench-Suite-Zero proxies |
| Latency | HW-NAS-Bench lookup table |
| Budget | 120 API calls per search = 6 niches × 10 generations × 2 stages (KB update + generation) |
| Metrics | HV and IGD of the discovered front against the true front |

## 3. What the paper leaves open, and the choice made here

All of these are in `configs/base.yaml` and tagged `[assumed]`. Report them in your thesis, and run a
sensitivity check on the ones that matter (HV normalisation, `n_child`, predictor training size).

| Item | Choice | Rationale |
|---|---|---|
| Initial population per niche | 5 random archs | Algorithm 1 samples one but gives no size |
| Children per Stage-2 call | 4 | gives 240 LLM proposals over 120 calls |
| Parents shown in a prompt | ≤ 6 archive members (half newest, rest random) | Fig. 4 shows "selected Pareto from last cycle" plus "random selected from Pareto archive" |
| How p_c is applied | each child slot is assigned crossover with prob. 0.5 (if ≥ 2 parents), and the prompt states the operation per child | Appendix E asks the LLM to decide, Sec. 4.1 fixes p_c; this satisfies both |
| Knowledge base scope | one KB per niche | the Stage-1 prompt contains the niche definition |
| Stage 1 in generation 1 | runs on the initial population (no rationales yet) | keeps exactly 120 calls |
| Invalid / duplicate / out-of-niche proposals | discarded, no retry, counted in `proposal_stats` | keeps the budget fixed |
| `{latency_limit}` in the prompt | omitted (`null`); `niche_max` or a number available | the paper never gives a value |
| Temperature | 0.7 | not stated |
| Predictor training set | 1,000 random archs, labels = NB201 validation accuracy | "trained on a sample"; validation labels avoid touching test data |
| Accuracy used for HV/IGD and plots | NB201 200-epoch **test** accuracy (mean over seeds) | Table 2 reports 94.37 %, the CIFAR-10 test optimum |
| Discovered set S | final front of Algorithm 1 (selected by *predicted* accuracy), re-scored with true accuracy | Algorithm 1 returns P_final; set `evaluation.front_source: all` to score every evaluated arch |
| HV / IGD normalisation | min-max by the true front's ideal/nadir points, reference point (1.1, 1.1), HV reported as HV(S)/HV(P\*) | paper reports HV ≤ 1 ("near-perfect"), implying a ratio; not stated |
| "Without partitioning" budget | one niche covering the whole space with 30 initial samples and 24 children per call, so the same number of candidates is evaluated (20 calls instead of 120) | `search.match_budget` |
| Synflow ablation | Synflow mapped to the accuracy scale with an isotonic fit (ranking = Synflow's) | the LLM is shown "accuracy" in every variant |
| Seeds | 3 per setting | paper reports ± but not the number of runs |

## 4. Data-alignment checks (`python -m llmnas check-data`)

These confirm that the local tables are the ones the paper used:

* best test accuracy CIFAR-10 / CIFAR-100 / ImageNet16-120 = 94.37 / 73.51 / 47.31;
* each LLM-NAS row of Table 2 (accuracy, latency) is looked up in the benchmark. A hit means the
  paper's numbers come from the same HW-NAS-Bench and NB201 tables;
* the Figure 2 example rows are matched to their closest (dataset, device);
* the ZC-ensemble predictor's Spearman correlation is compared with the paper's ≈ 0.90.

Results (full output in `results/reports/check_data.txt`):

* Best accuracies match exactly (94.37 / 73.51 / 47.31).
* 11 of the 12 Table-2 LLM-NAS points exist in the benchmark with matching accuracy *and* latency.
  The exception is **Edge GPU 94.37 % @ 4.35 ms**: in HW-NAS-Bench, the 94.37 % architecture
  `(3, 3, 3, 1, 3, 2)` runs at 6.81 ms on Edge GPU, and no architecture has that (accuracy, latency) pair.
  The same row is printed for PRP-NAS-BA, so it may have been copied from there.
* All five Figure-2 example rows match CIFAR-100 / Edge GPU to within rounding. This also confirms that
  the paper's 6-tuple encoding is the NB201 arch-string order with ops `0 none, 1 skip, 2 conv1x1,
  3 conv3x3, 4 avg_pool`, as implemented in `search_space.py`.
* The ZC-ensemble predictor reaches a Spearman of 0.96 on all three datasets with 1,000 training
  architectures, above the paper's ≈ 0.90. Synflow alone reaches 0.75–0.78. The paper's predictor setup
  (training size, labels) is unknown. To mimic a weaker predictor, lower `predictor.n_train`.

**Failed Edge TPU compilations.** HW-NAS-Bench stores `-1` as the Edge TPU latency of 4,699 architectures (30 %;
the same architectures for all three datasets; the other devices have none). The benchmark's maintainers state that
`-1` marks a model that did not pass the compiler
([GATECH-EIC/HW-NAS-Bench#11](https://github.com/GATECH-EIC/HW-NAS-Bench/issues/11), 12 Jan 2024; the reply says
"Edge GPU compiler" but links the Coral Edge TPU compiler docs). Failures concentrate in architectures with
`none` edges (0.4 % with no `none` edge, 26 % with one, 100 % with all six) but also include 18 with no `none` edge, so the
cause is undocumented. They are treated as infeasible on that device: excluded from Edge TPU's search space and
true front, and discarded (`proposal_stats.no_latency`) if proposed. Treating `-1` as a real latency would put fake
0-cost points on the true front and make Edge TPU HV meaningless. The paper does not say how it handled this.
The full list is in `results/edgetpu_missing_latency_architectures.csv`.

The NASLib → NB201 encoding (op order `[skip, none, conv3, conv1, pool]`, edge permutation `[0,1,3,2,4,5]`)
was verified on all 15,625 architectures: validation accuracies match exactly.

## 5. Not reproduced: ViT search (Sec. 4.4, Table 6, Fig. 5)

This needs an AutoFormer supernet, the ViT-Bench-101 Auto-Proxy predictor, and latency profiling on an
NVIDIA A6000. None of these is a lookup table. The niche idea carries over directly: partition by
`(embed_dim, depth)` bins (App. C), then reuse `search.py` with a new search space and latency function.
That is a natural extension for the thesis, but it is outside a benchmark-based reproduction.

## 6. Inconsistencies noticed in the paper

* Table 2's Edge GPU top row (94.37 % @ 4.35 ms) does not exist in HW-NAS-Bench (see §4).
* Table 5 reports a CIFAR-100 average HV of 0.978 for the full model, but the mean of the six Table 3
  CIFAR-100 values is 0.982. The average IGD (0.0246) does match.
* In Table 2, PRP-NAS-BA's Edge GPU row (94.37 %, 4.35 ms) is identical to LLM-NAS's top row. Both are
  presumably the NB201 CIFAR-10 optimum.
* "3 mins (120 API calls)" is per dataset and device. A 3 × 6 × 3-seed grid is about 6,500 calls.

## 7. Recommended protocol for thesis-grade numbers

1. Pin the code: tag a commit (`git tag repro-v1`) before running.
2. Run the grids with `llm.mode=record` (the default).
3. Archive `results/runs/` (each run holds `config.yaml`, `run.json` and the full `llm_calls.jsonl`), for
   example in a Zenodo or Drive bundle. These files are the evidence behind every number in the thesis.
4. Regenerate the tables at any time with `report`. To re-execute a search bit-for-bit without API access,
   use `--set llm.mode=replay`. A replay run writes `run_replay.json` next to the original, and
   `replay_mismatches` must be 0.
5. Report mean ± std over seeds, and state the `[assumed]` settings from §3.
