# Blend: LLM-NAS × LiteTransformerSearch × SLM survey

**One-sentence idea.** Use LLM-NAS's search engine (complexity niches plus LLM co-evolution of
prompts and architectures) to find hardware-efficient **small language models**. Candidates are
scored with LiteTransformerSearch's training-free quality proxy, and on-device cost is modelled the
way the SLM survey breaks it down: prefill, decode and memory.

| Paper | Reference | Used for |
|---|---|---|
| LLM-NAS | Zhu, Zhang & Huang, arXiv:2510.01472 | search engine: niches, Stage-1 knowledge base, Stage-2 generation, ablations |
| LTS | Javaheripi et al., *LiteTransformerSearch*, NeurIPS 2022, arXiv:2203.02094 | transformer search space, decoder-parameter proxy, 3-objective front, on-device measurement, baseline search |
| SLM survey | Lu et al., *Small Language Models: Survey, Measurements, and Insights*, arXiv:2409.15790 | modern SLM design space, cost structure, device settings, reference models, prior knowledge |

## 1. Why these three fit together

* **LTS (Sec. 4.3, Fig. 7):** at a fixed decoder-parameter count, perplexity barely changes with
  shape (< 7–13 % spread), but latency varies up to 2.8× and memory up to 5.5×. Quality is "free"
  given the size, and the real search problem is finding the cheapest *shape* for a device.
* **SLM survey (Sec. 4):** explains *why* shape matters. Prefill is compute-bound and favours wide,
  shallow models. Decode is memory-bound and roughly linear in parameter bytes. Grouped-query
  attention shrinks the KV cache. Vocabulary size inflates the compute buffer. Each layer adds
  fixed overhead.
* **LLM-NAS:** an LLM that reads results and writes design rules is well suited to a problem whose
  structure is known qualitatively. Its niches stop the LLM collapsing onto familiar sizes. The SLM
  survey even shows the same clustering LLM-NAS used to justify its niches: on-device latency falls
  into 0.1–1B / 1–2B / 2–3B bands (survey Sec. 4.1.1), just as CNN parameters cluster by the number
  of 3×3 convolutions (LLM-NAS Fig. 3).

## 2. Method (what the code does)

| Component | Implementation | From |
|---|---|---|
| Search space `slm` | homogeneous decoder: `n_layer` 8–40, `d_model` 512–3072, gated FFN with ratio 2–8, `n_head` with head_dim ∈ {64, 128}, `n_kv_head` ∈ {1, 2, 4, 8, MHA}; vocab 32K, tied embeddings | survey Table 1, Fig. 2 |
| Search space `lts_gpt2` | heterogeneous per-layer GPT-2 backbone: `n_layer` 2–16, `d_model` 128–1024, `d_inner` 256–4096 (≥ 2·d_model), `n_head` ∈ {2, 4, 8}, vocab 50257 (adaptive embeddings omitted) | LTS Sec. 3 |
| Niches | disjoint bands of decoder parameters, e.g. `<250M, 250–500M, 0.5–1B, 1–2B, 2–3B, ≥3B` | LLM-NAS Table 1 idea, survey's latency bands |
| Quality objective | decoder (non-embedding) parameter count. It is shown to the LLM as the Kaplan scaling-law loss (N_c/N)^0.076, which is monotone, so the ranking is exactly LTS's proxy | LTS Sec. 3.1 |
| Latency objective | end-to-end time for a 50-token prompt plus 50 generated tokens = prefill + 50 × decode | survey Sec. 4 setup |
| Memory objective | quantised weights + KV cache (2,048 context) + compute buffer (512 × vocab × 4 B) | survey Sec. 4.1.2, 4.3.2 |
| Hardware model | roofline: prefill = max(FLOPs/throughput, weight bytes/bandwidth) + per-layer overhead; decode = bytes/bandwidth + per-layer overhead | survey Sec. 4 insights |
| Device profiles | Jetson Orin NX 16 GB and Snapdragon 8 Gen 1 CPU (the survey's devices), your RTX 3050 Ti laptop GPU, your CPU | survey Table 3 |
| Measurement | random-weight PyTorch model, timed prefill and KV-cached decode, peak memory, cached per device | LTS Sec. 4.4 |
| Calibration | least-squares fit of the latency model to measurements (`slm-calibrate`) | new |
| LLM operator | the Appendix-E two-stage prompts, with transformer JSON in place of 6-tuples | LLM-NAS |
| Prior knowledge | `kb_seed: survey` starts the knowledge base from the survey's "Insights" boxes and LTS's shape finding | survey and LTS → LLM-NAS |
| Reference models | 11 real SLMs (SmolLM, Qwen2.5, Llama-3.2, TinyLlama, Pythia, Phi-2) scored with the same model | survey Table 1 |

### Variants (`configs/slm/experiments/slm_main.yaml`)

| Variant | What it isolates |
|---|---|
| `llmnas` | LLM-NAS transferred unchanged (empty knowledge base) |
| `llmnas_survey_kb` | adds the survey's insights as prior knowledge (the blend's contribution) |
| `no_partitioning` | LLM without niches: does exploration bias exist in transformer spaces? |
| `lts_search` | **LTS's own algorithm**: evolution over the global Pareto front, mutation p = 0.3, plus crossover, same budget |
| `no_llm_pea` | niches + random evolution (LLM-NAS's PEA ablation) |
| `random_sampling` | floor |

All variants get the same evaluation budget: 30 initial samples plus 240 proposals.

## 3. Research questions

1. **RQ1:** At an equal evaluation budget, does LLM co-evolution find a better front than LTS's
   evolutionary search (`llmnas*` vs `lts_search`)?
2. **RQ2:** Does seeding the knowledge base with published SLM insights help (`llmnas_survey_kb` vs `llmnas`)?
3. **RQ3:** Does the LLM exploration bias, and its fix by partitioning, carry over from CNN cells to
   transformers (`no_partitioning`)?
4. **RQ4:** At equal decoder parameters (equal proxy quality), how much faster or smaller are the
   searched shapes than hand-designed SLMs? (LTS made the same comparison against OPT-350M.)
   The report's "Hand-designed SLMs vs …" table answers this.
5. **RQ5:** How faithful is a calibrated analytical cost model to measured latency? This decides
   whether searches can use the model (fast, any machine) instead of in-loop measurement.
6. **RQ6 (from the CNN reproduction):** does the benefit depend on the LLM (GPT-4.1 vs Qwen2.5-7B)?

## 4. Evaluation

The space has no exhaustive ground truth, so HV and IGD are computed against the **pooled reference
front**: all feasible architectures evaluated by any run of the experiment for that (space, device).
Objectives are min-max normalised over the pool, with reference point 1.1 per objective, and HV is
reported as HV(run) / HV(pool). These numbers compare methods *within* an experiment only. The
report also gives per-generation curves and the reference-model comparison.

## 5. Validation done so far

| Check | Result |
|---|---|
| Parameter formula vs official sizes of 11 SLMs (`slm-reference`, `tests/test_slm.py`) | all within 0.4 % |
| Modelled Jetson / phone latency vs survey Figs. 8 and 11 | in range: e.g. Qwen2.5-0.5B 27 ms prefill / 20 ms per token; Phi-2 149 ms / 78 ms per token on Jetson |
| CPU calibration, 20 random GPT-2-space models (`configs/slm/devices/local_cpu.json`) | Spearman 0.977, mean absolute error 4.8 % |
| In-loop measured search (`slm_lts_measured`, 2 generations) | 78 architectures measured in 7.5 min on CPU |
| Offline end-to-end runs, replay, operators stay valid, 3-D HV | 43 tests pass |

## 6. Limitations (state these in the thesis)

* **The quality proxy is a ranking proxy.** It ignores data and tokenizer, which the survey finds
  matter more than architecture (Sec. 2.3). Claims are about *architecture at fixed training recipe*.
  The Kaplan-loss numbers are illustrative, not predictions. To claim real quality, train a handful
  of front models, as LTS trained 20.
* **Fixed vocabulary.** It is held constant within a space. Reference SLMs are compared at the
  space's vocabulary (architecture-only comparison); `slm-reference` also lists their own-vocabulary costs.
* **The Jetson and phone profiles are fitted by eye** to the survey's plots. Calibrate on real
  devices before making device-specific claims.
* **PyTorch eager timing ≠ llama.cpp at Q4.** Measured and analytical-Q4 latencies are different
  quantities; calibrate with the same dtype you search with (the runner warns on a mismatch).
* **3-objective fronts are large**, because memory and the quality proxy both track parameter count.
  If HV fails to separate methods, run with `--set objectives=[quality,latency]` and
  `hardware.latency_limit_ms`, or use memory as a constraint only.
* **Omitted:** adaptive input embeddings (LTS), per-layer scaling (OpenELM), MLA, and
  head_dim ≠ d_model/n_head (Gemma-2).

## 7. How to run

```bash
python -m llmnas slm-reference                 # parameter check + modelled costs of real SLMs
python -m llmnas run slm_smoke                 # offline, mock LLM
python -m llmnas run slm_main --seeds 0        # one seed of everything (LLM variants need an API key or Ollama)
python -m llmnas report slm_main               # results/reports/slm_main/
python -m llmnas slm-calibrate --device cpu --n 20 --max-decoder-m 60 --out configs/slm/devices/local_cpu.json
python -m llmnas run slm_lts_measured          # LTS-faithful: latency measured in the loop on this machine
```

GPU measurement needs a CUDA build of PyTorch. Your `hw-nas` virtual environment has one (torch
2.6 + cu124); install this package into it with
`~/hw-nas/.venv/Scripts/python -m pip install -e ~/llm-nas-repro`, then pass `--device cuda`.
