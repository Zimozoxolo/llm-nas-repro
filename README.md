# LLM-NAS reproduction

A reproduction of **LLM-NAS: LLM-Driven Hardware-Aware Neural Architecture Search**
(Hengyi Zhu, Grace Li Zhang, Shaoyi Huang; [arXiv:2510.01472](https://arxiv.org/abs/2510.01472)) on
HW-NAS-Bench, built for repeatable re-runs.

* One package (`src/llmnas`), one config system, one command per paper table or figure.
* Every LLM call is logged, so any run can be **replayed exactly** without an API key.
* Runs are keyed by a hash of their config. A finished run is never recomputed, and grids resume after interruptions.
* Local runs and the [Colab notebook](notebooks/LLM_NAS_Reproduction_Colab.ipynb) call the same code.

**Extension:** the same engine searches hardware-efficient *small language models*, blended with
LiteTransformerSearch and the 2024 SLM survey. See **[BLEND.md](BLEND.md)** and `configs/slm/`.

See **[REPRODUCTION.md](REPRODUCTION.md)** for how each paper result maps to a command, and for every
detail the paper leaves unspecified and the choice made here.

## Quick start

```bash
pip install -e ".[anthropic,dev]"
python -m llmnas prepare-data          # ~2 GB download once -> data/processed/*.csv
python -m llmnas check-data            # data matches the paper? predictor Spearman ~0.90?
python -m pytest                       # unit + offline end-to-end tests
python -m llmnas run smoke             # offline pipeline check (mock LLM, no API key)
```

Set your key and run the paper experiments (GPT-4.1 is the paper's engine):

```bash
export OPENAI_API_KEY=...              # PowerShell: $env:OPENAI_API_KEY="..."
python -m llmnas run fig1_bias         # Figure 1 / 9  - exploration bias
python -m llmnas run table3_main       # Table 3 (+ Table 2, Table 4)
python -m llmnas run table5_ablation   # Table 5 + Figures 6-8
python -m llmnas report table3_main    # -> results/reports/table3_main/report.md + PNGs
python -m llmnas fig3                  # Figure 3
```

Useful flags:

```bash
# run a slice of a grid
python -m llmnas run table3_main --datasets cifar10 --devices edgegpu --seeds 0
# override any config value
python -m llmnas run table3_main --set llm.model=gpt-4o search.n_child=6
# other LLMs: Claude, OpenRouter, a local Ollama model
python -m llmnas run table3_main --set llm.backend=anthropic llm.model=claude-opus-5-5
python -m llmnas run table3_main --set llm.base_url=https://openrouter.ai/api/v1 llm.api_key_env=OPENROUTER_API_KEY llm.model=openai/gpt-4.1
python -m llmnas run table3_main --set llm.base_url=http://localhost:11434/v1 llm.model=qwen2.5:14b
# re-execute a finished run from its saved LLM responses (no API calls)
python -m llmnas run table3_main --set llm.mode=replay
```

### Local models with Ollama (free, slower)

```bash
ollama pull qwen2.5:7b
ollama create qwen2.5-7b-8k -f configs/ollama/qwen2.5-7b-8k.Modelfile   # 8K context, avoids prompt truncation
python -m llmnas run table3_main --datasets cifar10 --devices edgegpu --seeds 0 --set llm.base_url=http://localhost:11434/v1 llm.model=qwen2.5-7b-8k
```

On a 4 GB laptop GPU (RTX 3050 Ti), one call takes ~80-95 s, so a 120-call search takes about 3 hours.
A local model is a *substituted LLM*: report its runs separately from GPT-4.1 runs (they get their own
config hash and folder automatically). For long runs, start Python with `-u` (or set `PYTHONUNBUFFERED=1`)
so the log file updates live.

Changing any setting other than `llm.mode` produces a new config hash, so the run goes to a new directory
and earlier results are never overwritten.

## Layout

```
configs/base.yaml                 all defaults, each tagged [paper] or [assumed]
configs/experiments/*.yaml        one file per paper table/figure (grid + variants)
configs/slm/                      SLM-blend defaults, experiments, device calibrations
src/llmnas/
  search_space.py                 NB201 encoding (paper Fig. 2), niches (Table 1), PEA operators
  data.py                         download + build tidy benchmark tables
  benchmark.py                    latency / accuracy / ZC lookups
  predictor.py                    XGBoost-on-13-ZC-proxies predictor, Synflow ablation
  prompts.py                      Stage-1 / Stage-2 prompts (Appendix E) + response parsing
  llm.py, llm_anthropic.py        LLM backends, call log, replay
  search.py                       Algorithm 1 + ablation switches
  bias.py                         exploration-bias study (Fig. 1 / 9)
  metrics.py                      Pareto front, HV, IGD
  experiments.py                  config merging, grids, run directories
  report.py                       Tables 2-5, Figures 1, 3, 6-8
  paper_results.py                the paper's reported numbers, for side-by-side tables
  checks.py                       data-alignment checks
  slm/                            blended SLM search (BLEND.md): space, costs, measure, prompts, search, report
notebooks/                        Colab notebook (+ the script that generates it)
tests/                            pytest; runs on synthetic data, no download needed
results/runs/<variant>-<hash>/<dataset>/<device>/seed<k>/
    config.yaml  run.json  llm_calls.jsonl
results/reports/<experiment>/     report.md, runs.csv, trajectories.csv, *.png
```

## Data sources

| Source | Used for | License |
|---|---|---|
| [HW-NAS-Bench](https://github.com/RICE-EIC/HW-NAS-Bench) | latency on 6 devices | MIT |
| [NATS-Bench / NAS-Bench-201](https://github.com/D-X-Y/NATS-Bench) | 200-epoch test accuracy | MIT |
| [NAS-Bench-Suite-Zero](https://github.com/automl/NASLib/tree/zerocost) | 13 zero-cost proxies, validation accuracy | Apache-2.0 |

## Citation

```bibtex
@article{zhu2025llmnas,
  title   = {LLM-NAS: LLM-Driven Hardware-Aware Neural Architecture Search},
  author  = {Zhu, Hengyi and Zhang, Grace Li and Huang, Shaoyi},
  journal = {arXiv preprint arXiv:2510.01472},
  year    = {2025}
}
```
