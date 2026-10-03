"""Regenerates LLM_NAS_Reproduction_Colab.ipynb (edit here, not in the .ipynb)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# LLM-NAS reproduction (Colab)

Reproduces **LLM-NAS: LLM-Driven Hardware-Aware Neural Architecture Search**
(Zhu, Zhang & Huang, arXiv:2510.01472) on HW-NAS-Bench.

| Paper item | Section in this notebook | Experiment config |
|---|---|---|
| Figure 3 (niche rationale) | 4 | `fig3` command |
| Figure 1 / Figure 9 (exploration bias) | 6 | `configs/experiments/fig1_bias.yaml` |
| Table 3 (HV / IGD) + Table 2 + Table 4 | 7 | `configs/experiments/table3_main.yaml` |
| Table 5 + Figures 6–8 (ablations) | 8 | `configs/experiments/table5_ablation.yaml` |
| Table 6 / Figure 5 (ViT) | not reproduced | see `REPRODUCTION.md` |

Everything the notebook does is a thin call to the `llmnas` package, so results are identical
to running the same commands locally. Runs are **resumable**: re-running a cell skips any
`(variant, dataset, device, seed)` that already has a `run.json`.

**Runtime:** CPU is enough (no training happens). Keep data and results on Google Drive so
they survive disconnects.
""")

md("## 1. Setup")
code("""
# --- edit these two lines -------------------------------------------------
REPO_URL = "https://github.com/Zimozoxolo/llm-nas-repro.git"   # your GitHub copy of this repo
USE_DRIVE = True                                                # keep data/results on Google Drive
# ----------------------------------------------------------------------------
import os, subprocess, sys
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/llm-nas-repro-work"
else:
    WORK = "/content/llm-nas-repro-work"
os.makedirs(WORK, exist_ok=True)
os.environ["LLMNAS_DATA"] = f"{WORK}/data"          # benchmark data location
RESULTS = f"{WORK}/results"                         # experiment outputs

if not os.path.exists("/content/llm-nas-repro"):
    subprocess.run(["git", "clone", REPO_URL, "/content/llm-nas-repro"], check=True)
%cd /content/llm-nas-repro
!git log -1 --format="commit %h  %ad  %s" --date=short
!pip install -q -e ".[anthropic,dev]"
""")

md("""
## 2. API key
Add your key in Colab's **Secrets** panel (key icon on the left) as `OPENAI_API_KEY`
(or `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY`), and enable notebook access. The paper used **GPT-4.1**.
""")
code("""
from google.colab import userdata
for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
    try:
        os.environ[k] = userdata.get(k)
        print(f"{k} loaded")
    except Exception:
        pass

# Pick the LLM. Default = paper setting. Examples of alternatives:
#   LLM = "llm.backend=anthropic llm.model=claude-opus-5-5"
#   LLM = "llm.base_url=https://openrouter.ai/api/v1 llm.api_key_env=OPENROUTER_API_KEY llm.model=openai/gpt-4.1"
LLM = "llm.backend=openai llm.model=gpt-4.1"
SET = f"--set results_dir={RESULTS} {LLM}"
print(SET)
""")

md("""
## 3. Benchmark data
Downloads HW-NAS-Bench, NAS-Bench-Suite-Zero and NATS-Bench (~2 GB, once) and builds
`processed/nb201_<dataset>.csv`. Skipped automatically when the processed tables exist.
Then checks that the data matches the paper (best accuracies, Table-2 points, predictor Spearman ≈ 0.90).
""")
code("""
if not os.path.exists(f"{os.environ['LLMNAS_DATA']}/processed/nb201_ImageNet16-120.csv"):
    !python -m llmnas prepare-data
!python -m llmnas check-data
""")

md("## 4. Figure 3 – parameter clusters by number of 3×3 convolutions")
code("""
!python -m llmnas fig3 --out {RESULTS}/reports/fig3_params.png
from IPython.display import Image, Markdown, display
display(Image(f"{RESULTS}/reports/fig3_params.png", width=500))
""")

md("""
## 5. Offline smoke test
Runs the full pipeline with the **mock** backend (a random operator, no API calls) to verify
the install. Its numbers are meaningless.
""")
code("""
!python -m llmnas run smoke --set results_dir={RESULTS}
!python -m llmnas report smoke --set results_dir={RESULTS}
""")

md("""
## 6. Figure 1 / Figure 9 – exploration bias
Normal prompt vs latency-optimised prompt vs LLM-NAS on CIFAR-10. Start with Edge GPU (Figure 1),
then drop `--devices` for all six devices (Figure 9). ~10 calls per prompting strategy + 120 for LLM-NAS.
""")
code("""
!python -m llmnas run fig1_bias {SET} --devices edgegpu
!python -m llmnas report fig1_bias {SET}
display(Image(f"{RESULTS}/reports/fig1_bias/bias_edgegpu.png"))
""")

md("""
## 7. Table 3 (+ Table 2, Table 4) – main results
Full grid = 3 datasets × 6 devices × 3 seeds × 120 calls ≈ 6,500 GPT-4.1 calls.
Run one slice first to check cost and behaviour, then the rest (it resumes).
""")
code("""
# one slice first
!python -m llmnas run table3_main {SET} --datasets cifar10 --devices edgegpu --seeds 0
""")
code("""
# the full grid (resumable - safe to re-run after a disconnect)
!python -m llmnas run table3_main {SET}
""")
code("""
!python -m llmnas report table3_main {SET}
display(Markdown(open(f"{RESULTS}/reports/table3_main/report.md").read().split("## Per-run")[0]))
display(Image(f"{RESULTS}/reports/table3_main/fronts_cifar10.png"))
""")

md("""
## 8. Table 5 + Figures 6–8 – ablations
Paper: CIFAR-100. The `llmnas` runs are shared with Table 3 (same config ⇒ same run directory).
Add `--datasets cifar10 cifar100 ImageNet16-120` for all three ablation figures.
""")
code("""
!python -m llmnas run table5_ablation {SET}
!python -m llmnas report table5_ablation {SET}
display(Markdown(open(f"{RESULTS}/reports/table5_ablation/report.md").read().split("## Per-run")[0]))
display(Image(f"{RESULTS}/reports/table5_ablation/fronts_cifar100.png"))
display(Image(f"{RESULTS}/reports/table5_ablation/convergence.png"))
""")

md("""
## 9. Exact replay (no API calls)
Every LLM call is saved in `llm_calls.jsonl`. Replaying feeds the saved responses back and
must give identical fronts. Use this to regenerate numbers for the thesis without spending money.
""")
code("""
!python -m llmnas run table3_main {SET} llm.mode=replay --datasets cifar10 --devices edgegpu --seeds 0
import json, glob
for f in glob.glob(f"{RESULTS}/runs/llmnas-*/cifar10/edgegpu/seed0/run*.json"):
    r = json.load(open(f)); print(os.path.basename(f), "HV", r["hv"], "IGD", r["igd"])
""")

md("""
## 10. Extension – LLM-NAS for small language models (BLEND.md)
Blends LLM-NAS with LiteTransformerSearch (decoder-parameter proxy, transformer space) and the
2024 SLM survey (prefill/decode/memory cost model, reference SLMs, prior knowledge).
`datasets` = search space (`slm`, `lts_gpt2`), `devices` = hardware profile.
""")
code("""
!python -m llmnas slm-reference
!python -m llmnas run slm_smoke --set results_dir={RESULTS}
!python -m llmnas run slm_main {SET} --seeds 0
!python -m llmnas report slm_main {SET}
display(Markdown(open(f"{RESULTS}/reports/slm_main/report.md").read().split("## Per-run")[0]))
display(Image(f"{RESULTS}/reports/slm_main/fronts_slm_jetson_orin_nx_16gb.png"))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"colab": {"provenance": []},
                                                 "kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "LLM_NAS_Reproduction_Colab.ipynb")
print("wrote LLM_NAS_Reproduction_Colab.ipynb")
