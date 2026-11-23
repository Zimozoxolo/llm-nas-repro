# Report: table3_main

1 finished runs.

## Table 3 - HV / IGD (mean ± std over seeds)

| Dataset | Method | Edge GPU HV↑ | IGD↓ | Raspi 4 HV↑ | IGD↓ | Edge TPU HV↑ | IGD↓ | Pixel 3 HV↑ | IGD↓ | Eyeriss HV↑ | IGD↓ | FPGA HV↑ | IGD↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | **Ours (reproduced)** | 0.964 | 0.063 | - | - | - | - | - | - | - | - | - | - |
| | LLMatic (paper) | 0.191 | 0.542 | 0.549 | 0.296 | 0.354 | 0.514 | 0.551 | 0.337 | 0.512 | 0.331 | 0.586 | 0.370 |
| | FairNAS (paper) | 0.892 | 0.073 | 0.962 | 0.035 | 0.947 | 0.089 | 0.971 | 0.033 | 0.958 | 0.068 | 0.918 | 0.091 |
| | PRP-NAS (paper) | 0.843 | 0.116 | 0.926 | 0.133 | 0.916 | 0.123 | 0.926 | 0.124 | 0.928 | 0.145 | 0.903 | 0.241 |
| | LLM-NAS (paper) | 0.997 | 0.006 | 0.997 | 0.013 | 0.955 | 0.057 | 0.996 | 0.011 | 0.961 | 0.037 | 0.931 | 0.046 |

## Table 2 - selected CIFAR-10 architectures (pooled over seeds)

| Device | Row | Ours: acc (%) | Ours: lat (ms) | arch | Paper: acc | Paper: lat |
|---|---|---|---|---|---|---|
| Edge GPU | top accuracy | 94.37 | 6.77 | (3, 3, 3, 1, 2, 3) | 94.37±0.02 | 4.35 |
| Edge GPU | knee | 92.98 | 4.31 | (3, 1, 3, 3, 2, 0) | 93.88±0.10 | 3.36 |
| Edge GPU | fastest ≥89% | 89.84 | 3.46 | (2, 1, 2, 1, 1, 0) | 89.18±0.15 | 1.78 |

## Table 4 - search cost

| Variant | API calls / search | input tokens | output tokens | wall time / search | runs |
|---|---|---|---|---|---|
| llmnas | 120 | 160,319 | 66,747 | 8.6 min | 1 |

Paper: LLMatic: 17 GPU days; FairNAS: 10 GPU days; DARTS: 4 GPU days; PRP-NAS-BA: 2 GPU days; LLM-NAS: 3 mins (120 API calls)

## Per-run results

| variant   | dataset   | device   |   seed |     hv |    igd |   n_front |   n_evaluated |   api_calls |   input_tokens |   output_tokens |   search_time_s |   predictor_spearman |
|:----------|:----------|:---------|-------:|-------:|-------:|----------:|--------------:|------------:|---------------:|----------------:|----------------:|---------------------:|
| llmnas    | cifar10   | edgegpu  |      0 | 0.9638 | 0.0631 |        11 |           129 |         120 |         160319 |           66747 |           516.5 |               0.9594 |