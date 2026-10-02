# Report: table5_ablation

36 finished runs.

## Table 5 - variants on CIFAR-100 (seed-mean, ± across devices)

| Variant | Avg HV↑ (ours) | Avg IGD↓ (ours) | Avg HV (paper) | Avg IGD (paper) | #runs |
|---|---|---|---|---|---|
| no_llm_pea | 0.946 ± 0.031 | 0.0623 ± 0.0220 | 0.843 ± 0.075 | 0.1649 ± 0.0311 | 18 |
| random_sampling | 0.931 ± 0.047 | 0.0614 ± 0.0330 | - | - | 18 |

## Table 4 - search cost

| Variant | API calls / search | input tokens | output tokens | wall time / search | runs |
|---|---|---|---|---|---|
| no_llm_pea | 0 | 0 | 0 | 0.0 min | 18 |
| random_sampling | 0 | 0 | 0 | 0.0 min | 18 |

Paper: LLMatic: 17 GPU days; FairNAS: 10 GPU days; DARTS: 4 GPU days; PRP-NAS-BA: 2 GPU days; LLM-NAS: 3 mins (120 API calls)

## Per-run results

| variant         | dataset   | device   |   seed |     hv |    igd |   n_front |   n_evaluated |   api_calls |   input_tokens |   output_tokens |   search_time_s |   predictor_spearman |
|:----------------|:----------|:---------|-------:|-------:|-------:|----------:|--------------:|------------:|---------------:|----------------:|----------------:|---------------------:|
| no_llm_pea      | cifar100  | edgegpu  |      0 | 0.9679 | 0.0957 |         8 |           173 |           0 |              0 |               0 |            1.82 |               0.9635 |
| no_llm_pea      | cifar100  | edgegpu  |      1 | 0.9878 | 0.0256 |        15 |           190 |           0 |              0 |               0 |            0.74 |               0.9649 |
| no_llm_pea      | cifar100  | edgegpu  |      2 | 0.9827 | 0.0415 |         9 |           201 |           0 |              0 |               0 |            0.74 |               0.9636 |
| no_llm_pea      | cifar100  | raspi4   |      0 | 0.9793 | 0.0283 |        16 |           174 |           0 |              0 |               0 |            0.83 |               0.9635 |
| no_llm_pea      | cifar100  | raspi4   |      1 | 0.9887 | 0.0256 |        13 |           192 |           0 |              0 |               0 |            0.75 |               0.9649 |
| no_llm_pea      | cifar100  | raspi4   |      2 | 0.9551 | 0.0563 |        14 |           191 |           0 |              0 |               0 |            0.76 |               0.9636 |
| no_llm_pea      | cifar100  | edgetpu  |      0 | 0.9218 | 0.0594 |        13 |           161 |           0 |              0 |               0 |            0.88 |               0.9635 |
| no_llm_pea      | cifar100  | edgetpu  |      1 | 0.9192 | 0.066  |        11 |           144 |           0 |              0 |               0 |            0.75 |               0.9649 |
| no_llm_pea      | cifar100  | edgetpu  |      2 | 0.9078 | 0.0883 |         7 |           148 |           0 |              0 |               0 |            0.77 |               0.9636 |
| no_llm_pea      | cifar100  | pixel3   |      0 | 0.977  | 0.0287 |        15 |           191 |           0 |              0 |               0 |            0.81 |               0.9635 |
| no_llm_pea      | cifar100  | pixel3   |      1 | 0.9839 | 0.0273 |        16 |           195 |           0 |              0 |               0 |            0.74 |               0.9649 |
| no_llm_pea      | cifar100  | pixel3   |      2 | 0.9661 | 0.0512 |        12 |           186 |           0 |              0 |               0 |            0.72 |               0.9636 |
| no_llm_pea      | cifar100  | eyeriss  |      0 | 0.9348 | 0.0755 |        13 |           173 |           0 |              0 |               0 |            0.78 |               0.9635 |
| no_llm_pea      | cifar100  | eyeriss  |      1 | 0.9275 | 0.0924 |        11 |           158 |           0 |              0 |               0 |            0.67 |               0.9649 |
| no_llm_pea      | cifar100  | eyeriss  |      2 | 0.9159 | 0.1161 |         7 |           171 |           0 |              0 |               0 |            0.7  |               0.9636 |
| no_llm_pea      | cifar100  | fpga     |      0 | 0.9119 | 0.0792 |        12 |           167 |           0 |              0 |               0 |            0.77 |               0.9635 |
| no_llm_pea      | cifar100  | fpga     |      1 | 0.9411 | 0.0582 |        10 |           159 |           0 |              0 |               0 |            0.87 |               0.9649 |
| no_llm_pea      | cifar100  | fpga     |      2 | 0.8594 | 0.1068 |         9 |           175 |           0 |              0 |               0 |            0.76 |               0.9636 |
| random_sampling | cifar100  | edgegpu  |      0 | 0.9859 | 0.0283 |        10 |           270 |           0 |              0 |               0 |            0.79 |               0.9635 |
| random_sampling | cifar100  | edgegpu  |      1 | 0.9791 | 0.0329 |        10 |           270 |           0 |              0 |               0 |            0.75 |               0.9649 |
| random_sampling | cifar100  | edgegpu  |      2 | 0.963  | 0.0334 |         9 |           268 |           0 |              0 |               0 |            0.8  |               0.9636 |
| random_sampling | cifar100  | raspi4   |      0 | 0.9776 | 0.0286 |        16 |           270 |           0 |              0 |               0 |            0.76 |               0.9635 |
| random_sampling | cifar100  | raspi4   |      1 | 0.98   | 0.0344 |        10 |           270 |           0 |              0 |               0 |            0.78 |               0.9649 |
| random_sampling | cifar100  | raspi4   |      2 | 0.9763 | 0.0336 |        15 |           268 |           0 |              0 |               0 |            0.84 |               0.9636 |
| random_sampling | cifar100  | edgetpu  |      0 | 0.9046 | 0.0624 |         9 |           270 |           0 |              0 |               0 |            0.88 |               0.9635 |
| random_sampling | cifar100  | edgetpu  |      1 | 0.8915 | 0.0646 |         8 |           270 |           0 |              0 |               0 |            0.94 |               0.9649 |
| random_sampling | cifar100  | edgetpu  |      2 | 0.9123 | 0.072  |         9 |           268 |           0 |              0 |               0 |            0.87 |               0.9636 |
| random_sampling | cifar100  | pixel3   |      0 | 0.9769 | 0.0286 |        16 |           269 |           0 |              0 |               0 |            0.85 |               0.9635 |
| random_sampling | cifar100  | pixel3   |      1 | 0.9747 | 0.0306 |        10 |           269 |           0 |              0 |               0 |            0.82 |               0.9649 |
| random_sampling | cifar100  | pixel3   |      2 | 0.9791 | 0.0334 |        14 |           270 |           0 |              0 |               0 |            0.79 |               0.9636 |
| random_sampling | cifar100  | eyeriss  |      0 | 0.9043 | 0.1076 |        12 |           270 |           0 |              0 |               0 |            0.8  |               0.9635 |
| random_sampling | cifar100  | eyeriss  |      1 | 0.8574 | 0.1266 |         6 |           270 |           0 |              0 |               0 |            0.76 |               0.9649 |
| random_sampling | cifar100  | eyeriss  |      2 | 0.8643 | 0.1107 |        10 |           269 |           0 |              0 |               0 |            0.78 |               0.9636 |
| random_sampling | cifar100  | fpga     |      0 | 0.8632 | 0.0893 |        10 |           270 |           0 |              0 |               0 |            0.76 |               0.9635 |
| random_sampling | cifar100  | fpga     |      1 | 0.8806 | 0.1018 |         9 |           270 |           0 |              0 |               0 |            0.76 |               0.9649 |
| random_sampling | cifar100  | fpga     |      2 | 0.8895 | 0.087  |        12 |           268 |           0 |              0 |               0 |            0.79 |               0.9636 |