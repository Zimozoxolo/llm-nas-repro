# Duplicate-fix test (pre-registered analysis)

Rules: PREREGISTRATION_duplicate_fix.md, Sections 4-5 and amendments 1-2. HV = default convention, read at N evaluations from the trajectory (final value if the run ended earlier). CIFAR-10.

Runs found: PEA with refill (no LLM): 60, Random sampling (no LLM): 60, NSGA-II (no LLM): 60, Reg. evolution (no LLM): 60, gpt-4.1: 6, gpt-4.1 [history + re-prompt]: 6, gpt-4.1 [history]: 6

## Controls at N = 270 (Q2): mean HV, s.d. over seeds, margin

| Control | edgegpu | raspi4 | edgetpu | pixel3 | eyeriss | fpga |
|---|---|---|---|---|---|---|
| PEA with refill (no LLM) | 0.991 ± 0.003 (m 0.010, n=10) | 0.993 ± 0.002 (m 0.010, n=10) | 0.942 ± 0.015 (m 0.029, n=10) | 0.992 ± 0.002 (m 0.010, n=10) | 0.919 ± 0.015 (m 0.029, n=10) | 0.919 ± 0.020 (m 0.041, n=10) |
| Random sampling (no LLM) | 0.979 ± 0.024 (m 0.048, n=10) | 0.990 ± 0.001 (m 0.010, n=10) | 0.933 ± 0.011 (m 0.023, n=10) | 0.989 ± 0.003 (m 0.010, n=10) | 0.895 ± 0.028 (m 0.057, n=10) | 0.904 ± 0.019 (m 0.038, n=10) |
| NSGA-II (no LLM) | 0.994 ± 0.001 (m 0.010, n=10) | 0.995 ± 0.001 (m 0.010, n=10) | 0.949 ± 0.010 (m 0.020, n=10) | 0.993 ± 0.004 (m 0.010, n=10) | 0.945 ± 0.015 (m 0.030, n=10) | 0.943 ± 0.014 (m 0.027, n=10) |
| Reg. evolution (no LLM) | 0.994 ± 0.002 (m 0.010, n=10) | 0.994 ± 0.001 (m 0.010, n=10) | 0.964 ± 0.011 (m 0.021, n=10) | 0.993 ± 0.002 (m 0.010, n=10) | 0.932 ± 0.012 (m 0.025, n=10) | 0.939 ± 0.009 (m 0.017, n=10) |

**gpt-4.1 [history + re-prompt] at N = 270 (Q2)** (primary)

| Control | verdict | devices above / below margin | above where a win is possible (amendment 2) | mean diff | p (better) | p (worse) |
|---|---|---|---|---|---|---|
| PEA with refill (no LLM) (primary comparator) | **Worse** | 0 / 4 of 6 | 0 of 3 (edgetpu, eyeriss, fpga) | -0.0277 | 1.000 | 0.016 |
| Random sampling (no LLM) | **Worse** | 0 / 2 of 6 | 0 of 5 (raspi4, edgetpu, pixel3, eyeriss, fpga) | -0.0168 | 0.984 | 0.031 |
| NSGA-II (no LLM) | **Worse** | 0 / 5 of 6 | 0 of 3 (edgetpu, eyeriss, fpga) | -0.0384 | 1.000 | 0.016 |
| Reg. evolution (no LLM) | **Worse** | 0 / 5 of 6 | 0 of 3 (edgetpu, eyeriss, fpga) | -0.0379 | 1.000 | 0.016 |

**gpt-4.1 [history] at N = 270 (Q2)** (secondary)

| Control | verdict | devices above / below margin | above where a win is possible (amendment 2) | mean diff | p (better) | p (worse) |
|---|---|---|---|---|---|---|
| PEA with refill (no LLM) (primary comparator) | **Worse** | 0 / 2 of 6 | 0 of 3 (edgetpu, eyeriss, fpga) | -0.0168 | 1.000 | 0.016 |
| Random sampling (no LLM) | **Parity** | 0 / 1 of 6 | 0 of 5 (raspi4, edgetpu, pixel3, eyeriss, fpga) | -0.0059 | 0.922 | 0.094 |
| NSGA-II (no LLM) | **Worse** | 0 / 5 of 6 | 0 of 3 (edgetpu, eyeriss, fpga) | -0.0275 | 1.000 | 0.016 |
| Reg. evolution (no LLM) | **Worse** | 0 / 5 of 6 | 0 of 3 (edgetpu, eyeriss, fpga) | -0.0270 | 1.000 | 0.016 |

## Controls at N = 120 (Q3): mean HV, s.d. over seeds, margin

| Control | edgegpu | raspi4 | edgetpu | pixel3 | eyeriss | fpga |
|---|---|---|---|---|---|---|
| PEA with refill (no LLM) | 0.969 ± 0.022 (m 0.043, n=10) | 0.987 ± 0.003 (m 0.010, n=10) | 0.898 ± 0.021 (m 0.042, n=10) | 0.980 ± 0.009 (m 0.017, n=10) | 0.874 ± 0.020 (m 0.041, n=10) | 0.882 ± 0.025 (m 0.049, n=10) |
| Random sampling (no LLM) | 0.952 ± 0.046 (m 0.093, n=10) | 0.983 ± 0.012 (m 0.024, n=10) | 0.892 ± 0.024 (m 0.048, n=10) | 0.982 ± 0.007 (m 0.015, n=10) | 0.846 ± 0.034 (m 0.068, n=10) | 0.858 ± 0.025 (m 0.050, n=10) |
| NSGA-II (no LLM) | 0.953 ± 0.056 (m 0.112, n=10) | 0.988 ± 0.008 (m 0.017, n=10) | 0.919 ± 0.026 (m 0.052, n=10) | 0.981 ± 0.009 (m 0.018, n=10) | 0.903 ± 0.054 (m 0.108, n=10) | 0.902 ± 0.029 (m 0.058, n=10) |
| Reg. evolution (no LLM) | 0.979 ± 0.020 (m 0.041, n=10) | 0.988 ± 0.009 (m 0.017, n=10) | 0.940 ± 0.014 (m 0.027, n=10) | 0.989 ± 0.005 (m 0.010, n=10) | 0.918 ± 0.020 (m 0.040, n=10) | 0.922 ± 0.016 (m 0.033, n=10) |

**gpt-4.1 [history + re-prompt] at N = 120 (Q3)** (primary)

| Control | verdict | devices above / below margin | above where a win is possible (amendment 2) | mean diff | p (better) | p (worse) |
|---|---|---|---|---|---|---|
| PEA with refill (no LLM) (primary comparator) | **Parity** | 0 / 1 of 6 | 0 of 5 (raspi4, edgetpu, pixel3, eyeriss, fpga) | -0.0122 | 0.875 | 0.141 |
| Random sampling (no LLM) | **Parity** | 0 / 0 of 6 | 0 of 4 (edgetpu, pixel3, eyeriss, fpga) | +0.0005 | 0.484 | 0.531 |
| NSGA-II (no LLM) | **Parity** | 0 / 1 of 6 | 0 of 3 (edgetpu, pixel3, fpga) | -0.0217 | 0.906 | 0.109 |
| Reg. evolution (no LLM) | **Worse** | 0 / 4 of 6 | 0 of 4 (edgetpu, pixel3, eyeriss, fpga) | -0.0365 | 1.000 | 0.016 |

**gpt-4.1 [history] at N = 120 (Q3)** (secondary)

| Control | verdict | devices above / below margin | above where a win is possible (amendment 2) | mean diff | p (better) | p (worse) |
|---|---|---|---|---|---|---|
| PEA with refill (no LLM) (primary comparator) | **Parity** | 0 / 1 of 6 | 0 of 5 (raspi4, edgetpu, pixel3, eyeriss, fpga) | -0.0055 | 0.625 | 0.391 |
| Random sampling (no LLM) | **Parity** | 0 / 0 of 6 | 0 of 4 (edgetpu, pixel3, eyeriss, fpga) | +0.0072 | 0.312 | 0.703 |
| NSGA-II (no LLM) | **Worse** | 0 / 0 of 6 | 0 of 3 (edgetpu, pixel3, fpga) | -0.0149 | 1.000 | 0.016 |
| Reg. evolution (no LLM) | **Worse** | 0 / 4 of 6 | 0 of 4 (edgetpu, pixel3, eyeriss, fpga) | -0.0298 | 1.000 | 0.016 |

**gpt-4.1 at N = 120 (Q3)** (secondary)

| Control | verdict | devices above / below margin | above where a win is possible (amendment 2) | mean diff | p (better) | p (worse) |
|---|---|---|---|---|---|---|
| PEA with refill (no LLM) (primary comparator) | **Parity** | 0 / 1 of 6 | 0 of 5 (raspi4, edgetpu, pixel3, eyeriss, fpga) | -0.0104 | 0.875 | 0.141 |
| Random sampling (no LLM) | **Parity** | 0 / 0 of 6 | 0 of 4 (edgetpu, pixel3, eyeriss, fpga) | +0.0023 | 0.375 | 0.641 |
| NSGA-II (no LLM) | **Parity** | 0 / 1 of 6 | 0 of 3 (edgetpu, pixel3, fpga) | -0.0198 | 0.922 | 0.094 |
| Reg. evolution (no LLM) | **Worse** | 0 / 3 of 6 | 0 of 4 (edgetpu, pixel3, eyeriss, fpga) | -0.0347 | 1.000 | 0.016 |

## Q1: mechanism

| Method | min N | mean N | duplicate rate | out of niche | slots unfilled | calls | cost ($) |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | 129 | 145 | 42.7% | 8.3% | 0 | 720 | 5.05 |
| gpt-4.1 [history + re-prompt] | 231 | 243 | 48.3% | 10.4% | 161 | 1644 | 14.07 |
| gpt-4.1 [history] | 140 | 164 | 35.9% | 7.1% | 0 | 720 | 5.53 |
| PEA with refill (no LLM) | 270 | 270 | 79.8% | 0.0% | 0 | 0 | 0.00 |

Arm A passes Q1: **no** (needs N >= 265 on all six devices and duplicate rate <= 22.7%).
Arm B passes Q1: **no** (needs duplicate rate <= 32.7%).

## Q4: arm B against the reference runs (120 calls each, margin 0.010)

**Parity**: 3 / 1 devices above / below, mean diff +0.0140, p (better) 0.078, p (worse) 0.938.

