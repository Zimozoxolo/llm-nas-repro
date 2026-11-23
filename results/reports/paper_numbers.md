# Numbers for the paper (CIFAR-10)

Writes results/reports/paper_numbers.md. No search is run and no LLM is called. Conventions:
* HV / IGD: the default convention (llmnas.metrics, as stored in run.json).
* "(>=80%)": re-scored with architectures below 80% test accuracy removed from the true front and the found front
  (llmnas.compare, as ``compare-llms --min-acc 80``).
* s.d. is the sample s.d. over seeds (ddof = 1).
* Matched budget: a control's HV read from its trajectory at the LLM run's number of evaluated architectures N
  (linear interpolation; llmnas.dupfix.hv_at), averaged over the control's seeds.


## 1. Edge GPU, per method

| Method | seeds | HV | IGD | HV>=80 | IGD>=80 | cell | usable | N | minutes | cost $ | tokens in/out (k) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1 | 1 | 0.964 | 0.063 | 0.752 | 0.171 | 0/1 | 99/240 (41%) | 129 | 8.6 | 0.85 | 160/67 |
| claude-opus-5-5 | 1 | 0.980 | 0.052 | 0.961 | 0.041 | 0/1 | 192/240 (80%) | 222 | 70.8 | 10.77 | 438/451 |
| claude-haiku-4-5 | 1 | 0.913 | 0.076 | 0.710 | 0.242 | 0/1 | 66/240 (28%) | 96 | 43.0 | 1.62 | 311/263 |
| deepseek-coder-v2-16b-8k | 3 | 0.983 ± 0.008 | 0.063 ± 0.010 | 0.825 ± 0.091 | 0.131 ± 0.017 | 3/3 | 36/240 (15%) | 66 | 124.8 | 0.00 | 100/46 |
| qwen2.5-7b-8k | 3 | 0.949 ± 0.053 | 0.069 ± 0.017 | 0.781 ± 0.106 | 0.165 ± 0.065 | 2/3 | 78/240 (32%) | 108 | 144.9 | 0.00 | 134/64 |
| PEA (no LLM) | 10 | 0.979 ± 0.026 | 0.053 ± 0.035 | 0.870 ± 0.052 | 0.119 ± 0.060 | 10/10 | 150/240 (62%) | 180 | 0.0 | 0.00 | 0/0 |
| PEA with refill (no LLM) | 10 | 0.991 ± 0.003 | 0.057 ± 0.046 | 0.922 ± 0.039 | 0.096 ± 0.051 | 10/10 | 240/1076 (24%) | 270 | 0.0 | 0.00 | 0/0 |
| Random sampling (no LLM) | 10 | 0.979 ± 0.024 | 0.049 ± 0.011 | 0.911 ± 0.055 | 0.100 ± 0.056 | 10/10 | 239/240 (100%) | 269 | 0.0 | 0.00 | 0/0 |
| NSGA-II (no LLM) | 10 | 0.994 ± 0.001 | 0.065 ± 0.041 | 0.953 ± 0.011 | 0.089 ± 0.050 | 10/10 | 240/314 (77%) | 270 | 0.0 | 0.00 | 0/0 |
| Reg. evolution (no LLM) | 10 | 0.994 ± 0.002 | 0.054 ± 0.038 | 0.951 ± 0.020 | 0.081 ± 0.046 | 10/10 | 240/307 (78%) | 270 | 0.1 | 0.00 | 0/0 |

Per-seed LLM runs on Edge GPU:

- claude-haiku-4-5 seed 0: HV 0.913, IGD 0.076, HV80 0.710, IGD80 0.242, N 96, usable 66/240, 43.0 min, cell False
- claude-opus-5-5 seed 0: HV 0.980, IGD 0.052, HV80 0.961, IGD80 0.041, N 222, usable 192/240, 70.8 min, cell False
- deepseek-coder-v2-16b-8k seed 0: HV 0.988, IGD 0.075, HV80 0.901, IGD80 0.124, N 85, usable 55/240, 70.8 min, cell True
- deepseek-coder-v2-16b-8k seed 1: HV 0.986, IGD 0.058, HV80 0.850, IGD80 0.119, N 49, usable 19/240, 149.6 min, cell True
- deepseek-coder-v2-16b-8k seed 2: HV 0.973, IGD 0.057, HV80 0.723, IGD80 0.151, N 64, usable 34/240, 154.1 min, cell True
- gpt-4.1 seed 0: HV 0.964, IGD 0.063, HV80 0.752, IGD80 0.171, N 129, usable 99/240, 8.6 min, cell False
- qwen2.5-7b-8k seed 0: HV 0.888, IGD 0.088, HV80 0.709, IGD80 0.230, N 125, usable 95/240, 152.8 min, cell False
- qwen2.5-7b-8k seed 1: HV 0.985, IGD 0.059, HV80 0.902, IGD80 0.099, N 89, usable 59/240, 143.3 min, cell True
- qwen2.5-7b-8k seed 2: HV 0.975, IGD 0.059, HV80 0.731, IGD80 0.165, N 110, usable 80/240, 138.5 min, cell True
- PEA (no LLM): HV80 range 0.799-0.964, IGD80 range 0.048-0.275
- PEA with refill (no LLM): HV80 range 0.842-0.970, IGD80 range 0.038-0.165
- Random sampling (no LLM): HV80 range 0.796-0.957, IGD80 range 0.054-0.232
- NSGA-II (no LLM): HV80 range 0.932-0.967, IGD80 range 0.039-0.165
- Reg. evolution (no LLM): HV80 range 0.900-0.972, IGD80 range 0.036-0.162

## 2. HV (default) on six devices: mean ± s.d. over seeds (seeds in brackets)

| Method | edgegpu | raspi4 | edgetpu | pixel3 | eyeriss | fpga | mean over devices |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | 0.964 (1) | 0.983 (1) | 0.902 (1) | 0.986 (1) | 0.896 (1) | 0.839 (1) | 0.928 |
| claude-haiku-4-5 | 0.913 (1) | 0.986 (1) | 0.844 (1) | 0.938 (1) | 0.840 (1) | 0.807 (1) | 0.888 |
| deepseek-coder-v2-16b-8k | 0.983 ± 0.008 (3) | 0.974 ± 0.016 (3) | 0.801 ± 0.027 (3) | 0.977 ± 0.006 (3) | 0.768 ± 0.058 (3) | 0.838 ± 0.027 (3) | 0.890 |
| qwen2.5-7b-8k | 0.949 ± 0.053 (3) | 0.983 ± 0.003 (3) | 0.845 ± 0.050 (3) | 0.980 ± 0.001 (3) | 0.787 ± 0.035 (3) | 0.819 ± 0.035 (3) | 0.894 |
| PEA (no LLM) | 0.979 ± 0.026 (10) | 0.990 ± 0.003 (10) | 0.921 ± 0.020 (10) | 0.988 ± 0.005 (10) | 0.908 ± 0.019 (10) | 0.913 ± 0.017 (10) | 0.950 |
| PEA with refill (no LLM) | 0.991 ± 0.003 (10) | 0.993 ± 0.002 (10) | 0.942 ± 0.015 (10) | 0.992 ± 0.002 (10) | 0.919 ± 0.015 (10) | 0.919 ± 0.020 (10) | 0.959 |
| Random sampling (no LLM) | 0.979 ± 0.024 (10) | 0.990 ± 0.001 (10) | 0.933 ± 0.011 (10) | 0.989 ± 0.003 (10) | 0.895 ± 0.028 (10) | 0.904 ± 0.019 (10) | 0.948 |
| NSGA-II (no LLM) | 0.994 ± 0.001 (10) | 0.995 ± 0.001 (10) | 0.949 ± 0.010 (10) | 0.993 ± 0.004 (10) | 0.945 ± 0.015 (10) | 0.943 ± 0.014 (10) | 0.970 |
| Reg. evolution (no LLM) | 0.994 ± 0.002 (10) | 0.994 ± 0.001 (10) | 0.964 ± 0.011 (10) | 0.993 ± 0.002 (10) | 0.932 ± 0.012 (10) | 0.939 ± 0.009 (10) | 0.969 |

## 2. HV (>= 80% accuracy) on six devices: mean ± s.d. over seeds (seeds in brackets)

| Method | edgegpu | raspi4 | edgetpu | pixel3 | eyeriss | fpga | mean over devices |
|---|---|---|---|---|---|---|---|
| gpt-4.1 | 0.752 (1) | 0.875 (1) | 0.902 (1) | 0.895 (1) | 0.896 (1) | 0.839 (1) | 0.860 |
| claude-haiku-4-5 | 0.710 (1) | 0.865 (1) | 0.844 (1) | 0.858 (1) | 0.840 (1) | 0.807 (1) | 0.821 |
| deepseek-coder-v2-16b-8k | 0.825 ± 0.091 (3) | 0.838 ± 0.034 (3) | 0.801 ± 0.027 (3) | 0.808 ± 0.044 (3) | 0.768 ± 0.058 (3) | 0.838 ± 0.027 (3) | 0.813 |
| qwen2.5-7b-8k | 0.781 ± 0.106 (3) | 0.827 ± 0.047 (3) | 0.845 ± 0.050 (3) | 0.833 ± 0.011 (3) | 0.787 ± 0.035 (3) | 0.819 ± 0.035 (3) | 0.815 |
| PEA (no LLM) | 0.870 ± 0.052 (10) | 0.901 ± 0.033 (10) | 0.921 ± 0.020 (10) | 0.895 ± 0.027 (10) | 0.908 ± 0.019 (10) | 0.913 ± 0.017 (10) | 0.901 |
| PEA with refill (no LLM) | 0.922 ± 0.039 (10) | 0.922 ± 0.023 (10) | 0.942 ± 0.015 (10) | 0.912 ± 0.016 (10) | 0.919 ± 0.015 (10) | 0.919 ± 0.020 (10) | 0.923 |
| Random sampling (no LLM) | 0.911 ± 0.055 (10) | 0.903 ± 0.010 (10) | 0.933 ± 0.011 (10) | 0.894 ± 0.025 (10) | 0.895 ± 0.028 (10) | 0.904 ± 0.019 (10) | 0.907 |
| NSGA-II (no LLM) | 0.953 ± 0.011 (10) | 0.942 ± 0.016 (10) | 0.949 ± 0.010 (10) | 0.938 ± 0.011 (10) | 0.945 ± 0.015 (10) | 0.943 ± 0.014 (10) | 0.945 |
| Reg. evolution (no LLM) | 0.951 ± 0.020 (10) | 0.935 ± 0.013 (10) | 0.964 ± 0.011 (10) | 0.923 ± 0.015 (10) | 0.932 ± 0.012 (10) | 0.939 ± 0.009 (10) | 0.941 |

## 3. LLM runs above each control's device mean (every LLM run = one device x seed)

| Convention | LLM runs | PEA | PEA+refill | RS | NSGA-II | RegEvo |
|---|---|---|---|---|---|---|
| default | 48 | 3 (DeepSeek/edgegpu/s0, DeepSeek/edgegpu/s1, Qwen/edgegpu/s1) | 0 | 4 (DeepSeek/edgegpu/s0, DeepSeek/edgegpu/s1, GPT-4.1/eyeriss/s0, Qwen/edgegpu/s1) | 0 | 0 |
| >=80% | 48 | 3 (DeepSeek/edgegpu/s0, GPT-4.1/pixel3/s0, Qwen/edgegpu/s1) | 0 | 2 (GPT-4.1/eyeriss/s0, GPT-4.1/pixel3/s0) | 0 | 0 |

Range of the LLM runs per device (all seeds) and of the control means:

- hv edgegpu: LLM runs 0.888-0.988; control means 0.979-0.994
- hv raspi4: LLM runs 0.955-0.987; control means 0.990-0.995
- hv edgetpu: LLM runs 0.778-0.902; control means 0.921-0.964
- hv pixel3: LLM runs 0.938-0.986; control means 0.988-0.993
- hv eyeriss: LLM runs 0.713-0.896; control means 0.895-0.945
- hv fpga: LLM runs 0.779-0.866; control means 0.904-0.943
- hv80 edgegpu: LLM runs 0.709-0.902; control means 0.870-0.953
- hv80 raspi4: LLM runs 0.788-0.879; control means 0.901-0.942
- hv80 edgetpu: LLM runs 0.778-0.902; control means 0.921-0.964
- hv80 pixel3: LLM runs 0.770-0.895; control means 0.894-0.938
- hv80 eyeriss: LLM runs 0.713-0.896; control means 0.895-0.945
- hv80 fpga: LLM runs 0.779-0.866; control means 0.904-0.943

## 4. Matched budget: controls read at each LLM run's N (mean over LLM seeds and control seeds)

| LLM | device | seeds | N | LLM HV | PEA@N | PEA+refill@N | RS@N | NSGA-II@N | RegEvo@N |
|---|---|---|---|---|---|---|---|---|---|
| GPT-4.1 | edgegpu | 1 | 129 | 0.964 | **0.962** | 0.970 | **0.959** | **0.957** | 0.981 |
| GPT-4.1 | raspi4 | 1 | 161 | 0.983 | 0.990 | 0.990 | 0.988 | 0.992 | 0.990 |
| GPT-4.1 | edgetpu | 1 | 141 | 0.902 | 0.916 | 0.911 | **0.900** | 0.924 | 0.946 |
| GPT-4.1 | pixel3 | 1 | 151 | 0.986 | 0.987 | **0.984** | **0.984** | 0.986 | 0.991 |
| GPT-4.1 | eyeriss | 1 | 140 | 0.896 | **0.894** | **0.884** | **0.857** | 0.917 | 0.922 |
| GPT-4.1 | fpga | 1 | 148 | 0.839 | 0.911 | 0.891 | 0.862 | 0.920 | 0.927 |
| Haiku 4.5 | edgegpu | 1 | 96 | 0.913 | 0.946 | 0.963 | 0.926 | 0.942 | 0.972 |
| Haiku 4.5 | raspi4 | 1 | 90 | 0.986 | **0.982** | **0.984** | **0.981** | **0.981** | **0.984** |
| Haiku 4.5 | edgetpu | 1 | 82 | 0.844 | 0.872 | 0.863 | 0.870 | 0.897 | 0.927 |
| Haiku 4.5 | pixel3 | 1 | 92 | 0.938 | 0.978 | 0.978 | 0.975 | 0.972 | 0.984 |
| Haiku 4.5 | eyeriss | 1 | 86 | 0.840 | **0.840** | 0.847 | **0.820** | 0.867 | 0.905 |
| Haiku 4.5 | fpga | 1 | 68 | 0.807 | 0.841 | 0.829 | 0.812 | 0.832 | 0.887 |
| DeepSeek | edgegpu | 3 | 66 | 0.983 | **0.921** | **0.926** | **0.894** | **0.908** | **0.937** |
| DeepSeek | raspi4 | 3 | 61 | 0.974 | 0.975 | 0.975 | **0.970** | **0.964** | 0.977 |
| DeepSeek | edgetpu | 3 | 61 | 0.801 | 0.837 | 0.836 | 0.839 | 0.859 | 0.894 |
| DeepSeek | pixel3 | 3 | 62 | 0.977 | **0.965** | **0.966** | **0.962** | **0.956** | **0.972** |
| DeepSeek | eyeriss | 3 | 57 | 0.768 | 0.780 | 0.784 | 0.783 | 0.802 | 0.869 |
| DeepSeek | fpga | 3 | 70 | 0.838 | **0.832** | **0.824** | **0.811** | **0.833** | 0.878 |
| Qwen | edgegpu | 3 | 108 | 0.949 | 0.951 | 0.965 | **0.939** | **0.946** | 0.975 |
| Qwen | raspi4 | 3 | 118 | 0.983 | 0.987 | 0.987 | **0.983** | 0.988 | 0.987 |
| Qwen | edgetpu | 3 | 105 | 0.845 | 0.896 | 0.883 | 0.884 | 0.912 | 0.934 |
| Qwen | pixel3 | 3 | 115 | 0.980 | 0.985 | 0.980 | 0.981 | 0.980 | 0.989 |
| Qwen | eyeriss | 3 | 123 | 0.787 | 0.882 | 0.875 | 0.848 | 0.905 | 0.919 |
| Qwen | fpga | 3 | 117 | 0.819 | 0.886 | 0.876 | 0.855 | 0.899 | 0.921 |

| LLM | mean N | LLM HV | PEA@N | PEA+refill@N | RS@N | NSGA-II@N | RegEvo@N | devices LLM above (of 6) |
|---|---|---|---|---|---|---|---|---|
| GPT-4.1 | 145 | 0.928 | 0.943 | 0.938 | 0.925 | 0.949 | 0.960 | PEA 2 / PEA+refill 2 / RS 4 / NSGA-II 1 / RegEvo 0 |
| Haiku 4.5 | 86 | 0.888 | 0.910 | 0.911 | 0.897 | 0.915 | 0.943 | PEA 2 / PEA+refill 1 / RS 2 / NSGA-II 1 / RegEvo 1 |
| DeepSeek | 63 | 0.890 | 0.885 | 0.885 | 0.877 | 0.887 | 0.921 | PEA 3 / PEA+refill 3 / RS 4 / NSGA-II 4 / RegEvo 2 |
| Qwen | 114 | 0.894 | 0.931 | 0.928 | 0.915 | 0.938 | 0.954 | PEA 0 / PEA+refill 0 / RS 2 / NSGA-II 1 / RegEvo 0 |

## 5. Controls over devices at N = 145 and N = 270 (default HV, mean over seeds then devices)

- PEA (no LLM): N=86 0.912, N=145 0.945, N=270 0.950 (gain 145->270 +0.005); mean seed s.d. at 145: 0.017; mean N evaluated 168
- PEA with refill (no LLM): N=86 0.913, N=145 0.940, N=270 0.959 (gain 145->270 +0.020); mean seed s.d. at 145: 0.016; mean N evaluated 270
- Random sampling (no LLM): N=86 0.899, N=145 0.926, N=270 0.948 (gain 145->270 +0.022); mean seed s.d. at 145: 0.022; mean N evaluated 269
- NSGA-II (no LLM): N=86 0.919, N=145 0.951, N=270 0.970 (gain 145->270 +0.019); mean seed s.d. at 145: 0.024; mean N evaluated 270
- Reg. evolution (no LLM): N=86 0.947, N=145 0.960, N=270 0.969 (gain 145->270 +0.009); mean seed s.d. at 145: 0.012; mean N evaluated 270

## 6. Seed-to-seed spread of the LLM runs with three seeds (default HV and HV>=80, s.d. per device)

- deepseek-coder-v2-16b-8k hv: s.d. edgegpu 0.008, raspi4 0.016, edgetpu 0.027, pixel3 0.006, eyeriss 0.058, fpga 0.027; range over devices 0.006-0.058; N 47-91 (mean 63)
- deepseek-coder-v2-16b-8k hv80: s.d. edgegpu 0.091, raspi4 0.034, edgetpu 0.027, pixel3 0.044, eyeriss 0.058, fpga 0.027; range over devices 0.027-0.091; N 47-91 (mean 63)
- qwen2.5-7b-8k hv: s.d. edgegpu 0.053, raspi4 0.003, edgetpu 0.050, pixel3 0.001, eyeriss 0.035, fpga 0.035; range over devices 0.001-0.053; N 89-136 (mean 114)
- qwen2.5-7b-8k hv80: s.d. edgegpu 0.106, raspi4 0.047, edgetpu 0.050, pixel3 0.011, eyeriss 0.035, fpga 0.035; range over devices 0.011-0.106; N 89-136 (mean 114)

Search cost per model (all runs): Haiku 4.5 6 runs, 38-43 min, $1.44-1.62; Opus 5.5 1 runs, 71-71 min, $10.77-10.77; DeepSeek 18 runs, 63-177 min, $0.00-0.00; GPT-4.1 6 runs, 8-9 min, $0.80-0.88; Qwen 18 runs, 124-191 min, $0.00-0.00
