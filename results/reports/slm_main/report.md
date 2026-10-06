# Report: slm_main

36 finished runs. HV/IGD are relative to the pooled reference front of this experiment (see `slm/report.py`).

**Primary metric - latency overhead at matched quality:** for 40 log-spaced decoder-parameter levels N, the fastest model with >= N decoder params found by the run, relative to the fastest found by any run (geometric mean; 0% = best everywhere). Coverage = share of levels the run reaches at all.

## slm on jetson_orin_nx_16gb

HV/IGD use the configured objectives ['quality', 'latency', 'memory']; HV₂/IGD₂ re-score the same runs on (quality, latency) only.

| Variant | latency overhead↓ | coverage | HV↑ | IGD↓ | HV₂↑ | IGD₂↓ | front | valid proposals | API calls | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| llmnas | +6.7% ± 1.1% | 95% | 0.985 ± 0.002 | 0.018 ± 0.002 | 0.985 ± 0.004 | 0.016 ± 0.004 | 124.0 | 48% | 120 | 3 |
| llmnas_survey_kb | +9.1% ± 3.4% | 97% | 0.983 ± 0.004 | 0.018 ± 0.004 | 0.986 ± 0.003 | 0.014 ± 0.003 | 125.3 | 51% | 120 | 3 |
| no_partitioning | +8.7% ± 1.9% | 88% | 0.966 ± 0.017 | 0.047 ± 0.027 | 0.956 ± 0.023 | 0.059 ± 0.037 | 109.7 | 56% | 20 | 3 |
| lts_search | +5.8% ± 0.9% | 92% | 0.984 ± 0.004 | 0.023 ± 0.007 | 0.979 ± 0.007 | 0.026 ± 0.011 | 193.3 | 94% | 0 | 3 |
| no_llm_pea | +6.5% ± 3.5% | 98% | 0.989 ± 0.004 | 0.013 ± 0.004 | 0.991 ± 0.003 | 0.009 ± 0.004 | 198.7 | 97% | 0 | 3 |
| random_sampling | +5.0% ± 0.4% | 96% | 0.991 ± 0.000 | 0.011 ± 0.001 | 0.991 ± 0.001 | 0.009 ± 0.001 | 179.7 | 100% | 0 | 3 |

Latency overhead per niche (mean over seeds; blank = level not reached):

| Variant | S0 <250M | S1 250-500M | S2 0.5-1B | S3 1-2B | S4 2-3B | S5 >=3B |
|---|---|---|---|---|---|---|
| llmnas | +7.1% | +6.3% | +10.9% | +6.2% | +5.2% | +3.5% |
| llmnas_survey_kb | +11.7% | +7.4% | +12.3% | +7.1% | +3.8% | +4.6% |
| no_partitioning | +9.8% | +8.6% | +8.0% | +6.1% | +9.8% | +6.1% |
| lts_search | +10.6% | +2.3% | +2.2% | +1.9% | +2.2% | +2.7% |
| no_llm_pea | +10.0% | +5.3% | +4.8% | +5.3% | +3.0% | +2.5% |
| random_sampling | +7.6% | +4.4% | +3.6% | +2.5% | +1.8% | +3.6% |

**Hand-designed SLMs vs the fastest LLM-NAS model with at least as many decoder parameters** (same cost model and vocabulary):

| model          |   decoder_params_m |   latency_ms |   memory_mb | feasible   |   found_decoder_params_m |   found_latency_ms |   found_memory_mb |   speedup | found_arch                                                 |
|:---------------|-------------------:|-------------:|------------:|:-----------|-------------------------:|-------------------:|------------------:|----------:|:-----------------------------------------------------------|
| SmolLM-135M    |              106.2 |        555.3 |         174 | True       |                    107   |              393.6 |               149 |      1.41 | L8 d768 {'d_inner': 5120, 'n_head': 12, 'n_kv_head': 4}    |
| SmolLM-360M    |              314.6 |        873.9 |         328 | True       |                    316.6 |              737.2 |               273 |      1.19 | L10 d1792 {'d_inner': 4608, 'n_head': 14, 'n_kv_head': 1}  |
| Qwen2.5-0.5B   |              357.9 |        864.3 |         294 | True       |                    362.5 |              800.6 |               308 |      1.08 | L10 d1792 {'d_inner': 5376, 'n_head': 14, 'n_kv_head': 2}  |
| TinyLlama-1.1B |              969   |       1732.5 |         661 | True       |                    981.5 |             1740.8 |               627 |      1    | L24 d1536 {'d_inner': 7808, 'n_head': 24, 'n_kv_head': 1}  |
| Llama-3.2-1B   |              973.1 |       1693.9 |         684 | True       |                    981.5 |             1740.8 |               627 |      0.97 | L24 d1536 {'d_inner': 7808, 'n_head': 24, 'n_kv_head': 1}  |
| Pythia-1.4B    |             1208.1 |       2102.2 |        1130 | True       |                   1237.1 |             2105.8 |               779 |      1    | L26 d1536 {'d_inner': 9216, 'n_head': 24, 'n_kv_head': 2}  |
| Qwen2.5-1.5B   |             1310.3 |       2223.4 |         848 | True       |                   1313.8 |             2210.4 |               820 |      1.01 | L26 d1536 {'d_inner': 9856, 'n_head': 24, 'n_kv_head': 2}  |
| SmolLM-1.7B    |             1610.7 |       2651.4 |        1346 | True       |                   1614.2 |             2672.2 |              1021 |      0.99 | L31 d1792 {'d_inner': 8320, 'n_head': 28, 'n_kv_head': 4}  |
| Phi-2          |             2516.7 |       3991.2 |        2097 | True       |                   2549.5 |             3893.9 |              1514 |      1.02 | L20 d2560 {'d_inner': 14720, 'n_head': 20, 'n_kv_head': 2} |
| Qwen2.5-3B     |             2774.7 |       4304.7 |        1658 | True       |                   2812.9 |             4307   |              1723 |      1    | L29 d2048 {'d_inner': 14080, 'n_head': 32, 'n_kv_head': 8} |
| Llama-3.2-3B   |             2818.7 |       4359.2 |        1851 | True       |                   2968.4 |             4553.5 |              1727 |      0.96 | L33 d2304 {'d_inner': 11392, 'n_head': 18, 'n_kv_head': 1} |

## slm on snapdragon_8gen1_cpu

HV/IGD use the configured objectives ['quality', 'latency', 'memory']; HV₂/IGD₂ re-score the same runs on (quality, latency) only.

| Variant | latency overhead↓ | coverage | HV↑ | IGD↓ | HV₂↑ | IGD₂↓ | front | valid proposals | API calls | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| llmnas | +10.0% ± 5.7% | 94% | 0.983 ± 0.005 | 0.021 ± 0.004 | 0.983 ± 0.005 | 0.016 ± 0.005 | 126.7 | 49% | 120 | 3 |
| llmnas_survey_kb | +6.5% ± 2.0% | 94% | 0.986 ± 0.004 | 0.019 ± 0.004 | 0.984 ± 0.005 | 0.014 ± 0.004 | 127.0 | 49% | 120 | 3 |
| no_partitioning | +11.5% ± 2.0% | 88% | 0.966 ± 0.010 | 0.046 ± 0.016 | 0.959 ± 0.016 | 0.043 ± 0.019 | 95.0 | 43% | 20 | 3 |
| lts_search | +6.4% ± 2.8% | 92% | 0.983 ± 0.001 | 0.023 ± 0.002 | 0.977 ± 0.003 | 0.020 ± 0.002 | 185.7 | 94% | 0 | 3 |
| no_llm_pea | +6.1% ± 1.4% | 98% | 0.991 ± 0.002 | 0.011 ± 0.002 | 0.993 ± 0.003 | 0.006 ± 0.002 | 223.3 | 98% | 0 | 3 |
| random_sampling | +5.3% ± 1.0% | 96% | 0.992 ± 0.001 | 0.010 ± 0.002 | 0.991 ± 0.004 | 0.007 ± 0.003 | 199.7 | 100% | 0 | 3 |

Latency overhead per niche (mean over seeds; blank = level not reached):

| Variant | S0 <250M | S1 250-500M | S2 0.5-1B | S3 1-2B | S4 2-3B | S5 >=3B |
|---|---|---|---|---|---|---|
| llmnas | +18.0% | +6.0% | +5.5% | +4.5% | +3.0% | +3.2% |
| llmnas_survey_kb | +9.5% | +5.8% | +6.1% | +3.8% | +3.1% | +2.5% |
| no_partitioning | +16.8% | +5.3% | +10.2% | +7.5% | +8.5% | +6.4% |
| lts_search | +12.0% | +2.0% | +3.2% | +0.9% | +2.1% | +3.5% |
| no_llm_pea | +10.7% | +3.6% | +3.9% | +2.4% | +1.4% | +2.9% |
| random_sampling | +9.7% | +4.3% | +1.5% | +1.9% | +1.0% | +2.0% |

**Hand-designed SLMs vs the fastest LLM-NAS model with at least as many decoder parameters** (same cost model and vocabulary):

| model          |   decoder_params_m |   latency_ms |   memory_mb | feasible   |   found_decoder_params_m |   found_latency_ms |   found_memory_mb |   speedup | found_arch                                                 |
|:---------------|-------------------:|-------------:|------------:|:-----------|-------------------------:|-------------------:|------------------:|----------:|:-----------------------------------------------------------|
| SmolLM-135M    |              106.2 |        595.9 |         174 | True       |                    108.1 |              498.6 |               142 |      1.2  | L10 d640 {'d_inner': 5120, 'n_head': 10, 'n_kv_head': 2}   |
| SmolLM-360M    |              314.6 |       1223.7 |         328 | True       |                    331.8 |             1195.5 |               270 |      1.02 | L15 d1280 {'d_inner': 4864, 'n_head': 20, 'n_kv_head': 1}  |
| Qwen2.5-0.5B   |              357.9 |       1294.5 |         294 | True       |                    362.5 |             1288.5 |               308 |      1    | L10 d1792 {'d_inner': 5376, 'n_head': 14, 'n_kv_head': 2}  |
| TinyLlama-1.1B |              969   |       3086.1 |         661 | True       |                    992.9 |             3147.3 |               647 |      0.98 | L30 d1280 {'d_inner': 7680, 'n_head': 20, 'n_kv_head': 2}  |
| Llama-3.2-1B   |              973.1 |       3069   |         684 | True       |                    992.9 |             3147.3 |               647 |      0.98 | L30 d1280 {'d_inner': 7680, 'n_head': 20, 'n_kv_head': 2}  |
| Pythia-1.4B    |             1208.1 |       3813.1 |        1130 | True       |                   1228.4 |             3765.3 |               767 |      1.01 | L15 d1792 {'d_inner': 13952, 'n_head': 28, 'n_kv_head': 2} |
| Qwen2.5-1.5B   |             1310.3 |       4054   |         848 | True       |                   1336   |             4093.7 |               830 |      0.99 | L18 d1920 {'d_inner': 11520, 'n_head': 30, 'n_kv_head': 2} |
| SmolLM-1.7B    |             1610.7 |       4953   |        1346 | True       |                   1637.8 |             4951   |               992 |      1    | L20 d1792 {'d_inner': 13952, 'n_head': 28, 'n_kv_head': 2} |
| Phi-2          |             2516.7 |       7620.7 |        2097 | True       |                   2532.4 |             7540.3 |              1512 |      1.01 | L30 d1792 {'d_inner': 14336, 'n_head': 14, 'n_kv_head': 2} |
| Qwen2.5-3B     |             2774.7 |       8274.3 |        1658 | True       |                   2787.1 |             8297.7 |              1710 |      1    | L27 d2560 {'d_inner': 11392, 'n_head': 20, 'n_kv_head': 4} |
| Llama-3.2-3B   |             2818.7 |       8437.2 |        1851 | True       |                   2818.8 |             8356.2 |              1643 |      1.01 | L20 d2816 {'d_inner': 14720, 'n_head': 44, 'n_kv_head': 2} |

## Per-run results

| variant          | space   | device               |   seed |     hv |    igd |   n_front |   n_evaluated |   valid_rate |   out_of_niche |   duplicate |   repaired |   api_calls |   time_s |
|:-----------------|:--------|:---------------------|-------:|-------:|-------:|----------:|--------------:|-------------:|---------------:|------------:|-----------:|------------:|---------:|
| llmnas           | slm     | jetson_orin_nx_16gb  |      0 | 0.987  | 0.0155 |       132 |           154 |       0.5254 |             33 |          78 |        115 |         120 | 11582.7  |
| llmnas           | slm     | jetson_orin_nx_16gb  |      1 | 0.9817 | 0.0182 |       122 |           148 |       0.5    |             29 |          88 |        128 |         120 | 11721.8  |
| llmnas           | slm     | jetson_orin_nx_16gb  |      2 | 0.9849 | 0.0191 |       118 |           131 |       0.428  |             33 |          94 |        147 |         120 | 10603.1  |
| llmnas_survey_kb | slm     | jetson_orin_nx_16gb  |      0 | 0.9849 | 0.0168 |       127 |           155 |       0.5297 |             19 |          91 |        127 |         120 | 13278.7  |
| llmnas_survey_kb | slm     | jetson_orin_nx_16gb  |      1 | 0.9771 | 0.0226 |       116 |           145 |       0.4792 |             31 |          94 |        119 |         120 | 11014.2  |
| llmnas_survey_kb | slm     | jetson_orin_nx_16gb  |      2 | 0.9871 | 0.0138 |       133 |           151 |       0.5216 |             29 |          79 |        135 |         120 | 11779.5  |
| no_partitioning  | slm     | jetson_orin_nx_16gb  |      0 | 0.9428 | 0.0855 |        82 |           134 |       0.5    |              0 |         104 |        125 |          20 |  7516.45 |
| no_partitioning  | slm     | jetson_orin_nx_16gb  |      1 | 0.9794 | 0.0275 |       116 |           164 |       0.5702 |              0 |         101 |        164 |          20 |  7162.26 |
| no_partitioning  | slm     | jetson_orin_nx_16gb  |      2 | 0.9772 | 0.0275 |       131 |           166 |       0.6182 |              0 |          82 |        135 |          20 |  6867.84 |
| lts_search       | slm     | jetson_orin_nx_16gb  |      0 | 0.9778 | 0.0327 |       182 |           250 |       0.9167 |              0 |          20 |          0 |           0 |     0.13 |
| lts_search       | slm     | jetson_orin_nx_16gb  |      1 | 0.9844 | 0.0186 |       196 |           258 |       0.95   |              0 |          12 |          0 |           0 |     0.13 |
| lts_search       | slm     | jetson_orin_nx_16gb  |      2 | 0.9884 | 0.0163 |       202 |           259 |       0.9542 |              0 |          11 |          0 |           0 |     0.14 |
| no_llm_pea       | slm     | jetson_orin_nx_16gb  |      0 | 0.9919 | 0.0095 |       190 |           265 |       0.9792 |              0 |           5 |          0 |           0 |     0.14 |
| no_llm_pea       | slm     | jetson_orin_nx_16gb  |      1 | 0.9834 | 0.0176 |       203 |           265 |       0.9792 |              0 |           5 |          0 |           0 |     0.14 |
| no_llm_pea       | slm     | jetson_orin_nx_16gb  |      2 | 0.9913 | 0.0106 |       203 |           260 |       0.9583 |              0 |          10 |          0 |           0 |     0.15 |
| random_sampling  | slm     | jetson_orin_nx_16gb  |      0 | 0.9911 | 0.0108 |       170 |           270 |       1      |              0 |           0 |          0 |           0 |     0.21 |
| random_sampling  | slm     | jetson_orin_nx_16gb  |      1 | 0.9901 | 0.012  |       176 |           269 |       0.9958 |              0 |           1 |          0 |           0 |     0.21 |
| random_sampling  | slm     | jetson_orin_nx_16gb  |      2 | 0.9912 | 0.0106 |       193 |           270 |       1      |              0 |           0 |          0 |           0 |     0.21 |
| llmnas           | slm     | snapdragon_8gen1_cpu |      0 | 0.9863 | 0.0172 |       127 |           151 |       0.5042 |             26 |          88 |        140 |         120 | 12067.9  |
| llmnas           | slm     | snapdragon_8gen1_cpu |      1 | 0.9762 | 0.0272 |       115 |           132 |       0.425  |             24 |         107 |        108 |         120 | 11384.1  |
| llmnas           | slm     | snapdragon_8gen1_cpu |      2 | 0.9854 | 0.0192 |       138 |           162 |       0.55   |             48 |          53 |        146 |         120 | 10891.9  |
| llmnas_survey_kb | slm     | snapdragon_8gen1_cpu |      0 | 0.9888 | 0.0139 |       132 |           146 |       0.5    |             24 |          90 |        126 |         120 | 12137.6  |
| llmnas_survey_kb | slm     | snapdragon_8gen1_cpu |      1 | 0.9804 | 0.0224 |       124 |           142 |       0.4746 |             34 |          85 |        124 |         120 | 10690.7  |
| llmnas_survey_kb | slm     | snapdragon_8gen1_cpu |      2 | 0.9873 | 0.0201 |       125 |           145 |       0.4873 |             43 |          77 |        133 |         120 | 11916.5  |
| no_partitioning  | slm     | snapdragon_8gen1_cpu |      0 | 0.9527 | 0.0678 |       113 |           143 |       0.4768 |              0 |         124 |        171 |          20 |  6989.81 |
| no_partitioning  | slm     | snapdragon_8gen1_cpu |      1 | 0.967  | 0.0435 |        71 |            97 |       0.3284 |              0 |         137 |        149 |          20 |  5781.24 |
| no_partitioning  | slm     | snapdragon_8gen1_cpu |      2 | 0.9783 | 0.0282 |       101 |           139 |       0.4977 |              0 |         110 |         95 |          20 |  6164.58 |
| lts_search       | slm     | snapdragon_8gen1_cpu |      0 | 0.9824 | 0.0209 |       184 |           251 |       0.9208 |              0 |          19 |          0 |           0 |     0.14 |
| lts_search       | slm     | snapdragon_8gen1_cpu |      1 | 0.9849 | 0.0244 |       190 |           253 |       0.9292 |              0 |          17 |          0 |           0 |     0.12 |
| lts_search       | slm     | snapdragon_8gen1_cpu |      2 | 0.983  | 0.0239 |       183 |           260 |       0.9583 |              0 |          10 |          0 |           0 |     0.13 |
| no_llm_pea       | slm     | snapdragon_8gen1_cpu |      0 | 0.991  | 0.0115 |       219 |           266 |       0.9833 |              0 |           4 |          0 |           0 |     0.14 |
| no_llm_pea       | slm     | snapdragon_8gen1_cpu |      1 | 0.9894 | 0.013  |       227 |           267 |       0.9875 |              0 |           3 |          0 |           0 |     0.15 |
| no_llm_pea       | slm     | snapdragon_8gen1_cpu |      2 | 0.9936 | 0.0084 |       224 |           264 |       0.975  |              0 |           6 |          0 |           0 |     0.15 |
| random_sampling  | slm     | snapdragon_8gen1_cpu |      0 | 0.9918 | 0.0093 |       194 |           270 |       1      |              0 |           0 |          0 |           0 |     0.21 |
| random_sampling  | slm     | snapdragon_8gen1_cpu |      1 | 0.9902 | 0.0133 |       193 |           269 |       0.9958 |              0 |           1 |          0 |           0 |     0.23 |
| random_sampling  | slm     | snapdragon_8gen1_cpu |      2 | 0.9934 | 0.0088 |       212 |           270 |       1      |              0 |           0 |          0 |           0 |     0.22 |