# Report: slm_smoke

3 finished runs. HV/IGD are relative to the pooled reference front of this experiment (see `slm/report.py`).

## slm on jetson_orin_nx_16gb

| Variant | HV↑ | IGD↓ | front | valid proposals | API calls | runs |
|---|---|---|---|---|---|---|
| llmnas | 0.993 | 0.011 | 83.0 | 97% | 36 | 1 |
| lts_search | 0.884 | 0.146 | 54.0 | 96% | 0 | 1 |
| no_llm_pea | 0.991 | 0.011 | 87.0 | 99% | 0 | 1 |

**Hand-designed SLMs vs the fastest LLM-NAS model with at least as many decoder parameters** (same cost model and vocabulary):

| model          |   decoder_params_m |   latency_ms |   memory_mb | feasible   |   found_decoder_params_m |   found_latency_ms |   found_memory_mb |   speedup | found_arch                                                      |
|:---------------|-------------------:|-------------:|------------:|:-----------|-------------------------:|-------------------:|------------------:|----------:|:----------------------------------------------------------------|
| SmolLM-135M    |              106.2 |        555.3 |         174 | True       |                    110.1 |              418.4 |               148 |      1.33 | L8 d1280 {'d_inner': 2688, 'n_head': 20, 'n_kv_head': 1}        |
| SmolLM-360M    |              314.6 |        873.9 |         328 | True       |                    334   |              754.2 |               384 |      1.16 | L8 d1792 {'d_inner': 5376, 'n_head': 14, 'n_kv_head': 'n_head'} |
| Qwen2.5-0.5B   |              357.9 |        864.3 |         294 | True       |                    362.5 |              800.6 |               308 |      1.08 | L10 d1792 {'d_inner': 5376, 'n_head': 14, 'n_kv_head': 2}       |
| TinyLlama-1.1B |              969   |       1732.5 |         661 | True       |                   1150.2 |             1976.7 |               756 |      0.88 | L23 d1792 {'d_inner': 7936, 'n_head': 28, 'n_kv_head': 4}       |
| Llama-3.2-1B   |              973.1 |       1693.9 |         684 | True       |                   1150.2 |             1976.7 |               756 |      0.86 | L23 d1792 {'d_inner': 7936, 'n_head': 28, 'n_kv_head': 4}       |
| Pythia-1.4B    |             1208.1 |       2102.2 |        1130 | True       |                   1387.2 |             2288   |               919 |      0.92 | L17 d2560 {'d_inner': 8576, 'n_head': 20, 'n_kv_head': 4}       |
| Qwen2.5-1.5B   |             1310.3 |       2223.4 |         848 | True       |                   1387.2 |             2288   |               919 |      0.97 | L17 d2560 {'d_inner': 8576, 'n_head': 20, 'n_kv_head': 4}       |
| SmolLM-1.7B    |             1610.7 |       2651.4 |        1346 | True       |                   1636.9 |             2679.7 |               998 |      0.99 | L29 d1664 {'d_inner': 10112, 'n_head': 26, 'n_kv_head': 2}      |
| Phi-2          |             2516.7 |       3991.2 |        2097 | True       |                   2632.1 |             4100.5 |              1582 |      0.97 | L32 d2560 {'d_inner': 8832, 'n_head': 20, 'n_kv_head': 2}       |
| Qwen2.5-3B     |             2774.7 |       4304.7 |        1658 | True       |                   2782.5 |             4260.7 |              1648 |      1.01 | L29 d2048 {'d_inner': 14080, 'n_head': 32, 'n_kv_head': 4}      |
| Llama-3.2-3B   |             2818.7 |       4359.2 |        1851 | True       |                   3084.5 |             4753.7 |              2048 |      0.92 | L37 d2048 {'d_inner': 11520, 'n_head': 16, 'n_kv_head': 8}      |

## Per-run results

| variant    | space   | device              |   seed |     hv |    igd |   n_front |   n_evaluated |   valid_rate |   out_of_niche |   duplicate |   repaired |   api_calls |   time_s |
|:-----------|:--------|:--------------------|-------:|-------:|-------:|----------:|--------------:|-------------:|---------------:|------------:|-----------:|------------:|---------:|
| llmnas     | slm     | jetson_orin_nx_16gb |      0 | 0.993  | 0.0106 |        83 |           100 |       0.9722 |              0 |           2 |          0 |          36 |     0.09 |
| lts_search | slm     | jetson_orin_nx_16gb |      0 | 0.8844 | 0.1459 |        54 |            74 |       0.9583 |              0 |           3 |          0 |           0 |     0.03 |
| no_llm_pea | slm     | jetson_orin_nx_16gb |      0 | 0.9907 | 0.0115 |        87 |           101 |       0.9861 |              0 |           1 |          0 |           0 |     0.06 |