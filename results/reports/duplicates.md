# Where duplicate LLM proposals come from

Replayed from the archived prompts and replies (no LLM called). Shares are of all proposals.

| Model | proposals | new | duplicate | out_of_niche | no_latency | invalid |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 1432 | 23.3% | 65.6% | 9.3% | 1.5% | 0.3% |
| claude-opus-5-5 | 240 | 80.0% | 20.0% | 0.0% | 0.0% | 0.0% |
| deepseek-coder-v2-16b-8k | 1440 | 19.0% | 57.1% | 22.4% | 0.6% | 0.9% |
| gpt-4.1 | 1440 | 47.9% | 42.7% | 8.3% | 1.1% | 0.0% |
| qwen2.5-7b-8k | 1440 | 38.8% | 28.1% | 32.1% | 1.0% | 0.1% |

| Model | proposals | duplicates | parent shown | last cycle shown | sibling | hidden history |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 1432 | 65.6% | 32.5% | 3.8% | 0.6% | 28.6% |
| claude-opus-5-5 | 240 | 20.0% | 0.0% | 0.0% | 0.0% | 20.0% |
| deepseek-coder-v2-16b-8k | 1440 | 57.1% | 41.1% | 12.6% | 0.3% | 3.1% |
| gpt-4.1 | 1440 | 42.7% | 5.8% | 2.8% | 1.7% | 32.3% |
| qwen2.5-7b-8k | 1440 | 28.1% | 11.7% | 6.1% | 0.5% | 9.7% |

## Duplicate rate by niche (all models)

| Niche | size | proposals | duplicate rate | mean share of niche already evaluated |
|---|---|---|---|---|
| S0 | 729 | 998 | 53.2% | 1.8% |
| S1 | 3367 | 1000 | 52.7% | 0.4% |
| S2 | 6144 | 1000 | 52.4% | 0.2% |
| S3 | 3840 | 1000 | 46.0% | 0.3% |
| S4 | 1280 | 1000 | 39.3% | 0.9% |
| S5 | 265 | 994 | 39.5% | 4.5% |

## Duplicate rate by requested operation

| Model | mutation | crossover |
|---|---|---|
| claude-haiku-4-5 | 57.7% | 70.3% |
| claude-opus-5-5 | 22.9% | 16.0% |
| deepseek-coder-v2-16b-8k | 53.3% | 60.8% |
| gpt-4.1 | 39.6% | 46.9% |
| qwen2.5-7b-8k | 22.7% | 36.0% |

## Duplicate rate by generation (all models)

| Generation | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| duplicate rate | 20% | 26% | 33% | 46% | 50% | 57% | 58% | 56% | 64% | 63% |
