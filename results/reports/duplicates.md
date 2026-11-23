# Where duplicate LLM proposals come from

Replayed from the archived prompts and replies (no LLM called). Shares are of all proposals.

| Model | proposals | new | duplicate | out_of_niche | no_latency | invalid |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 1432 | 23.3% | 65.6% | 9.3% | 1.5% | 0.3% |
| claude-opus-5-5 | 240 | 80.0% | 20.0% | 0.0% | 0.0% | 0.0% |
| deepseek-coder-v2-16b-8k | 4320 | 13.7% | 73.5% | 11.9% | 0.6% | 0.3% |
| gpt-4.1 | 1440 | 47.9% | 42.7% | 8.3% | 1.1% | 0.0% |
| gpt-4.1 [history + re-prompt] | 1440 | 60.0% | 29.7% | 8.1% | 2.3% | 0.0% |
| gpt-4.1 [history] | 1440 | 55.6% | 35.9% | 7.1% | 1.4% | 0.0% |
| qwen2.5-7b-8k | 4320 | 35.2% | 42.1% | 22.3% | 0.4% | 0.0% |

| Model | proposals | duplicates | parent shown | last cycle shown | sibling | hidden history |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 1432 | 65.6% | 32.5% | 3.8% | 0.6% | 28.6% |
| claude-opus-5-5 | 240 | 20.0% | 0.0% | 0.0% | 0.0% | 20.0% |
| deepseek-coder-v2-16b-8k | 4320 | 73.5% | 58.4% | 11.4% | 0.2% | 3.5% |
| gpt-4.1 | 1440 | 42.7% | 5.8% | 2.8% | 1.7% | 32.3% |
| gpt-4.1 [history + re-prompt] | 1440 | 29.7% | 2.0% | 2.4% | 3.4% | 21.8% |
| gpt-4.1 [history] | 1440 | 35.9% | 3.1% | 2.3% | 2.8% | 27.8% |
| qwen2.5-7b-8k | 4320 | 42.1% | 19.6% | 7.2% | 0.4% | 14.9% |

## Duplicate rate by niche (all models, published method only)

| Niche | size | proposals | duplicate rate | mean share of niche already evaluated |
|---|---|---|---|---|
| S0 | 729 | 1958 | 59.4% | 1.7% |
| S1 | 3367 | 1960 | 60.9% | 0.4% |
| S2 | 6144 | 1960 | 59.2% | 0.2% |
| S3 | 3840 | 1960 | 55.2% | 0.3% |
| S4 | 1280 | 1960 | 50.3% | 0.8% |
| S5 | 265 | 1954 | 51.9% | 4.2% |

## Duplicate rate by requested operation

| Model | mutation | crossover |
|---|---|---|
| claude-haiku-4-5 | 57.7% | 70.3% |
| claude-opus-5-5 | 22.9% | 16.0% |
| deepseek-coder-v2-16b-8k | 67.9% | 76.6% |
| gpt-4.1 | 39.6% | 46.9% |
| gpt-4.1 [history + re-prompt] | 27.3% | 32.4% |
| gpt-4.1 [history] | 34.0% | 38.3% |
| qwen2.5-7b-8k | 32.5% | 56.5% |

## Duplicate rate by generation

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| all models, published method | 37% | 37% | 47% | 54% | 57% | 62% | 66% | 64% | 69% | 68% |
| claude-haiku-4-5 | 5% | 31% | 49% | 75% | 74% | 85% | 87% | 83% | 85% | 83% |
| claude-opus-5-5 | 0% | 0% | 0% | 4% | 12% | 17% | 38% | 29% | 58% | 42% |
| deepseek-coder-v2-16b-8k | 87% | 59% | 69% | 71% | 73% | 77% | 77% | 72% | 76% | 75% |
| gpt-4.1 | 3% | 14% | 23% | 34% | 44% | 55% | 58% | 65% | 66% | 65% |
| gpt-4.1 [history + re-prompt] | 1% | 9% | 19% | 25% | 33% | 31% | 37% | 42% | 47% | 53% |
| gpt-4.1 [history] | 2% | 7% | 13% | 30% | 40% | 42% | 51% | 53% | 59% | 61% |
| qwen2.5-7b-8k | 12% | 26% | 35% | 40% | 43% | 44% | 52% | 53% | 57% | 59% |
