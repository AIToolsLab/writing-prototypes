# Prediction latency log

Written by `predict_bench.py`: server time per keystroke (ms), typing a 73-character
sentence with one typo into the middle of a response. Target: under 200 ms at the page.
Commit is the working tree's (`+`: server.cpp uncommitted), which is only the running server's if the container was rebuilt from it. "words" counts distinct first words among all continuations returned, partial ones included.

| date | commit | model | request extras | note | prompt tokens | median | p90 | max | words |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | a64538f | Qwen3.8-27B@4ca7207 | `{}` | before units (old behavior) | 148 | 60 | 115 | 185 | 6 |
| 2026-11-11 | a64538f | Qwen3.8-27B@4ca7207 | `{}` | before units (old behavior) | 688 | 65 | 92 | 185 | 7 |
| 2026-11-11 | a64538f | Qwen3.8-27B@4ca7207 | `{"units": 5}` | units=5 (a64538f) | 148 | 87 | 128 | 204 | 19 |
| 2026-11-11 | a64538f | Qwen3.8-27B@4ca7207 | `{"units": 5}` | units=5 (a64538f) | 688 | 91 | 144 | 191 | 19 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `{"units": 5, "stop": "words"}` | units=5, new stop pattern | 148 | 87 | 126 | 193 | 20.5 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `{"units": 5, "stop": "words"}` | units=5, new stop pattern | 688 | 90 | 142 | 210 | 19.5 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `{"units": 5, "corrections": 1, "stop": "words"}` | units=5, corrections=1, new stop pattern | 148 | 109 | 212 | 460 | 25 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `{"units": 5, "corrections": 1, "stop": "words"}` | units=5, corrections=1, new stop pattern | 688 | 104 | 351 | 497 | 23 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `page` | corrections capped at 60 ms | 148 | 105 | 190 | 362 | 25 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `page` | corrections capped at 60 ms | 688 | 104 | 221 | 348 | 23 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `page` | units and corrections share batches | 148 | 96 | 190 | 282 | 24.5 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `page` | units and corrections share batches | 688 | 100 | 198 | 315 | 23 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `page + {"time_ms": 150}` | time_ms 150 | 148 | 97 | 165 | 208 | 23.5 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@4ca7207 | `page + {"time_ms": 150}` | time_ms 150 | 688 | 101 | 164 | 198 | 23 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@ce331ef | `page` | Unsloth Dynamic 3.1 quant | 148 | 98 | 174 | 232 | 23.5 |
| 2026-11-11 | 85d3618+ | Qwen3.8-27B@ce331ef | `page` | Unsloth Dynamic 3.1 quant | 688 | 99 | 171 | 202 | 22 |
| 2026-11-11 | b56f8e6 | Qwen3.8-27B@ce331ef | `page` | punctuation inside words; markup chars end words | 148 | 116 | 180 | 232 | 23 |
| 2026-11-11 | b56f8e6 | Qwen3.8-27B@ce331ef | `page` | punctuation inside words; markup chars end words | 688 | 104 | 170 | 205 | 22 |
