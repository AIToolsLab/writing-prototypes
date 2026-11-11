# Writing engine

A small inference server for the writing prototypes, built on [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
library. It answers one question quickly and honestly: *given this context and the characters typed so far, what
characters are likely to come next?* The page uses it through the "Tokens" engine (`docs/engine-tokens.js`; called "Trie" before 2026-11-11).

## Build and run

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89   # fetches and builds a pinned llama.cpp
cmake --build build
./build/writing-engine --model /path/to/model.gguf --alias Qwen3.8-27B   # http://127.0.0.1:52409
```

Options: `--host`, `--port`, `--ctx` (context tokens, default 16384), `--seqs` (default 24), `--gpu-layers`.
The page uses https://llm.thoughtful-ai.com (Caddy, proxying to localhost:52409) by default; add `?engine-server=http://localhost:52409` to use a local one directly.

On the RTX 4090, Qwen3.8-27B (Q4_K_XL) takes about 21 GB with the defaults, so it can't run alongside the vLLM
server (`docker compose stop vllm` in `~/vllm-server` first). Qwen3.8's recurrent layers cost about 150 MB of
GPU memory per sequence, which is what limits `--seqs`.

## Docker

`docker compose up -d` here builds and starts the engine on `127.0.0.1:52409`, restarting it unless it's stopped
by hand. It reads the GGUF from `/storage/HuggingFace`, pinned to a revision in `compose.yaml`;
`./download-model.sh` fetches it if it's missing. It and vLLM (`~/vllm-server`) can't share the GPU, so stop one
before starting the other.

## Why it's fast

One model context stays warm, with a cache for each writer.

- **Reuse.** A request reuses the longest prefix it shares with the cached prompt and decodes only the rest.
- **Checkpoints.** Models with recurrent state (Qwen3.5 and later) can't be rewound token by token. So we park
  checkpoints: extra sequence ids at chosen points (every 256 tokens, and wherever a request ended), which share
  memory with the main sequence. Backspacing or editing earlier text restores the nearest one in under a millisecond.
- **Forks.** Each branch of the search is a sequence forked from its parent; it shares the whole prefix rather
  than copying it. One batched decode advances every branch.

Measured 2026-11-11 on the 4090 (`experiments/token-healing/trie_check.mjs` and a typing simulation):
50–200 ms per prediction, median 64 ms per keystroke at 700 tokens of context and 85 ms at 3,300. The vLLM path
took about 2 s per keystroke at 900 tokens, because that model's prefix cache only reuses 1568-token blocks.

## Several writers at once

Each writer has a *session*: a main sequence and its checkpoints. Name it with `session` in the request (the
page uses a random id per tab); unnamed requests are matched to the session whose text they continue.
Requests still run one at a time on the GPU.

What's scarce is sequences. The attention cache is one pool that all sequences share, but recurrent state costs
about 150 MB per sequence, so with the default 24 sequences, 12 are for sessions and 12 for search branches.
Sessions share theirs like this:

- A lone writer gets a main sequence and up to 11 checkpoints.
- As writers arrive, checkpoints are thinned, down to two each. A session keeps one checkpoint a little behind
  the text, so backspacing a word or two stays cheap.
- Beyond that (more than about four active writers, or documents that together overflow `--ctx`), the least
  recently used session moves to host memory and comes back when its writer types again. So does any session
  that has been idle for 32 requests.

Measured 2026-11-11 with `bench/sessions_check.py`, writers taking turns one keystroke at a time:

| Writers | Document | Median keystroke | Of which restoring from host memory |
|---|---|---|---|
| 3 | ~550 tokens | 82 ms | none needed |
| 16 | ~550 tokens | 177 ms | 62 ms, on every keystroke |
| 5 | ~4,400 tokens | 273 ms | 142 ms, on every keystroke |

Taking strict turns is the worst case for the last two rows; real writers pause, and a session that's still in
the GPU pays nothing. A restored session gives exactly the same probabilities as before it was moved out. It
loses its checkpoints, so a jump far back right after restoring decodes the prompt again (about 1.6 s for 4,400 tokens).

`GET /v1/sessions` lists the sessions and where they are.

## Why the probabilities are honest

The model predicts tokens; the writer typed characters. "Use de" tokenizes as `Use`, ` de`, and a model shown
` de` concludes the word isn't "deep" (which would have been the single token ` deep`).

So the engine treats the end of the draft as characters. It backs up to where the tokenization is settled (the
whitespace before the last word; at most 4 tokens), and from there sums over every token path that spells out
the typed characters and then continues. Reported probabilities are the model's own, conditioned on the typed
text. Nothing is renormalized over the candidates that happened to be kept; what was dropped is simply missing
from the list, and `unresolved` bounds the part of the typed text's probability that wasn't explored.

Probabilities are reproducible for a given cache history; the same request after a different history can differ
in the second decimal of a logit, because the GPU computes different batch shapes slightly differently.

## API

All bodies are JSON. `session` is optional everywhere it appears; see above.

**`POST /v1/continuations`** `{session?, prompt, text, stop?, max_chars?, min_prob?, max_rounds?, max_batch?, branch?, max_depth?, time_ms?}`

`prompt` is the templated context (special tokens are parsed); `text` is the draft so far. A continuation stops
growing when it matches the `stop` regex, reaches `max_chars` characters or `max_depth` tokens, or ends the turn.
Branches under `min_prob` (default 0.01) aren't followed.

```json
{"typed": " net", "prefix_logprob": -0.01, "unresolved": 4e-7,
 "continuations": [{"text": "works are", "prob": 0.45, "end": false, "done": true}, …],
 "stats": {"prompt_tokens": 50, "reused_tokens": 47, "decoded_tokens": 3, "rounds": 2, "total_ms": 85, …}}
```

Continuations are disjoint, so the probability that the text continues with some word is the sum over the
continuations that start with it (a lower bound, since unlikely branches were dropped). A newer request
from the same session supersedes its older ones, which return 409.

**`POST /v1/score`** `{session?, prompt, text}` → for each token of `text`: its `logprob`, and the model's `top` choice
there with `top_logprob`. Tokens that split a character are merged, so the `text` fields concatenate to `text`.

**`POST /v1/template`** `{messages, add_generation_prompt?, enable_thinking?}` → `{prompt}`, using the model's
chat template.

`GET /health`, `GET /v1/models`, `GET /v1/sessions`.

## Limits

- Requests run one at a time, so many people typing at once queue behind each other.
- The settled-tokenization rule is a conservative stand-in for the tokenizer's own chunking rule, capped at 4
  tokens; a word of 5 or more tokens is conditioned on its last 4.
- Scoring (highlights) decodes the whole document after an edit point, so it isn't faster than vLLM's.
