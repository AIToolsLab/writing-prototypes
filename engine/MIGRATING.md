# Moving from the vLLM approach to the writing engine

For people (and their coding agents) who built on `docs/engine-tokens.js` and a vLLM server. This explains
what changed and why, and how old code maps to new. [README.md](README.md) has the build steps and full API;
[`docs/engine-trie.js`](../docs/engine-trie.js) is a complete working client in about 150 lines.

## What was wrong with the old way

**Speed.** Every keystroke sent the whole prompt to vLLM three times (two "spines", then a batch that finished
the words). vLLM's prefix cache was supposed to make that cheap, but for Qwen3.5 and later it only reuses
1568-token blocks, because those models have recurrent layers whose state can't be rewound. A one-page document
never hit the cache: about 2 s per keystroke.

**Correctness.** The model predicts tokens; the writer types characters. "Use de" is the tokens `Use`, ` de`,
and a model that sees ` de` infers the word isn't "deep" (that would have been the single token ` deep`) and
predicts "depleting". Token healing (back up, constrain the output with a regex, merge two searches by
probability) fixed most cases, but needed a patched vLLM and still let odd tokenizations through.

## The new strategy

We run our own small server on llama.cpp's library, holding one warm model context.

1. **Keep the cache and rewind it ourselves.** A request reuses the longest prefix it shares with what's cached.
   To allow rewinding past recurrent state, the server parks *checkpoints*: extra sequence ids at chosen points
   that share memory with the main sequence. Backspacing restores one in under a millisecond.
2. **Fork instead of resending.** Each search branch is a sequence forked from its parent, sharing the prefix.
   One batched model call advances all branches.
3. **Condition on characters, not tokens.** The server backs up to the whitespace before the last word and
   treats everything after as characters. It sums over every token path that spells those characters and then
   continues. This replaces token healing entirely; the client does none.
4. **Report the model's own probabilities.** Nothing is renormalized over the candidates kept. Dropped
   branches are simply missing, so sums over the list are lower bounds.
5. **Answer in characters, group in the client.** The server returns continuations as text. The client decides
   what a "unit" is (a word, two words, a segment in a script without spaces) and adds up the continuations
   that start with it.

Result on one RTX 4090 with Qwen3.8-27B: 50–200 ms per prediction, and all 11 token-healing test cases pass.

## How old code maps to new

| Old (vLLM) | New (engine) |
|---|---|
| `/tokenize` + `/detokenize` to render the chat prefix | `POST /v1/template {messages}` → `{prompt}` |
| `/v1/completions` with `logprobs`, a `structured_outputs` regex, pre-token splitting, a second batched request | `POST /v1/continuations {prompt, text, stop}` |
| `/v1/completions` with `prompt_logprobs` (highlights) | `POST /v1/score {prompt, text}` |
| `PRETOKENIZE_REGEX`, `splitLastPretoken`, `spineBranches`, `noEndBias` | delete; the server does this |
| Chat completions, JSON schema, `choice`, free generation | not in the engine; keep using vLLM for these |

## The API in brief

`prompt` is the templated context from `/v1/template`; `text` is what the writer has typed after it.

**`POST /v1/continuations`** `{prompt, text, stop}` where `stop` is a regex saying when a continuation is long
enough, e.g. `^\s*\S+(?=\s)` for one word.

```json
{"typed": " net",
 "continuations": [{"text": "works are", "prob": 0.45, "end": false, "done": true},
                   {"text": "work technology", "prob": 0.11, "end": false, "done": true}],
 "unresolved": 4e-7, "stats": {"decoded_tokens": 3, "total_ms": 85}}
```

- `text` in each continuation is what to append to the draft. `prob` is P(the reply continues this way | the
  characters typed).
- Continuations are disjoint. P(next word is "works") = the sum of `prob` over continuations starting with
  "works" followed by a space.
- `end: true` means the model would end the message after that text.
- For scripts without spaces, also send `max_chars` and cut units with `Intl.Segmenter`.
- A newer request supersedes older ones, which return 409; treat that like an abort.

**`POST /v1/score`** `{prompt, text}` → `{tokens: [{text, logprob, top, top_logprob}]}`. One item per token of
`text`: how likely it was, and what the model expected most there. The `text` fields concatenate to the input.

**`POST /v1/template`** `{messages, add_generation_prompt}` → `{prompt}`. For a turn with no generation prompt
(writing as the user), render a placeholder message and cut the prompt there; see `triePrefix`.

## Things to know

- **One writer at a time.** There is one cache. Two people typing at once will each evict the other's.
- **It can't share the GPU with vLLM** on the 4090; one or the other runs.
- **Order prompts so the changing part is last.** Anything before an edit is reused; everything after is recomputed.
- **Send requests as `Content-Type: text/plain`** from a browser, to skip the CORS preflight on every keystroke.
- **Check your work with** `node experiments/token-healing/trie_check.mjs "your draft"`.
