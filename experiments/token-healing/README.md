# Predicting continuations of a partly typed word

Experiment log for predictive-text mode in `docs/engine-tokens.js`. These notes cover the
problem, each approach we tried, what it gets right and wrong, and the API we wish we had.

- `healing_options.py mechanism` demonstrates the two server behaviors that shape everything below.
- `healing_options.py compare [drafts...]` runs every option on the same drafts and checks the top suggestion.
- `engine_check.mjs [drafts...]` runs the app's actual engine (`docs/engine-tokens.js`) in Node
  on the same drafts, including the second request that finishes words.
- `cases.json` holds the drafts and expectations, shared by both scripts.
- Snapshots, all from Qwen/Qwen3.8-27B on vllm.thoughtful-ai.com on 2026-10-28:
  - `results-2026-10-28-masked.txt`: stock vLLM, where logprobs are reported after the grammar mask.
  - `results-2026-10-28-raw.txt`: vLLM patched to report raw logprobs under a grammar.
  - `engine-2026-10-28-raw.txt`: `engine_check.mjs` against the patched server.

The scripts use only the standard library (Python) or Node built-ins, and need no browser. Each
option is a short function with a docstring stating its tradeoff.

## The problem

The draft is the start of the assistant's reply, and we want the next few words. Two facts make that hard:

1. **A typed token implies the word ends there.** "Use de" tokenizes as `Use` + ` de`. Because
   ` deep` is its own token, a model that sees ` de` infers the word *isn't* "deep" and predicts
   "depleting". Likewise, after ` net` it predicts " advancements", not "works". Continuing
   from the full draft (`naive`) gets the top suggestion right for only 3 of 11 drafts.

2. **Healing with a regex constraint fixes the text, not its tokenization.** The committed fix
   (`healed_only`, commit 2826b23) backs up to the last pre-tokenizer boundary. It then constrains
   the output to start with the backed-over text (`structured_outputs: {regex: "Used[\s\S]*"}`)
   and lets the model choose the tokens. But greedy decoding under the constraint picks each token
   without looking ahead. For typed "Used" it picks `Use`, because the model was heading for "Use
   deep learning…". It then has to produce a `d`, so it takes `deep`, and every branch after that
   is "Usedeep …". Meanwhile `Used` itself is the 62nd most likely token there (logprob −8.65).

   Backing up to the pre-token boundary (rather than one token, as in classic token healing)
   already covers words whose typed part spans several tokens: BPE merges never cross pre-token
   boundaries. So the problem isn't how far to back up. It's that the constrained search can wander.

The fix for (2) is to also continue from the full draft, and rank both sets of branches by joint
probability (`merge`). That needs **raw logprobs under a grammar**, and stock vLLM doesn't give
them: it reports logprobs after the grammar mask, so the `Use` → `deep` path reports ≈ 0 for both
tokens and looks certain. `--logprobs-mode raw_logprobs` doesn't help, because the grammar mask is
applied before the sampler code that the flag controls. `logit_bias` *is* applied in the sampler,
so end tokens still appear in the "raw" alternatives. **Our server is now patched to report raw
logprobs under a grammar.** Then `Use` scores −0.96 and `deep` −12.94, so the wandering path
totals ≈ −13.9, below "Used ethically" at ≈ −9.

## Options

Scores: how many of 11 drafts got a top suggestion of the expected kind, under each server.

| option | requests (all in parallel) | stock | patched | strengths | weaknesses |
|---|---|---|---|---|---|
| `naive` | unhealed spine | 3 | 3 | simple; right when the draft ends at a word boundary | wrong whenever the draft ends mid-word |
| `healed_only` (commit 2826b23) | healed spine | 10 | 10 | handles mid-word and multi-token words ("revolutionizi" → "ng") | wanders off the usual tokenization ("Used" → "Usedeep") |
| **`merge`** (current `docs/engine-tokens.js`) | healed + unhealed spines, ranked by joint probability | 10 | **11** | simple; every score a raw joint probability; covers both mid-word and word-boundary drafts | needs patched vLLM: with masked logprobs, wandering healed paths score ≈ 1 and win. A wandering path can still fill a low slot ("Used" → 5th suggestion "eep learning", −13.9) |
| `continuation_mass` | healed + unhealed; healed weighted by the unhealed request's P(word continues) | 10 | 8 | fixes "Used" | P(word continues) comes from the very conditioning that fact 1 says is biased: after ` net` it's 0.05, so "net advancements" beats "networks" |
| `boundary_topk` | unhealed spine + raw top-20 at the boundary | 9 | 9 | works with stock vLLM; every score is a raw joint probability | mid-word branches are one token deep (little variety); misses words whose covering token isn't in the top 20 ("questionab") or that need several tokens ("revolutionizi") |
| `hybrid` | all three | 9 | 9 | like `boundary_topk`, plus the healed spine's later steps for variety, used only if its first token covers the typed text and is in the top 20 | same misses as `boundary_topk` |
| `hybrid_bounded` | all three | 10 | 10 | like `hybrid`, but a covering first token outside the top 20 is scored with the 20th logprob (an upper bound), which recovers "questionable" | overrates such tokens; still misses multi-token words ("revolutionizi" → "revolutioniziing") |

Failures with stock vLLM: `healed_only` and `merge` fail "Used"; `continuation_mass` fails
"Deep neural net"; `boundary_topk` and `hybrid` fail "questionab" and "revolutionizi";
`hybrid_bounded` fails "revolutionizi". With the patch, `merge` passes everything, and
`continuation_mass` also fails "questionab" and "revolutionizi". `naive` fails every draft that
ends mid-word in both.

**Decision:** `merge`, on the patched server. In the patched results its margins are wide: 4.6
nats on "Used", 12 on "questionab", 16 on "revolutionizi". If we have to run on stock vLLM,
`hybrid_bounded` is the best of the workarounds tried, though the cases are hand-picked.

### Ideas not tried

- **Drop branches that leave the usual tokenization.** `/tokenize` the healed spine's text
  (cheap, but sequential) and drop it if its tokens differ ("Used" → `Used`, not `Use` + `deep`).
  With raw logprobs this would only remove the low-ranked junk that `merge` still shows.
- **Restrict only the first step with `allowed_token_ids`** (classic token healing). vLLM applies
  `allowed_token_ids` at every step, and it needs the vocabulary on the client.

## The ideal API

What we're really asking for: given a context and a typed text prefix (characters, not tokens),
the k most likely distinct continuations up to a word boundary, with probabilities conditioned on
the *text*, regardless of how it's tokenized. The model is a distribution over token sequences,
and the user typed characters. Everything above is a workaround for that mismatch.

One endpoint would do it all server-side, sharing the KV cache across branches:

```jsonc
POST /v1/continuations
{
  "prompt": "<|im_start|>user\n…<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "text_prefix": "Deep neural net",          // characters; the server handles token boundaries
  "branches": 5,                              // distinct continuations to return
  "branch_until": {"regex": "^\\s*\\S+(\\s+\\S+){1}(?=\\s)"},  // e.g. two whole words
  "logit_bias": {"248044": -100}
}
→ {
  "text_prefix_logprob": -8.1,                // log P(reply starts with text_prefix), summed over tokenizations
  "continuations": [
    {"text": "works are", "logprob": -0.5},   // log P(text | context, text_prefix)
    {"text": "work technology", "logprob": -0.9},
    …
  ]
}
```

The semantics that matter:

1. **Condition on characters.** P(continuation | context, the reply starts with
   `text_prefix`), summed over every tokenization of the prefix, or at least every tokenization
   that is the usual one for some completion of it. This is the exact form of token healing. It
   has a published treatment: converting token-level LMs into character- or byte-level ones by
   summing over the tokenizations that cover a prefix (e.g. Vieira et al., 2024, "From Language
   Models over Tokens to Language Models over Characters"; Phan et al., 2024, on exact byte-level
   probabilities). Within a pre-token chunk, the set of candidate tokenizations is small.
2. **Branch deterministically, with joint probabilities.** Fork at the top alternatives (not
   random samples, as `n` gives), extend each greedily until `branch_until`, and drop duplicate
   texts. vLLM's beam search is close, but it optimizes for likely whole paths, not distinct next
   words.
3. **Report true probabilities.** Every logprob is the model's own, never renormalized by a
   constraint or bias.

### Smaller additions to vLLM, by how much they'd simplify our code

1. **Logprobs taken before *any* masking**, grammar included, ideally per request. Our patched
   server does this, and it's what makes `merge` work. It doesn't stop the wandering, but the
   wandering paths score low.
2. **A lookahead-aware constraint for text prefixes**: when choosing each token, weight it by
   whether the rest of the prefix can still be completed with the usual tokenization. A
   `text_prefix` request option could do this, implemented as server-side token healing.
3. **Logprobs for specific token ids** (e.g. `logprobs_for: [ids]`), not just the top k. On stock
   vLLM, this would fix the "covering token not in the top 20" miss without the upper-bound guess.
4. **Deterministic top-k forking plus a per-branch stop regex.** This replaces our second request,
   the batched greedy extension of unfinished branches to a word boundary.
5. **`allowed_token_ids` for the first step only.** This gives one-token token healing in one
   request, though the client would need the vocabulary.
