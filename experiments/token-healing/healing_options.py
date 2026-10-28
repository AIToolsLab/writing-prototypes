"""Compare ways of predicting continuations of a partly typed draft (see README.md).

    python3 healing_options.py mechanism    # the two server behaviors that shape the options
    python3 healing_options.py compare      # every option on every case, with pass/fail
    python3 healing_options.py compare "Deep neural net" "Use de"   # just these drafts

Talks to a vLLM server (override with VLLM_SERVER=https://...). Whether its logprobs are raw or
taken after the grammar mask changes the results; see README.md. Standard library only.
Each option only builds and ranks the "branches" (continuations past the typed text); the
app's second request, which extends branches to finish their words, is the same for all of
them and is left out.
"""

import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SERVER = os.environ.get("VLLM_SERVER", "https://vllm.thoughtful-ai.com")
END_BIAS = {"248044": -100, "248046": -100}  # <|im_end|>, <|endoftext|> for Qwen3.5-family tokenizers

DOC = ("Deep learning neural network technology advances are pretty cool if you are careful "
       "to use it in ways that don't take stuff from people.")
PROMPT = ("<|im_start|>user\nRewrite this document to be clear and concise.\n\n" + DOC +
          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def post(path, body):
    req = urllib.request.Request(SERVER + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


MODEL = None


def complete(**body):
    global MODEL
    if MODEL is None:
        MODEL = json.load(urllib.request.urlopen(SERVER + "/v1/models", timeout=30))["data"][0]["id"]
    return post("/v1/completions", {"model": MODEL, "temperature": 0, "logit_bias": END_BIAS, **body})["choices"][0]


def parallel(*calls):
    with ThreadPoolExecutor() as ex:
        return [f.result() for f in [ex.submit(lambda c=c: c and complete(**c)) for c in calls]]


def split_last_pretoken(draft):
    """[text before the last word-ish pre-token, that pre-token]. An ASCII approximation of the
    Qwen pre-tokenizer split in docs/engine-tokens.js; enough for these cases."""
    m = re.search(r"[^\r\n\w]?[^\W\d_]+$", draft)
    return (draft[:m.start()], draft[m.start():]) if m else (draft, "")


def starts_with(text):
    return {"structured_outputs": {"regex": re.escape(text) + r"[\s\S]*"}}


SPECIAL = re.compile(r"^<\|.*\|>$")  # logprobs are raw, so end tokens show up despite END_BIAS


def spine_branches(logprobs, typed, path="", path_lp=0.0, skip=0):
    """Each top alternative at each step of a greedy path, appended to the path so far.
    Keeps those that get past the typed text to something visible."""
    out = []
    toks, lps, tops = logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"]
    for j in range(skip, len(toks)):
        for tok, lp in tops[j].items():
            forced = path + tok
            if not SPECIAL.match(tok) and forced.startswith(typed) and forced[len(typed):].strip():
                out.append((forced, path_lp + lp))
        path += toks[j]
        path_lp += lps[j]
    return out


def typed_logprob(unhealed, n_base):
    """log P(typed tokens | base), from the unhealed request's prompt logprobs."""
    ids = unhealed["prompt_token_ids"]
    return sum(unhealed["prompt_logprobs"][i][str(ids[i])]["logprob"] for i in range(n_base, len(ids)))


# ── The options ────────────────────────────────────────────────────────────
# Each takes (base, typed, requests) and returns [(text including typed, logprob)]. `requests`
# holds every request any option needs, made once per draft in parallel:
#   unhealed: prompt ends with the whole draft; greedy spine; prompt_logprobs for P(typed)
#   healed:   prompt ends at the boundary; greedy spine constrained to start with typed
#   boundary: prompt ends at the boundary; one unconstrained step, top-20 alternatives

def naive(base, typed, r):
    """No healing: continue from the whole draft. Wrong mid-word: the typed tokens imply the
    word ends there ("Use de" → " de" + "pleting")."""
    return spine_branches(r["unhealed"]["logprobs"], typed, typed, 0)


def healed_only(base, typed, r):
    """The committed version: back up to the pre-token boundary, constrain output to start with
    typed. Greedy under the constraint can pick a non-canonical tokenization ("Used" →
    "Use" + "deep") and every branch after that is junk; masked logprobs hide it."""
    if not typed:
        return naive(base, typed, r)
    return spine_branches(r["healed"]["logprobs"], typed)


def merge(base, typed, r):
    """Healed + unhealed spines, merged by joint probability. What docs/engine-tokens.js does.
    Correct only if logprobs are raw (from before the grammar mask); stock vLLM reports them after
    the mask, so wandering healed paths ("Use" + "deep") score ≈ 1 and win."""
    if not typed:
        return naive(base, typed, r)
    n_base = len(r["boundary"]["prompt_token_ids"])
    return (spine_branches(r["unhealed"]["logprobs"], typed, typed, typed_logprob(r["unhealed"], n_base)) +
            spine_branches(r["healed"]["logprobs"], typed))


def continuation_mass(base, typed, r):
    """Unhealed new-word branches, plus healed branches weighted by the unhealed request's
    probability that the word continues. Fails when the typed tokens themselves imply a word
    end ("net" → P(continue) ≈ 0.05, so "networks" loses to "net advancements")."""
    if not typed:
        return naive(base, typed, r)
    n_base = len(r["boundary"]["prompt_token_ids"])
    t_lp = typed_logprob(r["unhealed"], n_base)
    first = r["unhealed"]["logprobs"]["top_logprobs"][0]
    p_new = sum(math.exp(lp) for tok, lp in first.items() if not re.match(r"\w", tok))
    p_cont = max(1 - p_new, 1e-9)
    unhealed = [b for b in spine_branches(r["unhealed"]["logprobs"], typed, typed, t_lp)
                if not re.match(r"\w", b[0][len(typed):])]
    healed = [(f, lp + math.log(p_cont) + t_lp) for f, lp in spine_branches(r["healed"]["logprobs"], typed)]
    return unhealed + healed


def boundary_topk(base, typed, r):
    """Unhealed spine, plus the raw top-20 tokens at the boundary that cover the typed text. All
    raw joint probabilities, so correct ranking, but mid-word branches are one token deep, and
    missing if the covering token isn't in the top 20 or no single token covers the typed text."""
    if not typed:
        return naive(base, typed, r)
    n_base = len(r["boundary"]["prompt_token_ids"])
    out = spine_branches(r["unhealed"]["logprobs"], typed, typed, typed_logprob(r["unhealed"], n_base))
    for tok, lp in r["boundary"]["logprobs"]["top_logprobs"][0].items():
        if tok.startswith(typed) and tok[len(typed):].strip():
            out.append((tok, lp))
    return out


def hybrid(base, typed, r):
    """A workaround for masked logprobs: boundary_topk, plus the healed spine's later steps
    when its first token covers the typed text, re-scored with that token's raw probability.
    Later steps are unconstrained, so their logprobs are already raw."""
    out = boundary_topk(base, typed, r)
    if not typed:
        return out
    raw = r["boundary"]["logprobs"]["top_logprobs"][0]
    first = r["healed"]["logprobs"]["tokens"][0]
    if first.startswith(typed) and first in raw:
        out += spine_branches(r["healed"]["logprobs"], typed, first, raw[first], skip=1)
    return out


def hybrid_bounded(base, typed, r):
    """hybrid, but if the healed spine's covering first token isn't in the boundary's top 20,
    score it with the 20th-best logprob, an upper bound on its own. Recovers words the model
    didn't expect at the boundary ("questionab" → "questionable") at the cost of overrating them."""
    out = boundary_topk(base, typed, r)
    if not typed:
        return out
    raw = r["boundary"]["logprobs"]["top_logprobs"][0]
    first = r["healed"]["logprobs"]["tokens"][0]
    if first.startswith(typed):
        first_lp = raw.get(first, min(raw.values()))
        if first not in raw:
            out.append((first, first_lp))
        out += spine_branches(r["healed"]["logprobs"], typed, first, first_lp, skip=1)
    return out


OPTIONS = [naive, healed_only, merge, continuation_mass, boundary_topk, hybrid, hybrid_bounded]

# ── Cases ──────────────────────────────────────────────────────────────────
# Drafts, what a good top suggestion looks like, and the text it should start with after the
# typed part (non-empty, if ""). Shared with engine_check.mjs.
CASES = [(c["draft"], c["want"], lambda s, p=c["starts"]: bool(s.strip()) and s.startswith(p))
         for c in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json")))]


def requests_for(draft):
    base, typed = split_last_pretoken(draft)
    spine = {"logprobs": 20, "return_token_ids": True}
    unhealed, healed, boundary = parallel(
        {"prompt": PROMPT + draft, "max_tokens": 3, "prompt_logprobs": 1, **spine},
        typed and {"prompt": PROMPT + base, "max_tokens": 3 + math.ceil(len(typed) / 4), **spine, **starts_with(typed)},
        typed and {"prompt": PROMPT + base, "max_tokens": 1, **spine},
    )
    return base, typed, {"unhealed": unhealed, "healed": healed, "boundary": boundary}


def ranked(branches, n=4):
    best = {}
    for text, lp in sorted(branches, key=lambda b: -b[1]):
        best.setdefault(text, lp)
    return list(best.items())[:n]


def compare(drafts):
    cases = [c for c in CASES if not drafts or c[0] in drafts] + [(d, "?", None) for d in drafts if d not in {c[0] for c in CASES}]
    score = {opt.__name__: 0 for opt in OPTIONS}
    for draft, want, ok in cases:
        base, typed, r = requests_for(draft)
        spine = r["healed"]["logprobs"]["tokens"] if r["healed"] else None
        print(f"\n{draft!r}  typed={typed!r}  want: {want}" + (f"\n  healed spine tokens: {spine}" if spine else ""))
        for opt in OPTIONS:
            top = ranked(opt(base, typed, r))
            passed = ok is not None and bool(top) and ok(top[0][0][len(typed):])
            score[opt.__name__] += passed
            mark = "?" if ok is None else "✓" if passed else "✗"
            shown = " | ".join(f"{t[len(typed):]!r} {lp:.1f}" for t, lp in top) or "(none)"
            print(f"  {mark} {opt.__name__:18} {shown}")
    print(f"\nTop suggestion as wanted, out of {sum(c[2] is not None for c in cases)}:")
    for name, n in score.items():
        print(f"  {name:18} {n}")


def mechanism():
    print("1. The regex fixes the text, not its tokenization. For typed 'Used', greedy decoding under")
    print("   the constraint picks 'Use' (the model was heading for 'Use deep learning…'), then must")
    print("   produce a 'd', so takes 'deep'. With stock vLLM, the reported logprobs are after the grammar")
    print("   mask, so this path looks certain (≈ 0 for 'Use' and 'deep'); with raw logprobs, 'deep' scores low:")
    c = complete(prompt=PROMPT, max_tokens=4, logprobs=20, **starts_with("Used"))
    for tok, lp in zip(c["logprobs"]["tokens"], c["logprobs"]["token_logprobs"]):
        print(f"     {tok!r:16} logprob {lp:.2f}")
    c = complete(prompt=PROMPT + "Used", max_tokens=1, logprobs=5, prompt_logprobs=1, return_token_ids=True)
    ids = c["prompt_token_ids"]
    e = c["prompt_logprobs"][-1][str(ids[-1])]
    print(f"   Unconstrained, the raw P('Used') at that point is {e['logprob']:.2f} (rank {e['rank']}), and after it")
    print(f"   the model continues with {list(c['logprobs']['top_logprobs'][0])}.")
    print()
    print("2. Without healing, a typed token implies the word ends there. 'Use de' is 'Use' + ' de', and")
    print("   ' deep' is its own token, so after ' de' the model avoids 'ep':")
    c = complete(prompt=PROMPT + "Use de", max_tokens=1, logprobs=8)
    print(f"     after ' de':      {[f'{t!r} {lp:.1f}' for t, lp in c['logprobs']['top_logprobs'][0].items()]}")
    c = complete(prompt=PROMPT + "Use", max_tokens=1, logprobs=5)
    print(f"     after 'Use' (raw): {[f'{t!r} {lp:.1f}' for t, lp in c['logprobs']['top_logprobs'][0].items()]}")
    print("   Same for 'Deep neural net': after the token ' net', 'works' is unlikely, though ' networks'")
    print("   is the likeliest token at the boundary:")
    c = complete(prompt=PROMPT + "Deep neural net", max_tokens=1, logprobs=5)
    print(f"     after ' net':      {[f'{t!r} {lp:.1f}' for t, lp in c['logprobs']['top_logprobs'][0].items()]}")
    c = complete(prompt=PROMPT + "Deep neural", max_tokens=1, logprobs=5)
    print(f"     after 'Deep neural': {[f'{t!r} {lp:.1f}' for t, lp in c['logprobs']['top_logprobs'][0].items()]}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["compare"]
    {"mechanism": lambda: mechanism(), "compare": lambda: compare(args)}[cmd]()
