// Token-level engine: reads the model's next-token log-probabilities directly.
//
// Both engines have the same interface:
//   highlights({instruction, doc}, signal) → {ranges: [{start, end, alternative, score?, detail}], status, warning?}
//   suggest({messages, words, onUpdate}, signal) → {suggestions: [{text, prob?}], status, warning?}
// where `messages` ends with the message being written, and each suggestion's text is to be
// appended to that message as-is.

const tokensEngine = {
  name: "tokens",
  highlightDelay: 150,
  highlights: tokenHighlights,
  suggest: tokenSuggestions,
};

// ── Highlights ─────────────────────────────────────────────────────────────

// Treat the document itself as the assistant's response, and mark tokens where the model's top
// choice differs, scored by how much more likely that choice was.
async function tokenHighlights({instruction, doc}, signal) {
  const t0 = performance.now();
  const prefix = await chatPrefix(taskMessages(instruction, doc));
  const res = await completions({prompt: prefix.text + doc, max_tokens: 1, prompt_logprobs: 5, return_token_ids: true}, signal);
  const {prompt_logprobs: promptLogprobs, prompt_token_ids: promptIds} = res.choices[0];

  // prompt_logprobs[i] is the distribution at position i, keyed by token id; it always
  // includes the actual token, plus the top-k (rank 1 is the model's top choice).
  const ranges = [];
  let pos = 0;
  const docIds = promptIds.slice(prefix.ids.length);
  docIds.forEach((id, j) => {
    const dist = promptLogprobs[prefix.ids.length + j];
    const actual = dist[String(id)];
    const top = Object.values(dist).find(e => e.rank === 1);
    const text = actual.decoded_token;
    const start = pos;
    pos += text.length;
    const gap = top.logprob - actual.logprob;  // 0 when the actual token is the top choice
    if (gap <= 0) return;
    // Don't highlight the space before a word; it reads as highlighting the gap between words.
    const lead = text.length - text.trimStart().length;
    const ratio = Math.exp(gap);
    ranges.push({
      start: lead < text.length ? start + lead : start,
      end: pos,
      alternative: top.decoded_token,
      score: gap,
      detail: `${ratio < 10 ? ratio.toFixed(1) : Math.round(ratio)}× more likely`,
    });
  });
  // Tokens that split a multi-byte character decode to replacement characters, which throws off
  // our character offsets.
  const warning = pos === doc.length ? "" :
    "The tokens don't line up with the document's characters (probably non-ASCII text), so highlights may be misplaced.";
  return {ranges, warning, status: `Scored ${docIds.length} tokens in ${Math.round(performance.now() - t0)} ms`};
}

// ── Predictions ────────────────────────────────────────────────────────────

const NUM_SUGGESTIONS = 5;

// Qwen3.5's pre-tokenizer split (transformers tokenization_qwen3_5.py), with the (?i:...)
// group expanded because not all browsers support inline modifiers. BPE merges never cross
// these boundaries, so typing more text can only change the tokens of the last chunk.
const PRETOKENIZE_REGEX = /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

// Split text into [everything before the last pre-token chunk, the last chunk].
function splitLastPretoken(text) {
  let start = text.length;
  for (const match of text.matchAll(PRETOKENIZE_REGEX)) start = match.index;
  return [text.slice(0, start), text.slice(start)];
}

// Constrain generation to output that starts with `text`.
function startsWith(text) {
  const escaped = text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return {structured_outputs: {regex: escaped + "[\\s\\S]*"}};
}

async function tokenSuggestions({messages, words, onUpdate}, signal) {
  // If the draft ends partway through a token (e.g. "Use de" = "Use" + "Ġde", where "deep"
  // would be the single token "Ġdeep"), the model infers the word *isn't* one that has a longer
  // token, and predicts "depleting". So back up to the last pre-token boundary, which every
  // tokenization of the full text shares, and constrain the output to start with the text we
  // backed over. The model then picks its own tokenization (e.g. "Ġdeep").
  const {role, content: draft} = messages.at(-1);
  const [base, typed] = splitLastPretoken(draft);
  const t0 = performance.now();
  const [prefix, logitBias] = await Promise.all([turnPrefix(messages.slice(0, -1), role), noEndBias()]);
  const prompt = prefix.text + base;

  // Request 1: short greedy paths ("spines") with the top alternatives at each step. Each
  // alternative starts a branch, scored by its whole path's probability. Looking past the first
  // step matters when the typed text fixes the first token (e.g. typed " learning"): the choices
  // then come from the step after. Two spines, in parallel:
  // - healed: from the boundary, constrained to start with the typed text, for when it ends
  //   mid-word. The constraint fixes the text, not its tokenization, so greedy can wander off the
  //   usual tokenization: for "Used" it picks "Use" (heading for "Use deep learning…"), then
  //   "deep" to supply the "d". Such paths get low scores, since logprobs are the model's raw
  //   ones, from before the constraint (this needs a patched vLLM; stock vLLM reports them after
  //   the grammar mask, which makes the wandering paths look certain).
  // - unhealed: from the end of the typed text, for when it ends at a token boundary ("Used" →
  //   " ethically"). Its prompt logprobs give the probability of the typed tokens, so its
  //   branches' probabilities are also of the whole path from the boundary, and comparable.
  // See experiments/token-healing for the alternatives we tried.
  const spineParams = {logprobs: 20, temperature: 0, logit_bias: logitBias, return_token_ids: true};
  const [unhealedRes, healedRes] = await Promise.all([
    completions({prompt: prefix.text + draft, max_tokens: 3, prompt_logprobs: 1, ...spineParams}, signal),
    typed && completions({prompt, max_tokens: 3 + Math.ceil(typed.length / 4), ...spineParams, ...startsWith(typed)}, signal),
  ]);
  const unhealed = unhealedRes.choices[0];
  // The draft's tokens are only for checking our pre-tokenizer split.
  const baseIds = typed ? healedRes.choices[0].prompt_token_ids : unhealed.prompt_token_ids;
  const warning = boundaryWarning(baseIds, unhealed.prompt_token_ids, base) ||
    (!typed || healedRes.choices[0].text.startsWith(typed) ? "" : "The server ignored the constraint to continue the partial word, so predictions may be missing.");
  const typedLogprob = unhealed.prompt_token_ids.slice(baseIds.length)
    .reduce((sum, id, j) => sum + unhealed.prompt_logprobs[baseIds.length + j][String(id)].logprob, 0);
  let branches = spineBranches(unhealed.logprobs, typed, typed, typedLogprob);
  if (typed) branches.push(...spineBranches(healedRes.choices[0].logprobs, typed, "", 0));
  branches.sort((a, b) => b.prob - a.prob);
  // The spines overlap when the typed text is a whole token (e.g. " is").
  branches = branches.filter((b, i) => branches.findIndex(c => c.forced === b.forced) === i);

  // Branches that already contain the words we'll show are final; keep the most probable one per
  // shown text (e.g. " deep learning" and " deep neural" both show as "ep" after typed " de").
  // The rest end mid-word and may still grow into something distinct.
  const shown = new Set();
  branches = branches.filter(b => {
    b.final = completeWords(b.forced.slice(typed.length), words);
    if (b.final === null) return true;
    if (shown.has(b.final)) return false;
    shown.add(b.final);
    return true;
  }).slice(0, NUM_SUGGESTIONS + 2);  // a little extra, since extended branches may turn out to be duplicates
  const toSuggestions = () => uniqueSuggestions(branches.map(b => ({
    text: b.final ?? firstWords(b.forced.slice(typed.length), words),
    prob: b.prob,
  })));
  // Show the branches right away; the lookahead below finishes their words.
  onUpdate(toSuggestions());

  // Request 2 (if needed): extend the unfinished branches greedily, in one batch, so we can cut
  // them at a word boundary. Each branch already contains the typed text, so no constraint.
  // (On our server, decoding gets slower per step as the batch grows, so extend as few as we can.)
  const unfinished = branches.filter(b => b.final === null);
  if (unfinished.length > 0) {
    const extRes = await completions({
      prompt: unfinished.map(b => prompt + b.forced),
      max_tokens: 2 * words,
      temperature: 0,
      logit_bias: logitBias,
    }, signal);
    for (const choice of extRes.choices) {
      const b = unfinished[choice.index];
      b.final = firstWords((b.forced + choice.text).slice(typed.length), words);
    }
  }
  return {suggestions: toSuggestions(), warning, status: `Predicted in ${Math.round(performance.now() - t0)} ms`};
}

// Branches off a spine: each top alternative at each step, appended to the spine's path so far
// (which starts as pathText, with log-probability pathLogprob). Only branches that get past the
// typed text to something visible are suggestions.
function spineBranches({tokens, token_logprobs, top_logprobs}, typed, pathText, pathLogprob) {
  const branches = [];
  for (let j = 0; j < tokens.length; j++) {
    for (const [tok, logprob] of Object.entries(top_logprobs[j])) {
      const forced = pathText + tok;
      if (isSpecialToken(tok) || !forced.startsWith(typed) || !forced.slice(typed.length).trim()) continue;
      branches.push({forced, prob: Math.exp(pathLogprob + logprob)});
    }
    pathText += tokens[j];
    pathLogprob += token_logprobs[j];
  }
  return branches;
}

// If our pre-tokenizer split matches the server's, the tokens of `base` are a prefix of the
// tokens of the whole draft. Warn if not.
function boundaryWarning(baseIds, draftIds, base) {
  if (baseIds.every((id, i) => id === draftIds[i])) return "";
  return `Tokenization warning: the draft's tokens don't start with the tokens of ${JSON.stringify(base.slice(-20))}, ` +
    `so this page's pre-tokenizer split doesn't match the server's. Mid-word predictions may be off.`;
}

// The first n words of text, if text already contains them followed by whitespace; else null.
function completeWords(text, n) {
  const match = text.match(new RegExp(`^\\s*\\S+(?:[ \\t]+\\S+){${n - 1}}(?=\\s)`));
  return match ? match[0] : null;
}

function uniqueSuggestions(suggestions) {
  return suggestions
    .filter((s, i) => s.text.trim() && suggestions.findIndex(t => t.text === s.text) === i)
    .slice(0, NUM_SUGGESTIONS);
}

// Leading whitespace plus the first n whitespace-delimited words.
function firstWords(text, n) {
  const match = text.match(new RegExp(`^\\s*\\S+(?:[ \\t]+\\S+){0,${n - 1}}`));
  return match ? match[0] : text;
}
