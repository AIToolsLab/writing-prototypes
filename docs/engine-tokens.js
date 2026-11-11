// Tokens engine: reads the model's probabilities from our own inference server (engine/server.cpp).
// (Until 2026-11 this was the "Trie" engine, and "Tokens" read them from vLLM.)
//
// The server answers in characters, not tokens: given the prompt and the draft, it returns likely
// continuations with the model's probability of each, conditioned on the draft's *characters*
// (summed over the ways they could be tokenized). So there's no token healing here. This file
// only groups those continuations into the units we show (words), and narrows the last answer
// locally while the next one is on its way.
//
// Both engines have the same interface:
//   highlights({instruction, doc}, signal) → {ranges: [{start, end, alternative, score?, detail}], status, warning?}
//   suggest({messages, words, onUpdate}, signal) → {suggestions: [{text, prob?}], status, warning?}
// where `messages` ends with the message being written, and each suggestion's text is to be
// appended to that message as-is. Uses helpers from llm.js.

const ENGINE_URL = new URLSearchParams(location.search).get("engine-server") || "https://llm.thoughtful-ai.com";

// Names this page's cache on the server, so several people can write at once without sharing one.
const SESSION = globalThis.crypto?.randomUUID?.() ?? String(Math.random()).slice(2);

const tokensEngine = {
  name: "tokens",
  server: ENGINE_URL,
  highlightDelay: 150,
  suggestDelay: 0,  // the server drops requests that a newer one supersedes
  ready: once(async () => {
    const res = await fetch(ENGINE_URL + "/v1/models");
    if (!res.ok) throw new Error(`Listing models failed: ${res.status}`);
  }),
  highlights: tokenHighlights,
  suggest: tokenSuggestions,
  // Only this engine has these two (draft mode):
  //   moves({assignment, draft}, signal) → {moves: [{text, prob}], status}
  //   mirror({assignment, draft}, signal) → {words: [{start, end, prob, alternative}], status, warning?}
  moves: tokenMoves,
  mirror: tokenMirror,
};

async function enginePost(path, body, signal) {
  const res = await fetch(ENGINE_URL + path, {
    method: "POST",
    headers: {"Content-Type": "text/plain"},  // a "simple" request, so no CORS preflight per keystroke
    body: JSON.stringify(body),
    signal,
  });
  if (res.status === 409) throw new DOMException("superseded", "AbortError");
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

// The chat-templated prompt up to where `role`'s message starts, after `history`.
function enginePrefix(history, role) {
  return cached(JSON.stringify(["engine", role, history]), async () => {
    if (role === "assistant") return (await enginePost("/v1/template", {messages: history, add_generation_prompt: true})).prompt;
    // There's no generation prompt for a user turn, so render the turn with a placeholder and cut there.
    const {prompt} = await enginePost("/v1/template", {messages: [...history, {role, content: PLACEHOLDER}], add_generation_prompt: false});
    const cut = prompt.lastIndexOf(PLACEHOLDER);
    if (cut === -1) throw new Error("Couldn't find the start of the message in the chat template");
    return prompt.slice(0, cut);
  });
}

// ── Highlights ─────────────────────────────────────────────────────────────

async function tokenHighlights({instruction, doc}, signal) {
  const t0 = performance.now();
  const prompt = await enginePrefix(taskMessages(instruction, doc), "assistant");
  const {tokens, stats} = await enginePost("/v1/score", {session: SESSION, prompt, text: doc}, signal);
  const ranges = [];
  let pos = 0;
  for (const {text, logprob, top, top_logprob} of tokens) {
    const start = pos;
    pos += text.length;
    const gap = top_logprob - logprob;  // 0 when the document's token was the model's top choice
    if (gap <= 0) continue;
    const lead = text.length - text.trimStart().length;
    const ratio = Math.exp(gap);
    ranges.push({
      start: lead < text.length ? start + lead : start,
      end: pos,
      alternative: top,
      score: gap,
      detail: `${ratio < 10 ? ratio.toFixed(1) : Math.round(ratio)}× more likely`,
    });
  }
  const warning = pos === doc.length ? "" : "The scored text doesn't line up with the document, so highlights may be misplaced.";
  return {ranges, warning, status: `Scored ${stats.text_tokens} tokens in ${Math.round(performance.now() - t0)} ms`};
}

// ── Predictions ────────────────────────────────────────────────────────────

// Scripts written without spaces between words. For these we find words with the browser's
// segmenter, and ask the server for a few characters rather than "up to the next space".
const NO_SPACE_SCRIPT = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]/u;
const segmenter = typeof Intl !== "undefined" && Intl.Segmenter ? new Intl.Segmenter(undefined, {granularity: "word"}) : null;

// The first n words of a continuation, and whether we've seen enough text after them to know they're whole.
function firstUnits(text, n) {
  if (!segmenter || !NO_SPACE_SCRIPT.test(text)) {
    const whole = completeWords(text, n);
    return whole === null ? {text: firstWords(text, n), complete: false} : {text: whole, complete: true};
  }
  // A unit is a word plus any punctuation after it; it ends at whitespace or where the next word starts.
  const segments = [...segmenter.segment(text)];
  let count = 0, end = 0, i = 0;
  for (; i < segments.length; i++) {
    const {segment, isWordLike} = segments[i];
    if (count === n && (isWordLike || /\s/.test(segment))) break;
    if (isWordLike) count++;
    end = segments[i].index + segment.length;
  }
  // The last segment may be a word that's still growing, so it doesn't confirm that the unit ended.
  const complete = count === n && (i < segments.length - 1 || (i < segments.length && /\s/.test(segments[i].segment)));
  return {text: text.slice(0, end), complete};
}

// Group continuations by the unit they start with, adding up their probabilities. A unit's total is
// a lower bound: continuations the server didn't follow (too unlikely) aren't counted.
function unitSuggestions(continuations, words) {
  const units = new Map();
  for (const {text, prob} of continuations) {
    const unit = firstUnits(text, words);
    if (!unit.text.trim()) continue;  // ending the message here isn't a suggestion
    units.set(unit.text, (units.get(unit.text) ?? 0) + prob);
  }
  // Under half a percent is noise: mostly words the server stopped following partway.
  return [...units].map(([text, prob]) => ({text, prob})).filter(u => u.prob >= 0.005)
    .sort((a, b) => b.prob - a.prob).slice(0, NUM_SUGGESTIONS);
}

// The continuations that are still possible after typing `extra` more characters. Their new
// probabilities are lower bounds: continuations we never saw might match `extra` too.
function narrow({continuations}, extra) {
  const matching = continuations.filter(c => c.text.length > extra.length && c.text.startsWith(extra));
  const seen = continuations.reduce((sum, c) => sum + c.prob, 0);
  const total = matching.reduce((sum, c) => sum + c.prob, 0) + Math.max(0, 1 - seen);
  return matching.map(c => ({...c, text: c.text.slice(extra.length), prob: c.prob / total}));
}

let lastAnswer = null;  // {prompt, words, draft, continuations}

async function tokenSuggestions({messages, words, onUpdate}, signal) {
  const {role, content: draft} = messages.at(-1);
  const t0 = performance.now();
  const prompt = await enginePrefix(messages.slice(0, -1), role);

  // Until the server answers, show what the last answer implies for the characters typed since.
  const last = lastAnswer;
  if (last && last.prompt === prompt && last.words === words && draft.length > last.draft.length && draft.startsWith(last.draft)) {
    const local = unitSuggestions(narrow(last, draft.slice(last.draft.length)), words);
    if (local.length) onUpdate(local);
  }

  const body = {session: SESSION, prompt, text: draft, stop: `^\\s*\\S+(?:[ \\t]+\\S+){${words - 1}}(?=\\s)`};
  if (NO_SPACE_SCRIPT.test((messages.map(m => m.content).join("\n")).slice(-200))) body.max_chars = 4 * words + 2;
  const res = await enginePost("/v1/continuations", body, signal);
  lastAnswer = {prompt, words, draft, continuations: res.continuations};
  const {stats} = res;
  return {
    suggestions: unitSuggestions(res.continuations, words),
    warning: res.unresolved > 0.05 ? "Some ways of reading the last word weren't explored, so probabilities may be a little high." : "",
    status: `Predicted in ${Math.round(performance.now() - t0)} ms ` +
      `(server ${Math.round(stats.total_ms)} ms: ${stats.decoded_tokens} new tokens, ${stats.rounds} search steps)`,
  };
}

// ── Moves ──────────────────────────────────────────────────────────────────

// What kind of sentence could come next: the model's own short answers to that question, with the
// probability it gives each. (Asking it to pick from a fixed lettered list mostly measures the
// letter's position; see design/interactions.md.) A separate session, since this prompt has the
// draft in the user turn and would otherwise evict the predictions' cache.
async function tokenMoves({assignment, draft}, signal) {
  const t0 = performance.now();
  const prompt = await enginePrefix([{role: "user", content:
    `${assignment}\n\nHere is my draft so far:\n\n${draft.trim() || "(nothing yet)"}\n\n` +
    "In five words or fewer, name the rhetorical move the next sentence should make."}], "assistant");
  const res = await enginePost("/v1/continuations", {
    session: SESSION + ":moves", prompt, text: "",
    stop: "[.;:\\n]|^\\s*\\S+(?:[ \\t]+\\S+){5}(?=\\s)",
    max_rounds: 32, max_depth: 16, time_ms: 2500,
  }, signal);
  return {
    // Only whole answers, and not the start of a longer one ("Op-Ed:").
    moves: groupMoves(res.continuations.filter(c => c.done && !/:\s*$/.test(c.text))),
    status: `Asked in ${Math.round(performance.now() - t0)} ms (server ${Math.round(res.stats.total_ms)} ms)`,
  };
}

// The same move comes back in several wordings ("Concede a point", "concede the point."). Add up
// those that differ only in case, articles, punctuation, or a final plural.
function groupMoves(continuations) {
  const groups = new Map();
  for (const {text, prob} of continuations) {
    const label = text.replace(/[*"“”`]/g, "").replace(/[.;:\s]+$/, "").trim().toLowerCase();
    const key = label.replace(/\b(an?|the)\s+/g, "").replace(/s$/, "");
    if (!/[a-z]{3}/.test(key)) continue;
    const group = groups.get(key) ?? {text: label, best: 0, prob: 0};
    if (prob > group.best) Object.assign(group, {text: label, best: prob});
    group.prob += prob;
    groups.set(key, group);
  }
  return [...groups.values()].map(({text, prob}) => ({text, prob}))
    .filter(m => m.prob >= 0.01).sort((a, b) => b.prob - a.prob).slice(0, 6);
}

// ── Mirror ─────────────────────────────────────────────────────────────────

// How predictable each word of the draft was, given the assignment and the draft before it.
// A word's probability is the product of its tokens'; shading tokens instead would mark the
// second half of every long word as predictable, which says nothing about the writer.
// `alternative` is what the model thought most likely where the word starts, if not this word.
async function tokenMirror({assignment, draft}, signal) {
  const t0 = performance.now();
  const prompt = await enginePrefix([{role: "user", content: assignment}], "assistant");
  const {tokens, stats} = await enginePost("/v1/score", {session: SESSION + ":mirror", prompt, text: draft}, signal);
  const words = [...draft.matchAll(/\S+/g)].map(m => ({start: m.index, end: m.index + m[0].length, logprob: 0, tokens: 0, alternative: ""}));
  let pos = 0, w = 0;
  for (const {text, logprob, top, top_logprob} of tokens) {
    const firstChar = pos + text.length - text.trimStart().length;
    pos += text.length;
    if (firstChar === pos) continue;  // only whitespace, such as a paragraph break
    while (w < words.length && words[w].end <= firstChar) w++;
    if (w === words.length) break;
    if (words[w].tokens++ === 0 && top_logprob > logprob) words[w].alternative = top;
    words[w].logprob += logprob;
  }
  return {
    words: words.filter(word => word.tokens).map(({start, end, logprob, alternative}) => ({start, end, prob: Math.exp(logprob), alternative})),
    warning: pos === draft.length ? "" : "The scored text doesn't line up with the draft, so shading may be misplaced.",
    status: `Read ${stats.text_tokens} tokens in ${Math.round(performance.now() - t0)} ms`,
  };
}
