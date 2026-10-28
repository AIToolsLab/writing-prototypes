// Thin client for a stock vLLM server (OpenAI-compatible API plus /tokenize).
// Shared by both engines.

const BASE_URL = new URLSearchParams(location.search).get("server") || "https://vllm.thoughtful-ai.com";
let MODEL = null;

// Thinking off: we want the model's direct response, and the thinking-disabled
// generation prompt ends in "<think>\n\n</think>\n\n" with no surprises after it.
const CHAT_TEMPLATE_KWARGS = {enable_thinking: false};

async function post(path, body, signal) {
  const res = await fetch(BASE_URL + path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

async function loadModel() {
  if (MODEL) return MODEL;
  const res = await fetch(BASE_URL + "/v1/models");
  if (!res.ok) throw new Error(`Listing models failed: ${res.status}`);
  MODEL = (await res.json()).data[0].id;
  return MODEL;
}

// A few recent results of `fn`, keyed by `key`. Not tied to any one caller's AbortSignal,
// since results are shared; failed results are dropped so the next call retries.
const prefixCache = new Map();
function cached(key, fn) {
  if (!prefixCache.has(key)) {
    const promise = fn();
    promise.catch(() => { if (prefixCache.get(key) === promise) prefixCache.delete(key); });
    prefixCache.set(key, promise);
    if (prefixCache.size > 8) prefixCache.delete(prefixCache.keys().next().value);
  }
  return prefixCache.get(key);
}

// The chat-templated prompt for `messages`, ending with the assistant generation prompt,
// as both token ids and text. We append the response text to `text` ourselves rather than
// passing it through the chat template, so Jinja can't trim its whitespace. The server parses
// special-token strings in text prompts, so `text + response` tokenizes the same as the
// template's ids followed by the response's ids.
function chatPrefix(messages) {
  return cached(JSON.stringify(["assistant", messages]), async () => {
    const {tokens} = await post("/tokenize", {
      model: MODEL,
      messages,
      add_generation_prompt: true,
      chat_template_kwargs: CHAT_TEMPLATE_KWARGS,
    });
    const {prompt} = await post("/detokenize", {model: MODEL, tokens});
    return {ids: tokens, text: prompt};
  });
}

// Like chatPrefix, but for a message from `role` after `history`. There's no generation prompt
// for a user turn, so render the turn with a placeholder as its content and cut there.
const PLACEHOLDER = "XYZZYPLACEHOLDER";
function turnPrefix(history, role) {
  if (role === "assistant") return chatPrefix(history);
  return cached(JSON.stringify([role, history]), async () => {
    const {tokens} = await post("/tokenize", {
      model: MODEL,
      messages: [...history, {role, content: PLACEHOLDER}],
      add_generation_prompt: false,
      chat_template_kwargs: CHAT_TEMPLATE_KWARGS,
    });
    const {prompt} = await post("/detokenize", {model: MODEL, tokens});
    const cut = prompt.lastIndexOf(PLACEHOLDER);
    if (cut === -1) throw new Error("Couldn't find the start of the message in the chat template");
    return {text: prompt.slice(0, cut)};
  });
}

// A logit_bias that keeps generation from ending the turn, so lookahead paths don't stop short
// where ending is likely (e.g. a user turn after a complete word); ending is never a suggestion
// anyway. The reported logprobs are from before the bias, so they're unaffected.
function noEndBias() {
  return cached("noEndBias", async () => {
    const {tokens} = await post("/tokenize", {model: MODEL, prompt: "<|im_end|><|endoftext|>", add_special_tokens: false});
    return Object.fromEntries(tokens.map(id => [id, -100]));
  });
}

async function completions(body, signal) {
  return post("/v1/completions", {model: MODEL, ...body}, signal);
}

// Chat completion constrained to a JSON schema; returns the parsed object.
async function chatJSON(messages, schema, {maxTokens = 400, signal} = {}) {
  const data = await post("/v1/chat/completions", {
    model: MODEL,
    messages,
    temperature: 0,
    max_tokens: maxTokens,
    response_format: {type: "json_schema", json_schema: {name: "result", schema}},
    chat_template_kwargs: CHAT_TEMPLATE_KWARGS,
  }, signal);
  return {result: JSON.parse(data.choices[0].message.content), usage: data.usage};
}

// The single-turn conversation for an instruction about a document.
function taskMessages(instruction, doc) {
  return [{role: "user", content: `${instruction}\n\n${doc}`}];
}

function isSpecialToken(tok) {
  return /^<\|.*\|>$/.test(tok);
}

function showToken(tok) {
  if (isSpecialToken(tok)) return "⏹ (end)";
  return tok.replace(/\n/g, "↵").replace(/\t/g, "⇥");
}
