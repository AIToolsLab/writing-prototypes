// Helpers shared by both engines, and a small client for an OpenAI-style chat server
// (the Structured engine's). By default that's Calvin's Open WebUI, which needs each
// person's own API key; the key is kept in this browser's localStorage and sent only there.

const CHAT_URL = (new URLSearchParams(location.search).get("chat-server") || "https://localai.calvin.edu/api").replace(/\/+$/, "");
const CHAT_KEY_HOME = "https://localai.calvin.edu";  // where to get a key for the default server
const DEFAULT_CHAT_MODEL = "nvidia-qwen3-6-35b-a3b-nvfp4";  // about 0.5 s per suggestion in 2026-11

const chatSettings = {
  get key() { return localStorage.getItem("chat-api-key") ?? ""; },
  set key(value) { localStorage.setItem("chat-api-key", value.trim()); },
  get model() { return localStorage.getItem("chat-model") || DEFAULT_CHAT_MODEL; },
  set model(value) { localStorage.setItem("chat-model", value); },
};

// Thinking off: we want the model's direct response.
const CHAT_TEMPLATE_KWARGS = {enable_thinking: false};

function chatFetch(path, init = {}) {
  return fetch(CHAT_URL + path, {
    ...init,
    headers: {...init.headers, "Authorization": `Bearer ${chatSettings.key}`},
  });
}

// The ids of the server's models. Throws a readable error if there's no key or it's refused.
async function listChatModels() {
  if (!chatSettings.key) throw new Error("this engine needs an API key; paste one above");
  const res = await chatFetch("/models");
  if (res.status === 401 || res.status === 403) throw new Error("the server refused the API key");
  if (!res.ok) throw new Error(`listing models failed: HTTP ${res.status}`);
  return (await res.json()).data.map(m => m.id);
}

// Chat completion constrained to a JSON schema; returns the parsed object.
async function chatJSON(messages, schema, {maxTokens = 400, signal} = {}) {
  const res = await chatFetch("/chat/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      model: chatSettings.model,
      messages,
      stream: false,
      temperature: 0,
      max_tokens: maxTokens,
      response_format: {type: "json_schema", json_schema: {name: "result", schema}},
      chat_template_kwargs: CHAT_TEMPLATE_KWARGS,
    }),
    signal,
  });
  if (!res.ok) throw new Error(`chat: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
  const {content, usage} = await readChat(res);
  return {result: JSON.parse(content), usage};
}

// Open WebUI streams even when asked not to, so accept either a JSON body or server-sent events.
async function readChat(res) {
  const body = await res.text();
  if (!body.startsWith("data:")) {
    const data = JSON.parse(body);
    return {content: data.choices[0].message.content, usage: data.usage};
  }
  let content = "", usage;
  for (const line of body.split("\n")) {
    if (!line.startsWith("data: {")) continue;
    const chunk = JSON.parse(line.slice(6));
    content += chunk.choices?.[0]?.delta?.content ?? "";
    usage = chunk.usage ?? usage;
  }
  return {content, usage};
}

// ── Shared helpers ─────────────────────────────────────────────────────────

const NUM_SUGGESTIONS = 5;

// Wrap an async function so it runs once; a failure is forgotten, so the next call retries.
function once(fn) {
  let promise = null;
  return () => promise ??= fn().catch(err => { promise = null; throw err; });
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

// Stands in for a message's content when rendering a chat template, to find where it starts.
const PLACEHOLDER = "XYZZYPLACEHOLDER";

// The single-turn conversation for an instruction about a document.
function taskMessages(instruction, doc) {
  return [{role: "user", content: `${instruction}\n\n${doc}`}];
}

// The first n words of text, if text already contains them followed by whitespace; else null.
function completeWords(text, n) {
  const match = text.match(new RegExp(`^\\s*\\S+(?:[ \\t]+\\S+){${n - 1}}(?=\\s)`));
  return match ? match[0] : null;
}

// Leading whitespace plus the first n whitespace-delimited words.
function firstWords(text, n) {
  const match = text.match(new RegExp(`^\\s*\\S+(?:[ \\t]+\\S+){0,${n - 1}}`));
  return match ? match[0] : text;
}

function isSpecialToken(tok) {
  return /^<\|.*\|>$/.test(tok);
}

function showToken(tok) {
  if (isSpecialToken(tok)) return "⏹ (end)";
  return tok.replace(/\n/g, "↵").replace(/\t/g, "⇥");
}
