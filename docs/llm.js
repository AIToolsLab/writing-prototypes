// Thin client for a stock vLLM server (OpenAI-compatible API plus /tokenize).
// Shared by tokens.html and structured.html.

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

// The chat-templated prompt for `messages`, ending with the assistant generation prompt,
// as both token ids and text. We append the response text to `text` ourselves rather than
// passing it through the chat template, so Jinja can't trim its whitespace. The server parses
// special-token strings in text prompts, so `text + response` tokenizes the same as the
// template's ids followed by the response's ids.
let prefixCache = {key: null, promise: null};
function chatPrefix(messages) {
  const key = JSON.stringify(messages);
  if (prefixCache.key !== key) {
    // Not tied to any one caller's AbortSignal, since the result is shared.
    const promise = (async () => {
      const {tokens} = await post("/tokenize", {
        model: MODEL,
        messages,
        add_generation_prompt: true,
        chat_template_kwargs: CHAT_TEMPLATE_KWARGS,
      });
      const {prompt} = await post("/detokenize", {model: MODEL, tokens});
      return {ids: tokens, text: prompt};
    })();
    promise.catch(() => { if (prefixCache.promise === promise) prefixCache = {key: null, promise: null}; });
    prefixCache = {key, promise};
  }
  return prefixCache.promise;
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

// The single-turn conversation both pages use: an instruction about a document.
function taskMessages(instruction, doc) {
  return [{role: "user", content: `${instruction}\n\n${doc}`}];
}

function showToken(tok) {
  return tok.replace(/\n/g, "↵").replace(/\t/g, "⇥");
}

// Run `fn` after the user pauses typing; abort the previous in-flight call.
function debounced(fn, ms = 150) {
  let timer = null, controller = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        await fn(controller.signal, ...args);
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error(err);
          setStatus("Error: " + err.message);
        }
      }
    }, ms);
  };
}

function setStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
}
