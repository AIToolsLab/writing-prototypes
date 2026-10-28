// The page: two modes (highlights, predictive text) over one engine at a time.
// Predictive text has two views of the same thing: a one-turn rewrite, and a conversation.

const $ = id => document.getElementById(id);
const ALT_KEY = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌥" : "Alt+";
const SEND_KEY = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘↵" : "Ctrl+↵";

const ENGINES = {tokens: tokensEngine, structured: structuredEngine};

const REWRITE_ADJECTIVES = [
  "clear and concise", "more detailed and engaging", "more formal and professional",
  "more casual and conversational", "more technical and precise", "more creative and imaginative",
  "more persuasive and compelling",
];
const GENERATE_PROMPTS = [
  "Summarize this document in one sentence.",
  "Translate this document into Spanish.",
  "Write a concise essay according to this outline.",
  "Write a detailed essay according to this outline.",
];

const state = {
  mode: "highlights",  // or "predict"
  view: "rewrite",     // or "chat"; only for predict mode
  engine: "tokens",
  instruction: "Rewrite this document to be clear and concise.",
  // The last highlights result, for the text it scored. Kept while the text changes,
  // shifted to follow the edits, until a new result arrives.
  highlights: {text: "", ranges: []},
  wantedHighlights: 8,
  chat: [{role: "user", content: ""}],  // the last message is the one being written
  suggestions: {draft: null, items: []},
};

// Conversation view always uses the tokens engine: the structured prompts are written for
// one request and one response.
function engine() {
  return state.mode === "predict" && state.view === "chat" ? tokensEngine : ENGINES[state.engine];
}

function setText(id, text) {
  $(id).textContent = text ?? "";
}

// ── Layout and URL ─────────────────────────────────────────────────────────

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  if (["highlights", "predict"].includes(params.get("mode"))) state.mode = params.get("mode");
  if (["rewrite", "chat"].includes(params.get("view"))) state.view = params.get("view");
  if (params.get("engine") in ENGINES) state.engine = params.get("engine");
}

function writeHash() {
  const params = {mode: state.mode, engine: state.engine};
  if (state.mode === "predict") params.view = state.view;
  history.replaceState(null, "", location.pathname + location.search + "#" + new URLSearchParams(params));
}

function layout() {
  const highlighting = state.mode === "highlights";
  const chatting = !highlighting && state.view === "chat";
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.setAttribute("aria-selected", tab.dataset.mode === state.mode);
  }
  $("highlights-view").hidden = !highlighting;
  $("rewrite-view").hidden = highlighting || chatting;
  $("chat-view").hidden = !chatting;
  $("prompt-section").hidden = chatting;
  document.body.classList.toggle("mode-predict", !highlighting);

  // One document editor and one suggestions panel, moved to wherever they're needed.
  $(highlighting ? "doc-slot-highlights" : "doc-slot-rewrite").append($("doc-editor"));
  $("doc-editor").classList.toggle("plain", !highlighting);
  $(chatting ? "predict-slot-chat" : "predict-slot-rewrite").append($("predict-panel"));

  $("engine").value = engine().name;
  $("engine").disabled = chatting;
  setText("engine-note", chatting ? "(conversations use Tokens)" : "");
  hideTooltip();
  writeHash();
  for (const ta of document.querySelectorAll("textarea.editor")) autogrow(ta);
  renderSuggestions();
  update();
}

// Grow a textarea to fit its text, so it never scrolls: the page scrolls instead. This also
// keeps the highlight backdrop aligned without syncing scroll positions.
function autogrow(ta) {
  if (!ta.offsetParent) return;  // hidden
  ta.style.height = "auto";
  ta.style.height = ta.scrollHeight + (ta.offsetHeight - ta.clientHeight) + "px";
}

// ── Requests ───────────────────────────────────────────────────────────────

// Returns a function that runs `run` after the user pauses, unless `key()` hasn't changed since
// the last run. A change aborts the in-flight run, since its result would be stale.
function updater({key, run, delay, statusId}) {
  let timer = null, controller = null, lastKey = null;
  return () => {
    const k = key();
    if (k === lastKey) return;
    lastKey = k;
    clearTimeout(timer);
    controller?.abort();
    timer = setTimeout(async () => {
      controller = new AbortController();
      try {
        await run(controller.signal);
      } catch (err) {
        if (err.name === "AbortError") return;
        console.error(err);
        lastKey = null;  // allow a retry
        setText(statusId, "Error: " + err.message);
      }
    }, delay());
  };
}

const updateHighlights = updater({
  key: () => JSON.stringify([engine().name, state.instruction, $("doc").value]),
  delay: () => engine().highlightDelay,
  statusId: "highlights-status",
  run: async signal => {
    const doc = $("doc").value;
    if (!doc.trim()) {
      state.highlights = {text: doc, ranges: [], scored: false};
      setText("highlights-status", "");
      renderHighlights();
      return;
    }
    setText("highlights-status", "Scoring…");
    const result = await engine().highlights({instruction: state.instruction, doc}, signal);
    state.highlights = {text: doc, ranges: result.ranges, scored: true};
    setText("highlights-status", result.status);
    setText("highlights-warning", result.warning);
    renderHighlights();
  },
});

const updateSuggestions = updater({
  key: () => JSON.stringify([engine().name, currentMessages(), $("words").value]),
  delay: () => 150,
  statusId: "predict-status",
  run: async signal => {
    const messages = currentMessages();
    const draft = messages.at(-1).content;
    setText("predict-status", "Predicting…");
    const result = await engine().suggest({
      messages,
      words: +$("words").value,
      onUpdate: items => showSuggestions(draft, items),
    }, signal);
    showSuggestions(draft, result.suggestions);
    setText("predict-status", result.status);
    setText("predict-warning", result.warning);
  },
});

// Only the visible mode makes requests.
function update() {
  if (!MODEL) return;
  if (state.mode === "highlights") updateHighlights();
  else updateSuggestions();
}

// ── Prompt picker ──────────────────────────────────────────────────────────

function buildPromptPicker() {
  const chip = (label, instruction, container) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.dataset.instruction = instruction;
    btn.addEventListener("click", () => {
      setInstruction(instruction);
      $("prompt-picker").open = false;
    });
    $(container).append(btn);
  };
  for (const adj of REWRITE_ADJECTIVES) chip(adj, `Rewrite this document to be ${adj}.`, "rewrite-chips");
  for (const prompt of GENERATE_PROMPTS) chip(prompt.replace(/\.$/, ""), prompt, "generate-chips");
}

function setInstruction(instruction) {
  state.instruction = instruction;
  setText("prompt-text", instruction);
  if (document.activeElement !== $("instruction")) $("instruction").value = instruction;
  for (const btn of document.querySelectorAll(".chips button")) {
    btn.setAttribute("aria-pressed", btn.dataset.instruction === instruction);
  }
  update();
}

// ── Highlights ─────────────────────────────────────────────────────────────

function renderHighlights() {
  const doc = $("doc").value;
  const {text, ranges, scored} = state.highlights;

  // With scores, pick the top N; the slider ranges over all the candidates.
  const scoredRanges = ranges.filter(r => r.score != null);
  let selected = ranges;
  $("num-highlights-control").hidden = scoredRanges.length === 0;
  if (scoredRanges.length) {
    const n = Math.min(state.wantedHighlights, scoredRanges.length);
    $("num-highlights").max = scoredRanges.length;
    $("num-highlights").value = n;
    setText("num-highlights-label", `${n} of ${scoredRanges.length}`);
    const top = new Set([...scoredRanges].sort((a, b) => b.score - a.score).slice(0, n));
    selected = ranges.filter(r => top.has(r));
  }
  setText("highlights-message",
    !doc.trim() ? "Type or paste a document above." :
    scored && ranges.length === 0 ? "No possible edits found: the AI would have written this just as you did." : "");
  paintBackdrop(doc, followEdits(text, doc, selected));
}

// Ranges computed for `before`, adjusted to `after`: ranges outside the edited region shift
// along with the text; ranges that overlap it are dropped until the next result.
function followEdits(before, after, ranges) {
  if (before === after) return ranges;
  const max = Math.min(before.length, after.length);
  let prefix = 0;
  while (prefix < max && before[prefix] === after[prefix]) prefix++;
  let suffix = 0;
  while (suffix < max - prefix && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]) suffix++;
  const shift = after.length - before.length;
  const editEnd = before.length - suffix;
  return ranges.flatMap(r =>
    r.end <= prefix ? [r] :
    r.start >= editEnd ? [{...r, start: r.start + shift, end: r.end + shift}] : []);
}

// The backdrop has the same text, font, and box as the textarea, but transparent text, so its
// marks line up behind the textarea's (transparent-background) text.
function paintBackdrop(doc, ranges) {
  const backdrop = $("doc-backdrop");
  backdrop.textContent = "";
  let pos = 0;
  for (const range of ranges) {
    backdrop.append(doc.slice(pos, range.start));
    const mark = document.createElement("mark");
    mark.textContent = doc.slice(range.start, range.end);
    mark.range = range;
    // A highlighted newline or space has no visible area of its own, so give it a marker.
    if (!mark.textContent.trim()) {
      mark.className = "whitespace";
      const marker = document.createElement("span");
      marker.className = "marker";
      mark.append(marker);
    }
    const alt = document.createElement("span");
    alt.className = "alt";
    alt.textContent = showToken(range.alternative);
    mark.append(alt);
    backdrop.append(mark);
    pos = range.end;
  }
  backdrop.append(doc.slice(pos));
}

// The textarea is on top, so the marks can't get mouse events; hit-test them instead.
function hoverHighlight(e) {
  if (state.mode !== "highlights") return;
  for (const mark of $("doc-backdrop").querySelectorAll("mark")) {
    const marker = mark.querySelector(".marker");
    for (const rect of [...mark.getClientRects(), ...(marker ? marker.getClientRects() : [])]) {
      if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
        showTooltip(rect, mark.range);
        return;
      }
    }
  }
  hideTooltip();
}

function showTooltip(rect, range) {
  const tooltip = $("tooltip");
  tooltip.textContent = "";
  const alt = document.createElement("span");
  alt.className = "alt";
  alt.textContent = showToken(range.alternative);
  const detail = document.createElement("span");
  detail.className = "detail";
  detail.textContent = range.detail ?? "";
  tooltip.append(alt, detail);
  tooltip.style.left = `${rect.left + rect.width / 2}px`;
  tooltip.style.top = `${rect.top}px`;
  tooltip.hidden = false;
}

function hideTooltip() {
  $("tooltip").hidden = true;
}

// ── Predictions ────────────────────────────────────────────────────────────

function activeDraft() {
  return state.view === "chat" ? $("chat-draft") : $("draft");
}

// The conversation so far, ending with the message being written.
function currentMessages() {
  if (state.view === "chat") return state.chat;
  return [...taskMessages(state.instruction, $("doc").value), {role: "assistant", content: $("draft").value}];
}

function showSuggestions(draft, items) {
  state.suggestions = {draft, items};
  renderSuggestions();
}

// A fixed number of slots, so they don't jump around as they update. Suggestions for a draft
// other than the current one are shown dimmed until new ones arrive, and can't be picked.
function renderSuggestions() {
  const {draft, items} = state.suggestions;
  const fresh = draft === activeDraft().value;
  const container = $("suggestions");
  container.innerHTML = "";
  for (let i = 0; i < NUM_SUGGESTIONS; i++) {
    const btn = document.createElement("button");
    btn.type = "button";
    container.append(btn);
    const item = items[i];
    const key = document.createElement("kbd");
    key.textContent = ALT_KEY + (i + 1);
    btn.append(key);
    if (!item) { btn.disabled = true; continue; }
    const label = document.createElement("span");
    label.textContent = showToken(item.text);
    btn.append(label);
    if (item.prob != null) {
      const prob = document.createElement("small");
      prob.textContent = item.prob < 0.005 ? "<1%" : `${Math.round(item.prob * 100)}%`;
      btn.append(prob);
    }
    btn.title = item.text;
    btn.disabled = !fresh;
    btn.addEventListener("click", () => acceptSuggestion(i));
  }
}

function acceptSuggestion(i) {
  const {draft, items} = state.suggestions;
  const ta = activeDraft();
  if (!items[i] || ta.value !== draft) return;
  ta.value = draft + items[i].text;
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  ta.dispatchEvent(new Event("input"));
}

// ── Conversation ───────────────────────────────────────────────────────────

const ROLE_NAMES = {user: "User", assistant: "Assistant"};

function renderChat() {
  const history = $("chat-history");
  history.innerHTML = "";
  state.chat.slice(0, -1).forEach((message, i) => {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.role}`;
    const role = document.createElement("div");
    role.className = "role";
    role.append(ROLE_NAMES[message.role]);
    const edit = document.createElement("button");
    edit.className = "link";
    edit.textContent = "Edit";
    edit.title = "Go back to this message (removes the messages after it)";
    edit.addEventListener("click", () => rewindChat(i));
    role.append(edit);
    const content = document.createElement("div");
    content.className = "content";
    content.textContent = message.content;
    bubble.append(role, content);
    history.append(bubble);
  });
  const current = state.chat.at(-1);
  $("chat-current").className = `bubble in-progress ${current.role}`;
  setText("chat-role", `${ROLE_NAMES[current.role]} (you're writing)`);
  $("chat-draft").value = current.content;
  autogrow($("chat-draft"));
  renderSuggestions();
  update();
}

function rewindChat(i) {
  state.chat = state.chat.slice(0, i + 1);
  renderChat();
  $("chat-draft").focus();
}

function sendChat() {
  const role = state.chat.at(-1).role === "user" ? "assistant" : "user";
  state.chat = [...state.chat, {role, content: ""}];
  renderChat();
  $("chat-draft").focus();
}

// Start the conversation from the rewrite, unless one has already continued from this rewrite.
function continueAsChat() {
  const seed = currentMessages();
  if (!(state.chat.length > 2 && state.chat[0].content === seed[0].content)) state.chat = seed;
  state.view = "chat";
  layout();
  renderChat();
  $("chat-draft").focus();
}

// ── Wiring ─────────────────────────────────────────────────────────────────

for (const tab of document.querySelectorAll(".tabs button")) {
  tab.addEventListener("click", () => { state.mode = tab.dataset.mode; layout(); });
}
$("engine").addEventListener("change", e => {
  state.engine = e.target.value;
  state.highlights = {text: "", ranges: []};
  state.suggestions = {draft: null, items: []};
  for (const id of ["highlights-status", "highlights-warning", "predict-status", "predict-warning"]) setText(id, "");
  renderHighlights();
  layout();
});

$("instruction").addEventListener("input", e => setInstruction(e.target.value));
// Close the prompt picker on a click outside it, or Escape.
document.addEventListener("click", e => {
  if (!$("prompt-picker").contains(e.target)) $("prompt-picker").open = false;
});
document.addEventListener("keydown", e => {
  if (e.key !== "Escape" || !$("prompt-picker").open) return;
  const wasInside = $("prompt-picker").contains(document.activeElement);
  $("prompt-picker").open = false;
  if (wasInside) $("prompt-picker").querySelector("summary").focus();
});

$("doc").addEventListener("input", () => { autogrow($("doc")); renderHighlights(); update(); });
$("doc").addEventListener("mousemove", hoverHighlight);
$("doc").addEventListener("mouseleave", hideTooltip);
$("doc").addEventListener("scroll", hideTooltip);
window.addEventListener("scroll", hideTooltip);
window.addEventListener("resize", () => { for (const ta of document.querySelectorAll("textarea.editor")) autogrow(ta); });
$("num-highlights").addEventListener("input", e => { state.wantedHighlights = +e.target.value; renderHighlights(); });
$("show-alts").addEventListener("change", e => $("doc-backdrop").classList.toggle("show-alts", e.target.checked));

$("draft").addEventListener("input", () => { autogrow($("draft")); renderSuggestions(); update(); });
$("words").addEventListener("change", update);
$("chat-draft").addEventListener("input", e => {
  state.chat = [...state.chat.slice(0, -1), {...state.chat.at(-1), content: e.target.value}];
  autogrow(e.target);
  renderSuggestions();
  update();
});
$("chat-draft").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); sendChat(); }
});
$("chat-send").addEventListener("click", sendChat);
$("chat-reset").addEventListener("click", () => { state.chat = [{role: "user", content: ""}]; renderChat(); $("chat-draft").focus(); });
$("to-chat").addEventListener("click", e => { e.preventDefault(); continueAsChat(); });
$("to-rewrite").addEventListener("click", e => { e.preventDefault(); state.view = "rewrite"; layout(); });

// Alt+1–5 picks a suggestion. Use the key's position (code), since on a Mac Option+1 types "¡".
document.addEventListener("keydown", e => {
  const match = e.code.match(/^Digit([1-9])$/);
  if (state.mode !== "predict" || !match || !e.altKey || e.ctrlKey || e.metaKey) return;
  e.preventDefault();
  acceptSuggestion(+match[1] - 1);
});

readHash();
buildPromptPicker();
setText("server", BASE_URL);
setText("keys-hint", `${ALT_KEY}1–${ALT_KEY}${NUM_SUGGESTIONS} picks a suggestion`);
setText("send-key", SEND_KEY);
setInstruction(state.instruction);
layout();
renderChat();
loadModel().then(update, err => {
  setText("highlights-status", "Error: " + err.message);
  setText("predict-status", "Error: " + err.message);
});
