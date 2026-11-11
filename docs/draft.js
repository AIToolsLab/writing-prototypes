// Draft mode: writing from an assignment, with three aids meant to keep the AI from getting ahead
// of the writer's own thinking (design/interactions.md):
//   Pencil stub: a note about where the draft is heading, which the predictions lean toward.
//   Moves:       what kind of sentence could come next, never its words. Picking one sets the stub.
//   Mirror:      the writer's own words, shaded by how predictable the model found them.
// Uses the Tokens engine only. Loaded before app.js, which calls initDraft() and uses the rest.

const draftState = {
  stub: "",  // the note in effect; the input may hold one that isn't applied yet
  moves: {draft: null, items: []},
  mirrorOn: false,
  // The last mirror result, for the text it scored; shifted to follow edits until a new one arrives.
  mirror: {text: "", words: []},
};

// The stub goes after the assignment, so changing it leaves the assignment's cache alone.
function draftMessages() {
  const note = draftState.stub ? `\n\n[My notes on where the draft goes next: ${draftState.stub}]` : "";
  return [
    {role: "user", content: $("assignment").value.trim() + note},
    {role: "assistant", content: $("compose").value},
  ];
}

// ── Pencil stub ────────────────────────────────────────────────────────────

// A stub takes effect when the writer commits it (Enter, or leaving the field), not on every
// keystroke: everything after it in the prompt is the whole draft, which has to be read again.
function applyStub(text) {
  draftState.stub = text.trim();
  $("stub").value = draftState.stub;
  renderStub();
  update();
}

function renderStub() {
  const pending = $("stub").value.trim() !== draftState.stub;
  $("stub-row").classList.toggle("active", !!draftState.stub && !pending);
  setText("stub-hint", pending ? "Enter to apply" : draftState.stub ? "predictions lean toward this" : "");
  $("stub-clear").hidden = !$("stub").value;
}

// ── Moves ──────────────────────────────────────────────────────────────────

let movesController = null;

async function askMoves() {
  movesController?.abort();
  const controller = movesController = new AbortController();
  const draft = $("compose").value;
  setText("moves-status", "Asking…");
  try {
    const result = await tokensEngine.moves({assignment: $("assignment").value.trim(), draft}, controller.signal);
    draftState.moves = {draft, items: result.moves};
    setText("moves-status", result.moves.length ? result.status : "The model had no clear answer.");
    renderMoves();
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error(err);
    setText("moves-status", "Error: " + err.message);
  }
}

// Moves asked about an earlier state of the draft are dimmed, but can still be picked.
function renderMoves() {
  const {draft, items} = draftState.moves;
  const container = $("moves");
  container.classList.toggle("stale", draft !== $("compose").value);
  container.replaceChildren(...items.map(move => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.append(move.text);
    const prob = document.createElement("small");
    prob.textContent = `${Math.round(move.prob * 100)}%`;
    btn.append(prob);
    btn.title = "How often the model, asked as an editor, names this move. Click to pencil it in.";
    btn.addEventListener("click", () => { applyStub(move.text); $("compose").focus(); });
    return btn;
  }));
}

// ── Mirror ─────────────────────────────────────────────────────────────────

const MIRROR_EXPECTED = 0.5, MIRROR_SURPRISE = 0.05;

let updateMirror = () => {};

function renderMirror() {
  const draft = $("compose").value;
  const backdrop = $("compose-backdrop");
  backdrop.textContent = "";
  $("mirror-legend").hidden = !draftState.mirrorOn;
  if (!draftState.mirrorOn) {
    setText("mirror-summary", "");
    return;
  }
  let words = followEdits(draftState.mirror.text, draft, draftState.mirror.words);
  // The last word may still be being typed; a half-typed word is always a surprise.
  if (words.length && words.at(-1).end === draft.length && !/[.!?…]["”')\]]*$/.test(draft)) words = words.slice(0, -1);

  let pos = 0;
  for (const word of words) {
    const kind = word.prob > MIRROR_EXPECTED ? "expected" : word.prob < MIRROR_SURPRISE ? "surprise" : "";
    if (!kind) continue;
    backdrop.append(draft.slice(pos, word.start));
    const span = document.createElement("span");
    span.className = kind;
    span.textContent = draft.slice(word.start, word.end);
    span.range = {
      alternative: word.alternative.trim(),
      detail: (word.alternative.trim() ? "was the model's guess · " : "") + `your word: ${percent(word.prob)}`,
    };
    backdrop.append(span);
    pos = word.end;
  }
  backdrop.append(draft.slice(pos));
  setText("mirror-summary", mirrorSummary(draft, words));
}

function percent(p) {
  return p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`;
}

// The useful message is about sentences: which one the model would have written anyway.
function mirrorSummary(draft, words) {
  if (!words.length) return "";
  const expected = words.filter(w => w.prob > MIRROR_EXPECTED).length;
  const surprises = words.filter(w => w.prob < MIRROR_SURPRISE).length;
  let summary = `Of ${words.length} words, the model saw ${expected} coming and didn't expect ${surprises}.`;

  let most = null;
  for (const match of draft.matchAll(/[^.!?\n]+(?:[.!?]+["”')\]]*|$)/g)) {
    const inside = words.filter(w => w.start >= match.index && w.end <= match.index + match[0].length);
    if (inside.length < 6) continue;
    const share = inside.filter(w => w.prob > MIRROR_EXPECTED).length / inside.length;
    if (!most || share > most.share) most = {share, text: match[0].trim(), count: inside.length};
  }
  if (most && most.share >= 0.4) {
    const quote = most.text.length > 60 ? most.text.slice(0, 57).trimEnd() + "…" : most.text;
    summary += ` Most predictable sentence: “${quote}” (${Math.round(most.share * most.count)} of ${most.count} words). Is it doing work?`;
  }
  return summary;
}

// The textarea is on top of the shading, so hit-test the shaded words (as highlights do).
function hoverMirror(e) {
  for (const span of $("compose-backdrop").children) {
    for (const rect of span.getClientRects()) {
      if (e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom) {
        showTooltip(rect, span.range);
        return;
      }
    }
  }
  hideTooltip();
}

// ── Wiring ─────────────────────────────────────────────────────────────────

// What draft mode refreshes besides predictions, whenever something changes.
function updateDraftAids() {
  renderStub();
  renderMoves();
  renderMirror();
  updateMirror();
}

function initDraft() {
  updateMirror = updater({
    key: () => JSON.stringify([draftState.mirrorOn, $("assignment").value, $("compose").value]),
    delay: () => 300,
    statusId: "mirror-status",
    run: async signal => {
      const draft = $("compose").value;
      if (!draftState.mirrorOn || !draft.trim()) {
        draftState.mirror = {text: draft, words: []};
        for (const id of ["mirror-status", "mirror-warning"]) setText(id, "");
        renderMirror();
        return;
      }
      setText("mirror-status", "Reading…");
      const result = await tokensEngine.mirror({assignment: $("assignment").value.trim(), draft}, signal);
      draftState.mirror = {text: draft, words: result.words};
      setText("mirror-status", result.status);
      setText("mirror-warning", result.warning);
      renderMirror();
    },
  });

  $("compose").addEventListener("input", () => { autogrow($("compose")); renderSuggestions(); update(); });
  $("compose").addEventListener("mousemove", hoverMirror);
  $("compose").addEventListener("mouseleave", hideTooltip);
  $("assignment").addEventListener("input", () => { autogrow($("assignment")); update(); });

  $("stub").addEventListener("input", renderStub);
  $("stub").addEventListener("change", e => applyStub(e.target.value));
  $("stub").addEventListener("keydown", e => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    applyStub(e.target.value);
    $("compose").focus();
  });
  $("stub-clear").addEventListener("click", () => { applyStub(""); $("compose").focus(); });

  $("moves-ask").addEventListener("click", askMoves);
  $("mirror-on").addEventListener("change", e => { draftState.mirrorOn = e.target.checked; updateDraftAids(); });
}
