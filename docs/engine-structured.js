// Structured engine: uses ordinary chat completions, asking the model to *describe* its edits or
// suggestions as JSON (schema-constrained). Prompts put stable content first and the changing
// draft last so vLLM's prefix cache can reuse work. Same interface as engine-tokens.js.

const structuredEngine = {
  name: "structured",
  highlightDelay: 800,
  highlights: structuredHighlights,
  suggest: structuredSuggestions,
};

// ── Edits ──────────────────────────────────────────────────────────────────

const EDITS_SYSTEM = `You are a careful editor. The user gives an instruction and a document. Do not rewrite the whole document. Instead, list the specific edits you would make to carry out the instruction, as several small, separate edits rather than one large one. For each edit, "original" must be copied exactly from the document (a short span, a few words), "replacement" is what you would write instead, and "reason" is a few words.`;

const EDITS_SCHEMA = {
  type: "object",
  properties: {
    edits: {
      type: "array",
      items: {
        type: "object",
        properties: {
          original: {type: "string"},
          replacement: {type: "string"},
          reason: {type: "string"},
        },
        required: ["original", "replacement", "reason"],
      },
    },
  },
  required: ["edits"],
};

async function structuredHighlights({instruction, doc}, signal) {
  const t0 = performance.now();
  const {result, usage} = await chatJSON([
    {role: "system", content: EDITS_SYSTEM},
    {role: "user", content: `Instruction: ${instruction}\n\nDocument:\n${doc}`},
  ], EDITS_SCHEMA, {signal});
  const ranges = locateEdits(doc, result.edits);
  const missing = result.edits.length - ranges.length;
  return {
    ranges,
    status: `${ranges.length} edits in ${Math.round(performance.now() - t0)} ms (${usage.completion_tokens} tokens generated)` +
      (missing ? `; ${missing} quote(s) not found in the document` : ""),
  };
}

// Find each edit's quote in the document; drop ones that don't match or overlap an earlier one.
function locateEdits(doc, edits) {
  const placed = [];
  for (const edit of edits) {
    if (!edit.original) continue;
    let start = doc.indexOf(edit.original);
    while (start !== -1 && placed.some(p => start < p.end && p.start < start + edit.original.length)) {
      start = doc.indexOf(edit.original, start + 1);
    }
    if (start === -1) continue;
    placed.push({start, end: start + edit.original.length, alternative: edit.replacement, detail: edit.reason});
  }
  return placed.sort((a, b) => a.start - b.start);
}

// ── Predictions ────────────────────────────────────────────────────────────

const SUGGEST_SYSTEM = `You help a person write a response to a request. Given the request and the person's draft so far, suggest 5 different short phrases (WORDS each) that could come next, continuing directly from the end of the draft. Order them from most to least likely. Give only the continuation text, not the draft.`;

const SUGGEST_SCHEMA = {
  type: "object",
  properties: {suggestions: {type: "array", items: {type: "string"}}},
  required: ["suggestions"],
};

async function structuredSuggestions({messages, words}, signal) {
  const draft = messages.at(-1).content;
  const request = messages.slice(0, -1).map(m => m.content).join("\n\n");
  const t0 = performance.now();
  const wordCount = words === 1 ? "1 word" : `about ${words} words`;
  // Stable content (system, request) first; the draft, which changes on every keystroke, last.
  const {result, usage} = await chatJSON([
    {role: "system", content: SUGGEST_SYSTEM.replace("WORDS", wordCount)},
    {role: "user", content: `Request:\n${request}\n\nDraft so far:\n${draft}`},
  ], SUGGEST_SCHEMA, {maxTokens: 100, signal});
  return {
    suggestions: result.suggestions.slice(0, 5).map(phrase => ({text: spacedPhrase(draft, phrase)})),
    status: `Suggested in ${Math.round(performance.now() - t0)} ms (${usage.completion_tokens} tokens generated)`,
  };
}

// The model gives bare phrases, so we decide the spacing: add a space unless the draft
// already ends in whitespace or the phrase starts with punctuation.
function spacedPhrase(draft, phrase) {
  phrase = phrase.replace(/^ +/, "");
  const needsSpace = draft && !/\s$/.test(draft) && !/^[.,;:!?)'’"]/.test(phrase);
  return (needsSpace ? " " : "") + phrase;
}
