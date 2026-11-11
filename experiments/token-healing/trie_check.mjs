// Run the Tokens engine (docs/engine-tokens.js, called the Trie engine until 2026-11-11, against engine/server.cpp) on the cases in cases.json.
//
//   node trie_check.mjs                    # all cases
//   node trie_check.mjs "Deep neural net"  # just these drafts
//
// Set ENGINE_SERVER to use a server other than http://localhost:52409.

import fs from "fs";
import vm from "vm";

const here = new URL(".", import.meta.url).pathname;
const docs = new URL("../../docs/", import.meta.url).pathname;
const DOC = "Deep learning neural network technology advances are pretty cool if you are careful " +
  "to use it in ways that don't take stuff from people.";

const server = process.env.ENGINE_SERVER;
const ctx = vm.createContext({location: {search: `?engine-server=${server ?? "http://localhost:52409"}`}, URLSearchParams, fetch, performance, console, Intl, DOMException});
vm.runInContext(["llm.js", "engine-tokens.js"].map(f => fs.readFileSync(docs + f, "utf8")).join("\n") +
  "\nglobalThis.engine = {taskMessages, trieEngine: tokensEngine};", ctx);
const {taskMessages, trieEngine} = ctx.engine;

const args = process.argv.slice(2);
const cases = JSON.parse(fs.readFileSync(here + "cases.json", "utf8"))
  .filter(c => !args.length || args.includes(c.draft));
for (const draft of args) if (!cases.some(c => c.draft === draft)) cases.push({draft, want: "?", starts: null});

await trieEngine.ready();
let passed = 0;
for (const {draft, want, starts} of cases) {
  const messages = [...taskMessages("Rewrite this document to be clear and concise.", DOC), {role: "assistant", content: draft}];
  let local = null;
  const {suggestions, warning, status} = await trieEngine.suggest({messages, words: 2, onUpdate(items) { local = items; }});
  const top = suggestions[0]?.text ?? "";
  const ok = starts !== null && top.trim() !== "" && top.startsWith(starts);
  passed += ok;
  console.log(`\n${starts === null ? "?" : ok ? "✓" : "✗"} ${JSON.stringify(draft)}  want: ${want}  (${status})`);
  if (warning) console.log(`  warning: ${warning}`);
  if (local) console.log("  shown before the server answered: " + local.map(s => `${JSON.stringify(s.text)} ${Math.round(s.prob * 100)}%`).join(" | "));
  console.log("  " + suggestions.map(s => `${JSON.stringify(s.text)} ${Math.round(s.prob * 100)}%`).join(" | "));
}
console.log(`\nTop suggestion as wanted: ${passed} of ${cases.filter(c => c.starts !== null).length}`);
