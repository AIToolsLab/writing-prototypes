// Run the app's actual prediction engine (docs/engine-tokens.js) on the cases in cases.json.
//
//   node engine_check.mjs                    # all cases
//   node engine_check.mjs "Deep neural net"  # just these drafts
//
// Loads docs/llm.js and docs/engine-tokens.js into a VM context with the few browser globals
// they use. Set VLLM_SERVER to use a different server.

import fs from "fs";
import vm from "vm";

const here = new URL(".", import.meta.url).pathname;
const docs = new URL("../../docs/", import.meta.url).pathname;
const DOC = "Deep learning neural network technology advances are pretty cool if you are careful " +
  "to use it in ways that don't take stuff from people.";

const server = process.env.VLLM_SERVER;
const ctx = vm.createContext({location: {search: server ? `?server=${server}` : ""}, URLSearchParams, fetch, performance, console});
vm.runInContext(fs.readFileSync(docs + "llm.js", "utf8") + "\n" + fs.readFileSync(docs + "engine-tokens.js", "utf8") +
  "\nglobalThis.engine = {loadModel, taskMessages, tokenSuggestions};", ctx);
const {loadModel, taskMessages, tokenSuggestions} = ctx.engine;

const args = process.argv.slice(2);
const cases = JSON.parse(fs.readFileSync(here + "cases.json", "utf8"))
  .filter(c => !args.length || args.includes(c.draft));
for (const draft of args) if (!cases.some(c => c.draft === draft)) cases.push({draft, want: "?", starts: null});

await loadModel();
let passed = 0;
for (const {draft, want, starts} of cases) {
  const messages = [...taskMessages("Rewrite this document to be clear and concise.", DOC), {role: "assistant", content: draft}];
  const {suggestions, warning, status} = await tokenSuggestions({messages, words: 2, onUpdate() {}});
  const top = suggestions[0]?.text ?? "";
  const ok = starts !== null && top.trim() !== "" && top.startsWith(starts);
  passed += ok;
  console.log(`\n${starts === null ? "?" : ok ? "✓" : "✗"} ${JSON.stringify(draft)}  want: ${want}  (${status})`);
  if (warning) console.log(`  warning: ${warning}`);
  console.log("  " + suggestions.map(s => `${JSON.stringify(s.text)} ${Math.log(s.prob).toFixed(1)}`).join(" | "));
}
console.log(`\nTop suggestion as wanted: ${passed} of ${cases.filter(c => c.starts !== null).length}`);
