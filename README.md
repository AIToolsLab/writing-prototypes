## Static web version (`docs/`)

A plain HTML/JS page (`docs/index.html`, UI in `app.js`) with two modes, Highlights and Predictive text. Predictive text defaults to rewriting a document, and can continue as a multi-turn conversation where you write both sides. Two interchangeable engines:

- `engine-tokens.js` (default): reads the model's probabilities from our own inference server in [`engine/`](engine/README.md), public at https://llm.thoughtful-ai.com. Highlights score the document placed in the assistant turn; predictions are conditioned on the characters typed, so they need no token healing (about 50–200 ms each). Until 2026-11-11 this was the "Trie" engine, and "Tokens" read token logprobs from vLLM; that code is in git history (commit 323ad08). `#engine=trie` links still work.
- `engine-structured.js`: asks a chat model, via chat completions with a JSON schema, to list edits or suggest next phrases. Uses any OpenAI-style server, by default Calvin's Open WebUI (`https://localai.calvin.edu/api`), with each person's own API key pasted into the page (kept in localStorage). Not available in conversation view.

Run locally with `cd docs && python3 -m http.server`, then open http://localhost:8000. URL parameters: `?engine-server=http://localhost:52409` for a different engine server, `?chat-server=https://host/v1` for a different chat server.
