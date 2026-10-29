## Static web version (`docs/`)

A plain HTML/JS reimplementation that talks directly to a stock vLLM server (no custom inference code). One page (`docs/index.html`, UI in `app.js`) with two modes, Highlights and Predictive text. Predictive text defaults to rewriting a document, and can continue as a multi-turn conversation where you write both sides. Two interchangeable engines:

- `engine-tokens.js` (default): highlights use `prompt_logprobs` over the document placed in the assistant turn; predictions use next-token `logprobs` plus a short greedy lookahead to finish words.
- `engine-structured.js`: asks the model, via chat completions with a JSON schema, to list edits or suggest next phrases. Not available in conversation view.

To avoid the chat template trimming whitespace, the response text never goes through the template: we render the chat prefix once (`/tokenize` with `add_generation_prompt`, then `/detokenize`), cache it, and send `prefix + response` as a text prompt. vLLM parses the special-token strings in it, giving the same token ids as tokenizing the two parts separately.

Run locally with `cd docs && python3 -m http.server`, then open http://localhost:8000. Add `?server=https://your-vllm-host` to the URL to use a different server.
