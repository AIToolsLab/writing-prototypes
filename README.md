---
title: Writing Tools Prototypes
emoji: 🏢
colorFrom: yellow
colorTo: yellow
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Static web version (`web/`)

Plain HTML/JS reimplementations that talk directly to a stock vLLM server (no custom inference code):

- `web/tokens.html`: token-level. Highlights use `prompt_logprobs` over the document placed in the assistant turn; predictions use next-token `logprobs` plus a short greedy lookahead to finish words.
- `web/structured.html`: asks the model, via chat completions with a JSON schema, to list edits or suggest next phrases.

To avoid the chat template trimming whitespace, the chat prefix (with `add_generation_prompt`) and the response text are tokenized separately via `/tokenize` and their token ids concatenated.

Run locally with `cd web && python3 -m http.server`, then open http://localhost:8000. Add `?server=https://your-vllm-host` to a page URL to use a different server.
