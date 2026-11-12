# Handoff

The state of the machine and what's left to do. Read `engine/README.md` (the engine and its API) and
`design/interactions.md` (the interaction ideas) first. Ken Arnold (Calvin University) researches writing with
AI; the core idea is that there is a limit to how far the AI should get ahead of the writer's own thinking. He
wants short replies and likes to talk design through before building. The codebase is ours to change.

## State (2026-11-12)

- The **engine** runs in Docker (`engine/compose.yaml`, container `writing-engine`, `127.0.0.1:52409`, public as
  llm.thoughtful-ai.com via Caddy), serving Qwen3.8-27B Q4_K_XL from `/storage/HuggingFace` at the revision pinned in
  `compose.yaml`. The page's "Tokens" mode uses it; "Structured" uses the campus server with each person's own key.
- **vLLM is stopped** (`~/vllm-server`, port 52408, was vllm.thoughtful-ai.com). It and the engine don't both fit on
  the RTX 4090. To swap: `docker compose stop` in `engine/`, then `docker compose start vllm` in `~/vllm-server`.
- **Tests:** `cd engine && ENGINE_MODEL=<Qwen3.5-2B gguf> uv run pytest` (CPU, about 30 s; see "Tests" in the README).
  Speed: `engine/bench/` and `engine/bench/LOG.md`; the target is under 200 ms per keystroke.

## To do

### Now: unblock students on the old vLLM API

- [x] vLLM-compatible `/tokenize`, `/detokenize`, `/v1/completions` in the engine (2026-11-12; tested on the 2B model).
- [x] Deployed 2026-11-12. vllm.thoughtful-ai.com now goes to the engine too (Caddy), and the full suite passes
      through it. Students' `model` field can be anything.
- [ ] Tell them what's not there: chat completions, constrained decoding, streaming, penalties (all refused with a 400).

### Next: lm-internals

The client (https://cs.calvin.edu/courses/cs/376/cur/lm-internals.html) calls `/tokenize` with `messages` and
`continue_final_message`, then `/v1/completions` with `echo: true, logprobs: 10` on the token ids. That flow passes
(`tests/test_vllm_compat.py::test_lm_internals_flow`, mirroring `~/vllm-server/bench/client_check.py`).
- [ ] Point it at the engine (routing, as above) and try the real page in a browser; it may use options the check doesn't.
- [ ] Classroom load: requests run one at a time, and each new prompt makes a session that can push writers' caches to
      host memory. Consider a separate pool or no caching for compatibility requests, and measure with ~30 clients.
- [ ] Echo scores every prompt position. An extension that scores from a given position would cut the logits work.

### Engine gaps (each has an expected failure in `engine/tests/test_known_gaps.py`)

- [ ] `/v1/score` conditions on tokens, not characters: its last-word logprob is a lower bound and doesn't match what
      `/v1/continuations` reports. Matters for the mirror.
- [ ] Return the search tree with each node's pruned probability, so clients can bound conditionals inside a
      continuation (P(word 2 | word 1)). Today only P(continuation | typed text) is reliable.
- [ ] `/v1/compare {prompt, text, options}`: probability of named strings, one fork per option, summed over
      tokenizations like continuations. Needed for moves and either/or spans.
- [ ] A session restored from host memory has lost its checkpoints, so a jump back decodes from scratch.
- [ ] Scoring re-decodes everything after an edit; the search tree is rebuilt on every keystroke.
- [ ] No logging, authentication, or rate limiting (binds to localhost; Caddy is the only way in).

### Page

- [ ] Switch `docs/engine-tokens.js` to send `{messages, role}` instead of rendering prompts itself (`turnPrefix` and
      `PLACEHOLDER` go away), once the deployed engine has it.
- [ ] Not verified with the engine: the conversation view, and accepting suggestions (click or Alt+digit).

### New interactions

Ken liked all of these and hasn't picked one; ask. `design/interactions.md` has examples and measured probes. The
frame: **the AI may sketch ahead or spell the next word, never both**; when revising, **the model reads and the
human writes**. Suggested first: pencil stubs, then moves.

1. Pencil stubs: a note ahead of the cursor conditions predictions to head there (put the stub after stable text).
2. Moves, not words: what kind of thing could come next, with probabilities; picking one makes a stub. Needs `/v1/compare`.
3. The mirror: shade what the model would have predicted anyway. Needs character-level `/v1/score`.
4. Count choices, not bits: visible provenance (typed, picked from 5, accepted the default).
5. Old draft as the pencil layer: retype a sentence with predictions conditioned on the old version.
6. Ripples: after an edit, which later sentences became more surprising (two scores, diffed).
7. Either/or spans: which of two versions the following text reads better after. Needs `/v1/compare`.
8. Gist-back: what a reader takes each paragraph to mean. Generates text, so it uses the campus server.

Campus server: `https://localai.calvin.edu/api` (OpenAI-style; key in `~/vllm-server/localai.calvin.edu-key`, never
print or commit it). Others use it, so changes there go to the sysadmin as proposals.

### Model updates

`engine/check-model-update.sh` reports new Unsloth uploads at container start (logs only); Ken wants to hear about
them but not have them downloaded automatically. To switch: download, bump `MODEL_REVISION`, rerun the tests.

## Learned the hard way

- Qwen3.5+ are hybrids with recurrent layers whose state can't be partly rewound; hence checkpoints, and about 150 MB
  of GPU memory per sequence, which caps sequences at 24 on the 4090.
- `llama_memory_seq_cp` shares cells without copying: checkpoints and forks are free until they diverge.
- Single-character probes can't find the tokenization boundary (" pow" vs " power"); the engine backs up to the
  whitespace before the last word instead.
- An end-of-turn leaf can carry text ("le." then end); only an empty-text end leaf means "stop here".
- `time_ms` includes the prefill; before 2026-11-12 a slow prefill meant no search and an empty suggestion.
- **Never `pkill writing-engine` on the host:** it also kills the production engine in its container. Kill a dev
  engine by PID.
- Headless Firefox (a snap) needs its profile and screenshots under `~/snap/firefox/common/`; `--screenshot` fires
  at page load, before results arrive.
