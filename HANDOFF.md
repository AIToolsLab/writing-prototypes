# Handoff: writing engine and new interactions

Written 2026-11-11 for a fresh session picking this work up. Read `engine/README.md` and
`engine/MIGRATING.md` first; they describe what was built and why. This file covers what isn't in them:
the state of the machine, what to build next, the design thinking behind it, and what was learned the hard way.

## Who this is for and what they want

Ken Arnold (Calvin University) researches writing with AI. The core idea of this repo: **there is a limit to
how far the AI should get ahead of the writer's own thinking.** Highlights and predictive text are two examples.
Ken wants short replies, not walls of text, and likes to be asked questions and to talk through design. The
codebase is fully ours to change.

## State as of this handoff

- **Nothing from this work is committed.** `git status` shows the changes: `engine/` (new), `docs/engine-trie.js`
  (new), `experiments/token-healing/trie_check.mjs` (new), edits to `docs/{app,llm,engine-tokens,engine-structured}.js`,
  `docs/index.html`, `README.md`. Ask Ken before committing.
- **The engine is running** on `127.0.0.1:52409` as a bare process (started by hand, won't survive a reboot),
  serving `/storage/HuggingFace/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf`.
- **vLLM is stopped** (`~/vllm-server`, Docker Compose, port 52408, public as vllm.thoughtful-ai.com). The two
  don't fit on the RTX 4090 together. Ken said on 2026-11-11 that nobody is using vLLM right now and stopping it
  is fine; the public page's default "Tokens" engine is down while it's stopped. Swap back with
  `pkill -x writing-engine` then `docker compose start vllm` in `~/vllm-server`.
- **Verified:** 11 of 11 token-healing cases (`node experiments/token-healing/trie_check.mjs`); typing latency
  (`python3 engine/bench/typing_sim.py`); both page modes render results in headless Firefox.
- **Not verified:** the conversation view and accepting suggestions (click or Alt+digit) with the Trie engine.

## Task 1: Dockerize the engine

So it auto-starts like the vLLM container. Ken considers this trivial. Things that matter:

- `engine/CMakeLists.txt` fetches llama.cpp pinned at `f7b886f` and needs CUDA at build time
  (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`). The host has CUDA 13.3.
- Mount `/storage/HuggingFace` for the GGUF. Pass `--host 0.0.0.0` inside the container.
- It and vLLM can't both hold the GPU. Decide with Ken which one has `restart: unless-stopped`, and what
  happens to vllm.thoughtful-ai.com and to clients of vLLM (the `lm-internals` course client needs vLLM's
  prompt logprobs; see `~/vllm-server/BENCHMARKS.md`).
- The repo root has an untracked, older `Dockerfile` and `docker-compose.yml` from a previous approach; look
  before overwriting.

## Task 2: New interactions

### The frame Ken agreed to

The two existing modes are points in one space: how far ahead of the writer the AI is, and how finished its
output looks. What misleads a writer is fluent prose about ideas they haven't had. So the fence is:
**the AI may sketch ahead or spell the next word, never both.** Near and fine (predictive text) is allowed;
far and coarse (a move, a keyword, a question) is allowed; far and fine (a drafted paragraph) is not.

For revising one's own draft: **the model reads and the human writes.** It scores and reflects; it generates
no prose.

### Candidate features

Ken liked all of these in conversation but has chosen none to build first. Ask which. My suggested order is
the first two, since the engine nearly supports them already.

1. **Pencil stubs.** The writer drops a note ahead of the cursor ("→ but cost?"), and predictions between the
   cursor and the stub are conditioned to arrive at it. Thinking ahead is rewarded with better predictions, and
   the AI only heads where the human pointed. Needs the stub in the prompt, before the draft. Put stable text
   first and the stub after it, so a stub edit invalidates as little cache as possible.
2. **Moves, not words.** Offer what kind of thing could come next ("give an example", "concede", "define the
   term") with probabilities, never the prose. Picking one becomes a stub.
3. **The mirror** (Ken singled this out). Shade the parts of the writer's text that the model would have
   predicted anyway, and optionally show the generic draft the model would write from the prompt alone. It
   replaces "bits contributed by human vs. AI", which Ken's students are exploring but which is hard to measure,
   gameable, and hard to explain. `/v1/score` already returns the per-token logprobs this needs.
4. **Count choices, not bits.** Permanent visible provenance on text: typed, picked from 5, accepted the default.
5. **Your old draft as the pencil layer.** Select a sentence and retype it; predictions are conditioned on the
   old version, so kept words are one tap and effort goes where the writer diverges. The existing "rewrite this
   document" predictive mode is close to this already.
6. **Ripples.** After an edit, show which later sentences became more surprising. Two `/v1/score` calls, diffed.
7. **Either/or spans.** Keep two versions of a sentence; show which one the following text reads better after.
8. **Gist-back.** A margin note saying what a reader takes each paragraph to mean. This one generates text, so
   it needs a chat-capable server (campus, below), and the gist must stay coarse.

### What the engine lacks for these (my assessment, untested)

- **Comparing a fixed set of options** (moves, either/or): there is no endpoint for "score these N strings
  after this prompt". Calling `/v1/score` N times would thrash the single cache. A `/v1/compare {prompt, options}`
  that forks one sequence per option would fit the existing design.
- **Free generation** (generic draft, gist-back): the engine has none. Either add sampling, or use vLLM.
- **Text after the cursor.** The model only reads left to right, so anything ahead of the cursor has to be
  restated in the prompt.

### Bigger models on campus

Calvin has DGX Spark boxes behind Open WebUI at `https://localai.calvin.edu/api` (OpenAI-style; key in
`~/vllm-server/localai.calvin.edu-key`; never print or commit it). Models listed on 2026-11-11:
`nvidia-qwen3-6-35b-a3b-nvfp4`, `deepseek-ai/DeepSeek-V4-Flash-0731`, `qwen3.8-flash-next`,
`gemma-4-12B-it-qat-UD-Q4_K_XL`, `nemotron-3.5-super-vl`. Other people use these, so changes there go through
the sysadmin as proposals.

The plan discussed: the local engine does the fast, fine-grained work; a big campus model does the slow, coarse
work (moves, gist-back, the generic draft), where quality matters and latency doesn't. A second idea, untested:
since the engine answers in characters, a big model could rescore the local model's continuations in one
batched `prompt_logprobs` request. Scoring whole units avoids the partial-word problem. The cost is suggestions
reshuffling under the writer's finger. Ken would also like to upstream the `/v1/continuations` API to vLLM so
it could be deployed on campus.

## What was learned the hard way

- **Qwen3.5+ models are hybrids** with recurrent layers. Their state can't be partially rewound
  (`llama_memory_seq_rm` on a range returns false), which is why vLLM's cache only works in 1568-token blocks
  and why the engine parks checkpoints.
- **Recurrent state costs about 150 MB of GPU memory per sequence** on the 27B. 64 sequences failed to
  allocate; 24 sequences with a 16k context uses about 21.3 GB of 24.5. This caps checkpoints (10) and
  simultaneous search branches (13).
- **`llama_memory_seq_cp` is the whole trick.** It shares attention cells and the recurrent cell with no copy,
  so both checkpoints and forks are free until a sequence diverges. Rewinding to a checkpoint reproduces logits
  exactly; recomputing from scratch differs by up to about 0.25 in a logit (batch-shape numerics).
- **Single-character probes can't find the tokenization boundary.** " pow" survives every one-character probe,
  yet " power" is one token. The fix was backing up to the whitespace before the last word (capped at 4 tokens).
  The exact answer is the tokenizer's own pre-tokenization rule, which llama.cpp doesn't expose publicly.
- **A leaf that ends the turn still carries text.** "le." followed by end-of-turn is the unit "le.", not an
  "end" suggestion. Only an empty-text end leaf means "stop here".
- **`pkill -f writing-engine` kills the shell running it** (the pattern matches its own command line). Use
  `pkill -x writing-engine`.
- **Headless Firefox for checking the page:** it is a snap, so the profile and screenshot must live under
  `~/snap/firefox/common/`, with `--no-remote --profile <dir>`. `--screenshot` fires at page load, before
  results arrive; wrapping the page in an iframe beside an image that takes 6 s to load delays it enough.
- **For CPU-only development without stopping anything**, `unsloth/Qwen3.5-2B-GGUF` (`Qwen3.5-2B-Q4_K_M.gguf`,
  1.3 GB) has the same architecture and tokenizer; run with `CUDA_VISIBLE_DEVICES="" --gpu-layers 0`. The copy
  used in this session was in a temporary directory and is gone.

## Known gaps in the engine

- One cache, so one writer at a time. Per-session caches would need more sequences than the 27B leaves room for.
- Highlights (`/v1/score`) re-decode everything after an edit: about 230 ms for 433 tokens and about 1.0 s for 2,161, once the prompt is cached.
- The search tree is rebuilt on every keystroke. Keeping it between requests would cut the 2–5 search rounds
  on most keystrokes; the client's local narrowing hides most of this already.
- No request logging, authentication, or rate limiting; it binds to localhost by default.
