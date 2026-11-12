// A small inference server for writing tools, on libllama.
//
// What it's for: "given this context and the characters typed so far, what are the likely next
// characters?", answered fast and with honest probabilities.
//
// - Fast: one model context stays warm, with a cache per writer. Each request reuses the longest cached prefix; "checkpoints"
//   (parked sequence ids, which share memory with the main sequence) let us rewind to an earlier
//   point even for models with recurrent state, which can't be rewound token by token. Search
//   branches are forked sequences, which also share the prefix instead of copying it.
// - Honest: the model predicts tokens, but the writer typed characters. We condition on the
//   characters by summing over every tokenization that is consistent with them, from the last
//   point where the tokenization is settled (see `boundary`). Probabilities are the model's own;
//   nothing is renormalized over the candidates we happened to keep.
//
// Endpoints (JSON): GET /health, GET /v1/models, GET /v1/sessions, POST /v1/template, POST /v1/continuations,
// POST /v1/score.
// Requests run one at a time. Each writer ("session") has its own cache; see Engine for how they share the GPU.

#include "llama.h"
#include "chat.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;
using Tokens = std::vector<llama_token>;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct HttpError { int status; std::string message; };
struct Superseded {};

static bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// Number of trailing bytes that are the start of an unfinished UTF-8 character.
static size_t incomplete_utf8_tail(const std::string & s) {
    size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char c = s[n - back];
        if ((c & 0xC0) == 0x80) continue;                // continuation byte: keep looking for the lead byte
        size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? back : 0;
    }
    return 0;
}

static size_t utf8_length(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

static void parallel_for(size_t n, const std::function<void(size_t)> & fn) {
    size_t n_threads = std::min<size_t>(n, std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
    if (n_threads <= 1) { for (size_t i = 0; i < n; i++) fn(i); return; }
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < n_threads; t++) threads.emplace_back([&] { for (size_t i; (i = next++) < n;) fn(i); });
    for (auto & t : threads) t.join();
}

static double log_sum_exp(const float * logits, int n) {
    float max = logits[0];
    for (int i = 1; i < n; i++) max = std::max(max, logits[i]);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += std::exp(logits[i] - max);
    return max + std::log(sum);
}

// ── Engine ───────────────────────────────────────────────────────────────────

struct Checkpoint {
    llama_seq_id seq;
    size_t len;                 // the checkpoint holds the state after cur[0..len)
    std::vector<float> logits;  // next-token logits at that point, if we have them
};

// One writer's cache: a main sequence holding `cur`, and checkpoints along it. A "cold" session has been moved
// out of the GPU to make room (seq < 0); its state, if we kept it, is in `saved`.
struct Session {
    std::string id;
    llama_seq_id seq = -1;
    Tokens cur;
    std::vector<float> cur_logits;               // next-token logits after `cur`; empty if unknown
    std::vector<Checkpoint> checkpoints;         // sorted by len; each is a prefix of `cur`
    std::string last_text; Tokens last_tokens; std::vector<size_t> last_offsets;  // last tokenization, to redo only the changed tail
    std::vector<uint8_t> saved;
    uint64_t last_used = 0;
    bool automatic = false;                      // made for a request that didn't name a session
};

struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_memory_t mem = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr templates;
    int n_vocab = 0, n_ctx = 0, n_seq = 0;
    std::string alias;

    // Vocabulary tables.
    std::vector<std::string> piece;      // text bytes of each ordinary token ("" for control tokens)
    std::vector<std::string> rendered;   // text of each token as it appears in a prompt (control tokens included)
    std::vector<char> is_control, is_eog;
    std::vector<llama_token> by_piece;   // ordinary tokens sorted by piece, for prefix ranges
    std::unordered_map<std::string, Tokens> exact;  // piece → tokens
    llama_token bos = LLAMA_TOKEN_NULL;

    // Cache. Each session has a main sequence and some checkpoints; search branches borrow sequences for the
    // length of one request. Sequences are the scarce thing on models with recurrent state, which costs memory
    // per distinct sequence (about 150 MB each for Qwen3.8-27B); the attention cache is one pool that sequences
    // share. So sessions compete for sequences: checkpoints are thinned first, and then the least recently
    // used session is moved to host memory.
    static constexpr size_t CHUNK = 256;         // prefill chunk; a checkpoint is parked after each
    static constexpr size_t KEEP_RECENT = 4;
    static constexpr size_t LAG = 48;
    static constexpr uint64_t IDLE = 32;         // requests from others since a session's last one, after which it's idle
    static constexpr size_t MAX_SESSIONS = 64;
    static constexpr size_t MAX_COLD_BYTES = size_t(16) << 30;
    int fork_reserve = 0;                        // sequences that sessions never take, kept for search branches
    std::vector<llama_seq_id> free_seqs;
    std::map<std::string, std::unique_ptr<Session>> sessions;
    Session * s = nullptr;                       // the session of the request being handled
    uint64_t clock = 0, auto_count = 0;
    size_t cold_bytes = 0;
    double restore_ms = 0;

    std::mutex mutex;
    // A newer keystroke makes a writer's older requests pointless: they give up at their next check.
    std::mutex latest_mutex;
    std::map<std::string, uint64_t> latest;
    uint64_t stamp(const std::string & key) { std::lock_guard<std::mutex> lock(latest_mutex); return ++latest[key]; }
    bool is_latest(const std::string & key, uint64_t id) { std::lock_guard<std::mutex> lock(latest_mutex); return latest[key] == id; }

    void load(const std::string & path, int ctx_size, int n_gpu_layers, int seqs);

    // Tokenization
    Tokens tokenize_raw(const std::string & text) const;
    struct Tokenized { Tokens tokens; std::vector<size_t> offsets; };  // offsets[i] = byte offset of token i; one extra at the end
    Tokenized tokenize(const std::string & text);
    size_t safe_restart(const Tokenized & t, size_t before_byte) const;
    size_t boundary(const std::string & text, const Tokenized & t) const;

    // Cache
    double decode(const std::vector<std::array<int, 3>> & items, const std::vector<char> & want_logits);
    void add_checkpoint(bool with_logits);
    void drop_checkpoint(Session & x, size_t i);
    void thin(Session & x);
    void reset_cache();
    Session & use_session(const json & req, const std::string & text);
    void make_room(size_t cells, int seqs);
    void cool(Session & x);
    void forget(Session & x);
    llama_seq_id take_seq(bool for_checkpoint);
    json list_sessions();
    struct SyncStats { size_t reused = 0, decoded = 0; double ms = 0, decode_ms = 0; };
    // on_chunk(positions, logits): logits[k] predicts the token after position positions[k].
    using ChunkFn = std::function<void(const std::vector<size_t> &, const std::vector<const float *> &)>;
    SyncStats sync(const Tokens & target, const ChunkFn & on_chunk = nullptr, size_t logits_from = SIZE_MAX);

    // Scoring: the logprob of each token from `first` on, given the tokens before it, with the k likeliest
    // tokens there (and the actual token's rank, if asked). Leaves the session holding `tokens`.
    struct Alt { llama_token token; double logprob; };
    struct Scored { double logprob = 0; int rank = 0; std::vector<Alt> top; };
    Scored score_one(const float * logits, llama_token actual, int k, bool want_rank) const;
    std::vector<Scored> score_tokens(const Tokens & tokens, size_t first, int k, bool want_rank, SyncStats & stats, double & score_ms);

    // Chat templates
    std::map<std::string, std::string> prefix_cache;
    std::string apply_template(const json & messages, bool add_generation_prompt, bool enable_thinking) const;
    std::string turn_prefix(const json & messages, const std::string & role, bool enable_thinking);
    std::string prompt_of(const json & req);

    // Requests
    json continuations(const json & req, uint64_t id);
    json score(const json & req);
    json render_template(const json & req);

};

void Engine::load(const std::string & path, int ctx_size, int n_gpu_layers, int seqs) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) throw std::runtime_error("couldn't load model " + path);
    vocab = llama_model_get_vocab(model);
    n_vocab = llama_vocab_n_tokens(vocab);

    auto cp = llama_context_default_params();
    cp.n_ctx = ctx_size;
    cp.n_batch = 2048;
    cp.n_seq_max = seqs;
    cp.kv_unified = true;   // sequences share cells for their common prefix
    cp.no_perf = true;
    cp.n_threads = cp.n_threads_batch = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("couldn't create context");
    mem = llama_get_memory(ctx);
    n_ctx = llama_n_ctx(ctx);
    n_seq = seqs;
    fork_reserve = std::min(12, seqs / 2);
    for (int i = seqs - 1; i >= 0; i--) free_seqs.push_back(i);

    templates = common_chat_templates_init(model, "");

    piece.resize(n_vocab); rendered.resize(n_vocab); is_control.resize(n_vocab); is_eog.resize(n_vocab);
    auto to_piece = [&](llama_token t, bool special) {
        char buf[256];
        int n = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, special);
        return std::string(buf, std::max(n, 0));
    };
    for (llama_token t = 0; t < n_vocab; t++) {
        is_control[t] = llama_vocab_is_control(vocab, t);
        is_eog[t] = llama_vocab_is_eog(vocab, t);
        rendered[t] = to_piece(t, true);
        if (!is_control[t]) piece[t] = to_piece(t, false);
        if (!piece[t].empty()) { by_piece.push_back(t); exact[piece[t]].push_back(t); }
    }
    std::sort(by_piece.begin(), by_piece.end(), [&](llama_token a, llama_token b) { return piece[a] < piece[b]; });
    if (llama_vocab_get_add_bos(vocab)) bos = llama_vocab_bos(vocab);
}

// ── Tokenization ─────────────────────────────────────────────────────────────

Tokens Engine::tokenize_raw(const std::string & text) const {
    Tokens out(text.size() + 8);
    int n = llama_tokenize(vocab, text.data(), text.size(), out.data(), out.size(), false, true);
    if (n < 0) { out.resize(-n); n = llama_tokenize(vocab, text.data(), text.size(), out.data(), out.size(), false, true); }
    out.resize(std::max(n, 0));
    return out;
}

// A token index at or before `before_byte` where tokenizing can restart without changing the result:
// after a control token, or at the start of a line that begins with a non-space character.
size_t Engine::safe_restart(const Tokenized & t, size_t before_byte) const {
    for (size_t i = t.tokens.size(); i-- > 1;) {
        if (t.offsets[i] >= before_byte) continue;
        llama_token prev = t.tokens[i - 1], tok = t.tokens[i];
        if (is_control[prev]) return i;
        const std::string & p = rendered[prev], & q = rendered[tok];
        if (!p.empty() && p.back() == '\n' && !q.empty() && !isspace((unsigned char) q[0])) return i;
    }
    return 0;
}

Engine::Tokenized Engine::tokenize(const std::string & text) {
    Tokenized out;
    size_t common = 0, max = std::min(text.size(), s->last_text.size());
    while (common < max && text[common] == s->last_text[common]) common++;
    size_t restart = safe_restart({s->last_tokens, s->last_offsets}, common);
    size_t start_byte = restart ? s->last_offsets[restart] : 0;
    out.tokens.assign(s->last_tokens.begin(), s->last_tokens.begin() + restart);
    out.offsets.assign(s->last_offsets.begin(), s->last_offsets.begin() + restart);
    bool had_bos = restart == 0 && bos != LLAMA_TOKEN_NULL && !starts_with(text, rendered[bos]);
    if (had_bos) { out.tokens.push_back(bos); out.offsets.push_back(0); }
    size_t offset = start_byte;
    for (llama_token t : tokenize_raw(text.substr(start_byte))) {
        out.tokens.push_back(t); out.offsets.push_back(offset);
        offset += rendered[t].size();
    }
    out.offsets.push_back(offset);
    if (offset != text.size()) throw HttpError{500, "this tokenizer doesn't round-trip text exactly, which the engine relies on"};
    s->last_text = text; s->last_tokens = out.tokens; s->last_offsets = out.offsets;
    return out;
}

// How many leading tokens are settled, i.e. won't change whatever is typed next. Everything after
// that point is treated as characters (`typed`), not tokens.
//
// BPE tokenizers first split text into chunks (roughly, a word with the space before it) and merge
// only within a chunk, so tokens before the last chunk are settled. The chunking rule itself is
// model-specific, so we use a conservative stand-in: back up to the whitespace before the last word.
// Backing up too far is harmless (we sum over more tokenizations), but each backed-over token costs a
// search step, so we back up at most MAX_BACKUP tokens; that's what bounds the work in scripts
// without spaces, at a small cost in exactness. We also append probes of several character classes
// and back up past any token they change, which catches merges the whitespace rule misses (e.g.
// punctuation runs). Control tokens are never treated as characters.
size_t Engine::boundary(const std::string & text, const Tokenized & t) const {
    static constexpr size_t MAX_BACKUP = 4;
    size_t n = t.tokens.size();
    size_t floor = n > MAX_BACKUP ? n - MAX_BACKUP : 0;
    for (size_t i = n; i-- > floor;) if (is_control[t.tokens[i]]) { floor = i + 1; break; }

    // The last word and the whitespace run before it (or just the trailing whitespace run).
    size_t chunk = text.size();
    while (chunk > 0 && !isspace((unsigned char) text[chunk - 1])) chunk--;
    while (chunk > 0 && isspace((unsigned char) text[chunk - 1])) chunk--;
    size_t stable = 0;
    while (stable < n && t.offsets[stable + 1] <= chunk) stable++;

    std::vector<std::string> probes = {"a", "1", " ", "\n", ".", ",", "'", "-"};
    size_t last_char = text.size();
    while (last_char > 0 && (text[--last_char] & 0xC0) == 0x80) {}
    if (!text.empty()) probes.push_back(text.substr(last_char));
    size_t restart = safe_restart(t, t.offsets[std::min(stable, n)]);
    std::string tail = text.substr(t.offsets[restart]);
    for (const auto & probe : probes) {
        Tokens probed = tokenize_raw(tail + probe);
        size_t same = 0;
        while (restart + same < stable && same < probed.size() && probed[same] == t.tokens[restart + same]) same++;
        stable = std::min(stable, restart + same);
    }
    return std::max(stable, floor);
}

// ── Cache ────────────────────────────────────────────────────────────────────

// items: {token, position, sequence}. Returns elapsed ms.
double Engine::decode(const std::vector<std::array<int, 3>> & items, const std::vector<char> & want_logits) {
    llama_batch batch = llama_batch_init(items.size(), 0, 1);
    for (size_t i = 0; i < items.size(); i++) {
        batch.token[i] = items[i][0]; batch.pos[i] = items[i][1];
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = items[i][2];
        batch.logits[i] = want_logits[i];
    }
    batch.n_tokens = items.size();
    double t0 = now_ms();
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw HttpError{500, "llama_decode failed (" + std::to_string(rc) + "); the context may be full"};
    return now_ms() - t0;
}

void Engine::drop_checkpoint(Session & x, size_t i) {
    llama_memory_seq_rm(mem, x.checkpoints[i].seq, -1, -1);
    free_seqs.push_back(x.checkpoints[i].seq);
    x.checkpoints.erase(x.checkpoints.begin() + i);
}

// Give up one checkpoint: keep the first and the most recent ones if we can; of the rest, drop the one that
// leaves the smallest gap.
void Engine::thin(Session & x) {
    size_t n = x.checkpoints.size();
    if (n <= 2) { drop_checkpoint(x, 0); return; }
    size_t last = n - 2 > KEEP_RECENT ? n - 1 - KEEP_RECENT : n - 2;   // the last interior index we may drop
    size_t best = 1, best_gap = SIZE_MAX;
    for (size_t i = 1; i <= last; i++) {
        size_t gap = x.checkpoints[i + 1].len - x.checkpoints[i - 1].len;
        if (gap < best_gap) { best_gap = gap; best = i; }
    }
    drop_checkpoint(x, best);
}

// A sequence for the current session's main sequence or for a new checkpoint (-1 if it can't have another
// checkpoint). Sessions together hold at most n_seq - fork_reserve sequences. When they're all taken:
// - a session that has been idle for a while is moved to host memory;
// - otherwise the session with the most checkpoints gives one up, but each keeps two. With fewer, backspacing
//   past a word means decoding the whole prompt again, which costs more than bringing a session back from
//   host memory. So when every session is down to two, a new session pushes out the least recently used one.
// A session with only two checkpoints replaces the older one only once the newer one trails the text by
// LAG / 2 tokens, so there is always one a little way back to rewind to.
llama_seq_id Engine::take_seq(bool for_checkpoint) {
    for (;;) {
        int held = 0;
        Session * richest = nullptr, * oldest = nullptr;
        for (auto & [id, x] : sessions) {
            if (x->seq < 0) continue;
            held += 1 + x->checkpoints.size();
            if (!richest || x->checkpoints.size() > richest->checkpoints.size() ||
                (x->checkpoints.size() == richest->checkpoints.size() && x->last_used < richest->last_used)) richest = x.get();
            if (x.get() != s && (!oldest || x->last_used < oldest->last_used)) oldest = x.get();
        }
        if (held < n_seq - fork_reserve && !free_seqs.empty()) break;
        size_t own = s->checkpoints.size(), most = richest ? richest->checkpoints.size() : 0;
        size_t behind = own ? s->cur.size() - s->checkpoints.back().len : 0;
        if (oldest && clock - oldest->last_used > IDLE) cool(*oldest);
        else if (most > 2 && (!for_checkpoint || most > own)) thin(*richest);
        else if (!for_checkpoint && oldest) cool(*oldest);
        else if (!for_checkpoint) throw HttpError{500, "no sequence available"};
        else if (own > 2) thin(*s);
        else if (own == 2 && behind >= LAG / 2) drop_checkpoint(*s, 0);
        else if (own == 1 && behind >= LAG) drop_checkpoint(*s, 0);
        else return -1;
    }
    llama_seq_id seq = free_seqs.back(); free_seqs.pop_back();
    return seq;
}

// Park a checkpoint at the end of `cur`.
void Engine::add_checkpoint(bool with_logits) {
    if (s->cur.empty()) return;
    if (!s->checkpoints.empty() && s->checkpoints.back().len == s->cur.size()) {
        if (with_logits) s->checkpoints.back().logits = s->cur_logits;
        return;
    }
    llama_seq_id seq = take_seq(true);
    if (seq < 0) return;
    llama_memory_seq_cp(mem, s->seq, seq, -1, -1);
    s->checkpoints.push_back({seq, s->cur.size(), with_logits ? s->cur_logits : std::vector<float>()});
}

void Engine::reset_cache() {
    while (!s->checkpoints.empty()) drop_checkpoint(*s, s->checkpoints.size() - 1);
    llama_memory_seq_rm(mem, s->seq, -1, -1);
    s->cur.clear(); s->cur_logits.clear();
}

// Move a session out of the GPU, keeping its state in host memory so it can come back without recomputing.
// Its checkpoints are dropped.
void Engine::cool(Session & x) {
    if (x.seq < 0) return;
    while (!x.checkpoints.empty()) drop_checkpoint(x, x.checkpoints.size() - 1);
    size_t size = x.cur.empty() ? 0 : llama_state_seq_get_size_ext(ctx, x.seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    x.saved.resize(size);
    if (size && llama_state_seq_get_data_ext(ctx, x.saved.data(), size, x.seq, LLAMA_STATE_SEQ_FLAGS_NONE) != size) x.saved.clear();
    if (x.saved.empty()) { x.cur.clear(); x.cur_logits.clear(); }
    cold_bytes += x.saved.size();
    llama_memory_seq_rm(mem, x.seq, -1, -1);
    free_seqs.push_back(x.seq);
    x.seq = -1;
    // Host memory isn't unlimited either: the longest-idle cold sessions lose their state.
    while (cold_bytes > MAX_COLD_BYTES) {
        Session * oldest = nullptr;
        for (auto & [id, y] : sessions) if (!y->saved.empty() && (!oldest || y->last_used < oldest->last_used)) oldest = y.get();
        if (!oldest) break;
        cold_bytes -= oldest->saved.size();
        oldest->saved = {}; oldest->cur.clear(); oldest->cur_logits.clear();
    }
}

void Engine::forget(Session & x) {
    while (!x.checkpoints.empty()) drop_checkpoint(x, x.checkpoints.size() - 1);
    if (x.seq >= 0) { llama_memory_seq_rm(mem, x.seq, -1, -1); free_seqs.push_back(x.seq); }
    cold_bytes -= x.saved.size();
    sessions.erase(x.id);
}

// Make sure the attention cache has `cells` free for the current session and that search branches can get
// `seqs` sequences, by moving the least recently used other sessions out of the GPU.
void Engine::make_room(size_t cells, int seqs) {
    for (;;) {
        size_t used = 0;
        Session * oldest = nullptr;
        for (auto & [id, x] : sessions) {
            if (x->seq < 0 || x.get() == s) continue;
            used += x->cur.size();
            if (!oldest || x->last_used < oldest->last_used) oldest = x.get();
        }
        if ((used + cells <= (size_t) n_ctx && (int) free_seqs.size() >= seqs) || !oldest) return;
        if (used + cells <= (size_t) n_ctx && !oldest->checkpoints.empty()) thin(*oldest); else cool(*oldest);
    }
}

// The session for this request: the one it names, or else the one whose last text it most plausibly continues.
// Brings it back into the GPU if it was moved out.
Session & Engine::use_session(const json & req, const std::string & text) {
    std::string id = req.value("session", "");
    Session * found = nullptr;
    if (!id.empty()) {
        auto it = sessions.find("id:" + id);
        if (it != sessions.end()) found = it->second.get();
        id = "id:" + id;
    } else {
        // Unnamed requests continue an unnamed session if they keep at least half of its text (or of their own).
        size_t best = 0;
        for (auto & [key, x] : sessions) {
            if (!x->automatic) continue;
            size_t common = 0, max = std::min(text.size(), x->last_text.size());
            while (common < max && text[common] == x->last_text[common]) common++;
            if (common > best && 2 * common >= max) { best = common; found = x.get(); }
        }
        if (!found) id = "auto:" + std::to_string(++auto_count);
    }
    if (!found) {
        if (sessions.size() >= MAX_SESSIONS) {
            Session * oldest = nullptr;
            for (auto & [key, x] : sessions) if (!oldest || x->last_used < oldest->last_used) oldest = x.get();
            forget(*oldest);
        }
        auto made = std::make_unique<Session>();
        made->id = id; made->automatic = id.rfind("auto:", 0) == 0;
        found = made.get();
        sessions[id] = std::move(made);
    }
    s = found;
    s->last_used = ++clock;
    restore_ms = 0;
    if (s->seq < 0) {
        double t0 = now_ms();
        make_room(s->cur.size() + CHUNK, 0);
        s->seq = take_seq(false);
        bool restored = !s->saved.empty() && llama_state_seq_set_data_ext(ctx, s->saved.data(), s->saved.size(), s->seq, LLAMA_STATE_SEQ_FLAGS_NONE) > 0;
        cold_bytes -= s->saved.size();
        s->saved = {};
        if (!restored) { llama_memory_seq_rm(mem, s->seq, -1, -1); s->cur.clear(); s->cur_logits.clear(); }
        else { add_checkpoint(true); restore_ms = now_ms() - t0; }
    }
    return *s;
}

json Engine::list_sessions() {
    json list = json::array();
    for (auto & [id, x] : sessions) list.push_back({
        {"id", id}, {"tokens", x->cur.size()}, {"hot", x->seq >= 0}, {"checkpoints", x->checkpoints.size()},
        {"saved_mb", x->saved.size() / 1048576.0}, {"age", clock - x->last_used},
    });
    return {{"sessions", list}, {"free_sequences", free_seqs.size()}, {"fork_reserve", fork_reserve}};
}

// Make the session's main sequence hold exactly `target`, with next-token logits in cur_logits, reusing what we can.
// If on_chunk is given, it gets the logits at every decoded position i with i + 1 >= logits_from (except the
// last position of the target); the caller arranges, by syncing to a shorter target first, that all of them
// get decoded.
Engine::SyncStats Engine::sync(const Tokens & target, const ChunkFn & on_chunk, size_t logits_from) {
    if (target.empty()) throw HttpError{400, "empty prompt"};
    if ((int) target.size() + 64 > n_ctx) throw HttpError{400, "prompt too long for the context (" + std::to_string(target.size()) + " tokens)"};
    make_room(target.size() + CHUNK, fork_reserve / 2);
    SyncStats stats;
    double t0 = now_ms();
    size_t common = 0;
    while (common < s->cur.size() && common < target.size() && s->cur[common] == target[common]) common++;

    if (common == s->cur.size() && common == target.size() && !s->cur_logits.empty()) { stats.reused = common; return stats; }
    if (common < s->cur.size() || common == target.size()) {
        // Rewind to the latest usable checkpoint. One at the very end of the target is only usable if it has logits.
        int pick = -1;
        for (size_t i = 0; i < s->checkpoints.size(); i++) {
            const auto & c = s->checkpoints[i];
            if (c.len <= common && (c.len < target.size() || !c.logits.empty())) pick = i;
        }
        if (pick < 0) reset_cache();
        else {
            for (size_t i = s->checkpoints.size(); i-- > 0;) if (s->checkpoints[i].len > common) drop_checkpoint(*s, i);
            const auto & c = s->checkpoints[pick];
            llama_memory_seq_rm(mem, s->seq, -1, -1);
            llama_memory_seq_cp(mem, c.seq, s->seq, -1, -1);
            s->cur.resize(c.len);
            s->cur_logits = c.logits;
        }
    }
    stats.reused = s->cur.size();

    while (s->cur.size() < target.size()) {
        size_t begin = s->cur.size();
        size_t end = std::min(target.size(), (begin / CHUNK + 1) * CHUNK);
        std::vector<std::array<int, 3>> items;
        std::vector<char> want;
        for (size_t i = begin; i < end; i++) {
            items.push_back({target[i], (int) i, s->seq});
            want.push_back(i + 1 == end || (on_chunk && i + 1 >= logits_from));
        }
        stats.decode_ms += decode(items, want);
        if (on_chunk) {
            std::vector<size_t> positions; std::vector<const float *> logits;
            for (size_t i = begin; i < end; i++) if (want[i - begin] && i + 1 < target.size()) { positions.push_back(i); logits.push_back(llama_get_logits_ith(ctx, i - begin)); }
            on_chunk(positions, logits);
        }
        const float * last = llama_get_logits_ith(ctx, end - begin - 1);
        s->cur.insert(s->cur.end(), target.begin() + begin, target.begin() + end);
        s->cur_logits.assign(last, last + n_vocab);
        add_checkpoint(end == target.size());
    }
    stats.decoded = target.size() - stats.reused;
    stats.ms = now_ms() - t0;
    return stats;
}

// ── Continuations ────────────────────────────────────────────────────────────

struct Node {
    int parent;
    llama_token token;
    double logprob;        // of the whole path from the boundary
    std::string text;      // the path's bytes, from the boundary
    int depth;             // tokens from the boundary
    int past;              // tokens since the path first covered the typed text
    bool covers;           // text is at least as long as the typed text
    bool end = false;      // an end-of-turn token
    bool done = false;     // won't be extended
    bool reserve = false;  // under the probability bar: only expanded to reach `units` distinct continuations
    int start = -1;        // a correction: the offset in the typed text where it leaves it (and its text starts)
    bool expanded = false;
    llama_seq_id seq = -1; // holds this node's state while children may still fork from it
    int pending = 0;       // children that may still need that state
    int kept = 0;          // children kept as nodes
};

struct Expansion {
    struct Child { llama_token token; double logprob; bool reserve = false; int start = -1; };
    std::vector<Child> children;
    double covering_mass = 0;   // probability, within a not-yet-covering node, of tokens that reach past the typed text
};

json Engine::continuations(const json & req, uint64_t id) {
    std::string prompt = prompt_of(req), text = req.value("text", "");
    int max_rounds = req.value("max_rounds", 8), max_batch = req.value("max_batch", 8), branch = req.value("branch", 8);
    int max_depth = req.value("max_depth", 8), max_chars = req.value("max_chars", 0);
    double min_prob = req.value("min_prob", 0.01), time_ms = req.value("time_ms", 400.0);
    int units = req.value("units", 0);  // keep going (past min_prob if need be) until this many distinct finished units
    // Also follow paths that leave the typed text partway (corrections), if at least this many times as likely as all
    // the paths that keep to it. 0: don't.
    double corrections = req.value("corrections", 0.0);
    double corrections_ms = req.value("corrections_ms", 60.0);  // time for that, on top of the rest
    bool has_stop = req.contains("stop") && req["stop"].is_string();
    std::regex stop;
    if (has_stop) {
        try { stop = std::regex(req["stop"].get<std::string>(), std::regex::ECMAScript | std::regex::optimize); }
        catch (const std::regex_error & e) { throw HttpError{400, std::string("bad stop regex: ") + e.what()}; }
    }
    std::string key = req.value("session", "");
    auto superseded = [&] { return !is_latest(key, id); };

    double t0 = now_ms();
    std::string full = prompt + text;
    use_session(req, full);
    Tokenized tokenized = tokenize(full);
    size_t n_settled = boundary(full, tokenized);
    // The characters to condition on. They never reach back into the prompt past a control token, but may
    // include the end of the prompt (e.g. its final newlines), since the reply's first token can merge with it.
    std::string typed = full.substr(tokenized.offsets[n_settled]);
    // Corrections may only replace the draft, not the end of the prompt.
    size_t edit_from = prompt.size() > tokenized.offsets[n_settled] ? prompt.size() - tokenized.offsets[n_settled] : 0;
    Tokens target(tokenized.tokens.begin(), tokenized.tokens.begin() + n_settled);
    double t_tokenize = now_ms() - t0;
    if (superseded()) throw Superseded{};

    SyncStats sync_stats = sync(target);
    double t1 = now_ms();

    std::vector<Node> nodes;
    nodes.push_back({-1, LLAMA_TOKEN_NULL, 0.0, "", 0, 0, typed.empty()});
    nodes[0].seq = s->seq;
    double covered = 0;      // probability that the text continues past `typed` (or ends exactly there), found so far
    double unresolved = 0;   // probability of paths still inside `typed` that we haven't expanded
    if (typed.empty()) covered = 1;

    // A continuation's text: after the typed text, or for a correction, from where it leaves it.
    auto cont_of = [&](const Node & n) { return n.text.substr(n.start >= 0 ? n.start : typed.size()); };

    auto is_done = [&](const Node & n) {
        if (n.end) return true;
        if (!n.covers) return false;
        std::string cont = cont_of(n);
        if (incomplete_utf8_tail(cont)) return n.past >= max_depth + 2;
        if (n.past >= max_depth) return true;
        if (max_chars > 0 && (int) utf8_length(cont) >= max_chars) return true;
        return has_stop && std::regex_search(cont, stop);
    };

    // What to keep from one node's next-token distribution. Pure, so nodes can be done in parallel.
    // Up to `branch` children at or above `floor` are kept; those under `threshold` are kept only as reserves.
    auto expand = [&](const Node & node, const float * logits, double threshold, double floor) {
        Expansion out;
        double base = node.logprob - log_sum_exp(logits, n_vocab);
        std::vector<Expansion::Child> candidates;
        if (node.covers) {
            for (llama_token t = 0; t < n_vocab; t++) {
                double lp = base + logits[t];
                if (lp < floor) continue;
                if (is_eog[t] || !piece[t].empty()) candidates.push_back({t, lp, lp < threshold});
            }
        } else {
            std::string rest = typed.substr(node.text.size());
            // Tokens that spell out part (or exactly all) of the rest of the typed text: always kept.
            for (size_t len = 1; len <= rest.size(); len++) {
                auto it = exact.find(rest.substr(0, len));
                if (it != exact.end()) for (llama_token t : it->second) out.children.push_back({t, base + logits[t]});
            }
            // Tokens that reach past the typed text: all of them count toward the covered probability.
            auto lo = std::lower_bound(by_piece.begin(), by_piece.end(), rest, [&](llama_token t, const std::string & s) { return piece[t] < s; });
            for (auto it = lo; it != by_piece.end() && starts_with(piece[*it], rest); ++it) {
                if (piece[*it].size() == rest.size()) continue;
                double lp = base + logits[*it];
                out.covering_mass += std::exp(lp);
                if (lp >= floor) candidates.push_back({*it, lp, lp < threshold});
            }
            // Tokens that leave the typed text: candidate corrections, kept as reserves.
            if (corrections > 0) {
                std::vector<Expansion::Child> leaving;
                for (llama_token t = 0; t < n_vocab; t++) {
                    const std::string & pc = piece[t];
                    if (pc.empty() || is_eog[t]) continue;
                    size_t c = 0;
                    while (c < pc.size() && c < rest.size() && pc[c] == rest[c]) c++;
                    if (c == pc.size() || c == rest.size()) continue;  // keeps to the typed text
                    while (c > 0 && (rest[c] & 0xC0) == 0x80) c--;    // leave at a character boundary
                    if (node.text.size() + c < edit_from) continue;
                    leaving.push_back({t, base + logits[t], true, (int) (node.text.size() + c)});
                }
                size_t k = std::min<size_t>(leaving.size(), 4);
                std::partial_sort(leaving.begin(), leaving.begin() + k, leaving.end(), [](auto & a, auto & b) { return a.logprob > b.logprob; });
                out.children.insert(out.children.end(), leaving.begin(), leaving.begin() + k);
            }
        }
        size_t keep = std::min<size_t>(candidates.size(), branch);
        std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(), [](auto & a, auto & b) { return a.logprob > b.logprob; });
        out.children.insert(out.children.end(), candidates.begin(), candidates.begin() + keep);
        return out;
    };

    auto apply = [&](int ni, const Expansion & e) {
        nodes[ni].expanded = true;
        if (!nodes[ni].covers) { unresolved -= std::exp(nodes[ni].logprob); covered += e.covering_mass; }
        for (const auto & c : e.children) {
            Node child{ni, c.token, c.logprob, nodes[ni].text + piece[c.token], nodes[ni].depth + 1, 0, false};
            child.end = is_eog[c.token];
            child.start = c.start >= 0 ? c.start : nodes[ni].start;
            bool leaves_typed = child.start >= 0;
            child.covers = leaves_typed || nodes[ni].covers || child.text.size() >= typed.size();
            child.past = nodes[ni].covers ? nodes[ni].past + 1 : 0;
            if (!leaves_typed && !nodes[ni].covers && child.text.size() == typed.size()) covered += std::exp(c.logprob);
            if (!child.covers) unresolved += std::exp(c.logprob);
            child.done = is_done(child);
            child.reserve = c.reserve;
            if (!child.done && !child.reserve) nodes[ni].pending++;
            nodes[ni].kept++;
            nodes.push_back(std::move(child));
        }
    };
    auto release = [&](Node & n) {
        if (n.parent >= 0 && n.seq >= 0) { llama_memory_seq_rm(mem, n.seq, -1, -1); free_seqs.push_back(n.seq); }
        n.seq = -1;
    };
    auto cleanup = [&] {
        for (auto & n : nodes) release(n);
    };

    // The unit a continuation offers: through the end of its `stop` match (ending the turn counts as a space
    // after it), or all of it.
    auto unit_of = [&](const Node & n) {
        std::string cont = cont_of(n);
        cont.resize(cont.size() - incomplete_utf8_tail(cont));
        std::string probe = n.end ? cont + " " : cont;
        std::smatch m;
        if (has_stop && std::regex_search(probe, m, stop)) cont = probe.substr(0, m.position(0) + m.length(0));
        return cont;
    };

    // Expand nodes whose parents' states are gone, replaying each from the settled prompt (a few tokens).
    // All children are reserves, since nothing keeps this state; later rounds pick among them by probability.
    auto replay_expand = [&](std::vector<int> batch) {
        batch.resize(std::min<size_t>({batch.size(), (size_t) max_batch, free_seqs.size()}));
        if (batch.empty()) return false;
        std::vector<std::array<int, 3>> items;
        std::vector<char> want;
        std::vector<int> last_index;
        std::vector<llama_seq_id> seqs;
        for (int ni : batch) {
            llama_seq_id seq = free_seqs.back(); free_seqs.pop_back();
            seqs.push_back(seq);
            llama_memory_seq_cp(mem, s->seq, seq, -1, -1);
            Tokens path;
            for (int i = ni; i > 0; i = nodes[i].parent) path.push_back(nodes[i].token);
            std::reverse(path.begin(), path.end());
            for (size_t k = 0; k < path.size(); k++) {
                items.push_back({path[k], (int) (target.size() + k), seq});
                want.push_back(k + 1 == path.size());
            }
            last_index.push_back((int) items.size() - 1);
        }
        auto free_all = [&] { for (auto seq : seqs) { llama_memory_seq_rm(mem, seq, -1, -1); free_seqs.push_back(seq); } };
        try { decode(items, want); } catch (...) { free_all(); throw; }
        std::vector<Expansion> expansions(batch.size());
        std::vector<const float *> logits(batch.size());
        for (size_t i = 0; i < batch.size(); i++) logits[i] = llama_get_logits_ith(ctx, last_index[i]);
        parallel_for(batch.size(), [&](size_t i) {
            expansions[i] = expand(nodes[batch[i]], logits[i], INFINITY, nodes[batch[i]].logprob + std::log(1e-3));
        });
        for (size_t i = 0; i < batch.size(); i++) apply(batch[i], expansions[i]);
        free_all();
        return true;
    };

    int rounds = 0, extra_rounds = 0;
    try {
        if (!typed.empty()) unresolved = 1;  // the root itself
        apply(0, expand(nodes[0], s->cur_logits.data(), -INFINITY, -INFINITY));

        // At least one round, even if a slow prefill used up the time: otherwise the answer is just the typed text.
        for (; rounds < max_rounds && (rounds == 0 || now_ms() - t0 < time_ms); rounds++) {
            if (superseded()) throw Superseded{};
            double total = covered + std::max(unresolved, 0.0);
            if (total <= 0) break;
            // The most probable unfinished paths. Paths still inside the typed text hide probability, so they get a lower bar.
            std::vector<int> batch;
            for (size_t i = 1; i < nodes.size(); i++) {
                const Node & n = nodes[i];
                if (n.expanded || n.done || n.reserve) continue;
                if (std::exp(n.logprob) >= total * min_prob * (n.covers ? 1 : 0.1)) batch.push_back(i);
            }
            if (batch.empty()) break;
            std::sort(batch.begin(), batch.end(), [&](int a, int b) { return nodes[a].logprob > nodes[b].logprob; });
            if ((int) batch.size() > max_batch) batch.resize(max_batch);

            // Give each its own sequence: the last child to need a parent's state takes it over; others fork it.
            std::vector<std::array<int, 3>> items;
            std::vector<int> decoded;
            for (int ni : batch) {
                Node & n = nodes[ni]; Node & p = nodes[n.parent];
                if (p.pending == 1 && n.parent != 0) { n.seq = p.seq; p.seq = -1; }
                else {
                    if (free_seqs.empty()) continue;
                    n.seq = free_seqs.back(); free_seqs.pop_back();
                    llama_memory_seq_cp(mem, p.seq, n.seq, -1, -1);
                }
                p.pending--;
                items.push_back({n.token, (int) target.size() + n.depth - 1, n.seq});
                decoded.push_back(ni);
            }
            if (items.empty()) break;
            decode(items, std::vector<char>(items.size(), 1));

            double threshold = std::log(total * min_prob);
            std::vector<Expansion> expansions(decoded.size());
            std::vector<const float *> logits(decoded.size());
            for (size_t i = 0; i < decoded.size(); i++) logits[i] = llama_get_logits_ith(ctx, i);
            double floor = units > 0 ? threshold + std::log(1e-3) : threshold;
            parallel_for(decoded.size(), [&](size_t i) { expansions[i] = expand(nodes[decoded[i]], logits[i], threshold, floor); });
            for (size_t i = 0; i < decoded.size(); i++) {
                apply(decoded[i], expansions[i]);
                if (nodes[decoded[i]].pending == 0) release(nodes[decoded[i]]);
            }
        }

        // Then, sharing each batch: if there are too few distinct units, the likeliest reserves (and leftovers),
        // until there are enough; and corrections at least `corrections` times as likely as the typed text (a
        // path's probability only falls as it grows, so paths under that bar can't lead to one), for at most
        // `corrections_ms`. Their parents' states are gone by now, so each is replayed from the settled prompt.
        double t_extra = now_ms();
        for (; extra_rounds < max_rounds && now_ms() - t0 < time_ms; extra_rounds++) {
            if (superseded()) throw Superseded{};
            std::vector<int> more, fixes;
            if (units > 0) {
                std::set<std::string> found;
                for (const Node & n : nodes) {
                    if (!n.covers || !n.done || n.start >= 0) continue;
                    std::string u = unit_of(n);
                    if (u.find_first_not_of(" \t\n\r") != std::string::npos) found.insert(u);
                }
                if ((int) found.size() < units) {
                    for (size_t i = 1; i < nodes.size(); i++) {
                        const Node & n = nodes[i];
                        if (!n.covers || n.expanded || n.done || n.start >= 0) continue;
                        // Skip a word that is already complete and already found (e.g. " fun" once " fun pets" is in).
                        std::string probe = unit_of(n) + " ";
                        std::smatch m;
                        if (has_stop && std::regex_search(probe, m, stop) && found.count(probe.substr(0, m.position(0) + m.length(0)))) continue;
                        more.push_back(i);
                    }
                }
            }
            if (corrections > 0 && now_ms() - t_extra < corrections_ms) {
                for (size_t i = 1; i < nodes.size(); i++) {
                    const Node & n = nodes[i];
                    if (n.start >= 0 && !n.expanded && !n.done && std::exp(n.logprob) >= covered * corrections) fixes.push_back(i);
                }
            }
            if (more.empty() && fixes.empty()) break;
            auto likeliest = [&](int a, int b) { return nodes[a].logprob > nodes[b].logprob; };
            std::sort(more.begin(), more.end(), likeliest);
            std::sort(fixes.begin(), fixes.end(), likeliest);
            // Up to half the batch for each, and whatever the other leaves.
            size_t half = max_batch / 2;
            size_t n_fixes = std::min(fixes.size(), std::max(half, max_batch - std::min(more.size(), half)));
            std::vector<int> batch(fixes.begin(), fixes.begin() + n_fixes);
            batch.insert(batch.end(), more.begin(), more.begin() + std::min(more.size(), max_batch - n_fixes));
            if (!replay_expand(batch)) break;
        }
    } catch (...) { cleanup(); throw; }
    cleanup();

    // The leaves that cover the typed text are disjoint events; report each one's probability given the typed text.
    // `covered` only grows as more paths are expanded, so these are upper bounds by at most unresolved / covered.
    // Corrections aren't given the typed text, so theirs is a ratio: how many times as likely as the typed text.
    // A correction replaces the last `replace` bytes of the typed text with its text.
    std::map<std::tuple<std::string, bool, size_t>, std::pair<double, bool>> leaves;  // (text, end, replace) → (probability, done)
    for (const Node & n : nodes) {
        if (!n.covers || n.kept > 0 || (n.expanded && !n.end)) continue;
        if (n.start >= 0 && std::exp(n.logprob) < covered * corrections) continue;
        std::string cont = cont_of(n);
        cont.resize(cont.size() - incomplete_utf8_tail(cont));
        auto & leaf = leaves[{cont, n.end, n.start >= 0 ? typed.size() - n.start : 0}];
        leaf.first += std::exp(n.logprob) / covered;
        leaf.second = leaf.second || n.done;
    }
    json list = json::array();
    for (const auto & [key, value] : leaves) {
        const auto & [cont, end, replace] = key;
        json item = {{"text", cont}, {"prob", value.first}, {"end", end}, {"done", value.second}};
        if (replace > 0) item["replace"] = replace;
        list.push_back(item);
    }
    std::sort(list.begin(), list.end(), [](const json & a, const json & b) { return a["prob"].get<double>() > b["prob"].get<double>(); });

    return {
        {"typed", typed},
        {"prefix_logprob", covered > 0 ? json(std::log(covered)) : json(nullptr)},
        {"unresolved", covered > 0 ? std::max(unresolved, 0.0) / covered : 0.0},
        {"continuations", list},
        {"stats", {
            {"prompt_tokens", target.size()}, {"reused_tokens", sync_stats.reused}, {"decoded_tokens", sync_stats.decoded},
            {"rounds", rounds}, {"extra_rounds", extra_rounds}, {"nodes", nodes.size()}, {"restore_ms", restore_ms},
            {"tokenize_ms", t_tokenize}, {"prefill_ms", sync_stats.ms}, {"search_ms", now_ms() - t1}, {"total_ms", now_ms() - t0},
        }},
    };
}

// ── Scoring ──────────────────────────────────────────────────────────────────

Engine::Scored Engine::score_one(const float * logits, llama_token actual, int k, bool want_rank) const {
    Scored out;
    double lse = log_sum_exp(logits, n_vocab);
    out.logprob = logits[actual] - lse;
    using Entry = std::pair<float, llama_token>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;   // the k largest so far, smallest on top
    for (llama_token t = 0; t < n_vocab && k > 0; t++) {
        if ((int) heap.size() < k) heap.push({logits[t], t});
        else if (logits[t] > heap.top().first) { heap.pop(); heap.push({logits[t], t}); }
    }
    for (; !heap.empty(); heap.pop()) out.top.push_back({heap.top().second, heap.top().first - lse});
    std::reverse(out.top.begin(), out.top.end());
    if (want_rank) {
        out.rank = 1;
        for (llama_token t = 0; t < n_vocab; t++) out.rank += logits[t] > logits[actual];
    }
    return out;
}

std::vector<Engine::Scored> Engine::score_tokens(const Tokens & tokens, size_t first, int k, bool want_rank, SyncStats & stats, double & score_ms) {
    size_t n = tokens.size();
    first = std::max<size_t>(first, 1);
    std::vector<Scored> out(n);
    score_ms = 0;
    if (first >= n) { stats = sync(tokens); return out; }
    // Sync to just before the first scored token, so that every scored position gets decoded with logits.
    stats = sync(Tokens(tokens.begin(), tokens.begin() + first));
    out[first] = score_one(s->cur_logits.data(), tokens[first], k, want_rank);
    SyncStats rest = sync(tokens, [&](const std::vector<size_t> & positions, const std::vector<const float *> & logits) {
        double t1 = now_ms();
        parallel_for(positions.size(), [&](size_t j) { out[positions[j] + 1] = score_one(logits[j], tokens[positions[j] + 1], k, want_rank); });
        score_ms += now_ms() - t1;
    }, first + 1);
    stats.decoded += rest.decoded; stats.decode_ms += rest.decode_ms;
    return out;
}

// How likely was each token of `text`, given the prompt and the text before it, and what did the model expect instead?
json Engine::score(const json & req) {
    std::string prompt = prompt_of(req), text = req.value("text", "");
    int k = std::clamp(req.value("top_k", 1), 1, 20);
    double t0 = now_ms();
    std::string full = prompt + text;
    use_session(req, full);
    Tokenized t = tokenize(full);
    size_t n = t.tokens.size();
    // The first token with any of `text` in it. It may start inside the prompt (when the text's first characters
    // merge with the prompt's last ones); it's scored whole, but its item only shows its part of `text`.
    size_t first = 0;
    while (first < n && t.offsets[first + 1] <= prompt.size()) first++;

    SyncStats stats;
    double score_ms = 0;
    std::vector<Scored> scored = score_tokens(t.tokens, first, k, false, stats, score_ms);

    // Merge tokens that split a character, so every item is valid text.
    json items = json::array();
    auto alternatives = [&](const Scored & sc) {
        json alts = json::array();
        for (const auto & a : sc.top) alts.push_back({{"text", rendered[a.token]}, {"logprob", a.logprob}});
        return alts;
    };
    size_t begin = std::max<size_t>(first, 1);
    double lp = 0;
    for (size_t i = begin; i < n; i++) {
        lp += scored[i].logprob;
        size_t from = std::max(t.offsets[begin], prompt.size()), to = t.offsets[i + 1];
        std::string piece_text = full.substr(from, to - from);
        if (incomplete_utf8_tail(piece_text) && i + 1 < n) continue;
        const Scored & head = scored[begin];
        json item = {{"text", piece_text}, {"logprob", lp}, {"top", rendered[head.top[0].token]}, {"top_logprob", head.top[0].logprob}};
        if (k > 1) item["alternatives"] = alternatives(head);
        items.push_back(item);
        lp = 0; begin = i + 1;
    }
    return {
        {"tokens", items},
        {"stats", {{"prompt_tokens", first}, {"text_tokens", n - first}, {"reused_tokens", stats.reused}, {"decoded_tokens", stats.decoded}, {"decode_ms", stats.decode_ms}, {"score_ms", score_ms}, {"total_ms", now_ms() - t0}}},
    };
}

// ── Chat template ────────────────────────────────────────────────────────────

static const std::string PLACEHOLDER = "XYZZYPLACEHOLDER";

static std::string content_of(const json & message) {
    const json & c = message.at("content");
    if (c.is_string()) return c.get<std::string>();
    std::string out;   // OpenAI-style parts: [{type: "text", text}]
    if (c.is_array()) for (const auto & part : c) if (part.value("type", "") == "text") out += part.value("text", "");
    return out;
}

static bool enable_thinking_of(const json & req, bool fallback) {
    if (req.contains("enable_thinking")) return req["enable_thinking"].get<bool>();
    if (req.contains("chat_template_kwargs") && req["chat_template_kwargs"].is_object())
        return req["chat_template_kwargs"].value("enable_thinking", fallback);
    return fallback;
}

std::string Engine::apply_template(const json & messages, bool add_generation_prompt, bool enable_thinking) const {
    if (!messages.is_array()) throw HttpError{400, "messages must be a list"};
    common_chat_templates_inputs inputs;
    for (const auto & m : messages) {
        common_chat_msg msg;
        msg.role = m.at("role").get<std::string>();
        msg.content = content_of(m);
        inputs.messages.push_back(msg);
    }
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.enable_thinking = enable_thinking;
    inputs.use_jinja = true;
    return common_chat_templates_apply(templates.get(), inputs).prompt.str();
}

// The prompt up to where a new message from `role` starts, after `messages`. For the assistant that's the generation
// prompt. Other roles have none, so we render a placeholder message and cut there, and check that the cut is at a
// token boundary, so the message's text can be appended to the result.
std::string Engine::turn_prefix(const json & messages, const std::string & role, bool enable_thinking) {
    std::string key = role + (enable_thinking ? "+" : "-") + messages.dump();
    auto it = prefix_cache.find(key);
    if (it != prefix_cache.end()) return it->second;
    std::string out;
    if (role == "assistant") out = apply_template(messages, true, enable_thinking);
    else {
        json with = messages;
        with.push_back({{"role", role}, {"content", PLACEHOLDER}});
        std::string full = apply_template(with, false, enable_thinking);
        size_t cut = full.rfind(PLACEHOLDER);
        if (cut == std::string::npos) throw HttpError{500, "the chat template changes message text, so the start of a " + role + " message can't be found"};
        out = full.substr(0, cut);
        Tokens head = tokenize_raw(out), whole = tokenize_raw(full);
        if (head.size() > whole.size() || !std::equal(head.begin(), head.end(), whole.begin()))
            throw HttpError{500, "the start of a " + role + " message isn't at a token boundary in this chat template"};
    }
    if (prefix_cache.size() >= 32) prefix_cache.clear();
    prefix_cache[key] = out;
    return out;
}

// The prompt of a request: `prompt` as given, or the start of a new message from `role` (default assistant) after `messages`.
std::string Engine::prompt_of(const json & req) {
    if (!req.contains("messages")) return req.value("prompt", "");
    if (req.contains("prompt")) throw HttpError{400, "give either prompt or messages, not both"};
    return turn_prefix(req["messages"], req.value("role", "assistant"), enable_thinking_of(req, false));
}

json Engine::render_template(const json & req) {
    return {{"prompt", apply_template(req.at("messages"), req.value("add_generation_prompt", true), enable_thinking_of(req, false))}};
}

// ── HTTP ─────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    std::string model_path, host = "127.0.0.1", alias;
    int port = 52409, ctx_size = 16384, n_gpu_layers = 999, seqs = 24;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--model") model_path = v;
        else if (k == "--host") host = v;
        else if (k == "--port") port = std::stoi(v);
        else if (k == "--ctx") ctx_size = std::stoi(v);
        else if (k == "--gpu-layers") n_gpu_layers = std::stoi(v);
        else if (k == "--seqs") seqs = std::stoi(v);
        else if (k == "--alias") alias = v;
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
    if (model_path.empty() || seqs < 8) {
        fprintf(stderr, "usage: %s --model file.gguf [--host 127.0.0.1] [--port 52409] [--ctx 16384] [--gpu-layers 999] [--seqs 24] [--alias name]\n", argv[0]);
        return 1;
    }
    if (alias.empty()) alias = model_path.substr(model_path.find_last_of('/') + 1);

    llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr); }, nullptr);
    llama_backend_init();
    Engine engine;
    engine.alias = alias;
    try { engine.load(model_path, ctx_size, n_gpu_layers, seqs); }
    catch (const std::exception & e) { fprintf(stderr, "%s (out of GPU memory? try a smaller --ctx or --seqs)\n", e.what()); return 1; }

    httplib::Server server;
    server.set_default_headers({{"Access-Control-Allow-Origin", "*"}, {"Access-Control-Allow-Headers", "Content-Type"}});
    server.Options(".*", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });
    auto send = [](httplib::Response & res, int status, const json & body) {
        res.status = status;
        res.set_content(body.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    };
    auto handle = [&](const char * path, std::function<json(const json &, uint64_t)> fn, bool supersedes) {
        server.Post(path, [&, fn, supersedes](const httplib::Request & req, httplib::Response & res) {
            try {
                json body = json::parse(req.body);
                std::string key = body.value("session", "");
                uint64_t id = supersedes ? engine.stamp(key) : 0;
                std::lock_guard<std::mutex> lock(engine.mutex);
                if (supersedes && !engine.is_latest(key, id)) throw Superseded{};
                send(res, 200, fn(body, id));
            } catch (const Superseded &) { send(res, 409, {{"error", "superseded by a newer request"}});
            } catch (const HttpError & e) { send(res, e.status, {{"error", e.message}});
            } catch (const std::exception & e) { send(res, 400, {{"error", e.what()}}); }
        });
    };
    handle("/v1/continuations", [&](const json & b, uint64_t id) { return engine.continuations(b, id); }, true);
    handle("/v1/score", [&](const json & b, uint64_t) { return engine.score(b); }, false);
    handle("/v1/template", [&](const json & b, uint64_t) { return engine.render_template(b); }, false);
    server.Get("/health", [&](const httplib::Request &, httplib::Response & res) { send(res, 200, {{"status", "ok"}}); });
    server.Get("/v1/sessions", [&](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(engine.mutex);
        send(res, 200, engine.list_sessions());
    });
    server.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        send(res, 200, {{"object", "list"}, {"data", {{{"id", engine.alias}, {"object", "model"}, {"engine", "writing-engine"}}}}});
    });

    fprintf(stderr, "writing-engine: %s on http://%s:%d (context %d tokens)\n", alias.c_str(), host.c_str(), port, engine.n_ctx);
    if (!server.listen(host, port)) { fprintf(stderr, "couldn't listen on %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
