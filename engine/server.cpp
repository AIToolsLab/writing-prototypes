// A small inference server for writing tools, on libllama.
//
// What it's for: "given this context and the characters typed so far, what are the likely next
// characters?", answered fast and with honest probabilities.
//
// - Fast: one model context stays warm. Each request reuses the longest cached prefix; "checkpoints"
//   (parked sequence ids, which share memory with the main sequence) let us rewind to an earlier
//   point even for models with recurrent state, which can't be rewound token by token. Search
//   branches are forked sequences, which also share the prefix instead of copying it.
// - Honest: the model predicts tokens, but the writer typed characters. We condition on the
//   characters by summing over every tokenization that is consistent with them, from the last
//   point where the tokenization is settled (see `boundary`). Probabilities are the model's own;
//   nothing is renormalized over the candidates we happened to keep.
//
// Endpoints (JSON): GET /health, GET /v1/models, POST /v1/template, POST /v1/continuations, POST /v1/score.
// One writer at a time: all requests share one cache and are serialized.

#include "llama.h"
#include "chat.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
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

    // Cache: sequence 0 holds `cur`. Sequences 1..n_checkpoints are checkpoints; the rest are for forks.
    // Sequences are scarce on models with recurrent state, which costs memory per distinct sequence
    // (about 150 MB each for Qwen3.8-27B), unlike the attention cache, which sequences share.
    int n_checkpoints = 0;
    static constexpr size_t CHUNK = 256;         // prefill chunk; a checkpoint is parked after each
    static constexpr size_t KEEP_RECENT = 4;
    Tokens cur;
    std::vector<float> cur_logits;               // next-token logits after `cur`; empty if unknown
    std::vector<Checkpoint> checkpoints;         // sorted by len; each is a prefix of `cur`
    std::vector<llama_seq_id> free_checkpoint_seqs, free_fork_seqs;

    // Last tokenization, to retokenize only the changed tail.
    std::string last_text; Tokens last_tokens; std::vector<size_t> last_offsets;

    std::mutex mutex;
    std::atomic<uint64_t> latest_request{0};

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
    void drop_checkpoint(size_t i);
    void reset_cache();
    struct SyncStats { size_t reused = 0, decoded = 0; double ms = 0, decode_ms = 0; };
    // on_chunk(positions, logits): logits[k] predicts the token after position positions[k].
    using ChunkFn = std::function<void(const std::vector<size_t> &, const std::vector<const float *> &)>;
    SyncStats sync(const Tokens & target, const ChunkFn & on_chunk = nullptr, size_t logits_from = SIZE_MAX);

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
    n_checkpoints = std::min(31, (seqs - 1) / 2 - 1);
    for (int s = n_checkpoints; s >= 1; s--) free_checkpoint_seqs.push_back(s);
    for (int s = seqs - 1; s > n_checkpoints; s--) free_fork_seqs.push_back(s);

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
    size_t common = 0, max = std::min(text.size(), last_text.size());
    while (common < max && text[common] == last_text[common]) common++;
    size_t restart = safe_restart({last_tokens, last_offsets}, common);
    size_t start_byte = restart ? last_offsets[restart] : 0;
    out.tokens.assign(last_tokens.begin(), last_tokens.begin() + restart);
    out.offsets.assign(last_offsets.begin(), last_offsets.begin() + restart);
    bool had_bos = restart == 0 && bos != LLAMA_TOKEN_NULL && !starts_with(text, rendered[bos]);
    if (had_bos) { out.tokens.push_back(bos); out.offsets.push_back(0); }
    size_t offset = start_byte;
    for (llama_token t : tokenize_raw(text.substr(start_byte))) {
        out.tokens.push_back(t); out.offsets.push_back(offset);
        offset += rendered[t].size();
    }
    out.offsets.push_back(offset);
    if (offset != text.size()) throw HttpError{500, "this tokenizer doesn't round-trip text exactly, which the engine relies on"};
    last_text = text; last_tokens = out.tokens; last_offsets = out.offsets;
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

void Engine::drop_checkpoint(size_t i) {
    llama_memory_seq_rm(mem, checkpoints[i].seq, -1, -1);
    free_checkpoint_seqs.push_back(checkpoints[i].seq);
    checkpoints.erase(checkpoints.begin() + i);
}

// Park a checkpoint at the end of `cur`.
void Engine::add_checkpoint(bool with_logits) {
    if (cur.empty()) return;
    if (!checkpoints.empty() && checkpoints.back().len == cur.size()) {
        if (with_logits) checkpoints.back().logits = cur_logits;
        return;
    }
    if (free_checkpoint_seqs.empty()) {
        // Thin out: keep the first and the most recent ones; of the rest, drop the one that leaves the smallest gap.
        size_t best = 1, best_gap = SIZE_MAX;
        for (size_t i = 1; i + KEEP_RECENT < checkpoints.size(); i++) {
            size_t gap = checkpoints[i + 1].len - checkpoints[i - 1].len;
            if (gap < best_gap) { best_gap = gap; best = i; }
        }
        drop_checkpoint(best);
    }
    llama_seq_id seq = free_checkpoint_seqs.back(); free_checkpoint_seqs.pop_back();
    llama_memory_seq_cp(mem, 0, seq, -1, -1);
    checkpoints.push_back({seq, cur.size(), with_logits ? cur_logits : std::vector<float>()});
}

void Engine::reset_cache() {
    llama_memory_clear(mem, true);
    for (auto & c : checkpoints) free_checkpoint_seqs.push_back(c.seq);
    checkpoints.clear(); cur.clear(); cur_logits.clear();
}

// Make sequence 0 hold exactly `target`, with next-token logits in cur_logits, reusing what we can.
// If on_chunk is given, it gets the logits at every decoded position i with i + 1 >= logits_from (except the
// last position of the target); the caller arranges, by syncing to a shorter target first, that all of them
// get decoded.
Engine::SyncStats Engine::sync(const Tokens & target, const ChunkFn & on_chunk, size_t logits_from) {
    if (target.empty()) throw HttpError{400, "empty prompt"};
    if ((int) target.size() + 64 > n_ctx) throw HttpError{400, "prompt too long for the context (" + std::to_string(target.size()) + " tokens)"};
    SyncStats stats;
    double t0 = now_ms();
    size_t common = 0;
    while (common < cur.size() && common < target.size() && cur[common] == target[common]) common++;

    if (common == cur.size() && common == target.size() && !cur_logits.empty()) { stats.reused = common; return stats; }
    if (common < cur.size() || common == target.size()) {
        // Rewind to the latest usable checkpoint. One at the very end of the target is only usable if it has logits.
        int pick = -1;
        for (size_t i = 0; i < checkpoints.size(); i++) {
            const auto & c = checkpoints[i];
            if (c.len <= common && (c.len < target.size() || !c.logits.empty())) pick = i;
        }
        if (pick < 0) reset_cache();
        else {
            for (size_t i = checkpoints.size(); i-- > 0;) if (checkpoints[i].len > common) drop_checkpoint(i);
            const auto & c = checkpoints[pick];
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_memory_seq_cp(mem, c.seq, 0, -1, -1);
            cur.resize(c.len);
            cur_logits = c.logits;
        }
    }
    stats.reused = cur.size();

    while (cur.size() < target.size()) {
        size_t begin = cur.size();
        size_t end = std::min(target.size(), (begin / CHUNK + 1) * CHUNK);
        std::vector<std::array<int, 3>> items;
        std::vector<char> want;
        for (size_t i = begin; i < end; i++) {
            items.push_back({target[i], (int) i, 0});
            want.push_back(i + 1 == end || (on_chunk && i + 1 >= logits_from));
        }
        stats.decode_ms += decode(items, want);
        if (on_chunk) {
            std::vector<size_t> positions; std::vector<const float *> logits;
            for (size_t i = begin; i < end; i++) if (want[i - begin] && i + 1 < target.size()) { positions.push_back(i); logits.push_back(llama_get_logits_ith(ctx, i - begin)); }
            on_chunk(positions, logits);
        }
        const float * last = llama_get_logits_ith(ctx, end - begin - 1);
        cur.insert(cur.end(), target.begin() + begin, target.begin() + end);
        cur_logits.assign(last, last + n_vocab);
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
    bool expanded = false;
    llama_seq_id seq = -1; // holds this node's state while children may still fork from it
    int pending = 0;       // children that may still need that state
    int kept = 0;          // children kept as nodes
};

struct Expansion {
    struct Child { llama_token token; double logprob; };
    std::vector<Child> children;
    double covering_mass = 0;   // probability, within a not-yet-covering node, of tokens that reach past the typed text
};

json Engine::continuations(const json & req, uint64_t id) {
    std::string prompt = req.value("prompt", ""), text = req.value("text", "");
    int max_rounds = req.value("max_rounds", 8), max_batch = req.value("max_batch", 8), branch = req.value("branch", 8);
    int max_depth = req.value("max_depth", 8), max_chars = req.value("max_chars", 0);
    double min_prob = req.value("min_prob", 0.01), time_ms = req.value("time_ms", 400.0);
    bool has_stop = req.contains("stop") && req["stop"].is_string();
    std::regex stop;
    if (has_stop) {
        try { stop = std::regex(req["stop"].get<std::string>(), std::regex::ECMAScript | std::regex::optimize); }
        catch (const std::regex_error & e) { throw HttpError{400, std::string("bad stop regex: ") + e.what()}; }
    }
    auto superseded = [&] { return latest_request.load() != id; };

    double t0 = now_ms();
    std::string full = prompt + text;
    Tokenized tokenized = tokenize(full);
    size_t n_settled = boundary(full, tokenized);
    // The characters to condition on. They never reach back into the prompt past a control token, but may
    // include the end of the prompt (e.g. its final newlines), since the reply's first token can merge with it.
    std::string typed = full.substr(tokenized.offsets[n_settled]);
    Tokens target(tokenized.tokens.begin(), tokenized.tokens.begin() + n_settled);
    double t_tokenize = now_ms() - t0;
    if (superseded()) throw Superseded{};

    SyncStats sync_stats = sync(target);
    double t1 = now_ms();

    std::vector<Node> nodes;
    nodes.push_back({-1, LLAMA_TOKEN_NULL, 0.0, "", 0, 0, typed.empty()});
    nodes[0].seq = 0;
    double covered = 0;      // probability that the text continues past `typed` (or ends exactly there), found so far
    double unresolved = 0;   // probability of paths still inside `typed` that we haven't expanded
    if (typed.empty()) covered = 1;
    std::vector<llama_seq_id> used_seqs;

    auto is_done = [&](const Node & n) {
        if (n.end) return true;
        if (!n.covers) return false;
        std::string cont = n.text.substr(typed.size());
        if (incomplete_utf8_tail(cont)) return n.past >= max_depth + 2;
        if (n.past >= max_depth) return true;
        if (max_chars > 0 && (int) utf8_length(cont) >= max_chars) return true;
        return has_stop && std::regex_search(cont, stop);
    };

    // What to keep from one node's next-token distribution. Pure, so nodes can be done in parallel.
    auto expand = [&](const Node & node, const float * logits, double threshold) {
        Expansion out;
        double base = node.logprob - log_sum_exp(logits, n_vocab);
        std::vector<Expansion::Child> candidates;
        if (node.covers) {
            for (llama_token t = 0; t < n_vocab; t++) {
                double lp = base + logits[t];
                if (lp < threshold) continue;
                if (is_eog[t] || !piece[t].empty()) candidates.push_back({t, lp});
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
                if (lp >= threshold) candidates.push_back({*it, lp});
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
            child.covers = nodes[ni].covers || child.text.size() >= typed.size();
            child.past = nodes[ni].covers ? nodes[ni].past + 1 : 0;
            if (!nodes[ni].covers && child.text.size() == typed.size()) covered += std::exp(c.logprob);
            if (!child.covers) unresolved += std::exp(c.logprob);
            child.done = is_done(child);
            if (!child.done) nodes[ni].pending++;
            nodes[ni].kept++;
            nodes.push_back(std::move(child));
        }
    };
    auto release = [&](Node & n) {
        if (n.seq > 0) { llama_memory_seq_rm(mem, n.seq, -1, -1); free_fork_seqs.push_back(n.seq); }
        n.seq = -1;
    };
    auto cleanup = [&] {
        for (auto & n : nodes) release(n);
    };

    int rounds = 0;
    try {
        if (!typed.empty()) unresolved = 1;  // the root itself
        apply(0, expand(nodes[0], cur_logits.data(), -INFINITY));

        for (; rounds < max_rounds && now_ms() - t0 < time_ms; rounds++) {
            if (superseded()) throw Superseded{};
            double total = covered + std::max(unresolved, 0.0);
            if (total <= 0) break;
            // The most probable unfinished paths. Paths still inside the typed text hide probability, so they get a lower bar.
            std::vector<int> batch;
            for (size_t i = 1; i < nodes.size(); i++) {
                const Node & n = nodes[i];
                if (n.expanded || n.done) continue;
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
                    if (free_fork_seqs.empty()) continue;
                    n.seq = free_fork_seqs.back(); free_fork_seqs.pop_back();
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
            parallel_for(decoded.size(), [&](size_t i) { expansions[i] = expand(nodes[decoded[i]], logits[i], threshold); });
            for (size_t i = 0; i < decoded.size(); i++) {
                apply(decoded[i], expansions[i]);
                if (nodes[decoded[i]].pending == 0) release(nodes[decoded[i]]);
            }
        }
    } catch (...) { cleanup(); throw; }
    cleanup();

    // The leaves that cover the typed text are disjoint events; report each one's probability given the typed text.
    // `covered` only grows as more paths are expanded, so these are upper bounds by at most unresolved / covered.
    std::map<std::pair<std::string, bool>, std::pair<double, bool>> leaves;  // (text, end) → (probability, done)
    for (const Node & n : nodes) {
        if (!n.covers || n.kept > 0 || (n.expanded && !n.end)) continue;
        std::string cont = n.text.substr(typed.size());
        cont.resize(cont.size() - incomplete_utf8_tail(cont));
        auto & leaf = leaves[{cont, n.end}];
        leaf.first += std::exp(n.logprob) / covered;
        leaf.second = leaf.second || n.done;
    }
    json list = json::array();
    for (const auto & [key, value] : leaves) list.push_back({{"text", key.first}, {"prob", value.first}, {"end", key.second}, {"done", value.second}});
    std::sort(list.begin(), list.end(), [](const json & a, const json & b) { return a["prob"].get<double>() > b["prob"].get<double>(); });

    return {
        {"typed", typed},
        {"prefix_logprob", covered > 0 ? json(std::log(covered)) : json(nullptr)},
        {"unresolved", covered > 0 ? std::max(unresolved, 0.0) / covered : 0.0},
        {"continuations", list},
        {"stats", {
            {"prompt_tokens", target.size()}, {"reused_tokens", sync_stats.reused}, {"decoded_tokens", sync_stats.decoded},
            {"rounds", rounds}, {"nodes", nodes.size()},
            {"tokenize_ms", t_tokenize}, {"prefill_ms", sync_stats.ms}, {"search_ms", now_ms() - t1}, {"total_ms", now_ms() - t0},
        }},
    };
}

// ── Scoring ──────────────────────────────────────────────────────────────────

// How likely was each token of `text`, given `prompt` and the text before it, and what did the model expect instead?
json Engine::score(const json & req) {
    std::string prompt = req.value("prompt", ""), text = req.value("text", "");
    double t0 = now_ms();
    Tokenized t = tokenize(prompt + text);
    size_t first = 0;
    while (first < t.tokens.size() && t.offsets[first] < prompt.size()) first++;
    first = std::max<size_t>(first, 1);
    size_t n = t.tokens.size();

    struct Scored { double logprob; llama_token top; double top_logprob; };
    std::vector<Scored> scored(n);
    auto score_logits = [&](size_t i, const float * logits) {    // logits predict token i + 1
        double lse = log_sum_exp(logits, n_vocab);
        llama_token top = std::max_element(logits, logits + n_vocab) - logits;
        scored[i + 1] = {logits[t.tokens[i + 1]] - lse, top, logits[top] - lse};
    };

    SyncStats stats;
    double score_ms = 0;
    if (first < n) {
        // Sync to just before the text, so that every text position gets decoded with logits.
        Tokens head(t.tokens.begin(), t.tokens.begin() + first);
        stats = sync(head);
        score_logits(first - 1, cur_logits.data());
        SyncStats rest = sync(t.tokens, [&](const std::vector<size_t> & positions, const std::vector<const float *> & logits) {
            double t1 = now_ms();
            parallel_for(positions.size(), [&](size_t k) { score_logits(positions[k], logits[k]); });
            score_ms += now_ms() - t1;
        }, first + 1);
        stats.decoded += rest.decoded; stats.decode_ms += rest.decode_ms;
    }

    // Merge tokens that split a character, so every item is valid text.
    json items = json::array();
    std::string buf; double lp = 0; llama_token top = LLAMA_TOKEN_NULL; double top_lp = 0;
    for (size_t i = first; i < n; i++) {
        if (buf.empty()) { top = scored[i].top; top_lp = scored[i].top_logprob; }
        buf += rendered[t.tokens[i]]; lp += scored[i].logprob;
        if (incomplete_utf8_tail(buf) && i + 1 < n) continue;
        items.push_back({{"text", buf}, {"logprob", lp}, {"top", rendered[top]}, {"top_logprob", top_lp}});
        buf.clear(); lp = 0;
    }
    return {
        {"tokens", items},
        {"stats", {{"prompt_tokens", first}, {"text_tokens", n - first}, {"reused_tokens", stats.reused}, {"decoded_tokens", stats.decoded}, {"decode_ms", stats.decode_ms}, {"score_ms", score_ms}, {"total_ms", now_ms() - t0}}},
    };
}

// ── Chat template ────────────────────────────────────────────────────────────

json Engine::render_template(const json & req) {
    common_chat_templates_inputs inputs;
    for (const auto & m : req.at("messages")) {
        common_chat_msg msg;
        msg.role = m.at("role").get<std::string>();
        msg.content = m.at("content").get<std::string>();
        inputs.messages.push_back(msg);
    }
    inputs.add_generation_prompt = req.value("add_generation_prompt", true);
    inputs.enable_thinking = req.value("enable_thinking", false);
    inputs.use_jinja = true;
    std::string prompt = common_chat_templates_apply(templates.get(), inputs).prompt.str();
    return {{"prompt", prompt}};
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
            // A newer keystroke makes older requests pointless: they give up at their next check.
            uint64_t id = supersedes ? ++engine.latest_request : 0;
            try {
                json body = json::parse(req.body);
                std::lock_guard<std::mutex> lock(engine.mutex);
                if (supersedes && engine.latest_request.load() != id) throw Superseded{};
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
    server.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        send(res, 200, {{"object", "list"}, {"data", {{{"id", engine.alias}, {"object", "model"}, {"engine", "writing-engine"}}}}});
    });

    fprintf(stderr, "writing-engine: %s on http://%s:%d (context %d tokens)\n", alias.c_str(), host.c_str(), port, engine.n_ctx);
    if (!server.listen(host, port)) { fprintf(stderr, "couldn't listen on %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
