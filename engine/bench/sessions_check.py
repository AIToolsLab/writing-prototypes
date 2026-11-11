# Several writers at once: interleaved typing, eviction to host memory, and restore.
# Usage: python3 sessions_check.py [http://localhost:52409]
import json, sys, time, urllib.request, statistics
U = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:52409"
def call(path, body=None):
    req = urllib.request.Request(U + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    t = time.perf_counter()
    try: d = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e: d = {"error": e.read().decode()}
    return d, (time.perf_counter() - t) * 1000
STOP = r"^\s*\S+(?=\s)"
TOPICS = ["rivers", "bridges", "bread", "chess", "wind power", "typefaces", "honeybees", "tides", "violins", "glaciers",
          "railways", "coral reefs", "printing", "volcanoes", "lighthouses", "maps"]
def writer(i, paragraphs):
    topic = TOPICS[i % len(TOPICS)]
    doc = (f"Notes on {topic}, part {i}. People have studied {topic} for a long time, and there is still more to learn about how {topic} "
           f"shaped the places where we live and the way we work. ") * paragraphs
    prompt = call("/v1/template", {"messages": [{"role": "user", "content": "Rewrite this document to be clear and concise.\n\n" + doc}]})[0]["prompt"]
    return {"session": f"writer-{i}", "prompt": prompt, "text": f"The history of {topic}"}
def predict(w):
    d, ms = call("/v1/continuations", {**w, "stop": STOP})
    if "error" in d: raise SystemExit(d["error"])
    return d, ms
def summary():
    d = call("/v1/sessions")[0]
    hot = [x for x in d["sessions"] if x["hot"]]; cold = [x for x in d["sessions"] if not x["hot"]]
    return (f"{len(hot)} in GPU (checkpoints: {[x['checkpoints'] for x in hot]}), {len(cold)} in host memory "
            f"({sum(x['saved_mb'] for x in cold):.0f} MB), {d['free_sequences']} sequences free")

for n, paragraphs, label in [(3, 12, "3 writers, ~550-token documents"), (16, 12, "16 writers, ~550-token documents"), (5, 100, "5 writers, ~4,400-token documents")]:
    print(f"\n== {label}")
    writers = [writer(100 * paragraphs + i, paragraphs) for i in range(n)]
    first = {}
    for w in writers:
        d, ms = predict(w); first[w["session"]] = d
    print("  after everyone's first request:", summary())
    # Everyone types a few words, taking turns one character at a time.
    times, restores, decoded = [], [], []
    for ch in " is longer than":
        for w in writers:
            w["text"] += ch
            d, ms = predict(w); times.append(ms); restores.append(d["stats"]["restore_ms"]); decoded.append(d["stats"]["decoded_tokens"])
    came_back = [r for r in restores if r > 0]
    print(f"  interleaved typing, {len(times)} keystrokes: median {statistics.median(times):.0f} ms, p90 {sorted(times)[int(len(times) * .9)]:.0f}, max {max(times):.0f}")
    print(f"  keystrokes that had to bring a session back from host memory: {len(came_back)}" +
          (f" (restore median {statistics.median(came_back):.0f} ms)" if came_back else "") + f"; most tokens re-decoded on a keystroke: {max(decoded)}")
    print("  now:", summary())
    # Does the first writer get the same answer for their first draft as before all this?
    w = dict(writers[0]); w["text"] = w["text"][:-len(" is longer than")]
    d, ms = predict(w)
    a = {c["text"]: c["prob"] for c in first[w["session"]]["continuations"]}; b = {c["text"]: c["prob"] for c in d["continuations"]}
    diff = max(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))
    print(f"  first writer, first draft again: {ms:.0f} ms, largest change in any probability {diff:.4f}")
