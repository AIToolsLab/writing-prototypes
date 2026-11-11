# Per-keystroke prediction latency, as the page asks for it, logged to LOG.md so we can see what each
# change costs. Types a sentence (with one typo) one character at a time into the middle of a response,
# at two document sizes, and reports the server's own time per request.
#
# Usage: python3 predict_bench.py "what changed" [--page] [--server URL] [--body '{"units": 5}'] [--no-log]
# --page sends what the page sends (its stop pattern from docs/llm.js, units, corrections; needs node);
# --body adds or overrides fields of every /v1/continuations request.
import argparse, datetime, json, os, statistics, subprocess, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("note")
ap.add_argument("--server", default="http://127.0.0.1:52409")
ap.add_argument("--body", default="{}")
ap.add_argument("--no-log", action="store_true")
ap.add_argument("--page", action="store_true")
args = ap.parse_args()
extra = json.loads(args.body)
if args.page:
    llm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docs/llm.js")
    js = ("const fs=require('fs');eval(fs.readFileSync(process.argv[1],'utf8').replace(/^const CHAT_URL.*$/m,'')"
          ".replace(/location\\.search/g,'\"\"')+';process.stdout.write(wordUnitPattern(1))')")
    stop_pattern = subprocess.run(["node", "-e", js, llm], capture_output=True, text=True, check=True).stdout
    extra = {"units": 5, "corrections": 1, "time_ms": 150, "stop": stop_pattern, **extra}

def post(path, body):
    req = urllib.request.Request(args.server + path, json.dumps(body).encode(), {"Content-Type": "text/plain"})
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(req))
    return d, (time.perf_counter() - t) * 1000

para = ("Writing with AI raises a question about who is doing the thinking. When a model drafts a paragraph, "
        "the writer may feel they have reasoned through an idea that they have only read. ")
typing = " The limit is how far ahaed of the writer's own thinking the model gets."
stop = r"^\s*\S+(?=\s)"
session = f"bench-{os.getpid()}"
rows = []
for nrep in (2, 12):
    doc = para * nrep
    prompt = post("/v1/template", {"messages": [{"role": "user", "content": "Rewrite this document to be clear and concise.\n\n" + doc}],
                                   "add_generation_prompt": True})[0]["prompt"]
    draft = doc[: len(doc) // 2]
    body = {"session": session, "prompt": prompt, "stop": stop, **extra}
    post("/v1/continuations", {**body, "text": draft})  # warm the cache
    server, rt, filled = [], [], []
    for ch in typing:
        draft += ch
        d, ms = post("/v1/continuations", {**body, "text": draft})
        server.append(d["stats"]["total_ms"]); rt.append(ms)
        filled.append(len({c["text"].split()[0] if c["text"].split() else "" for c in d["continuations"] if c["text"].strip()}))
    p90 = lambda xs: sorted(xs)[int(len(xs) * 0.9)]
    rows.append((d["stats"]["prompt_tokens"], statistics.median(server), p90(server), max(server), statistics.median(rt), statistics.median(filled)))
    print(f"{d['stats']['prompt_tokens']} tokens: server median {statistics.median(server):.0f} ms, p90 {p90(server):.0f}, "
          f"max {max(server):.0f}; round trip median {statistics.median(rt):.0f}; distinct first words median {statistics.median(filled)}")

if not args.no_log:
    here = os.path.dirname(os.path.abspath(__file__))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    dirty = "+" if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "../server.cpp"], cwd=here).returncode else ""
    model = json.load(urllib.request.urlopen(args.server + "/v1/models"))["data"][0]["id"]
    env = subprocess.run(["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", "writing-engine"], capture_output=True, text=True).stdout
    rev = [line.split("=", 1)[1][:7] for line in env.splitlines() if line.startswith("MODEL_REVISION=")]
    if rev: model += f"@{rev[0]}"
    log = os.path.join(here, "LOG.md")
    if not os.path.exists(log):
        with open(log, "w") as f:
            f.write("# Prediction latency log\n\nWritten by `predict_bench.py`: server time per keystroke (ms), typing a 73-character\n"
                    "sentence with one typo into the middle of a response. Target: under 200 ms at the page.\n"
                    "Commit is the working tree's (`+`: server.cpp uncommitted), which is only the running server's if the container was rebuilt from it.\n"
                    "\"words\" counts distinct first words among all continuations returned, partial ones included.\n\n"
                    "| date | commit | model | request extras | note | prompt tokens | median | p90 | max | words |\n"
                    "|---|---|---|---|---|---|---|---|---|---|\n")
    shown = "page" + (" + " + args.body if args.body != "{}" else "") if args.page else args.body
    with open(log, "a") as f:
        for tokens, med, p, mx, _, words in rows:
            f.write(f"| {datetime.date.today()} | {commit}{dirty} | {model} | `{shown}` | {args.note} | {tokens} | {med:.0f} | {p:.0f} | {mx:.0f} | {words:g} |\n")
