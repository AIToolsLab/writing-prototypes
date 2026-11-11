# Typing simulation against the writing engine: scoring, per-keystroke prediction, backspace, and a jump
# back to mid-draft, at two document sizes. Usage: python3 typing_sim.py [http://localhost:52409]
import json,urllib.request,time,statistics
import sys
U=sys.argv[1] if len(sys.argv)>1 else "http://localhost:52409"
def post(p,b):
    r=urllib.request.Request(U+p,json.dumps(b).encode(),{"Content-Type":"application/json"})
    t=time.perf_counter()
    try: d=json.load(urllib.request.urlopen(r))
    except urllib.error.HTTPError as e: d={"error":e.read().decode()}
    return d,(time.perf_counter()-t)*1000
para="Writing with AI raises a question about who is doing the thinking. When a model drafts a paragraph, the writer may feel they have reasoned through an idea that they have only read. "
stop=r"^\s*\S+(?=\s)"
for nrep in (12,60):
    doc=para*nrep
    prompt=post("/v1/template",{"messages":[{"role":"user","content":"Rewrite this document to be clear and concise.\n\n"+doc}]})[0]["prompt"]
    # score (highlights)
    d,ms=post("/v1/score",{"prompt":prompt,"text":doc}); 
    if "error" in d: print(d); continue
    s=d["stats"]; worst=sorted(d["tokens"],key=lambda t:t["logprob"]-t["top_logprob"])[:3]
    print(f"score {s['text_tokens']} tokens: {ms:.0f} ms (decoded {s['decoded_tokens']});", [(t["text"],t["top"],round(t["top_logprob"]-t["logprob"],1)) for t in worst], "roundtrip", "".join(t["text"] for t in d["tokens"])==doc)
    d,ms=post("/v1/score",{"prompt":prompt,"text":doc}); print(f"  again: {ms:.0f} ms (decoded {d['stats']['decoded_tokens']})")
    # typing simulation: the response so far is half the doc; type a sentence one character at a time
    draft=doc[:len(doc)//2]
    d,ms=post("/v1/continuations",{"prompt":prompt,"text":draft,"stop":stop}); print(f"  first prediction after scoring: {ms:.0f} ms, decoded {d['stats']['decoded_tokens']}")
    times=[];dec=[]
    for ch in " The limit is how far ahead of the writer's own thinking the model gets.":
        draft+=ch
        d,ms=post("/v1/continuations",{"prompt":prompt,"text":draft,"stop":stop}); times.append(ms); dec.append(d["stats"]["decoded_tokens"])
    print(f"  typing {len(times)} chars at {d['stats']['prompt_tokens']} tokens: median {statistics.median(times):.0f} ms, p90 {sorted(times)[int(len(times)*.9)]:.0f}, max {max(times):.0f}; top:", [(c['text'],round(c['prob'],2)) for c in d["continuations"][:3]])
    # backspace over two words, then an edit in the middle of the draft
    d,ms=post("/v1/continuations",{"prompt":prompt,"text":draft[:-12],"stop":stop}); print(f"  backspace 12 chars: {ms:.0f} ms, decoded {d['stats']['decoded_tokens']}")
    mid=len(draft)//2
    d,ms=post("/v1/continuations",{"prompt":prompt,"text":draft[:mid],"stop":stop}); print(f"  jump to mid-draft: {ms:.0f} ms, decoded {d['stats']['decoded_tokens']}")
