import re


def units(resp, stop=r"^\s*\S+(?=\s)"):
    """P(unit) for each unit (the text through the end of its `stop` match), summed over continuations, as the
    page does. Corrections and a bare end of turn are left out."""
    out = {}
    for c in resp["continuations"]:
        if "replace" in c or not c["text"]: continue
        m = re.search(stop, c["text"] + (" " if c["end"] else ""))
        u = m.group(0) if m else c["text"]
        out[u] = out.get(u, 0) + c["prob"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def starting_with(resp, prefix):
    """P(the text continues with `prefix`), summed over the continuations that start with it (a lower bound)."""
    return sum(c["prob"] for c in resp["continuations"] if "replace" not in c and c["text"].startswith(prefix))
