"""Probes behind the measured results in design/interactions.md.

    python3 experiments/interactions/probe.py [server]    # default https://llm.thoughtful-ai.com

1. Moves as a multiple-choice letter, with the options in three orders.
2. Moves as free continuations of "The next sentence should".
3. Gists as free continuations of "This paragraph claims that", for two paragraphs.
"""
import json, sys, urllib.request

SERVER = sys.argv[1] if len(sys.argv) > 1 else "https://llm.thoughtful-ai.com"
ASSIGN = "Write a 500-word op-ed for the campus paper about a change you'd like to see on campus."
P1 = ("The library should stay open all night during finals week. Right now it closes at midnight, which pushes "
      "students into dorm lounges that are loud and crowded. A quiet place to study is not a luxury during the most "
      "stressful week of the semester.")
P2 = ("Of course, keeping the building open costs money. But the library already pays two staff members to stay until "
      "midnight; extending their shift for five nights a year is a small expense next to the tuition students pay for "
      "those same exams.")
MOVES = ["give an example", "concede a cost", "restate the claim", "cite evidence", "describe a scene"]


def post(path, body):
    req = urllib.request.Request(SERVER + path, json.dumps(body).encode(), {"Content-Type": "text/plain"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def prompt_for(user_message):
    return post("/v1/template", {"messages": [{"role": "user", "content": user_message}],
                                 "add_generation_prompt": True, "enable_thinking": False})["prompt"]


def words(n):  # stop regex: n whole words
    return rf"^\s*\S+(?:[ \t]+\S+){{{n - 1}}}(?=\s)"


def show(continuations, k=10):
    for c in continuations[:k]:
        print(f"   {c['prob']:.3f}  {c['text']!r}")


print("1. Moves as a letter")
for order in [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 4, 1, 3]]:
    options = "\n".join(f"{chr(65 + i)}. {MOVES[j]}" for i, j in enumerate(order))
    prompt = prompt_for(f"{ASSIGN}\n\nHere is a draft so far:\n\n{P1}\n\nWhat should the next sentence do?\n{options}\n\n"
                        "Answer with one letter.")
    res = post("/v1/continuations", {"session": "probe", "prompt": prompt, "text": "", "max_depth": 1, "branch": 12,
                                     "min_prob": 0.001})
    by_move = {}
    for c in res["continuations"]:
        t = c["text"].strip()
        if len(t) == 1 and "A" <= t <= "E":
            move = MOVES[order[ord(t) - 65]]
            by_move[move] = by_move.get(move, 0) + c["prob"]
    letters = sum(by_move.values())
    print(f"   order {order}: {letters:.0%} on a letter;",
          ", ".join(f"{m} {p:.3f}" for m, p in sorted(by_move.items(), key=lambda x: -x[1])))
    print("   top first tokens:", [(c["text"], round(c["prob"], 3)) for c in res["continuations"][:4]])

print("2. Moves as free phrases")
prompt = prompt_for(f"{ASSIGN}\n\nHere is a draft so far:\n\n{P1}\n\nIn a few words, what should the next sentence do?")
show(post("/v1/continuations", {"session": "probe", "prompt": prompt, "text": "The next sentence should",
                                "stop": words(4), "branch": 8, "max_rounds": 12, "time_ms": 1500})["continuations"])

print("3. Gists as free phrases")
for name, para in [("P1", P1), ("P2", P2)]:
    prompt = prompt_for(f"Here is a paragraph from a student op-ed:\n\n{para}\n\n"
                        "In a few words, what does a reader take this paragraph to claim?")
    print(f"  {name}")
    show(post("/v1/continuations", {"session": "probe", "prompt": prompt, "text": "This paragraph claims that",
                                    "stop": words(6), "branch": 8, "max_rounds": 16, "time_ms": 2000})["continuations"], 8)
