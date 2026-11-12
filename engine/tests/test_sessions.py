"""Sessions: several writers' caches, moved to host memory and back, and newer requests superseding older ones."""
import threading, time

import pytest

from conftest import WORD

PARA = ("Notes on {t}. People have studied {t} for a long time, and there is still more to learn about how {t} shaped "
        "the places where we live and the way we work. ")


def writer(engine, name, topic, paragraphs=4):
    prompt = engine("/v1/template", {"messages": [{"role": "user", "content": "Rewrite this.\n\n" + PARA.format(t=topic) * paragraphs}]})["prompt"]
    return {"session": name, "prompt": prompt, "text": f"The history of {topic}", "stop": WORD}


def hot(engine, name):
    return next(x["hot"] for x in engine("/v1/sessions")["sessions"] if x["id"] == "id:" + name)


def test_restored_session_gives_the_same_probabilities(engine, session):
    """A session moved to host memory (to make room for others) gives exactly the same answer when it comes back."""
    me = {**writer(engine, session, "rivers"), "time_ms": 60000}    # the same search both times, however slow the prefill
    before = engine("/v1/continuations", me)
    for i in range(80):
        engine("/v1/continuations", writer(engine, f"{session}-other{i}", f"topic {i}"))
        if not hot(engine, session): break
    else:
        pytest.skip("couldn't push the session out of the GPU")
    after = engine("/v1/continuations", me)
    assert after["stats"]["restore_ms"] > 0
    assert [(c["text"], c["prob"]) for c in after["continuations"]] == [(c["text"], c["prob"]) for c in before["continuations"]]


def test_newer_request_supersedes_older(engine, session):
    """While the engine is busy, a writer's older queued request gives way to a newer one (409)."""
    busy = writer(engine, session + "-busy", "glaciers", paragraphs=40)
    me = writer(engine, session, "tides")
    results = {}
    def run(key, body, path="/v1/continuations"):
        results[key] = engine.raw(path, body).status_code
    t0 = threading.Thread(target=run, args=("busy", {**busy, "text": busy["text"]}, "/v1/score"))
    t0.start()
    time.sleep(0.05)
    t1 = threading.Thread(target=run, args=("old", me)); t1.start(); time.sleep(0.05)
    t2 = threading.Thread(target=run, args=("new", {**me, "text": me["text"] + " is"})); t2.start()
    for t in (t0, t1, t2): t.join()
    assert results == {"busy": 200, "old": 409, "new": 200}
