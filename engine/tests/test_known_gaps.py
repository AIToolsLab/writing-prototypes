"""Properties the engine doesn't have yet. Each is expected to fail (xfail, strict): when one starts passing, pytest
reports it, and the marker should come off. The request shapes here are proposals, not settled APIs."""
import math

import pytest

from conftest import REWRITE, WORD


@pytest.mark.xfail(reason="/v1/score conditions on tokens: the last word's logprob is that of its standard tokenization, "
                          "not of its characters (' de' rather than ' de…', which includes ' deep')")
def test_score_conditions_on_characters(engine, session):
    text = "Use de"
    c = engine("/v1/continuations", {"session": session, "messages": REWRITE, "text": text, "stop": WORD})
    s = engine("/v1/score", {"session": session, "messages": REWRITE, "text": text})
    # P(" de" | "Use") from both: continuations give it for characters, score for the token " de".
    assert c["typed"] == " de"
    assert s["tokens"][-1]["logprob"] == pytest.approx(c["prefix_logprob"], abs=0.3)


@pytest.mark.xfail(reason="no tree in the response: a conditional inside a continuation, like P(word 2 | word 1), "
                          "can't be bounded, since both sums lose different pruned branches")
def test_conditionals_inside_continuations_are_bounded(engine, session):
    d = engine("/v1/continuations", {"session": session, "messages": REWRITE, "text": "Deep learning is", "stop": r"^\s*\S+\s+\S+(?=\s)", "tree": True})
    tree = d["tree"]   # proposed: [{text, logprob, pruned}], pruned = probability of the children not kept
    for node in tree:
        assert 0 <= node["pruned"] <= math.exp(node["logprob"])


@pytest.mark.xfail(reason="no /v1/compare: the probability of a string you name is only available if the search kept it")
def test_compare_named_options(engine, session):
    d = engine("/v1/compare", {"session": session, "messages": REWRITE, "text": "Deep learning is",
                                "options": [" powerful", " a", " banana"]})
    p = {o["text"]: o["logprob"] for o in d["options"]}
    assert p[" powerful"] > p[" banana"]


@pytest.mark.xfail(reason="no /v1/chat/completions; the old Structured engine used it with json_schema")
def test_chat_completions(engine):
    engine("/v1/chat/completions", {"messages": REWRITE, "max_tokens": 5})


@pytest.mark.xfail(reason="a session brought back from host memory has lost its checkpoints, so a jump back decodes from scratch")
def test_restored_session_keeps_checkpoints(engine, session):
    from test_sessions import hot, writer
    me = writer(engine, session, "lighthouses", paragraphs=12)
    engine("/v1/continuations", me)
    for i in range(80):
        engine("/v1/continuations", writer(engine, f"{session}-o{i}", f"subject {i}"))
        if not hot(engine, session): break
    engine("/v1/continuations", me)
    back = engine("/v1/continuations", {**me, "prompt": me["prompt"][: len(me["prompt"]) // 2]})
    assert back["stats"]["reused_tokens"] > 0
