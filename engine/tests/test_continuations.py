"""/v1/continuations: probabilities given the characters typed, not the tokens."""
import json, math, re
from pathlib import Path

import pytest

from conftest import REWRITE, WORD
from helpers import starting_with, units

CASES = json.loads((Path(__file__).parents[2] / "experiments/token-healing/cases.json").read_text())


def predict(engine, session, text, **kw):
    return engine("/v1/continuations", {"session": session, "messages": REWRITE, "text": text, "stop": WORD, **kw})


@pytest.mark.model
@pytest.mark.parametrize("case", CASES, ids=[c["draft"] or "(empty)" for c in CASES])
def test_token_healing_cases(engine, session, case):
    """The likeliest next word continues a partly typed word the way a writer would (e.g. "Use de" → "ep")."""
    d = predict(engine, session, case["draft"], units=5)
    top = next(iter(units(d)), "")
    assert top.strip() and top.startswith(case["starts"]), f"want {case['want']}; got {list(units(d).items())[:5]}"


@pytest.mark.parametrize("draft", ["", "Use de", "Deep learning is pow", "Deep learning is a powerful technology"])
def test_probabilities_are_disjoint(engine, session, draft):
    """Continuations are disjoint events given the typed text, so their probabilities sum to at most 1."""
    d = predict(engine, session, draft)
    probs = [c["prob"] for c in d["continuations"] if "replace" not in c]
    assert probs and all(0 < p <= 1 + 1e-6 for p in probs)
    assert sum(probs) <= 1 + 1e-6
    assert d["unresolved"] >= 0     # relative to the typed text's probability found, so it can exceed 1


def test_typed_is_the_unsettled_end_of_the_text(engine, session):
    d = predict(engine, session, "Deep learning is pow")
    assert d["typed"] == " pow"
    assert d["prefix_logprob"] < 0


def test_chain_rule_across_keystrokes(engine, session):
    """Typing more of a word conditions on it consistently. From "…is pow" to "…is power" (same settled point):
    P("er" | " pow") = exp(prefix_logprob(" power") − prefix_logprob(" pow")), the continuations that start with
    "er" add up to at most that, and each such continuation keeps its probability once "er" is typed."""
    a = predict(engine, session, "Deep learning is pow")
    b = predict(engine, session, "Deep learning is power")
    p_er = math.exp(b["prefix_logprob"] - a["prefix_logprob"])
    assert starting_with(a, "er") <= p_er * 1.05
    assert starting_with(a, "er") >= 0.5 * p_er
    after = {c["text"]: c["prob"] for c in b["continuations"]}
    for c in a["continuations"]:
        if c["text"].startswith("er") and c["prob"] > 0.05 and c["text"][2:] in after:
            assert math.isclose(c["prob"] / p_er, after[c["text"][2:]], rel_tol=0.3), c["text"]


def test_same_request_same_answer(engine, session):
    a = predict(engine, session, "Deep learning is")
    b = predict(engine, session, "Deep learning is")
    assert b["stats"]["decoded_tokens"] == 0
    assert [(c["text"], round(c["prob"], 6)) for c in a["continuations"]] == [(c["text"], round(c["prob"], 6)) for c in b["continuations"]]


def test_stop_and_limits(engine, session):
    d = predict(engine, session, "Deep learning is", max_chars=6)
    for c in d["continuations"]:
        if c["done"] and not c["end"]:
            assert re.search(WORD, c["text"]) or len(c["text"]) >= 6, c
    d = engine("/v1/continuations", {"session": session, "messages": REWRITE, "text": "Deep", "max_depth": 1})
    assert all(len(c["text"]) < 40 for c in d["continuations"])


def test_slow_prefill_still_offers_words(engine, session):
    """The time budget includes the prefill; when that alone uses it up, there's still one round of search."""
    d = predict(engine, session, "Deep learning is a powerful", time_ms=1)
    assert any(c["text"].strip() for c in d["continuations"]), d["continuations"]


def test_units_asks_for_enough_distinct_words(engine, session):
    d = predict(engine, session, "Deep learning is", units=5)
    assert len(units(d)) >= 5


def test_end_of_turn_is_marked(engine, session):
    """After a finished rewrite, ending the turn is likely; it's a continuation with end: true (empty text means
    "stop here"; text before the end is still text)."""
    d = predict(engine, session, "Deep learning is useful, but use it with care and respect for people.", units=5)
    ends = [c for c in d["continuations"] if c["end"]]
    assert ends and all(c["done"] for c in ends)


def test_corrections(engine, session):
    """A correction replaces the last `replace` bytes of the text; its prob is a ratio to the typed text's."""
    text = "Deep learning is pwo"
    d = predict(engine, session, text, corrections=0.5)
    fixes = [c for c in d["continuations"] if "replace" in c]
    for c in fixes:
        assert 0 < c["replace"] <= len(" pwo".encode())
        assert c["prob"] >= 0.5
    assert any((text.encode()[:-c["replace"]].decode() + c["text"]).startswith("Deep learning is pow") for c in fixes), fixes


def test_bad_stop_regex(engine, session):
    r = engine.raw("/v1/continuations", {"session": session, "prompt": "x", "text": "y", "stop": "("})
    assert r.status_code == 400
