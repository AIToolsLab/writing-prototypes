"""/v1/score: per-token logprobs of `text` after the prompt (highlights)."""
import pytest

from conftest import DOC, REWRITE


def test_pieces_concatenate_to_text(engine, session, rewrite_prompt):
    d = engine("/v1/score", {"session": session, "prompt": rewrite_prompt, "text": DOC})
    assert "".join(t["text"] for t in d["tokens"]) == DOC
    for t in d["tokens"]:
        assert t["logprob"] <= t["top_logprob"] + 1e-4


@pytest.mark.parametrize("prompt,text", [("Hello wor", "ld, how are you?"), ("<|im_start|>user\n", "\n\nHi"), ("A", "")])
def test_text_that_merges_with_the_prompt(engine, session, prompt, text):
    """When the text's first characters share a token with the prompt's end, the pieces still add up to the text."""
    d = engine("/v1/score", {"session": session, "prompt": prompt, "text": text})
    assert "".join(t["text"] for t in d["tokens"]) == text


def test_alternatives(engine, session, rewrite_prompt):
    d = engine("/v1/score", {"session": session, "prompt": rewrite_prompt, "text": DOC, "top_k": 5})
    for t in d["tokens"]:
        alts = t["alternatives"]
        assert len(alts) == 5 and alts[0]["text"] == t["top"] and alts[0]["logprob"] == pytest.approx(t["top_logprob"])
        assert [a["logprob"] for a in alts] == sorted((a["logprob"] for a in alts), reverse=True)


def test_rescoring_after_an_edit_matches(engine, session, rewrite_prompt):
    a = engine("/v1/score", {"session": session, "prompt": rewrite_prompt, "text": DOC})
    engine("/v1/score", {"session": session, "prompt": rewrite_prompt, "text": DOC.replace("pretty", "very")})
    b = engine("/v1/score", {"session": session, "prompt": rewrite_prompt, "text": DOC})
    for x, y in zip(a["tokens"], b["tokens"]):
        assert x["logprob"] == pytest.approx(y["logprob"], abs=0.3)


def test_messages_instead_of_prompt(engine, session, rewrite_prompt):
    a = engine("/v1/score", {"session": session, "prompt": rewrite_prompt, "text": DOC})
    b = engine("/v1/score", {"session": session, "messages": REWRITE, "text": DOC})
    assert a["tokens"] == b["tokens"]
