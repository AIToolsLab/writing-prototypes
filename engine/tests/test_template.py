"""Chat templates: /v1/template, and `messages` + `role` in place of `prompt`."""
import pytest

from conftest import REWRITE, WORD

PLACEHOLDER = "XYZZYPLACEHOLDER"


def client_side_prefix(engine, history, role):
    """What clients did before the engine took `messages`: render a placeholder message and cut there."""
    if role == "assistant": return engine("/v1/template", {"messages": history})["prompt"]
    p = engine("/v1/template", {"messages": history + [{"role": role, "content": PLACEHOLDER}], "add_generation_prompt": False})["prompt"]
    return p[:p.rindex(PLACEHOLDER)]


def test_generation_prompt_has_thinking_off(engine):
    p = engine("/v1/template", {"messages": REWRITE})["prompt"]
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_messages_match_the_client_side_cut(engine, session, role):
    history = REWRITE + [{"role": "assistant", "content": "Use deep learning carefully."}] if role == "user" else REWRITE
    prompt = client_side_prefix(engine, history, role)
    body = {"session": session, "text": "I think", "stop": WORD}
    a = engine("/v1/continuations", {**body, "prompt": prompt})
    b = engine("/v1/continuations", {**body, "messages": history, "role": role})
    assert a["continuations"] == b["continuations"]


def test_prompt_and_messages_together_is_an_error(engine, session):
    r = engine.raw("/v1/continuations", {"session": session, "prompt": "x", "messages": REWRITE, "text": ""})
    assert r.status_code == 400


def test_content_parts(engine):
    parts = [{"role": "user", "content": [{"type": "text", "text": REWRITE[0]["content"]}]}]
    assert engine("/v1/template", {"messages": parts}) == engine("/v1/template", {"messages": REWRITE})
