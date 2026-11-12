"""The vLLM-compatible subset: /tokenize, /detokenize, /v1/completions, as the old page and lm-internals used them."""
import math

import pytest

from conftest import REWRITE

FRANCE = [{"role": "user", "content": "What is the capital of France?"},
          {"role": "assistant", "content": "The capital of France is ___"}]


def test_models(engine):
    assert engine("/v1/models")["data"][0]["id"]


def test_lm_internals_flow(engine):
    """lm-internals: tokenize a chat continuing the final message, then echo + logprobs over those tokens."""
    t = engine("/tokenize", {"model": "x", "messages": FRANCE, "continue_final_message": True,
                             "add_generation_prompt": False, "return_token_strs": True})
    assert t["count"] == len(t["tokens"]) == len(t["token_strs"])
    assert t["token_strs"][-2:] == ["Ġis", "Ġ___"]            # Hugging Face's byte-level spelling
    r = engine("/v1/completions", {"model": "x", "prompt": t["tokens"], "max_tokens": 2, "logprobs": 10, "echo": True})
    lp = r["choices"][0]["logprobs"]
    assert len(lp["tokens"]) == len(t["tokens"]) + 2
    assert lp["token_logprobs"][0] is None and lp["top_logprobs"][0] is None
    after_is = lp["top_logprobs"][len(t["tokens"]) - 2]
    assert 10 <= len(after_is) <= 11 and all(v <= 0 for v in after_is.values())


def test_old_page_chat_prefix(engine):
    """The old page rendered prompts with /tokenize then /detokenize; that matches /v1/template."""
    t = engine("/tokenize", {"messages": REWRITE, "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})
    assert engine("/detokenize", {"tokens": t["tokens"]})["prompt"] == engine("/v1/template", {"messages": REWRITE})["prompt"]


def test_tokenize_round_trip(engine):
    text = "Use deep learning — carefully.<|im_end|>"
    t = engine("/tokenize", {"prompt": text})
    assert engine("/detokenize", {"tokens": t["tokens"]})["prompt"] == text
    assert len(engine("/tokenize", {"prompt": "<|im_end|>", "add_special_tokens": False})["tokens"]) == 1


def test_prompt_logprobs(engine):
    """The old highlights request: prompt_logprobs keyed by token id, with the actual token, its rank, and the top k."""
    r = engine("/v1/completions", {"prompt": "The capital of France is Paris", "max_tokens": 1, "prompt_logprobs": 5, "return_token_ids": True})
    c = r["choices"][0]
    ids, plp = c["prompt_token_ids"], c["prompt_logprobs"]
    assert plp[0] is None and len(plp) == len(ids)
    for i in range(1, len(ids)):
        d = plp[i]
        assert str(ids[i]) in d and 5 <= len(d) <= 6
        ranks = sorted(v["rank"] for v in d.values())
        assert ranks[:5] == [1, 2, 3, 4, 5]
        assert d[str(ids[i])]["rank"] >= 1


def test_prompt_logprobs_agree_with_score(engine, session):
    prompt, text = "The capital of France is", " Paris, and the capital of Spain is Madrid."
    s = engine("/v1/score", {"session": session, "prompt": prompt, "text": text})["tokens"]
    r = engine("/v1/completions", {"prompt": prompt + text, "max_tokens": 1, "prompt_logprobs": 0, "return_token_ids": True})["choices"][0]
    tail = r["prompt_logprobs"][-len(s):]
    for item, d in zip(s, tail):
        (v,) = d.values()
        assert v["decoded_token"] == item["text"] and v["logprob"] == pytest.approx(item["logprob"], abs=0.3)


def test_greedy_generation(engine):
    body = {"prompt": "The capital of France is", "max_tokens": 6, "temperature": 0, "logprobs": 5, "return_token_ids": True}
    a, b = engine("/v1/completions", body)["choices"][0], engine("/v1/completions", body)["choices"][0]
    assert a["text"] == b["text"] and a["text"]
    for tok, lp, top in zip(a["logprobs"]["tokens"], a["logprobs"]["token_logprobs"], a["logprobs"]["top_logprobs"]):
        assert lp == pytest.approx(max(top.values())), tok      # greedy picks the likeliest
    assert a["finish_reason"] in ("length", "stop")
    assert r_ok(a)


def r_ok(choice):
    return len(choice["token_ids"]) == len(choice["logprobs"]["tokens"])


def test_generated_logprobs_match_rescoring(engine):
    """The logprob reported for a generated token is the same as scoring it as part of a prompt."""
    a = engine("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 5, "temperature": 0, "logprobs": 0, "return_token_ids": True})["choices"][0]
    r = engine("/v1/completions", {"prompt": a["prompt_token_ids"] + a["token_ids"], "max_tokens": 1, "prompt_logprobs": 0})["choices"][0]
    rescored = [next(iter(d.values()))["logprob"] for d in r["prompt_logprobs"][-len(a["token_ids"]):]]
    assert rescored == pytest.approx(a["logprobs"]["token_logprobs"], abs=0.3)


def test_stop_strings(engine):
    c = engine("/v1/completions", {"prompt": "Count: 1, 2, 3, 4, 5,", "max_tokens": 20, "temperature": 0, "stop": [" 8"]})["choices"][0]
    assert c["finish_reason"] == "stop" and c["stop_reason"] == " 8" and "8" not in c["text"] and "7" in c["text"]
    c = engine("/v1/completions", {"prompt": "Count: 1, 2, 3, 4, 5,", "max_tokens": 20, "temperature": 0, "stop": " 8", "include_stop_str_in_output": True})["choices"][0]
    assert c["text"].endswith(" 8")


def test_batched_prompts_and_samples(engine):
    body = {"prompt": ["Once upon a time", [785, 6722]], "max_tokens": 4, "n": 2, "temperature": 1.0, "seed": 7}
    a = engine("/v1/completions", body)
    assert [c["index"] for c in a["choices"]] == [0, 1, 2, 3]
    assert a["usage"]["completion_tokens"] == sum(1 for _ in a["choices"]) * 4 or a["usage"]["completion_tokens"] <= 16
    b = engine("/v1/completions", body)
    assert [c["text"] for c in a["choices"]] == [c["text"] for c in b["choices"]]   # seeded


def test_logit_bias_changes_the_choice_but_not_the_reported_logprob(engine):
    prompt = "The capital of France is"
    plain = engine("/v1/completions", {"prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 5, "return_token_ids": True})["choices"][0]
    banned = plain["token_ids"][0]
    biased = engine("/v1/completions", {"prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 5, "return_token_ids": True,
                                        "logit_bias": {str(banned): -100}})["choices"][0]
    assert biased["token_ids"][0] != banned
    second = sorted(plain["logprobs"]["top_logprobs"][0].values(), reverse=True)[1]
    assert biased["logprobs"]["token_logprobs"][0] == pytest.approx(second)


def test_echo_without_generation(engine):
    c = engine("/v1/completions", {"prompt": "Hello there", "max_tokens": 0, "echo": True, "logprobs": 1})["choices"][0]
    assert c["text"] == "Hello there" and len(c["logprobs"]["tokens"]) == 2


@pytest.mark.parametrize("extra", [{"structured_outputs": {"regex": "a+"}}, {"stream": True}, {"response_format": {"type": "json_object"}},
                                   {"presence_penalty": 0.5}, {"logprobs": 50}])
def test_unsupported_options_are_refused(engine, extra):
    assert engine.raw("/v1/completions", {"prompt": "Hi", "max_tokens": 1, **extra}).status_code == 400
