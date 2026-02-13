"""Tests for the vLLM backend.

These tests mock both the vLLM server HTTP responses and the tokenizer,
so they can run without a live vLLM instance or HuggingFace authentication.
They verify that the backend correctly translates between its own API format
and vLLM's OpenAI-compatible API.
"""

import pytest
import httpx
import json
from unittest.mock import patch, MagicMock

# Patch argparse before importing vllm_backend so it doesn't try to parse CLI args
import sys
sys.argv = ["test"]


# --- Mock tokenizer ---

class MockTokenizer:
    """A minimal tokenizer mock that supports the methods used by vllm_backend."""

    def __init__(self):
        # Simple vocabulary for testing
        self._vocab = {
            "<bos>": 1, "<start>": 2, "<end>": 3, "<model>": 4,
            "Hello": 100, " world": 101, ".": 102,
            "The": 200, " quick": 201, " fox": 202,
            "A": 210, " fast": 211,
            " How": 300, " I": 301, " What": 302,
            " are": 310, "'m": 311, " is": 312,
            " Sure": 320, " Of": 321,
            "Rewrite": 400, " this": 401,
            "Test": 500, "Edit": 501,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def apply_chat_template(self, messages, *, tokenize=True, continue_final_message=False, add_generation_prompt=False):
        """Return a deterministic list of token IDs for the messages."""
        # Build a simple token sequence:
        # [1(bos), 2(start)] + user_content_tokens + [3(end), 4(model)] + assistant_content_tokens
        ids = [1, 2]  # bos, start_of_turn
        for msg in messages:
            content = msg["content"]
            # Tokenize content character by character into fake IDs (10000+ord)
            for ch in content:
                ids.append(10000 + ord(ch))
            if msg["role"] == "user":
                ids.append(3)  # end_of_turn
                ids.append(4)  # start model turn
        return ids

    def encode(self, text, add_special_tokens=False):
        """Encode text into token IDs."""
        # Use simple character-level encoding for predictability
        if text in self._vocab:
            return [self._vocab[text]]
        return [10000 + ord(ch) for ch in text]

    def decode(self, token_id):
        """Decode a single token ID back to text."""
        if isinstance(token_id, list):
            return "".join(self.decode(t) for t in token_id)
        if token_id in self._id_to_token:
            return self._id_to_token[token_id]
        if token_id >= 10000:
            return chr(token_id - 10000)
        return f"<unk:{token_id}>"

    def batch_decode(self, token_ids_list, **kwargs):
        return [self.decode(ids) for ids in token_ids_list]


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def app(tokenizer):
    """Create the FastAPI app with a pre-loaded mock tokenizer."""
    import vllm_backend
    vllm_backend.state["tokenizer"] = tokenizer
    return vllm_backend.app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


# --- Response builders ---

def _make_chat_completion_response(top_logprobs_data):
    """Helper to build a vLLM /v1/chat/completions response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "google/gemma-2-9b-it",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": top_logprobs_data[0]["token"] if top_logprobs_data else "",
                },
                "logprobs": {
                    "content": [
                        {
                            "token": top_logprobs_data[0]["token"],
                            "logprob": top_logprobs_data[0]["logprob"],
                            "bytes": list(top_logprobs_data[0]["token"].encode("utf-8")),
                            "top_logprobs": top_logprobs_data,
                        }
                    ]
                    if top_logprobs_data
                    else [],
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51},
    }


def _make_completion_response(texts, indices=None):
    """Helper to build a vLLM /v1/completions response."""
    if indices is None:
        indices = list(range(len(texts)))
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1234567890,
        "model": "google/gemma-2-9b-it",
        "choices": [
            {
                "index": idx,
                "text": text,
                "logprobs": None,
                "finish_reason": "length",
            }
            for idx, text in zip(indices, texts)
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": len(texts), "total_tokens": 50 + len(texts)},
    }


def _make_completion_response_with_prompt_logprobs(prompt_logprobs_list):
    """Helper to build a /v1/completions response with prompt_logprobs."""
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1234567890,
        "model": "google/gemma-2-9b-it",
        "choices": [
            {
                "index": 0,
                "text": ".",
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
        "prompt_logprobs": prompt_logprobs_list,
    }


# --- Mock HTTP client ---

class MockResponse:
    """Minimal mock for httpx.Response."""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)

    def json(self):
        return self._json_data


class MockAsyncClient:
    """Mock httpx.AsyncClient that records requests and returns canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, *, json=None, **kwargs):
        self.requests.append({"url": url, "json": json})
        if self.responses:
            return self.responses.pop(0)
        return MockResponse({"error": "no more mock responses"}, status_code=500)


# --- continue_messages tests ---

def test_continue_messages_response_format(client, tokenizer):
    """Test that continue_messages returns the expected response format."""
    top_logprobs = [
        {"token": " How", "logprob": -0.5, "bytes": [32, 72, 111, 119]},
        {"token": " I", "logprob": -1.0, "bytes": [32, 73]},
        {"token": " What", "logprob": -1.5, "bytes": [32, 87, 104, 97, 116]},
    ]
    chat_response = MockResponse(_make_chat_completion_response(top_logprobs))
    completion_response = MockResponse(
        _make_completion_response([" are", "'m", " is"], indices=[0, 1, 2])
    )

    mock_client = MockAsyncClient([chat_response, completion_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.post(
            "/api/continue_messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": ""},
                ],
                "n_branch_tokens": 3,
                "n_future_tokens": 2,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "continuations" in data
    assert len(data["continuations"]) == 3
    # Each continuation should combine the branch token + next token
    assert data["continuations"][0]["doc_text"] == " How are"
    assert data["continuations"][1]["doc_text"] == " I'm"
    assert data["continuations"][2]["doc_text"] == " What is"


def test_continue_messages_makes_two_vllm_calls(client, tokenizer):
    """Test that continue_messages makes exactly 2 calls to vLLM."""
    top_logprobs = [
        {"token": " Sure", "logprob": -0.3, "bytes": [32, 83, 117, 114, 101]},
        {"token": " Of", "logprob": -0.8, "bytes": [32, 79, 102]},
    ]
    chat_response = MockResponse(_make_chat_completion_response(top_logprobs))
    completion_response = MockResponse(
        _make_completion_response(["!", " course"], indices=[0, 1])
    )

    mock_client = MockAsyncClient([chat_response, completion_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.post(
            "/api/continue_messages",
            json={
                "messages": [
                    {"role": "user", "content": "Can you help?"},
                    {"role": "assistant", "content": ""},
                ],
                "n_branch_tokens": 2,
            },
        )

    assert response.status_code == 200
    # Should have made exactly 2 requests
    assert len(mock_client.requests) == 2
    # First: chat completions (for top-k branch tokens)
    assert "/v1/chat/completions" in mock_client.requests[0]["url"]
    # Second: completions (for next token of each branch)
    assert "/v1/completions" in mock_client.requests[1]["url"]


def test_continue_messages_second_call_batches_branches(client, tokenizer):
    """Test that the completions call sends k prompts as a batch of token ID arrays."""
    k = 3
    top_logprobs = [
        {"token": "A", "logprob": -0.5, "bytes": [65]},
        {"token": "B", "logprob": -1.0, "bytes": [66]},
        {"token": "C", "logprob": -1.5, "bytes": [67]},
    ]
    chat_response = MockResponse(_make_chat_completion_response(top_logprobs))
    completion_response = MockResponse(
        _make_completion_response(["a", "b", "c"], indices=[0, 1, 2])
    )

    mock_client = MockAsyncClient([chat_response, completion_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.post(
            "/api/continue_messages",
            json={
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": ""},
                ],
                "n_branch_tokens": k,
            },
        )

    assert response.status_code == 200
    # The completions call should have k prompts (token ID arrays)
    completions_payload = mock_client.requests[1]["json"]
    assert isinstance(completions_payload["prompt"], list)
    assert len(completions_payload["prompt"]) == k
    # Each prompt should be a list of token IDs (ints)
    for prompt in completions_payload["prompt"]:
        assert isinstance(prompt, list)
        assert all(isinstance(tid, int) for tid in prompt)


def test_continue_messages_empty_messages_returns_400(client):
    """Test that empty messages list returns a 400 error."""
    response = client.post(
        "/api/continue_messages",
        json={"messages": [], "n_branch_tokens": 5},
    )
    assert response.status_code == 400


def test_continue_messages_chat_completion_params(client, tokenizer):
    """Test that the chat completion call uses correct parameters for assistant continuation."""
    top_logprobs = [
        {"token": " Hi", "logprob": -0.5, "bytes": [32, 72, 105]},
    ]
    chat_response = MockResponse(_make_chat_completion_response(top_logprobs))
    completion_response = MockResponse(_make_completion_response(["!"]))

    mock_client = MockAsyncClient([chat_response, completion_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.post(
            "/api/continue_messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                "n_branch_tokens": 1,
            },
        )

    assert response.status_code == 200
    chat_payload = mock_client.requests[0]["json"]
    assert chat_payload["max_tokens"] == 1
    assert chat_payload["logprobs"] is True
    assert chat_payload["top_logprobs"] == 1
    assert chat_payload["temperature"] == 0
    # Last message is assistant, so continue_final_message should be set
    assert chat_payload["continue_final_message"] is True
    assert chat_payload["add_generation_prompt"] is False


def test_continue_messages_user_last_message(client, tokenizer):
    """Test that when the last message is from the user, add_generation_prompt is used."""
    top_logprobs = [
        {"token": " Sure", "logprob": -0.5, "bytes": [32, 83, 117, 114, 101]},
    ]
    chat_response = MockResponse(_make_chat_completion_response(top_logprobs))
    completion_response = MockResponse(_make_completion_response(["!"]))

    mock_client = MockAsyncClient([chat_response, completion_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.post(
            "/api/continue_messages",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "n_branch_tokens": 1,
            },
        )

    assert response.status_code == 200
    chat_payload = mock_client.requests[0]["json"]
    # Last message is user, so continue_final_message should NOT be set
    assert "continue_final_message" not in chat_payload
    assert "add_generation_prompt" not in chat_payload


def test_continue_messages_empty_logprobs(client, tokenizer):
    """Test graceful handling when vLLM returns empty logprobs."""
    chat_response = MockResponse({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "google/gemma-2-9b-it",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "logprobs": {"content": []},
            "finish_reason": "length",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    })

    mock_client = MockAsyncClient([chat_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.post(
            "/api/continue_messages",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "n_branch_tokens": 3,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"continuations": []}


# --- highlights tests ---

def test_highlights_response_format(client, tokenizer):
    """Test that highlights returns correctly structured response."""
    doc = "Hello"
    prompt = "Rewrite"
    updated_doc = "Hi"

    # Get the token IDs to build mock prompt_logprobs
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": updated_doc},
        ],
        tokenize=True,
        continue_final_message=True,
    )
    prefix_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": "."},
        ],
        tokenize=True,
        continue_final_message=True,
    )[:-1]
    prefix_len = len(prefix_ids)
    updated_ids = full_ids[prefix_len:]

    # Build mock prompt_logprobs
    prompt_logprobs_list = [None] * len(full_ids)
    for i, token_id in enumerate(updated_ids):
        idx = prefix_len + i
        decoded = tokenizer.decode(token_id)
        prompt_logprobs_list[idx] = {
            str(token_id): {
                "logprob": -0.5,
                "rank": 1,
                "decoded_token": decoded,
            },
            "99999": {
                "logprob": -2.0,
                "rank": 2,
                "decoded_token": "[ALT]",
            },
        }

    mock_response = MockResponse(
        _make_completion_response_with_prompt_logprobs(prompt_logprobs_list)
    )
    mock_client = MockAsyncClient([mock_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.get(
            "/api/highlights",
            params={"doc": doc, "prompt": prompt, "updated_doc": updated_doc, "k": 5},
        )

    assert response.status_code == 200
    data = response.json()
    assert "highlights" in data
    highlights = data["highlights"]
    assert len(highlights) == len(updated_ids)

    for h in highlights:
        assert "start" in h
        assert "end" in h
        assert "token" in h
        assert "token_loss" in h
        assert "most_likely_token" in h
        assert "topk_tokens" in h
        assert h["start"] >= 0
        assert h["end"] >= h["start"]
        assert isinstance(h["token_loss"], float)
        assert h["token_loss"] >= 0

    # Verify character offsets are contiguous
    reconstructed = "".join(h["token"] for h in highlights)
    assert reconstructed == updated_doc

    for i, h in enumerate(highlights):
        if i == 0:
            assert h["start"] == 0
        else:
            assert h["start"] == highlights[i - 1]["end"]


def test_highlights_defaults_to_doc_when_updated_doc_empty(client, tokenizer):
    """Test that highlights uses doc as updated_doc when updated_doc is empty."""
    doc = "Hi"
    prompt = "Edit"

    # When updated_doc is empty, the code uses doc instead
    messages = [
        {"role": "user", "content": f"{prompt}\n\n{doc}"},
        {"role": "assistant", "content": doc},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, continue_final_message=True
    )
    prefix_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": "."},
        ],
        tokenize=True,
        continue_final_message=True,
    )[:-1]
    prefix_len = len(prefix_ids)

    prompt_logprobs_list = [None] * len(full_ids)
    for idx in range(prefix_len, len(full_ids)):
        tid = full_ids[idx]
        prompt_logprobs_list[idx] = {
            str(tid): {
                "logprob": -1.0,
                "rank": 1,
                "decoded_token": tokenizer.decode(tid),
            }
        }

    mock_response = MockResponse(
        _make_completion_response_with_prompt_logprobs(prompt_logprobs_list)
    )
    mock_client = MockAsyncClient([mock_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.get(
            "/api/highlights",
            params={"doc": doc, "prompt": prompt, "updated_doc": ""},
        )

    assert response.status_code == 200
    highlights = response.json()["highlights"]
    reconstructed = "".join(h["token"] for h in highlights)
    assert reconstructed == doc


def test_highlights_uses_prompt_logprobs_param(client, tokenizer):
    """Test that the completions call passes prompt_logprobs=k."""
    doc = "Hi"
    prompt = "Edit"

    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": doc},
        ],
        tokenize=True,
        continue_final_message=True,
    )
    prefix_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": "."},
        ],
        tokenize=True,
        continue_final_message=True,
    )[:-1]

    prompt_logprobs_list = [None] * len(full_ids)
    for idx in range(len(prefix_ids), len(full_ids)):
        tid = full_ids[idx]
        prompt_logprobs_list[idx] = {
            str(tid): {
                "logprob": -0.1,
                "rank": 1,
                "decoded_token": tokenizer.decode(tid),
            }
        }

    mock_response = MockResponse(
        _make_completion_response_with_prompt_logprobs(prompt_logprobs_list)
    )
    mock_client = MockAsyncClient([mock_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.get(
            "/api/highlights",
            params={"doc": doc, "prompt": prompt, "k": 7},
        )

    assert response.status_code == 200
    payload = mock_client.requests[0]["json"]
    assert payload["prompt_logprobs"] == 7
    assert payload["max_tokens"] == 1


def test_highlights_token_loss_computed_correctly(client, tokenizer):
    """Test that token_loss = -logprob."""
    doc = "A"
    prompt = "Edit"

    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": doc},
        ],
        tokenize=True,
        continue_final_message=True,
    )
    prefix_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": "."},
        ],
        tokenize=True,
        continue_final_message=True,
    )[:-1]
    prefix_len = len(prefix_ids)
    updated_ids = full_ids[prefix_len:]

    test_logprob = -2.345
    prompt_logprobs_list = [None] * len(full_ids)
    for i, tid in enumerate(updated_ids):
        idx = prefix_len + i
        prompt_logprobs_list[idx] = {
            str(tid): {
                "logprob": test_logprob,
                "rank": 1,
                "decoded_token": tokenizer.decode(tid),
            }
        }

    mock_response = MockResponse(
        _make_completion_response_with_prompt_logprobs(prompt_logprobs_list)
    )
    mock_client = MockAsyncClient([mock_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.get(
            "/api/highlights",
            params={"doc": doc, "prompt": prompt, "updated_doc": doc},
        )

    assert response.status_code == 200
    for h in response.json()["highlights"]:
        assert abs(h["token_loss"] - (-test_logprob)) < 1e-6


def test_highlights_sends_token_ids_as_prompt(client, tokenizer):
    """Test that the completions call sends token IDs (not text) as the prompt."""
    doc = "Hi"
    prompt = "Edit"

    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": doc},
        ],
        tokenize=True,
        continue_final_message=True,
    )

    prompt_logprobs_list = [None] * len(full_ids)
    prefix_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": f"{prompt}\n\n{doc}"},
            {"role": "assistant", "content": "."},
        ],
        tokenize=True,
        continue_final_message=True,
    )[:-1]
    for idx in range(len(prefix_ids), len(full_ids)):
        tid = full_ids[idx]
        prompt_logprobs_list[idx] = {
            str(tid): {"logprob": -0.1, "rank": 1, "decoded_token": tokenizer.decode(tid)}
        }

    mock_response = MockResponse(
        _make_completion_response_with_prompt_logprobs(prompt_logprobs_list)
    )
    mock_client = MockAsyncClient([mock_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        response = client.get(
            "/api/highlights",
            params={"doc": doc, "prompt": prompt, "updated_doc": doc},
        )

    assert response.status_code == 200
    payload = mock_client.requests[0]["json"]
    # Prompt should be a list of ints (token IDs)
    assert isinstance(payload["prompt"], list)
    assert all(isinstance(tid, int) for tid in payload["prompt"])
    assert payload["prompt"] == full_ids
