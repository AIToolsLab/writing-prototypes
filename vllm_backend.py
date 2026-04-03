"""
vLLM Backend Server

A FastAPI server that provides the same API as custom_llm.py but uses a
vLLM server for inference instead of loading the model directly with
HuggingFace Transformers.

Only the tokenizer is loaded locally (lightweight, no GPU needed).
All model inference is delegated to the vLLM server via its OpenAI-compatible
REST API.

Usage:
    # 1. Start the vLLM server (separate process):
    #    Note: --no-enable-prefix-caching is required for prompt_logprobs
    #    support (used by the highlights endpoint). Once vLLM adds support
    #    for prompt_logprobs with prefix caching, this flag can be removed.
    vllm serve google/gemma-2-9b-it --no-enable-prefix-caching

    # 2. Start this backend:
    python vllm_backend.py
    python vllm_backend.py --vllm-base-url http://localhost:8000 --model google/gemma-2-9b-it --port 19570
"""

import argparse
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer


# --- Configuration ---

parser = argparse.ArgumentParser(description="vLLM-backed inference server")
parser.add_argument(
    "--vllm-base-url",
    default=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
    help="Base URL of the vLLM server",
)
parser.add_argument(
    "--model",
    default=os.getenv("MODEL_NAME", "google/gemma-2-9b-it"),
    help="Model name (must match what the vLLM server is serving)",
)
parser.add_argument(
    "--port",
    type=int,
    default=int(os.getenv("PORT", "19570")),
    help="Port for this backend server",
)
args = parser.parse_args()

VLLM_BASE_URL = args.vllm_base_url
MODEL_NAME = args.model
PORT = args.port

VLLM_TIMEOUT = 120.0  # seconds


# --- Global state ---

state = {}


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load only the tokenizer locally (lightweight, no GPU needed).
    # We need it for chat template formatting and token boundary detection.
    print(f"Loading tokenizer for {MODEL_NAME}...")
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded.")

    # Verify vLLM server is reachable
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")
            response.raise_for_status()
            models = response.json()
            model_ids = [m["id"] for m in models["data"]]
            print(f"Connected to vLLM server at {VLLM_BASE_URL}")
            print(f"Available models: {model_ids}")
            if MODEL_NAME not in model_ids:
                print(f"Warning: {MODEL_NAME} not found in vLLM server models.")
        except Exception as e:
            print(f"Warning: Could not connect to vLLM server at {VLLM_BASE_URL}: {e}")
            print("The server will start but requests will fail until vLLM is available.")

    yield
    state.clear()


# --- App setup ---

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/response models ---

class Message(BaseModel):
    role: str
    content: str


class ContinueMessagesRequest(BaseModel):
    messages: List[Message]
    n_branch_tokens: int = 5
    n_future_tokens: int = 5


# --- vLLM API helpers ---

async def vllm_chat_completion(
    messages: list[dict],
    *,
    max_tokens: int = 1,
    logprobs: bool = True,
    top_logprobs: int = 5,
    temperature: float = 0,
    continue_final_message: bool = False,
) -> dict:
    """Call vLLM's /v1/chat/completions endpoint."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    if continue_final_message:
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False

    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
        response = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions", json=payload
        )
        if response.status_code != 200:
            detail = response.text
            raise HTTPException(
                status_code=502,
                detail=f"vLLM chat completion error: {detail}",
            )
        return response.json()


async def vllm_completion(
    prompt,
    *,
    max_tokens: int = 1,
    temperature: float = 0,
    logprobs: Optional[int] = None,
    prompt_logprobs: Optional[int] = None,
    echo: bool = False,
) -> dict:
    """Call vLLM's /v1/completions endpoint.

    `prompt` can be:
    - str: a single text prompt
    - list[str]: batch of text prompts
    - list[int]: a single prompt as token IDs
    - list[list[int]]: batch of token ID prompts
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if prompt_logprobs is not None:
        payload["prompt_logprobs"] = prompt_logprobs
    if echo:
        payload["echo"] = True

    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
        response = await client.post(
            f"{VLLM_BASE_URL}/v1/completions", json=payload
        )
        if response.status_code != 200:
            detail = response.text
            raise HTTPException(
                status_code=502,
                detail=f"vLLM completion error: {detail}",
            )
        return response.json()


def _get_prefix_token_ids(tokenizer, prompt: str, doc: str) -> list[int]:
    """Get token IDs for the chat prefix (everything before the assistant's response content).

    Uses a workaround for empty assistant content: adds a "." then strips the last token.
    """
    messages = [
        {"role": "user", "content": f"{prompt}\n\n{doc}"},
        {"role": "assistant", "content": "."},
    ]
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, continue_final_message=True
    )
    # Remove the "." token to get just the prefix
    return token_ids[:-1]


# --- Endpoints ---

@app.post("/api/continue_messages")
async def continue_messages(request: ContinueMessagesRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if len(messages) == 0:
        raise HTTPException(
            status_code=400, detail="At least one message must be provided."
        )

    tokenizer = state["tokenizer"]
    k = request.n_branch_tokens

    # Determine whether to continue an existing assistant message or start a new one
    continue_final = messages[-1]["role"] == "assistant"

    # Step 1: Get top-k tokens at the next position
    result = await vllm_chat_completion(
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=k,
        temperature=0,
        continue_final_message=continue_final,
    )

    content_logprobs = result["choices"][0]["logprobs"]["content"]
    if not content_logprobs:
        return {"continuations": []}

    top_lps = content_logprobs[0]["top_logprobs"]
    branch_tokens_text = [lp["token"] for lp in top_lps[:k]]

    # Step 2: For each branch token, get the greedy next token.
    # We use the completions API with token ID arrays so we can batch all k
    # branches into a single request. vLLM's prefix caching (if enabled) will
    # reuse the KV cache for the shared prefix.
    if continue_final:
        base_token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, continue_final_message=True
        )
    else:
        base_token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

    branch_prompts = []
    for token_text in branch_tokens_text:
        branch_ids = tokenizer.encode(token_text, add_special_tokens=False)
        branch_prompts.append(base_token_ids + branch_ids)

    result2 = await vllm_completion(
        prompt=branch_prompts,
        max_tokens=1,
        temperature=0,
    )

    # Sort choices by index to match branch order
    choices = sorted(result2["choices"], key=lambda c: c["index"])

    continuations = []
    for i, choice in enumerate(choices):
        next_token_text = choice["text"]
        full_text = branch_tokens_text[i] + next_token_text
        continuations.append({"doc_text": full_text})

    return {"continuations": continuations}


@app.get("/api/highlights")
async def get_highlights(
    doc: str,
    prompt: Optional[str] = None,
    updated_doc: Optional[str] = "",
    k: Optional[int] = 5,
):
    """Analyze tokens in a document and identify which ones differ from the model's top prediction.

    Uses vLLM's prompt_logprobs to get per-token log probabilities without
    generating any new text.

    Note: Requires the vLLM server to be started with --no-enable-prefix-caching
    since prompt_logprobs is not yet supported with prefix caching in vLLM V1.
    """
    tokenizer = state["tokenizer"]

    if prompt is None:
        prompt = "Rewrite this document to be more concise."
    if updated_doc is None or len(updated_doc.strip()) == 0:
        updated_doc = doc

    # Build the full token sequence: [chat prefix] + [updated_doc tokens]
    messages = [
        {"role": "user", "content": f"{prompt}\n\n{doc}"},
        {"role": "assistant", "content": updated_doc},
    ]
    full_token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, continue_final_message=True
    )

    # Find where updated_doc tokens start
    prefix_token_ids = _get_prefix_token_ids(tokenizer, prompt, doc)
    prefix_len = len(prefix_token_ids)
    updated_doc_token_ids = full_token_ids[prefix_len:]

    # Get prompt logprobs from vLLM.
    # prompt_logprobs=k returns top-k logprobs at each prompt token position,
    # plus the actual token if it's not in the top-k.
    result = await vllm_completion(
        prompt=full_token_ids,
        max_tokens=1,  # Must generate at least 1 token; we ignore it
        prompt_logprobs=k,
    )

    # prompt_logprobs is a top-level field in the response (vLLM extension).
    # Format: list with one entry per prompt token.
    #   - First entry is null (no context for the first token)
    #   - Each subsequent entry: {token_id_str: {"logprob": float, "rank": int, "decoded_token": str}, ...}
    all_prompt_logprobs = result.get("prompt_logprobs", [])

    highlights = []
    length_so_far = 0

    for i, token_id in enumerate(updated_doc_token_ids):
        idx = prefix_len + i  # Position in the full token sequence
        token = tokenizer.decode(token_id)

        if idx < len(all_prompt_logprobs) and all_prompt_logprobs[idx] is not None:
            pos_logprobs = all_prompt_logprobs[idx]

            # Find the actual token's logprob (keyed by token ID as string)
            token_id_str = str(token_id)
            if token_id_str in pos_logprobs:
                token_logprob = pos_logprobs[token_id_str]["logprob"]
                token_loss = -token_logprob
            else:
                # Should not happen since vLLM includes the actual token,
                # but handle gracefully
                token_loss = float("inf")

            # Top-k tokens sorted by rank
            sorted_entries = sorted(
                pos_logprobs.values(), key=lambda x: x.get("rank", 999)
            )
            topk_tokens = [entry["decoded_token"] for entry in sorted_entries[:k]]
            most_likely_token = topk_tokens[0] if topk_tokens else token
        else:
            token_loss = 0.0
            topk_tokens = [token]
            most_likely_token = token

        highlights.append(
            {
                "start": length_so_far,
                "end": length_so_far + len(token),
                "token": token,
                "token_loss": token_loss,
                "most_likely_token": most_likely_token,
                "topk_tokens": topk_tokens,
            }
        )
        length_so_far += len(token)

    return {"highlights": highlights}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=PORT)
