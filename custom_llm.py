import os
os.environ["HF_HOME"] = r"D:\huggingface"

import argparse
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_llm_inference import get_highlights_inner, get_next_token_predictions_inner, continue_messages_inner

ml_models = {}

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="Enable GPU usage")
args = parser.parse_args()

USE_GPU = args.gpu

if not USE_GPU:
    print("Running without GPU. To enable GPU, run with the --gpu flag.")

@asynccontextmanager
async def models_lifespan(app: FastAPI):

    #model_name = 'google/gemma-1.1-7b-it'
    #model_name = 'google/gemma-1.1-2b-it'
    model_name = 'google/gemma-3-1b-it'
    #model_name = 'google/gemma-3-12b-it'
    #model_name = 'google/gemma-3-4b-it'

    if USE_GPU:
        dtype = torch.bfloat16
        from transformers import TorchAoConfig
        quantization_config = None#TorchAoConfig("int4_weight_only", group_size=128)
    else:
        dtype = torch.float16
        quantization_config = None

    ml_models["llm"] = llm = {
        'tokenizer': AutoTokenizer.from_pretrained(model_name),
        'model': AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if USE_GPU else "cpu",
            torch_dtype=dtype,
            attn_implementation='eager',
            quantization_config=quantization_config,
        )
    }
    print("Loaded llm with device map:")
    print(llm['model'].hf_device_map)

    # Print timing info for each endpoint
    print("\nRunning endpoint tests...")

    test_doc = "This is a test document that needs to be revised for clarity and conciseness."
    test_prompt = "Make this more clear and concise."

    client = TestClient(app)

    start = time.time()
    response = client.get("/api/highlights",
        params={"doc": test_doc, "prompt": test_prompt})
    print(f"Highlights endpoint: {time.time() - start:.2f}s")

    start = time.time()
    response = client.get("/api/next_token",
        params={"original_doc": test_doc, "prompt": test_prompt, "doc_in_progress": "This is"})
    print(f"Next token endpoint: {time.time() - start:.2f}s")

    if False:
        start = time.time()
        response = client.get("/api/gen_revisions",
            params={"doc": test_doc, "prompt": test_prompt, "n": 1, "max_length": 16})
        print(f"Gen revisions endpoint: {time.time() - start:.2f}s")

    yield

    # Release resources on exit
    ml_models.clear()

DEBUG = os.getenv("DEBUG") or False
PORT = int(os.getenv("PORT") or "19570")

app = FastAPI(lifespan=models_lifespan)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/highlights")
def get_highlights(doc: str, prompt: Optional[str] = None, updated_doc: Optional[str] = '', k: Optional[int] = 5):
    ''' Example of using this in JavaScript:

    let url = new URL('http://localhost:8000/api/highlights')
    url.searchParams.append('doc', 'This is a test document. It is a test document because it is a test document.')
    url.searchParams.append('prompt', 'Rewrite this document to be more concise.')
    url.searchParams.append('updated_doc', 'This is a test document.')
    let response = await fetch(url)
    '''

    llm = ml_models['llm']
    model = llm['model']
    tokenizer = llm['tokenizer']

    if prompt is None:
        prompt = "Rewrite this document to be more concise."

    highlights = get_highlights_inner(model, tokenizer, doc, prompt, updated_doc, k)

    return {'highlights': highlights}


@app.get('/api/next_token')
def get_next_token_predictions(original_doc: str,
                               prompt: str,
                               doc_in_progress: str,
                               k: Optional[int] = 5):


    model = ml_models['llm']['model']
    tokenizer = ml_models['llm']['tokenizer']

    decoded_next_tokens, next_token_logits = get_next_token_predictions_inner(
        model, tokenizer, original_doc, prompt, doc_in_progress, k)

    return {
        'next_tokens': decoded_next_tokens
    }


@app.get('/api/gen_revisions')
def gen_revisions(
        prompt: str,
        doc: str,
        n: Optional[int] = 5,
        max_length: Optional[int] = 1024,
        ):

    return {
        'ok': False,
        'message': 'This endpoint has been disabled.'
    }
    model = ml_models['llm']['model']
    tokenizer = ml_models['llm']['tokenizer']

    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\n{doc}",
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    generations = model.generate(
        tokenized_chat, num_return_sequences=n,
        max_new_tokens=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.5,
        return_dict_in_generate=True, output_scores=True)
    generated_docs = tokenizer.batch_decode(generations.sequences, skip_special_tokens=True)
    #print(generations.scores)

    # Remove prompt text. see https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/pipelines/text_generation.py#L37
    prompt_length = len(
        tokenizer.decode(
            tokenized_chat[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
    ))

    return {
        'revised_docs': [dict(doc_text=doc[prompt_length:]) for doc in generated_docs]
    }


class Message(BaseModel):
    role: str
    content: str

class ContinueMessagesRequest(BaseModel):
    messages: List[Message]
    n_branch_tokens: int = 5
    n_future_tokens: int = 5


@app.post('/api/continue_messages')
def continue_messages(request: ContinueMessagesRequest):

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if len(messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message must be provided.")
    n_branch_tokens = request.n_branch_tokens
    n_future_tokens = request.n_future_tokens

    model = ml_models['llm']['model']
    tokenizer = ml_models['llm']['tokenizer']

    generated_docs = continue_messages_inner(model, tokenizer, messages, n_branch_tokens, n_future_tokens)

    return {
        'continuations': [dict(doc_text=doc) for doc in generated_docs]
    }


@app.post('/api/logprobs')
def logprobs(request: ContinueMessagesRequest):

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if len(messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message must be provided.")

    model = ml_models['llm']['model']
    tokenizer = ml_models['llm']['tokenizer']

    # Work around a bug when the last message is empty
    trim_last_message = False
    if messages[-1]['content'] == '':
        messages[-1]['content'] = '.'
        trim_last_message = True
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", continue_final_message=True).to(model.device)
    if trim_last_message:
        tokenized_chat = tokenized_chat[:, :-1]


    # Compute all logits
    with torch.no_grad():
        logits = model(tokenized_chat).logits

    k = request.n_branch_tokens

    # Return a list of tokens:
    # {
    #     "token": "the",
    #     "logprobs": [{"the": -0.1, "a": -0.2, ...}]
    # }
    # logprobs are the top-k logprobs for each token, plus the chosen token in case it is not in the top-k
    # The very first token will have no logprobs, since it is the beginning of the document
    # The very last token will have "token" set to None, and "logprobs" will be the logprobs for the next token

    all_logprobs = []
    for idx in range(len(tokenized_chat[0]) + 1):
        if idx == len(tokenized_chat[0]):
            actual_token_id = None
            token = None
        else:
            actual_token_id = tokenized_chat[0, idx].item()
            token = tokenizer.decode(actual_token_id)

        if idx == 0:
            token_logprobs = []
        else:
            logprobs = logits[0, idx - 1].log_softmax(dim=-1)
            token_ids_to_return = logprobs.topk(k).indices.cpu().numpy().tolist()
            if actual_token_id is not None and actual_token_id not in token_ids_to_return:
                token_ids_to_return.append(actual_token_id)
            token_logprobs = {tokenizer.decode(i): logprobs[i].item() for i in token_ids_to_return}
        all_logprobs.append(dict(token=token, logprobs=token_logprobs))

    return {
        'logprobs': all_logprobs
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=PORT)
