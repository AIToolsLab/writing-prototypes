"""Fixtures for testing the engine over HTTP.

Point the tests at a running engine with ENGINE_URL, or have them start one with ENGINE_MODEL (a GGUF path; it runs
build/writing-engine on CPU unless ENGINE_GPU=1). Qwen3.5-2B (unsloth/Qwen3.5-2B-GGUF, Qwen3.5-2B-Q4_K_M.gguf) has
the same architecture and tokenizer as the production model and runs everything here in a minute or two on CPU.
"""
import os, socket, subprocess, time, uuid
from pathlib import Path

import httpx
import pytest

ENGINE = Path(__file__).resolve().parent.parent


class Client:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=300)

    def raw(self, path, body=None):
        if body is None: return self.http.get(self.url + path)
        return self.http.post(self.url + path, json=body)

    def __call__(self, path, body=None):
        r = self.raw(path, body)
        assert r.status_code == 200, f"{path}: HTTP {r.status_code} {r.text[:300]}"
        return r.json()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def engine():
    url, model = os.environ.get("ENGINE_URL"), os.environ.get("ENGINE_MODEL")
    if url:
        yield Client(url)
        return
    if not model:
        pytest.skip("set ENGINE_URL to a running engine, or ENGINE_MODEL to a GGUF file to start one")
    port = _free_port()
    env = dict(os.environ)
    args = [str(ENGINE / "build" / "writing-engine"), "--model", model, "--port", str(port), "--ctx", "8192", "--seqs", "16"]
    if os.environ.get("ENGINE_GPU") != "1":
        env["CUDA_VISIBLE_DEVICES"] = ""
        args += ["--gpu-layers", "0"]
    log = open(ENGINE / "build" / "pytest-engine.log", "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    client = Client(f"http://127.0.0.1:{port}")
    try:
        for _ in range(300):
            if proc.poll() is not None: pytest.fail(f"engine exited; see {log.name}")
            try:
                if client.raw("/health").status_code == 200: break
            except httpx.HTTPError: pass
            time.sleep(0.5)
        yield client
    finally:
        proc.terminate(); proc.wait(30)


@pytest.fixture
def session():
    """A fresh session id, so tests don't share caches."""
    return "pytest-" + uuid.uuid4().hex[:8]


DOC = ("Deep learning neural network technology advances are pretty cool if you are careful "
       "to use it in ways that don't take stuff from people.")
REWRITE = [{"role": "user", "content": "Rewrite this document to be clear and concise.\n\n" + DOC}]
WORD = r"^\s*\S+(?=\s)"


@pytest.fixture(scope="session")
def rewrite_prompt(engine):
    return engine("/v1/template", {"messages": REWRITE})["prompt"]
