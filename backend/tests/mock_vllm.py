"""A small in-process fake of vLLM's OpenAI-compatible server, with failure injection and call tracking."""
import asyncio
import json
import socket
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


@dataclass
class MockState:
    model: str = "mock-model"
    max_model_len: int = 4096
    ttft_s: float = 0.02
    token_s: float = 0.005
    num_gpu_blocks: int = 1000
    block_size: int = 16
    usage: bool = True
    prompt_token_offset: int = 0  # add to reported prompt_tokens to simulate mismatch
    mode: str = "ok"  # ok | http500 | oom | hang | drop | fail_every_other
    fail_first_n: int = 0
    metrics: bool = True
    requests: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    disconnects: int = 0
    completed: int = 0
    bodies: list = field(default_factory=list)


def build_app(st: MockState) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": st.model, "root": "org/Mock-Model", "max_model_len": st.max_model_len}]}

    @app.get("/version")
    def version():
        return {"version": "0.0.mock"}

    @app.get("/metrics")
    def metrics():
        if not st.metrics:
            return PlainTextResponse("nope", status_code=404)
        return PlainTextResponse(
            f'vllm:cache_config_info{{block_size="{st.block_size}",num_gpu_blocks="{st.num_gpu_blocks}",cache_dtype="auto"}} 1.0\n')

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        st.requests += 1
        st.bodies.append(body)
        n = st.requests
        if st.mode == "http500" or n <= st.fail_first_n or (st.mode == "fail_every_other" and n % 2 == 0):
            return JSONResponse({"error": {"message": "internal boom", "code": 500}}, status_code=500)
        if st.mode == "oom":
            return JSONResponse({"error": {"message": "CUDA out of memory. Tried to allocate 2.00 GiB", "code": 500}}, status_code=500)
        prompt = body["prompt"]
        n_prompt = len(prompt) if isinstance(prompt, list) else len(prompt.split())
        n_out = body["max_tokens"]

        async def gen():
            st.in_flight += 1
            st.max_in_flight = max(st.max_in_flight, st.in_flight)
            try:
                if st.mode == "hang":
                    await asyncio.sleep(60)
                await asyncio.sleep(st.ttft_s)
                for i in range(n_out):
                    if await request.is_disconnected():
                        st.disconnects += 1
                        return
                    last = i == n_out - 1
                    chunk = {"choices": [{"index": 0, "text": "x", "finish_reason": "length" if last else None}], "usage": None}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if st.mode == "drop" and i == 2:
                        raise ConnectionResetError("drop")
                    await asyncio.sleep(st.token_s)
                if st.usage:
                    u = {"prompt_tokens": n_prompt + st.prompt_token_offset, "completion_tokens": n_out,
                         "total_tokens": n_prompt + n_out}
                    yield f"data: {json.dumps({'choices': [], 'usage': u})}\n\n"
                yield "data: [DONE]\n\n"
                st.completed += 1
            except asyncio.CancelledError:
                st.disconnects += 1
                raise
            finally:
                st.in_flight -= 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


class MockServer:
    def __init__(self, port: int | None = None, **kw) -> None:
        self.state = MockState(**kw)
        if port is None:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(build_app(self.state), host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "MockServer":
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("mock server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
