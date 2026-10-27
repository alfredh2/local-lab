"""Adapter for an externally managed vLLM OpenAI-compatible server (/v1/completions, streaming)."""
import asyncio
import json
import logging
import re
import time

import httpx

from locallab.runtimes.base import (
    GenerationRequest, ModelInfo, RequestOutcome, RuntimeAdapter, RuntimeError_, ServerInfo,
)

log = logging.getLogger(__name__)
OOM_RE = re.compile(r"out of memory|\bOOM\b|OutOfMemory|CUDA error: out of memory", re.I)


def classify_http_error(status: int, body: str) -> str:
    if OOM_RE.search(body):
        return "oom"
    return "bad_request" if 400 <= status < 500 else "server_error"


def parse_cache_info(metrics_text: str) -> dict[str, str]:
    for line in metrics_text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            return dict(re.findall(r'(\w+)="([^"]*)"', line))
    return {}


class VllmOpenAIAdapter(RuntimeAdapter):
    name = "vllm"

    def __init__(self, base_url: str, max_connections: int = 256) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(None, connect=10.0),  # total deadline enforced per request
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

    async def probe(self) -> ServerInfo:
        try:
            r = await self._client.get("/v1/models", timeout=10.0)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError_(f"Cannot reach an OpenAI-compatible server at {self.base_url}: {e.__class__.__name__} {e}") from e
        models = [ModelInfo(m["id"], m.get("root"), m.get("max_model_len")) for m in r.json().get("data", [])]
        version = None
        try:
            v = await self._client.get("/version", timeout=5.0)
            version = v.json().get("version") if v.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            pass
        info = ServerInfo("vllm", self.base_url, version, models)
        try:
            m = await self._client.get("/metrics", timeout=5.0)
            if m.status_code == 200:
                cache = parse_cache_info(m.text)
                info.extra["cache_config"] = cache
                if cache.get("num_gpu_blocks", "").isdigit() and cache.get("block_size", "").isdigit():
                    info.kv_cache_tokens = int(cache["num_gpu_blocks"]) * int(cache["block_size"])
        except httpx.HTTPError:
            log.info("no /metrics at %s; KV capacity unknown", self.base_url)
        return info

    @staticmethod
    def _payload(req: GenerationRequest) -> dict:
        body: dict = {
            "model": req.model, "max_tokens": req.max_tokens,
            "temperature": req.temperature, "top_p": req.top_p, "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.messages is not None:
            body["messages"] = req.messages
            body["chat_template_kwargs"] = req.chat_template_kwargs or {}
        else:
            body["prompt"] = req.prompt
        if req.seed is not None:
            body["seed"] = req.seed
        if req.top_k is not None:
            body["top_k"] = req.top_k
        if req.force_length and req.messages is None:
            body["min_tokens"] = req.max_tokens
            body["ignore_eos"] = True
        return body

    async def generate(self, req: GenerationRequest) -> RequestOutcome:
        out = RequestOutcome(ok=False, t_start=time.perf_counter(), t_end=0.0)
        try:
            async with asyncio.timeout(req.timeout_s):
                await self._stream(req, out)
        except TimeoutError:
            out.error_type, out.error_message = "timeout", f"no completion within {req.timeout_s:g}s"
        except httpx.ConnectError as e:
            out.error_type, out.error_message = "connection", f"connect failed: {e}"
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as e:
            out.error_type, out.error_message = "connection_lost", f"connection lost mid-request: {e.__class__.__name__} {e}"
        except httpx.HTTPError as e:
            out.error_type, out.error_message = "connection", f"{e.__class__.__name__}: {e}"
        except asyncio.CancelledError:
            raise
        finally:
            out.t_end = time.perf_counter()
        if out.error_type is None and not out.ok:
            out.error_type, out.error_message = "incomplete_stream", "stream ended without a finish_reason"
        if out.error_message:
            out.error_message = out.error_message[:500]
        return out

    async def _stream(self, req: GenerationRequest, out: RequestOutcome) -> None:
        chat = req.messages is not None
        path = "/v1/chat/completions" if chat else "/v1/completions"
        async with self._client.stream("POST", path, json=self._payload(req)) as resp:
            out.http_status = resp.status_code
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                out.error_type = classify_http_error(resp.status_code, body)
                out.error_message = f"HTTP {resp.status_code}: {body}"
                return
            usage = None
            chunks = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    out.error_type, out.error_message = "incomplete_stream", f"unparseable stream chunk: {data[:100]}"
                    return
                if "error" in obj:
                    msg = json.dumps(obj["error"])
                    out.error_type = classify_http_error(500, msg)
                    out.error_message = f"stream error: {msg}"
                    return
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    now = time.perf_counter()
                    piece = ch.get("text") if not chat else (ch.get("delta") or {}).get("content")
                    if piece:
                        out.text += piece
                    if not chat or piece:  # a chat stream opens with a role-only delta that is not a generated token
                        chunks += 1
                        if out.t_first_token is None:
                            out.t_first_token = now
                        out.chunk_times.append(now)
                    if ch.get("finish_reason"):
                        out.finish_reason = ch["finish_reason"]
            if out.finish_reason is None:
                return  # caller reports incomplete_stream
            if usage:
                out.prompt_tokens = usage.get("prompt_tokens")
                out.completion_tokens = usage.get("completion_tokens")
                out.token_source = "server"
            else:
                out.completion_tokens, out.token_source = chunks, "estimated"
            out.ok = True

    async def count_tokens(self, model: str, text: str) -> int | None:
        """Token count by the server's own tokenizer (None if the server has no /tokenize)."""
        try:
            r = await self._client.post("/tokenize", json={"model": model, "prompt": text}, timeout=60.0)
            return int(r.json()["count"]) if r.status_code == 200 else None
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
