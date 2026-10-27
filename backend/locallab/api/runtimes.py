import asyncio
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from locallab.config import get_settings
from locallab.runtimes.base import RuntimeError_
from locallab.runtimes.vllm_openai import VllmOpenAIAdapter

router = APIRouter(prefix="/runtimes", tags=["runtimes"])


@router.get("/vllm/probe")
async def probe_vllm(url: str):
    """Describe an externally managed vLLM server. Only allowlisted hosts are contacted."""
    host = urlparse(url).hostname
    if host not in get_settings().allowed_runtime_hosts:
        raise HTTPException(422, f"host {host!r} is not in LOCALLAB_ALLOWED_RUNTIME_HOSTS")
    adapter = VllmOpenAIAdapter(url, max_connections=4)
    try:
        info = await asyncio.wait_for(adapter.probe(), timeout=20)
    except RuntimeError_ as e:
        raise HTTPException(502, str(e)) from e
    finally:
        await adapter.aclose()
    return {"runtime": info.runtime, "base_url": info.base_url, "version": info.version,
            "kv_cache_tokens": info.kv_cache_tokens, "models": [m.__dict__ for m in info.models]}
