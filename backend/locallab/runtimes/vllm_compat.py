"""Validation of a managed vLLM configuration against the model checkpoint, GPU and image.

Only facts that can be established are checked here (checkpoint config, hardware capability,
memory arithmetic, vLLM's own supported-architecture list). Anything else is left to the real
launch, whose logs are classified (see classify_launch_failure) rather than assumed to work.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from locallab.config import Settings
from locallab.runtimes.docker import CUDA_COMPAT_FIX_LD_LIBRARY_PATH, DockerAPI, DockerError

log = logging.getLogger(__name__)
_MB = 1024 * 1024
ACTIVATION_OVERHEAD_MB = 1536  # CUDA context, activations, cudagraphs: rough, labelled as an estimate
CKPT_QUANT = {"awq": "awq", "gptq": "gptq", "fp8": "fp8", "compressed-tensors": "compressed-tensors"}


class ManagedVllmConfig(BaseModel):
    max_model_len: int = Field(8192, ge=256, le=262_144)
    gpu_memory_utilization: float = Field(0.30, ge=0.05, le=0.95)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    quantization: Literal["none", "awq", "gptq", "fp8"] = Field("none", description="none = use the checkpoint as-is")
    kv_cache_dtype: Literal["auto", "fp8"] = "auto"
    max_num_seqs: int = Field(64, ge=1, le=1024)
    enforce_eager: bool = False
    gpu_index: int = Field(0, ge=0, le=15)
    startup_timeout_s: float = Field(600, ge=5, le=3600)
    allow_contention: bool = Field(False, description="Launch even if other work is using the GPU (nothing is stopped)")
    image: str | None = Field(None, description="Docker image; defaults to LOCALLAB_VLLM_IMAGE")

    @field_validator("image")
    @classmethod
    def _image_ok(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"vllm/[A-Za-z0-9._-]+(:[A-Za-z0-9._-]+)?", v):
            raise ValueError("image must look like vllm/<name>[:tag]")
        return v


def checkpoint_info(snapshot: Path) -> dict[str, Any]:
    cfg = json.loads((snapshot / "config.json").read_text())
    text = cfg.get("text_config") or cfg
    q = cfg.get("quantization_config") or text.get("quantization_config") or {}
    layer_types = text.get("layer_types")
    n_layers = text.get("num_hidden_layers")
    full_layers = layer_types.count("full_attention") if layer_types else n_layers
    heads = text.get("num_attention_heads")
    head_dim = text.get("head_dim") or (text["hidden_size"] // heads if heads and text.get("hidden_size") else None)
    weights = sum(p.stat().st_size for p in snapshot.iterdir() if p.suffix == ".safetensors")
    return {
        "architectures": cfg.get("architectures") or [], "quant_method": q.get("quant_method"),
        "max_position_embeddings": text.get("max_position_embeddings"), "kv_layers": full_layers,
        "kv_heads": text.get("num_key_value_heads") or heads, "head_dim": head_dim, "weights_mb": weights / _MB,
        "dtype": text.get("dtype") or text.get("torch_dtype") or cfg.get("torch_dtype"),
    }


def kv_bytes_per_token(ck: dict[str, Any], kv_dtype: str) -> int | None:
    if not (ck["kv_layers"] and ck["kv_heads"] and ck["head_dim"]):
        return None
    return int(2 * ck["kv_layers"] * ck["kv_heads"] * ck["head_dim"] * (1 if kv_dtype == "fp8" else 2))


def validate_runtime(cfg: ManagedVllmConfig, ck: dict[str, Any], capability: tuple[int, int] | None,
                     total_mb: float, supported_archs: list[str] | None) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    cc = capability[0] * 10 + capability[1] if capability else None
    method = ck["quant_method"]
    if supported_archs is not None:
        bad = [a for a in ck["architectures"] if a not in supported_archs]
        if bad or not ck["architectures"]:
            errors.append(f"Architecture {ck['architectures'] or '(none)'} is not supported by this vLLM image.")
    else:
        warnings.append("vLLM's supported-architecture list is not cached yet; it is checked at job start.")
    if cfg.quantization in ("awq", "gptq"):
        if method != cfg.quantization:
            errors.append(f"quantization={cfg.quantization} requires a {cfg.quantization} checkpoint, but this checkpoint's "
                          f"quantization is {method or 'none (unquantized)'}.")
    elif cfg.quantization == "fp8":
        if method and method != "fp8":
            errors.append(f"quantization=fp8 cannot be applied to a {method}-quantized checkpoint.")
        elif method is None:
            warnings.append("Online fp8 quantization of an unquantized checkpoint: speed/quality differ from the bf16 baseline.")
    elif method and method not in CKPT_QUANT.values():
        warnings.append(f"Checkpoint quantization {method!r} is not one LocalLab has validated.")
    if (cfg.quantization == "fp8" or cfg.kv_cache_dtype == "fp8") and cc is not None and cc < 89:
        errors.append(f"fp8 needs compute capability >= 8.9; GPU {cfg.gpu_index} is {capability[0]}.{capability[1]}.")
    if cfg.dtype == "bfloat16" and cc is not None and cc < 80:
        errors.append("bfloat16 needs compute capability >= 8.0.")
    if cfg.dtype == "float32":
        warnings.append("float32 doubles weight memory and is rarely what you want for serving.")
    if cfg.kv_cache_dtype == "fp8":
        warnings.append("fp8 KV cache can change output quality; LocalLab measures speed, not quality.")
    mpe = ck["max_position_embeddings"]
    if mpe and cfg.max_model_len > mpe:
        errors.append(f"max_model_len {cfg.max_model_len} exceeds the model's max_position_embeddings {mpe}.")
    budget = cfg.gpu_memory_utilization * total_mb
    est: dict[str, Any] = {"weights_mb": round(ck["weights_mb"]), "budget_mb": round(budget),
                           "overhead_mb_assumed": ACTIVATION_OVERHEAD_MB, "kind": "estimate (actual KV size is read from the server after launch)"}
    kv_mb = budget - ck["weights_mb"] - ACTIVATION_OVERHEAD_MB
    est["kv_budget_mb"] = round(kv_mb)
    if kv_mb <= 0:
        errors.append(f"Weights ({ck['weights_mb']:.0f} MB) plus ~{ACTIVATION_OVERHEAD_MB} MB overhead do not fit in the "
                      f"{budget:.0f} MB budget (gpu_memory_utilization {cfg.gpu_memory_utilization:g}); no room for KV cache.")
    else:
        bpt = kv_bytes_per_token(ck, cfg.kv_cache_dtype)
        if bpt:
            est["kv_bytes_per_token"] = bpt
            est["kv_tokens"] = int(kv_mb * _MB / bpt)
            if est["kv_tokens"] < cfg.max_model_len:
                errors.append(f"Estimated KV capacity ({est['kv_tokens']} tokens) is below max_model_len {cfg.max_model_len}; "
                              "vLLM will refuse to start. Raise gpu_memory_utilization or lower max_model_len.")
        else:
            warnings.append("Could not estimate KV capacity from the checkpoint config.")
    return errors, warnings, est


def supported_architectures(docker: DockerAPI, image: str, image_id: str, settings: Settings,
                            allow_run: bool) -> list[str] | None:
    """vLLM's own architecture list, cached per image id. Returns None if unknown and allow_run is False."""
    cache = settings.data_dir / f"vllm_archs_{image_id.split(':')[-1][:16]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not allow_run:
        return None
    out = docker.run_oneoff(
        image, "python3",
        ["-c", "import json; from vllm import ModelRegistry; print(json.dumps(sorted(ModelRegistry.get_supported_archs())))"],
        {"LD_LIBRARY_PATH": CUDA_COMPAT_FIX_LD_LIBRARY_PATH}, timeout_s=180)
    archs = json.loads(out.strip().splitlines()[-1])
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(archs))
    return archs


FAILURE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("cuda_driver_mismatch", re.compile(r"Error 803|unsupported display driver"), "CUDA driver/library mismatch inside the container (see scripts/run_vllm_server.sh)."),
    ("gpu_memory_changed_during_startup", re.compile(r"Error in memory profiling"), "Another process changed its GPU memory use while vLLM was profiling at startup; the GPU is shared and unstable. Retry when it is idle."),
    ("insufficient_free_memory", re.compile(r"Free memory on device.*less than desired|less than desired GPU memory utilization", re.S), "Not enough free GPU memory for the requested gpu_memory_utilization."),
    ("insufficient_kv_cache", re.compile(r"No available memory for the cache blocks|larger than the maximum number of tokens that can be stored in KV cache|KV cache is needed", re.I), "Not enough memory left for the KV cache at this max_model_len; raise gpu_memory_utilization or lower max_model_len."),
    ("oom", re.compile(r"out of memory|OutOfMemoryError|\bOOM\b", re.I), "GPU out of memory while loading or warming up."),
    ("unsupported_model", re.compile(r"not supported|Unsupported|has no vLLM implementation", re.I), "vLLM does not support this model/architecture/quantization combination."),
    ("model_not_found", re.compile(r"not a local folder|Repository Not Found|couldn't connect to|OfflineModeIsEnabled|Cannot find the requested files", re.I), "Model files not found in the mounted Hugging Face cache (runs offline)."),
]


def classify_launch_failure(logs: str, oom_killed: bool = False) -> tuple[str, str]:
    if oom_killed:
        return "host_oom", "The container was killed by the host's out-of-memory killer (system RAM, not GPU memory)."
    for kind, rx, msg in FAILURE_PATTERNS:
        if rx.search(logs):
            return kind, msg
    return "launch_failed", "The server container exited; see the server log for details."
