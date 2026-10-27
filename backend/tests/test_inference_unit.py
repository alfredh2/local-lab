import pytest

from locallab.experiments.inference.config import InferenceBenchmarkConfig
from locallab.experiments.inference.metrics import Record, dist, percentile, summarize, verify_lengths
from locallab.experiments.inference.workload import request_seed, text_prompt, token_id_prompt
from locallab.runtimes.base import RequestOutcome
from locallab.runtimes.vllm_openai import classify_http_error, parse_cache_info


def test_percentile_linear_interpolation():
    v = [1.0, 2.0, 3.0, 4.0]
    assert percentile(v, 50) == 2.5
    assert percentile(v, 0) == 1.0 and percentile(v, 100) == 4.0
    assert percentile(v, 95) == pytest.approx(3.85)
    assert percentile([7.0], 95) == 7.0
    assert dist([]) is None


def test_token_prompts_exact_length_and_deterministic():
    a = token_id_prompt(42, 300)
    assert len(a) == 300 and a == token_id_prompt(42, 300) and a != token_id_prompt(43, 300)
    assert all(1000 <= t < 30000 for t in a)
    assert len(text_prompt(1, 130).split()) == 100  # ~1.3 tokens/word: approximate by design


def test_request_seeds_are_disjoint_between_phases_and_unique():
    seeds = {request_seed(0, ph, it, i) for ph in ("warmup", "measured") for it in range(3) for i in range(200)}
    assert len(seeds) == 2 * 3 * 200


def outcome(start=10.0, first=10.1, end=11.1, tokens=11, ok=True, prompt=128):
    n = tokens
    chunks = [first + i * (end - first) / (n - 1) for i in range(n)] if n > 1 else [first]
    return RequestOutcome(ok=ok, t_start=start, t_end=end, t_first_token=first, chunk_times=chunks,
                          prompt_tokens=prompt, completion_tokens=tokens, finish_reason="length")


def test_record_metric_definitions():
    r = Record(0, "measured", 0, 1, 128, 11, outcome(), t0=10.0).to_dict()
    assert r["ttft_ms"] == pytest.approx(100)
    assert r["e2e_ms"] == pytest.approx(1100)
    assert r["tpot_ms"] == pytest.approx(100)  # (1100-100)/(11-1)
    assert r["decode_tok_s"] == pytest.approx(10)
    assert r["output_tok_s"] == pytest.approx(11 / 1.1)  # includes TTFT
    assert r["itl_mean_ms"] == pytest.approx(100)


def test_single_token_has_no_tpot():
    r = Record(0, "measured", 0, 1, 8, 1, outcome(tokens=1, first=10.5, end=10.5), t0=0).to_dict()
    assert r["tpot_ms"] is None and r["decode_tok_s"] is None


def test_summary_throughput_and_failures():
    rows = [Record(i, "measured", 0, i, 128, 11, outcome(start=10 + i * 0.5, first=10.1 + i * 0.5, end=11.1 + i * 0.5), 10.0).to_dict()
            for i in range(4)]
    rows.append(Record(4, "measured", 0, 4, 128, 11, RequestOutcome(ok=False, t_start=10.0, t_end=10.2, error_type="oom"), 10.0).to_dict())
    s = summarize(rows)
    # wall = last end (12.6-10) - first start (0) = 2.6s ; 4 ok * 11 output tokens
    assert s["wall_s"] == pytest.approx(2.6)
    assert s["throughput"]["aggregate_output_tok_s"] == pytest.approx(44 / 2.6)
    assert s["throughput"]["total_tok_s"] == pytest.approx((44 + 4 * 128) / 2.6)
    assert s["requests"] == {"attempted": 5, "succeeded": 4, "failed": 1, "success_rate": 0.8, "errors_by_type": {"oom": 1}}
    assert s["latency"]["e2e_ms"]["n"] == 4  # failed request excluded from latency stats


def test_verify_lengths_detects_mismatch():
    good = Record(0, "measured", 0, 1, 128, 11, outcome(), 0).to_dict()
    bad = Record(1, "measured", 0, 2, 128, 11, outcome(prompt=130), 0).to_dict()
    assert verify_lengths([good])["input_length_exact"] is True
    v = verify_lengths([good, bad])
    assert v["input_length_exact"] is False and v["input_mismatches"] == 1


def test_error_classification_and_metrics_parsing():
    assert classify_http_error(500, "CUDA out of memory. Tried") == "oom"
    assert classify_http_error(400, "bad") == "bad_request" and classify_http_error(503, "x") == "server_error"
    info = parse_cache_info('vllm:cache_config_info{block_size="16",num_gpu_blocks="10"} 1.0')
    assert info == {"block_size": "16", "num_gpu_blocks": "10"}


def test_config_host_allowlist_and_bounds(settings):
    with pytest.raises(ValueError, match="not in LOCALLAB_ALLOWED_RUNTIME_HOSTS"):
        InferenceBenchmarkConfig(model="m", server_url="http://169.254.169.254:80")
    with pytest.raises(ValueError):
        InferenceBenchmarkConfig(model="m", concurrency=0)
    assert InferenceBenchmarkConfig(model="m", server_url="http://localhost:8100/").server_url == "http://localhost:8100"
