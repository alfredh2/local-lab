"""Deterministic prompt generation.

token_ids mode sends raw token ids to /v1/completions, so the input length is exact by
construction; the server's usage.prompt_tokens is compared against it. Ids are random in
[1000, 30000), valid for common vocabularies (>= 32k) and away from special tokens.
Every request gets a distinct seed so prefix caching cannot inflate prefill speed.
text mode builds pseudo-English from a word list; its token count is only approximate.
"""
import random

ID_LOW, ID_HIGH = 1000, 30_000
WORDS = (
    "system model request latency memory token cache batch queue server kernel layer vector tensor "
    "decode prefill attention stream device buffer thread schedule measure result metric sample value "
    "window range limit budget policy config route graph index signal input output prompt answer "
    "question reason context length speed power energy cluster network storage compute pipeline stage "
    "report review change update version branch commit deploy release monitor alert trace record"
).split()


def request_seed(base_seed: int, phase: str, iteration: int, index: int) -> int:
    """Stable per-request seed; phases occupy disjoint ranges so warmup never repeats measured prompts."""
    offset = {"warmup": 1, "measured": 2}[phase]
    return (base_seed * 1_000_003 + offset * 100_000_007 + iteration * 1_000_033 + index) & 0x7FFFFFFF


def token_id_prompt(seed: int, n: int) -> list[int]:
    return random.Random(seed).choices(range(ID_LOW, ID_HIGH), k=n)


def text_prompt(seed: int, approx_tokens: int) -> str:
    n_words = max(1, int(approx_tokens / 1.3))
    return " ".join(random.Random(seed).choices(WORDS, k=n_words))
