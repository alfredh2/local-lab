"""Long-context retrieval suite v1: one target fact hidden in a long document at a controlled position.

Documents are deterministic pseudo-prose; the target fact is an 'access code' with distractor codes for
other keys. Pass = the expected code appears in the answer and no distractor code does. Prompt length is
calibrated with the server's tokenizer (see calibrate_chars_per_token) and the server-reported prompt
token count is stored with every result, so reported lengths are verified, not assumed.
"""
import hashlib
import random
import re
from typing import Any

from locallab.evals.base import CaseScore, EvalCase, EvalSuite

VERSION = "1"
SYSTEM = "You answer questions using only the document provided. Reply with only the requested value."
ADJ = ["quiet", "northern", "weekly", "provisional", "regional", "annual", "internal", "revised", "final", "draft"]
NOUN = ["committee", "audit", "shipment", "survey", "workshop", "inventory", "forecast", "budget", "archive", "pilot"]
PLACE = ["the harbour office", "the east depot", "the main hall", "the records room", "the lab annex", "the north gate"]
CODE_A = ["amber", "birch", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo", "juniper"]
CODE_B = ["falcon", "lantern", "meadow", "nectar", "orchid", "pebble", "quartz", "raven", "saddle", "thistle"]
DEFAULT_LENGTHS = [2048, 4096]
DEFAULT_POSITIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
FILLER_SAMPLE_CHARS = 20_000


def _filler_sentence(r: random.Random, n: int) -> str:
    return (f"Note {n}: the {r.choice(ADJ)} {r.choice(NOUN)} was reviewed in {r.choice(PLACE)} and the {r.choice(ADJ)} "
            f"{r.choice(NOUN)} was postponed until {r.choice(['Monday', 'Tuesday', 'Thursday', 'Friday'])} week {r.randint(1, 52)}.")


def filler_text(seed: str, n_chars: int) -> str:
    r = random.Random(seed)
    out, total, i = [], 0, 0
    while total < n_chars:
        s = _filler_sentence(r, i)
        out.append(s)
        total += len(s) + 1
        i += 1
    return " ".join(out)


def _code(r: random.Random) -> str:
    return f"{r.choice(CODE_A)}-{r.choice(CODE_B)}-{r.randint(1000, 9999)}"


def build_case(split: str, length: int, position: float, repeat: int, chars_per_token: float) -> EvalCase:
    cid = f"ret-v{VERSION}-{split}-L{length}-p{int(position * 100):02d}-r{repeat}"
    seed = int(hashlib.sha256(cid.encode()).hexdigest()[:12], 16)
    r = random.Random(seed)
    keys = r.sample(range(100, 999), 7)
    codes = [_code(r) for _ in keys]
    while len(set(codes)) < len(codes):
        codes = [_code(r) for _ in keys]
    target_key, target_code = keys[0], codes[0]
    facts = [f"The access code for vault V{k} is {c}." for k, c in zip(keys, codes)]
    question = f"What is the access code for vault V{target_key}?"
    budget_chars = max(500, int((length - 120) * chars_per_token))  # leave room for system prompt + question
    body = filler_text(f"{cid}-filler", budget_chars)
    sentences = re.split(r"(?<=\.) ", body)
    n = len(sentences)
    slots = {min(n - 1, max(0, int(position * n))): facts[0]}
    for f in facts[1:]:
        for _ in range(50):
            pos = r.randrange(n)
            if abs(pos / n - position) > 0.06 and pos not in slots:
                slots[pos] = f
                break
    doc = " ".join(s + (" " + slots[i] if i in slots else "") for i, s in enumerate(sentences))
    return EvalCase(
        id=cid, suite="retrieval_needle", version=VERSION, split=split,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {question} Answer with only the code."}],
        max_tokens=64, meta={"expected": target_code, "distractors": codes[1:], "chars_per_token": chars_per_token},
        tags={"target_tokens": length, "position": position, "repeat": repeat})


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


class RetrievalSuite(EvalSuite):
    name, version, title, primary_metric = "retrieval_needle", VERSION, "Long-context retrieval", "accuracy"

    def cases(self, split: str, **params: Any) -> list[EvalCase]:
        lengths = params.get("lengths", DEFAULT_LENGTHS)
        positions = params.get("positions", DEFAULT_POSITIONS)
        repeats = params.get("repeats", 2)
        cpt = params.get("chars_per_token", 4.0)
        return [build_case(split, L, p, k, cpt) for L in lengths for p in positions for k in range(repeats)]

    def score(self, case: EvalCase, response: str, sandbox: Any = None) -> CaseScore:
        resp = normalize(response)
        has = normalize(case.meta["expected"]) in resp
        confused = [d for d in case.meta["distractors"] if normalize(d) in resp]
        comps = {"found": has, "confused": bool(confused)}
        if has and not confused:
            return CaseScore(True, comps)
        if confused:
            return CaseScore(False, comps, "wrong_fact", f"answer contains a different vault's code ({confused[0]})")
        return CaseScore(False, comps, "not_found", response.strip()[:120] or "(empty answer)")

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        grid: dict[str, dict[str, list[int]]] = {}
        by_len: dict[str, list[int]] = {}
        by_pos: dict[str, list[int]] = {}
        for r in records:
            L, p = str(r["tags"]["target_tokens"]), f"{r['tags']['position']:.2f}"
            v = 1 if r["passed"] else 0
            grid.setdefault(L, {}).setdefault(p, []).append(v)
            by_len.setdefault(L, []).append(v)
            by_pos.setdefault(p, []).append(v)
        rate = lambda xs: {"n": len(xs), "k": sum(xs), "rate": sum(xs) / len(xs)}  # noqa: E731
        fails: dict[str, int] = {}
        for r in records:
            if not r["passed"]:
                fails[r["failure_type"] or "unknown"] = fails.get(r["failure_type"] or "unknown", 0) + 1
        actual = [r["prompt_tokens"] for r in records if r.get("prompt_tokens")]
        return {"by_length": {k: rate(v) for k, v in sorted(by_len.items(), key=lambda kv: int(kv[0]))},
                "by_position": {k: rate(v) for k, v in sorted(by_pos.items())},
                "grid": {L: {p: rate(v) for p, v in sorted(ps.items())} for L, ps in sorted(grid.items(), key=lambda kv: int(kv[0]))},
                "failures_by_type": fails, "prompt_tokens_mean": sum(actual) / len(actual) if actual else None}
