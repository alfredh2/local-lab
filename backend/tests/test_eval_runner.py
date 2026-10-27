import asyncio
import json
import re

import pytest

from locallab.evals.datasets.coding import TASKS, CodingSuite
from locallab.evals.datasets.retrieval import RetrievalSuite
from locallab.evals.datasets.structured import StructuredSuite
from locallab.evals.runner import calibrate_chars_per_token, run_suite
from locallab.evals.sandbox import LocalSubprocessSandbox
from locallab.runtimes.vllm_openai import VllmOpenAIAdapter
from tests.mock_vllm import MockServer


def run(coro):
    return asyncio.run(coro)


def evaluate(srv, suite, cases, sandbox=None, **kw):
    async def go():
        a = VllmOpenAIAdapter(srv.url)
        try:
            return await run_suite(a, "mock-model", suite, cases, sandbox, **kw)
        finally:
            await a.aclose()

    return run(go())


def structured_responder(cases, wrong=()):
    lookup = {c.messages[-1]["content"]: c for c in cases}

    def respond(messages, body):
        c = lookup[messages[-1]["content"]]
        if c.id in wrong:
            return "Sure, here you go: not json"
        return json.dumps(c.meta["reference"])

    return respond


def test_structured_suite_end_to_end_with_mock_model():
    suite = StructuredSuite()
    cases = suite.cases("heldout")
    wrong = {cases[0].id, cases[1].id}
    with MockServer(responder=structured_responder(cases, wrong)) as srv:
        res, recs = evaluate(srv, suite, cases, concurrency=4)
    assert res["n"] == 18 and res["passed"] == 16 and res["split"] == "heldout" and len(recs) == 18
    assert res["interval"]["rate"] == pytest.approx(16 / 18) and res["interval"]["lo"] < 0.89 < res["interval"]["hi"]
    assert res["aggregate"]["failures_by_type"] == {"no_json": 2}
    assert res["performance"]["requests"] == {"attempted": 18, "succeeded": 18, "failed": 0, "errors_by_type": {}}
    assert "evaluation_prompts" in res["performance"]["workload_kind"] and res["performance"]["latency"]["e2e_ms"]["p50"] > 0
    assert res["performance"]["latency"]["e2e_ms"]["p99"] is None  # 18 samples cannot support p99
    assert srv.state.max_in_flight <= 4
    body = srv.state.bodies[0]
    assert body["temperature"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": False} and body["messages"][0]["role"] == "user"
    bad = next(r for r in recs if not r["passed"])
    assert bad["response"].startswith("Sure") and bad["failure_type"] == "no_json" and bad["case_id"] in wrong


def test_generation_failures_count_against_accuracy_and_are_classified():
    suite = StructuredSuite()
    cases = suite.cases("dev")
    with MockServer(responder=structured_responder(cases), mode="http500") as srv:
        res, recs = evaluate(srv, suite, cases, concurrency=2)
    assert res["passed"] == 0 and res["n"] == 6  # a model that cannot answer is not 'accurate on the ones it answered'
    assert res["performance"]["requests"]["failed"] == 6 and res["performance"]["requests"]["errors_by_type"] == {"server_error": 6}
    assert {r["failure_type"] for r in recs} == {"generation_server_error"} and res["performance"]["latency"]["e2e_ms"] is None


def test_timeouts_are_reported():
    suite = StructuredSuite()
    cases = suite.cases("dev")[:2]
    with MockServer(mode="hang") as srv:
        res, recs = evaluate(srv, suite, cases, timeout_s=5)
    assert res["performance"]["requests"]["errors_by_type"] == {"timeout": 2} and res["passed"] == 0


def test_truncated_outputs_are_flagged():
    suite = StructuredSuite()
    cases = suite.cases("dev")[:3]
    long_answer = lambda messages, body: json.dumps({"x": "y" * 5000})  # noqa: E731  (exceeds the 512-token budget in the mock)
    with MockServer(responder=long_answer, chat_token_s=0.0) as srv:
        res, recs = evaluate(srv, suite, cases)
    assert res["truncated"] == 3 and all(r["finish_reason"] == "length" for r in recs)


def test_coding_suite_scores_with_sandbox():
    suite = CodingSuite()
    cases = suite.cases("heldout")
    ref = {t.key: t.reference for t in TASKS}
    lookup = {c.messages[-1]["content"]: c.meta["key"] for c in cases}
    broken = {cases[0].meta["key"], cases[1].meta["key"], cases[2].meta["key"]}

    def respond(messages, body):
        key = lookup[messages[-1]["content"]]
        code = "def nope():\n    pass" if key in broken else ref[key]
        return f"```python\n{code}\n```"

    with MockServer(responder=respond) as srv:
        res, recs = evaluate(srv, suite, cases, LocalSubprocessSandbox(), concurrency=4)
    assert res["n"] == 13 and res["passed"] == 10
    assert res["aggregate"]["failures_by_type"] == {"missing_function": 3}
    assert res["interval"]["lo"] < 10 / 13 < res["interval"]["hi"]


def test_scorer_faults_are_not_blamed_on_the_model():
    class Boom(CodingSuite):
        def score(self, case, response, sandbox=None):
            raise RuntimeError("docker exploded")

    suite = Boom()
    with MockServer(responder=lambda m, b: "```python\nx=1\n```") as srv:
        res, recs = evaluate(srv, suite, suite.cases("dev")[:2], None)
    assert {r["failure_type"] for r in recs} == {"scoring_error"} and "docker exploded" in recs[0]["detail"]


def retrieval_oracle(blind_positions=()):
    """A 'perfect reader' that finds the code for the asked vault, except at blind positions."""
    def respond(messages, body):
        doc = messages[-1]["content"]
        key = re.search(r"vault (V\d+)\?", doc).group(1)
        code = re.search(rf"access code for vault {key} is ([a-z]+-[a-z]+-\d+)\.", doc).group(1)
        pos = doc.index(code) / doc.index("Question:")
        return "I don't know" if any(abs(pos - bp) < 0.1 for bp in blind_positions) else code
    return respond


def test_retrieval_calibration_and_position_breakdown():
    suite = RetrievalSuite()
    with MockServer(responder=retrieval_oracle(blind_positions=[0.5]), chars_per_token=3.0, max_model_len=8192) as srv:
        async def go():
            a = VllmOpenAIAdapter(srv.url)
            try:
                cpt, measured = await calibrate_chars_per_token(a, "mock-model")
                cases = suite.cases("heldout", lengths=[2048], repeats=2, chars_per_token=cpt)
                res, recs = await run_suite(a, "mock-model", suite, cases, None, concurrency=4)
                return cpt, measured, cases, res, recs
            finally:
                await a.aclose()
        cpt, measured, cases, res, recs = run(go())
    assert measured and cpt == pytest.approx(3.0, abs=0.05)
    assert res["n"] == 10 and res["passed"] == 8
    assert res["aggregate"]["by_position"]["0.50"]["rate"] == 0.0 and res["aggregate"]["by_position"]["0.05"]["rate"] == 1.0
    assert res["aggregate"]["grid"]["2048"]["0.95"]["k"] == 2
    # the server-reported prompt size is stored, so the claimed length is verifiable against the target
    assert all(abs(r["prompt_tokens"] - 2048) / 2048 < 0.1 for r in recs)


def test_calibration_falls_back_when_server_has_no_tokenizer():
    with MockServer(tokenize=False) as srv:
        async def go():
            a = VllmOpenAIAdapter(srv.url)
            try:
                return await calibrate_chars_per_token(a, "mock-model")
            finally:
                await a.aclose()
        assert run(go()) == (4.0, False)


def test_progress_callback_reports_every_case():
    suite = StructuredSuite()
    cases = suite.cases("dev")
    seen = []
    with MockServer(responder=structured_responder(cases)) as srv:
        evaluate(srv, suite, cases, concurrency=3, on_progress=lambda d, t, p: seen.append((d, t, p)))
    assert seen[-1] == (6, 6, 6) and [s[0] for s in seen] == sorted(s[0] for s in seen)
