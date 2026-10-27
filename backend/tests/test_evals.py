import json

import pytest
from jsonschema import Draft202012Validator

from locallab.evals.base import fingerprint, wilson
from locallab.evals.datasets import coding as C
from locallab.evals.datasets import retrieval as R
from locallab.evals.datasets import structured as S
from locallab.evals.registry import SUITES, get_suite
from locallab.evals.sandbox import SENTINEL, LocalSubprocessSandbox, parse_harness_output

SB = LocalSubprocessSandbox()


# ---------------- datasets: stability and splits ----------------
@pytest.mark.parametrize("name", list(SUITES))
def test_cases_are_stable_unique_and_split_disjoint(name):
    s = get_suite(name)
    dev, held = s.cases("dev"), s.cases("heldout")
    assert dev and held
    assert fingerprint(s.cases("heldout")) == fingerprint(held)  # deterministic
    ids = [c.id for c in dev + held]
    assert len(ids) == len(set(ids))
    assert {c.split for c in dev} == {"dev"} and {c.split for c in held} == {"heldout"}
    assert all(c.version == s.version and c.id.startswith(("json-v", "code-v", "ret-v")) for c in dev + held)
    assert fingerprint(dev) != fingerprint(held)


def test_expected_answers_never_appear_in_prompts_except_retrieval_document():
    for c in C.CodingSuite().cases("heldout"):
        assert "assert" not in json.dumps(c.messages) and C.TASKS[0].reference not in json.dumps(c.messages)
    for c in S.StructuredSuite().cases("heldout"):
        assert "expected" not in json.dumps(c.messages)


# ---------------- coding ----------------
@pytest.mark.parametrize("task", C.TASKS, ids=[t.key for t in C.TASKS])
def test_every_reference_solution_passes_its_hidden_tests(task):
    r = SB.run_python({"solution.py": task.reference, "harness.py": C.harness_source(task)}, "harness.py", 20)
    results = parse_harness_output(r.stdout)
    assert results and all(x["ok"] for x in results), (results, r.stderr)
    assert len(task.tests) >= 6


def test_hidden_tests_reject_plausible_wrong_solutions():
    t = next(x for x in C.TASKS if x.key == "rotate_list")
    wrong = "def rotate_list(lst, k):\n    return lst[k:] + lst[:k]"  # rotates left, no empty-list guard
    r = SB.run_python({"solution.py": wrong, "harness.py": C.harness_source(t)}, "harness.py", 20)
    res = parse_harness_output(r.stdout)
    assert any(not x["ok"] for x in res)


def code_case(key):
    return next(c for c in C.CodingSuite().cases("heldout") + C.CodingSuite().cases("dev") if c.meta["key"] == key)


def score(key, response):
    return C.CodingSuite().score(code_case(key), response, SB)


def fenced(code):
    return f"Here you go:\n```python\n{code}\n```\nHope it helps."


def test_coding_scoring_outcomes():
    ref = next(t for t in C.TASKS if t.key == "is_palindrome").reference
    assert score("is_palindrome", fenced(ref)).passed is True
    assert score("is_palindrome", "I cannot help with that.").failure_type == "no_code"
    assert score("is_palindrome", fenced("def is_palindrome(s:\n    return")).failure_type == "syntax_error"
    assert score("is_palindrome", fenced("def other(s):\n    return True")).failure_type == "missing_function"
    wrong = score("is_palindrome", fenced("def is_palindrome(s):\n    return s == s[::-1]"))
    assert wrong.passed is False and wrong.failure_type == "wrong_answer" and 0 < wrong.components["tests_passed"] < 1
    crash = score("is_palindrome", fenced("def is_palindrome(s):\n    return s.nonexistent()"))
    assert crash.failure_type == "runtime_error"
    assert score("is_palindrome", ref).passed is True  # bare code without a fence is accepted


def test_coding_infinite_loop_times_out_without_stalling():
    import time

    t0 = time.time()
    s = score("is_palindrome", fenced("def is_palindrome(s):\n    while True:\n        pass"))
    assert s.failure_type == "timeout" and time.time() - t0 < 10


def test_model_prints_cannot_forge_a_passing_result_by_accident():
    noisy = "print('hello')\ndef is_palindrome(s):\n    return False"
    assert score("is_palindrome", fenced(noisy)).passed is False


def test_extract_code_variants():
    assert C.extract_code("```python\nx=1\n```") == "x=1"
    assert C.extract_code("```\ndef f():\n  pass\n```") == "def f():\n  pass"
    assert C.extract_code("def f():\n  pass") == "def f():\n  pass"
    assert C.extract_code("just prose") is None


def test_parse_harness_output_ignores_model_prints():
    out = f"noise\n{SENTINEL}" + json.dumps({"results": [{"name": "t0", "ok": True}]}) + "\n"
    assert parse_harness_output(out) == [{"name": "t0", "ok": True}]
    assert parse_harness_output("no sentinel here") is None
    assert parse_harness_output(f"{SENTINEL}not json") is None


# ---------------- structured ----------------
def jcase(template="person", v=1):
    return next(c for c in S.StructuredSuite().cases("heldout") if c.tags["template"] == template and c.id.endswith(f"-{v}"))


def good_json(case):
    return json.loads(json.dumps(case.meta["reference"]))  # the dataset's own reference answer


def test_structured_reference_answers_validate_and_score_fully_for_every_case():
    suite = S.StructuredSuite()
    cases = suite.cases("heldout") + suite.cases("dev")
    assert len(cases) == 24
    for case in cases:
        Draft202012Validator.check_schema(case.meta["schema"])  # the schema itself is valid
        r = suite.score(case, json.dumps(case.meta["reference"]))
        assert r.passed and r.components["extraction"] == 1.0, (case.id, r)


def test_structured_failure_modes():
    suite = S.StructuredSuite()
    c = jcase("order")
    ok = good_json(c)
    ok_text = json.dumps(ok)
    assert suite.score(c, ok_text).passed
    assert suite.score(c, f"```json\n{ok_text}\n```").passed  # a single fence is tolerated
    chatty = suite.score(c, f"Sure! Here is the JSON:\n{ok_text}")
    assert chatty.passed is False and chatty.failure_type == "extra_text_or_invalid_json" and chatty.components["lenient_parse"] is True
    assert suite.score(c, "no json at all").failure_type == "no_json"
    assert suite.score(c, "[1,2,3]").failure_type == "no_json"
    bad_enum = {**ok, "status": "lost"}
    r = suite.score(c, json.dumps(bad_enum))
    assert r.failure_type == "schema_violation" and "[enum]" in r.detail
    missing = {k: v for k, v in ok.items() if k != "total"}
    assert "[required]" in suite.score(c, json.dumps(missing)).detail
    assert "[additionalProperties]" in suite.score(c, json.dumps({**ok, "extra": 1})).detail
    badsku = json.loads(ok_text)
    badsku["items"][0]["sku"] = "abc-1"
    assert "[pattern]" in suite.score(c, json.dumps(badsku)).detail


def test_structured_optional_and_nullable_semantics():
    suite = S.StructuredSuite()
    person = next(c for c in suite.cases("heldout") if c.tags["template"] == "person" and "email" not in c.meta["expected"])
    base = good_json(person)
    base["tags"] = ["x"]
    assert suite.score(person, json.dumps(base)).passed
    assert suite.score(person, json.dumps({**base, "email": None})).failure_type == "schema_violation"  # null is not an omitted optional
    ticket = next(c for c in suite.cases("heldout") if c.tags["template"] == "ticket" and c.meta["expected"]["assignee"] is None)
    t = good_json(ticket)
    assert t["assignee"] is None and suite.score(ticket, json.dumps(t)).passed  # nullable accepts null


def test_structured_wrong_but_valid_values_lower_extraction_only():
    suite = S.StructuredSuite()
    c = jcase("sensor")
    obj = good_json(c)
    obj["unit"] = "K" if obj["unit"] != "K" else "C"
    r = suite.score(c, json.dumps(obj))
    assert r.passed is True and r.components["extraction"] < 1.0  # compliant JSON, wrong content: visible separately


def test_structured_aggregate():
    suite = S.StructuredSuite()
    c = jcase("order")
    recs = []
    for resp in [json.dumps(good_json(c)), "nope", json.dumps({**good_json(c), "status": "x"})]:
        s = suite.score(c, resp)
        recs.append({"passed": s.passed, "components": s.components, "failure_type": s.failure_type, "detail": s.detail, "tags": c.tags})
    a = suite.aggregate(recs)
    assert a["parse_rate"] == pytest.approx(2 / 3) and a["schema_valid_rate"] == pytest.approx(1 / 3)
    assert a["failures_by_type"] == {"no_json": 1, "schema_violation": 1} and a["violations_by_keyword"] == {"enum": 1}


# ---------------- retrieval ----------------
def test_retrieval_target_present_once_at_requested_position_with_distractors():
    for p in R.DEFAULT_POSITIONS:
        c = R.build_case("heldout", 2048, p, 0, 4.0)
        doc = c.messages[1]["content"]
        assert doc.count(c.meta["expected"]) == 1 and all(d in doc for d in c.meta["distractors"]) and len(c.meta["distractors"]) == 6
        frac = doc.index(c.meta["expected"]) / doc.index("Question:")
        assert abs(frac - p) < 0.1, (p, frac)


def test_retrieval_length_scales_and_is_deterministic():
    a, b = R.build_case("dev", 2048, 0.5, 0, 4.0), R.build_case("dev", 2048, 0.5, 0, 4.0)
    assert a.messages == b.messages
    short, long_ = R.build_case("dev", 2048, 0.5, 0, 4.0), R.build_case("dev", 8192, 0.5, 0, 4.0)
    assert len(long_.messages[1]["content"]) > 3.5 * len(short.messages[1]["content"])
    denser = R.build_case("dev", 2048, 0.5, 0, 3.0)  # a worse chars/token ratio means fewer characters for the same tokens
    assert len(denser.messages[1]["content"]) < len(short.messages[1]["content"])
    assert R.build_case("dev", 2048, 0.5, 1, 4.0).meta["expected"] != a.meta["expected"]


def test_retrieval_scoring():
    suite = R.RetrievalSuite()
    c = R.build_case("heldout", 2048, 0.5, 0, 4.0)
    exp, dis = c.meta["expected"], c.meta["distractors"][0]
    assert suite.score(c, exp).passed and suite.score(c, f"The code is {exp.upper()}.").passed
    only_distractor = suite.score(c, dis)
    assert only_distractor.passed is False and only_distractor.failure_type == "wrong_fact"
    assert suite.score(c, "I could not find it.").failure_type == "not_found"
    both = suite.score(c, f"{exp} or {dis}")
    assert both.passed is False and both.failure_type == "wrong_fact"
    assert suite.score(c, "").failure_type == "not_found"


def test_retrieval_aggregate_grid():
    suite = R.RetrievalSuite()
    recs = [{"passed": p, "failure_type": None if p else "not_found", "tags": {"target_tokens": L, "position": pos}, "prompt_tokens": L}
            for L, pos, p in [(2048, 0.05, True), (2048, 0.5, False), (4096, 0.05, True), (4096, 0.5, True), (4096, 0.5, False)]]
    a = suite.aggregate(recs)
    assert a["by_length"]["2048"] == {"n": 2, "k": 1, "rate": 0.5} and a["by_length"]["4096"]["n"] == 3
    assert a["by_position"]["0.50"]["n"] == 3 and a["grid"]["4096"]["0.50"]["k"] == 1


def test_default_retrieval_cases_per_split():
    assert len(R.RetrievalSuite().cases("heldout")) == 2 * 5 * 2


# ---------------- statistics ----------------
def test_wilson_interval():
    w = wilson(8, 10)
    assert w["rate"] == 0.8 and w["lo"] == pytest.approx(0.490, abs=0.002) and w["hi"] == pytest.approx(0.943, abs=0.002)
    assert wilson(0, 10)["lo"] == 0 and wilson(10, 10)["hi"] == 1.0
    assert wilson(0, 0)["rate"] is None
    small, large = wilson(5, 10), wilson(500, 1000)
    assert (small["hi"] - small["lo"]) > 5 * (large["hi"] - large["lo"])  # small n is honestly wide
