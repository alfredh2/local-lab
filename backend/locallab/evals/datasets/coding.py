"""Python coding suite v1: small deterministic function-writing tasks with hidden asserts.

Each task ships a reference solution (validated in the test-suite against its own hidden tests) so a
broken task cannot silently depress every configuration's score.
"""
import json
import re
from dataclasses import dataclass
from typing import Any

from locallab.evals.base import CaseScore, EvalCase, EvalSuite, split_for
from locallab.evals.sandbox import SENTINEL, parse_harness_output

VERSION = "1"
PER_TEST_TIMEOUT_S = 2
SYSTEM = "You are an expert Python programmer. Reply with only the complete function in a single ```python code block."


@dataclass(frozen=True)
class Task:
    key: str
    entry: str
    prompt: str
    reference: str
    tests: tuple[str, ...]


T = Task
TASKS: list[Task] = [
    T("sum_odd_squares", "sum_odd_squares", "Write `sum_odd_squares(n)` returning the sum of the squares of all odd integers from 1 to n inclusive (0 if n < 1).",
      "def sum_odd_squares(n):\n    return sum(i*i for i in range(1, n+1, 2))",
      ("sum_odd_squares(1)==1", "sum_odd_squares(5)==35", "sum_odd_squares(0)==0", "sum_odd_squares(-4)==0", "sum_odd_squares(10)==165", "sum_odd_squares(2)==1")),
    T("is_palindrome", "is_palindrome", "Write `is_palindrome(s)` returning True if the string reads the same forwards and backwards, ignoring case and any non-alphanumeric characters.",
      "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
      ("is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('race a car') is False", "is_palindrome('') is True", "is_palindrome('No lemon, no melon') is True", "is_palindrome('ab') is False", "is_palindrome('12321') is True")),
    T("fizzbuzz_list", "fizzbuzz_list", "Write `fizzbuzz_list(n)` returning a list of strings for 1..n: 'FizzBuzz' for multiples of 15, 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, otherwise the number as a string.",
      "def fizzbuzz_list(n):\n    out = []\n    for i in range(1, n+1):\n        out.append('FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i))\n    return out",
      ("fizzbuzz_list(0)==[]", "fizzbuzz_list(3)==['1','2','Fizz']", "fizzbuzz_list(5)[-1]=='Buzz'", "fizzbuzz_list(15)[-1]=='FizzBuzz'", "len(fizzbuzz_list(20))==20", "fizzbuzz_list(7)[5]=='Fizz'")),
    T("flatten", "flatten", "Write `flatten(items)` that takes an arbitrarily nested list of integers and returns a single flat list preserving order.",
      "def flatten(items):\n    out = []\n    for x in items:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out",
      ("flatten([])==[]", "flatten([1,[2,[3,[4]]]])==[1,2,3,4]", "flatten([[1,2],[3],[]])==[1,2,3]", "flatten([[[]]])==[]", "flatten([1,2,3])==[1,2,3]", "flatten([[1,[2]],[[3],4],5])==[1,2,3,4,5]")),
    T("word_frequencies", "word_frequencies", "Write `word_frequencies(text)` returning a dict mapping each lowercase word to its count. Words are separated by whitespace; strip leading/trailing punctuation `.,!?;:` from each word and ignore empty words.",
      "def word_frequencies(text):\n    d = {}\n    for w in text.split():\n        w = w.strip('.,!?;:').lower()\n        if w:\n            d[w] = d.get(w, 0) + 1\n    return d",
      ("word_frequencies('')=={}", "word_frequencies('a a b')=={'a':2,'b':1}", "word_frequencies('Hello, hello! World.')=={'hello':2,'world':1}", "word_frequencies('... !!')=={}", "word_frequencies('x;  y: x')=={'x':2,'y':1}", "word_frequencies('The the THE')=={'the':3}")),
    T("two_sum", "two_sum", "Write `two_sum(nums, target)` returning a tuple (i, j) with i < j of the indices of two numbers adding up to target, or None if no such pair exists. If several pairs exist return the one with the smallest j, then the smallest i.",
      "def two_sum(nums, target):\n    for j in range(len(nums)):\n        for i in range(j):\n            if nums[i] + nums[j] == target:\n                return (i, j)\n    return None",
      ("two_sum([2,7,11,15],9)==(0,1)", "two_sum([3,2,4],6)==(1,2)", "two_sum([1,2,3],10) is None", "two_sum([],0) is None", "two_sum([3,3],6)==(0,1)", "two_sum([1,5,5,1],6)==(0,1)")),
    T("roman_to_int", "roman_to_int", "Write `roman_to_int(s)` converting a valid Roman numeral (I,V,X,L,C,D,M with subtractive notation) to an integer.",
      "def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t = 0\n    for i, c in enumerate(s):\n        if i+1 < len(s) and v[c] < v[s[i+1]]:\n            t -= v[c]\n        else:\n            t += v[c]\n    return t",
      ("roman_to_int('III')==3", "roman_to_int('IV')==4", "roman_to_int('IX')==9", "roman_to_int('LVIII')==58", "roman_to_int('MCMXCIV')==1994", "roman_to_int('MMXXIV')==2024")),
    T("int_to_roman", "int_to_roman", "Write `int_to_roman(n)` converting an integer from 1 to 3999 to its standard Roman numeral string.",
      "def int_to_roman(n):\n    pairs = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out = ''\n    for v, s in pairs:\n        while n >= v:\n            out += s\n            n -= v\n    return out",
      ("int_to_roman(1)=='I'", "int_to_roman(4)=='IV'", "int_to_roman(9)=='IX'", "int_to_roman(58)=='LVIII'", "int_to_roman(1994)=='MCMXCIV'", "int_to_roman(3999)=='MMMCMXCIX'")),
    T("gcd_list", "gcd_list", "Write `gcd_list(nums)` returning the greatest common divisor of a non-empty list of positive integers.",
      "from math import gcd\n\ndef gcd_list(nums):\n    g = 0\n    for n in nums:\n        g = gcd(g, n)\n    return g",
      ("gcd_list([12])==12", "gcd_list([12,18])==6", "gcd_list([7,13])==1", "gcd_list([100,75,50])==25", "gcd_list([8,16,24,40])==8", "gcd_list([9,6,3])==3")),
    T("rotate_list", "rotate_list", "Write `rotate_list(lst, k)` returning a new list rotated to the right by k positions (k may be larger than the length or negative for a left rotation). An empty list returns an empty list.",
      "def rotate_list(lst, k):\n    if not lst:\n        return []\n    k %= len(lst)\n    return lst[-k:] + lst[:-k] if k else list(lst)",
      ("rotate_list([],3)==[]", "rotate_list([1,2,3,4,5],2)==[4,5,1,2,3]", "rotate_list([1,2,3],0)==[1,2,3]", "rotate_list([1,2,3],4)==[3,1,2]", "rotate_list([1,2,3,4],-1)==[2,3,4,1]", "rotate_list([1],10)==[1]")),
    T("balanced_brackets", "balanced_brackets", "Write `balanced_brackets(s)` returning True if every (), [] and {} in the string is properly nested and closed. Other characters are ignored.",
      "def balanced_brackets(s):\n    pairs = {')':'(', ']':'[', '}':'{'}\n    st = []\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif c in pairs:\n            if not st or st.pop() != pairs[c]:\n                return False\n    return not st",
      ("balanced_brackets('')is True", "balanced_brackets('([]{})') is True", "balanced_brackets('(]') is False", "balanced_brackets('((') is False", "balanced_brackets('a(b)c[d]') is True", "balanced_brackets('}{') is False")),
    T("merge_intervals", "merge_intervals", "Write `merge_intervals(intervals)` taking a list of [start, end] pairs and returning a sorted list of merged, non-overlapping [start, end] lists. Intervals that touch (end == next start) are merged.",
      "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out",
      ("merge_intervals([])==[]", "merge_intervals([[1,3],[2,6],[8,10]])==[[1,6],[8,10]]", "merge_intervals([[1,4],[4,5]])==[[1,5]]", "merge_intervals([[5,6],[1,2]])==[[1,2],[5,6]]", "merge_intervals([[1,10],[2,3]])==[[1,10]]", "merge_intervals([[1,2]])==[[1,2]]")),
    T("count_vowels", "count_vowels", "Write `count_vowels(s)` returning the number of vowels (a, e, i, o, u, case-insensitive) in the string.",
      "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
      ("count_vowels('')==0", "count_vowels('hello')==2", "count_vowels('AEIOU')==5", "count_vowels('rhythm')==0", "count_vowels('Programming Language')==7", "count_vowels('y')==0")),
    T("caesar_shift", "caesar_shift", "Write `caesar_shift(s, k)` shifting each ASCII letter forward by k positions in the alphabet (wrapping, preserving case, k may be negative or exceed 26). Non-letters are unchanged.",
      "def caesar_shift(s, k):\n    out = []\n    for c in s:\n        if c.isascii() and c.isalpha():\n            base = ord('A') if c.isupper() else ord('a')\n            out.append(chr((ord(c) - base + k) % 26 + base))\n        else:\n            out.append(c)\n    return ''.join(out)",
      ("caesar_shift('abc',1)=='bcd'", "caesar_shift('xyz',3)=='abc'", "caesar_shift('Hello, World!',13)=='Uryyb, Jbeyq!'", "caesar_shift('abc',-1)=='zab'", "caesar_shift('abc',27)=='bcd'", "caesar_shift('',5)==''")),
    T("longest_common_prefix", "longest_common_prefix", "Write `longest_common_prefix(strs)` returning the longest common prefix of a list of strings ('' if none or the list is empty).",
      "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    p = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(p):\n            p = p[:-1]\n    return p",
      ("longest_common_prefix([])==''", "longest_common_prefix(['flower','flow','flight'])=='fl'", "longest_common_prefix(['dog','car'])==''", "longest_common_prefix(['a'])=='a'", "longest_common_prefix(['abc','abc'])=='abc'", "longest_common_prefix(['','b'])==''")),
    T("matrix_transpose", "matrix_transpose", "Write `matrix_transpose(m)` returning the transpose of a rectangular list-of-lists matrix (an empty matrix returns []).",
      "def matrix_transpose(m):\n    return [list(r) for r in zip(*m)] if m else []",
      ("matrix_transpose([])==[]", "matrix_transpose([[1,2,3],[4,5,6]])==[[1,4],[2,5],[3,6]]", "matrix_transpose([[1]])==[[1]]", "matrix_transpose([[1,2]])==[[1],[2]]", "matrix_transpose([[1],[2],[3]])==[[1,2,3]]", "matrix_transpose([[1,2],[3,4]])==[[1,3],[2,4]]")),
    T("prime_factors", "prime_factors", "Write `prime_factors(n)` returning the sorted list of prime factors of n >= 2, with multiplicity (e.g. 12 -> [2, 2, 3]).",
      "def prime_factors(n):\n    out, d = [], 2\n    while d * d <= n:\n        while n % d == 0:\n            out.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        out.append(n)\n    return out",
      ("prime_factors(2)==[2]", "prime_factors(12)==[2,2,3]", "prime_factors(97)==[97]", "prime_factors(360)==[2,2,2,3,3,5]", "prime_factors(49)==[7,7]", "prime_factors(1001)==[7,11,13]")),
    T("chunk_list", "chunk_list", "Write `chunk_list(lst, size)` splitting a list into consecutive chunks of length `size` (the last chunk may be shorter). Raise ValueError if size < 1.",
      "def chunk_list(lst, size):\n    if size < 1:\n        raise ValueError('size must be >= 1')\n    return [lst[i:i+size] for i in range(0, len(lst), size)]",
      ("chunk_list([],3)==[]", "chunk_list([1,2,3,4,5],2)==[[1,2],[3,4],[5]]", "chunk_list([1,2,3],3)==[[1,2,3]]", "chunk_list([1,2,3],10)==[[1,2,3]]", "chunk_list([1,2],1)==[[1],[2]]", "chunk_list([1,2,3,4],2)==[[1,2],[3,4]]")),
    T("dedupe_preserve_order", "dedupe_preserve_order", "Write `dedupe_preserve_order(items)` returning a list of the unique items of a list in order of first appearance.",
      "def dedupe_preserve_order(items):\n    seen, out = set(), []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
      ("dedupe_preserve_order([])==[]", "dedupe_preserve_order([1,2,1,3,2])==[1,2,3]", "dedupe_preserve_order(['a','a','a'])==['a']", "dedupe_preserve_order([3,2,1])==[3,2,1]", "dedupe_preserve_order([1,1,2,2,3,3,1])==[1,2,3]", "dedupe_preserve_order([0,False,'',None])==[0,'',None]")),
    T("max_subarray_sum", "max_subarray_sum", "Write `max_subarray_sum(nums)` returning the largest sum of any non-empty contiguous subarray of a non-empty list of integers.",
      "def max_subarray_sum(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
      ("max_subarray_sum([1])==1", "max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4])==6", "max_subarray_sum([-3,-1,-2])==-1", "max_subarray_sum([5,4,-1,7,8])==23", "max_subarray_sum([0,0,0])==0", "max_subarray_sum([2,-1,2,-1,2])==4")),
]


def harness_source(task: Task) -> str:
    """Trusted driver run inside the sandbox: imports the model's solution and runs every hidden assert."""
    return f'''import json, signal, sys, traceback
RESULTS = []
def _alarm(signum, frame):
    raise TimeoutError("test exceeded {PER_TEST_TIMEOUT_S}s")
signal.signal(signal.SIGALRM, _alarm)
ns = {{}}
try:
    signal.alarm({PER_TEST_TIMEOUT_S * 2})
    src = open("solution.py").read()
    exec(compile(src, "solution.py", "exec"), ns)
    signal.alarm(0)
    loaded = True
except BaseException as e:
    signal.alarm(0)
    loaded = False
    kind = "syntax_error" if isinstance(e, SyntaxError) else "load_error"
    RESULTS.append({{"name": "load", "ok": False, "kind": kind, "error": f"{{type(e).__name__}}: {{e}}"[:300]}})
if loaded and {task.entry!r} not in ns:
    loaded = False
    RESULTS.append({{"name": "load", "ok": False, "kind": "missing_function", "error": "function {task.entry} is not defined"}})
TESTS = {json.dumps(list(task.tests))}
dead = False
if loaded:
    for i, t in enumerate(TESTS):
        if dead:
            RESULTS.append({{"name": f"t{{i}}", "ok": False, "kind": "timeout", "error": "skipped after an earlier timeout"}})
            continue
        try:
            signal.alarm({PER_TEST_TIMEOUT_S})
            ok = bool(eval(t, ns))
            signal.alarm(0)
            RESULTS.append({{"name": f"t{{i}}", "ok": ok, "kind": None if ok else "wrong_answer", "error": None if ok else "assertion false"}})
        except TimeoutError as e:
            signal.alarm(0)
            dead = True
            RESULTS.append({{"name": f"t{{i}}", "ok": False, "kind": "timeout", "error": str(e)}})
        except BaseException as e:
            signal.alarm(0)
            RESULTS.append({{"name": f"t{{i}}", "ok": False, "kind": "runtime_error", "error": f"{{type(e).__name__}}: {{e}}"[:300]}})
print({SENTINEL!r} + json.dumps({{"results": RESULTS}}))
'''


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(response: str) -> str | None:
    m = _FENCE.search(response)
    if m:
        return m.group(1).strip("\n")
    stripped = response.strip()
    return stripped if re.match(r"^(def |import |from |class )", stripped) else None


class CodingSuite(EvalSuite):
    name, version, title, primary_metric = "coding_python", VERSION, "Python coding (hidden tests)", "pass_rate"

    def cases(self, split: str, **params: Any) -> list[EvalCase]:
        out = []
        for i, t in enumerate(TASKS):
            sp = split_for(i, 3)
            if sp != split:
                continue
            out.append(EvalCase(
                id=f"code-v{VERSION}-{sp}-{t.key}", suite=self.name, version=VERSION, split=sp,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": t.prompt}],
                max_tokens=768, meta={"key": t.key}, tags={"task": t.key}))
        return out

    def score(self, case: EvalCase, response: str, sandbox: Any = None) -> CaseScore:
        task = next(t for t in TASKS if t.key == case.meta["key"])
        code = extract_code(response)
        if code is None:
            return CaseScore(False, {"tests_passed": 0.0}, "no_code", "response contained no code block")
        if sandbox is None:
            raise RuntimeError("coding scoring requires a sandbox")
        res = sandbox.run_python({"solution.py": code, "harness.py": harness_source(task)}, "harness.py", 12.0)
        if res.error:
            return CaseScore(False, {}, "sandbox_error", res.error)
        if res.timed_out:
            return CaseScore(False, {"tests_passed": 0.0}, "timeout", "execution exceeded the time limit")
        results = parse_harness_output(res.stdout)
        if results is None:
            kind = "oom" if res.oom_killed else "crash"
            return CaseScore(False, {"tests_passed": 0.0}, kind, (res.stderr or res.stdout)[-300:] or f"exit code {res.exit_code}")
        tests = [r for r in results if r["name"].startswith("t")]
        load_fail = next((r for r in results if r["name"] == "load"), None)
        if load_fail:
            return CaseScore(False, {"tests_passed": 0.0}, load_fail["kind"], load_fail["error"])
        passed = sum(1 for r in tests if r["ok"])
        if passed == len(task.tests):
            return CaseScore(True, {"tests_passed": 1.0})
        first = next(r for r in tests if not r["ok"])
        return CaseScore(False, {"tests_passed": passed / len(task.tests)}, first["kind"], f"{passed}/{len(task.tests)} tests passed; {first['error']}")

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        fails: dict[str, int] = {}
        for r in records:
            if not r["passed"]:
                fails[r["failure_type"] or "unknown"] = fails.get(r["failure_type"] or "unknown", 0) + 1
        tp = [r["components"].get("tests_passed", 0.0) for r in records]
        return {"mean_tests_passed": sum(tp) / len(tp) if tp else None, "failures_by_type": fails}
