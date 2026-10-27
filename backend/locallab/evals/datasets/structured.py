"""Structured-output suite v1: extract facts from text into JSON that must match a JSON Schema.

Primary metric = schema compliance (strict parse of the whole response, then Draft 2020-12
validation). Also reports extraction accuracy on the fields the source text determines, so a model
that emits valid-but-wrong JSON is visible. Cases are generated from seeded templates (stable ids).
"""
import json
import random
from typing import Any

from jsonschema import Draft202012Validator

from locallab.evals.base import CaseScore, EvalCase, EvalSuite

VERSION = "1"
FIRST = ["Alice", "Bruno", "Chen", "Dana", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jamal", "Keiko", "Luca"]
LAST = ["Novak", "Okafor", "Petrov", "Quinn", "Rossi", "Santos", "Tanaka", "Umar", "Vega", "Wright"]
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Cairo", "Perth", "Dublin", "Quito"]
ITEMS = ["bolt", "hinge", "gasket", "valve", "bracket", "spring", "washer", "clamp"]
FOODS = ["flour", "butter", "sugar", "milk", "salt", "rice", "lentils", "basil"]
ROOMS = ["Atlas", "Borealis", "Cedar", "Delta"]
LABELS = ["backend", "ui", "billing", "infra", "docs", "security"]


def _name(r: random.Random) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def person(r: random.Random):
    name, age, city, zip_ = _name(r), r.randint(18, 90), r.choice(CITIES), f"{r.randint(10000, 99999)}"
    interests = r.sample(["chess", "climbing", "pottery", "sailing", "baking"], r.randint(1, 3))
    email = f"{name.split()[0].lower()}@example.org" if r.random() < 0.5 else None
    text = (f"{name} is {age} years old and lives in {city} (postal code {zip_}). Interests: {', '.join(interests)}. "
            + (f"Contact: {email}." if email else "No email address is on file."))
    schema = {"type": "object", "additionalProperties": False, "required": ["name", "age", "address", "tags"], "properties": {
        "name": {"type": "string", "minLength": 1}, "age": {"type": "integer", "minimum": 0, "maximum": 130},
        "email": {"type": "string", "pattern": "^[^@\\s]+@[^@\\s]+$"},
        "address": {"type": "object", "additionalProperties": False, "required": ["city", "zip"],
                    "properties": {"city": {"type": "string"}, "zip": {"type": "string", "pattern": "^[0-9]{5}$"}}},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}}}
    ref = {"name": name, "age": age, **({"email": email} if email else {}), "address": {"city": city, "zip": zip_}, "tags": interests}
    paths = ["name", "age", "address.city", "address.zip", "tags.0"] + (["email"] if email else [])
    return schema, text, ref, paths, ["optional email omitted when absent"]


def order(r: random.Random):
    n = r.randint(1, 3)
    its = [{"sku": f"{''.join(r.choice('ABCDEFGH') for _ in range(3))}-{r.randint(100, 999)}", "qty": r.randint(1, 20), "price": round(r.uniform(1, 90), 2)} for _ in range(n)]
    status = r.choice(["pending", "shipped", "delivered", "cancelled"])
    total = round(sum(i["qty"] * i["price"] for i in its), 2)
    lines = "; ".join(f"{i['qty']} x {r.choice(ITEMS)} (SKU {i['sku']}) at {i['price']:.2f} each" for i in its)
    text = f"Order lines: {lines}. Current status: {status}. Order total: {total:.2f}."
    schema = {"type": "object", "additionalProperties": False, "required": ["items", "status", "total"], "properties": {
        "items": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["sku", "qty", "price"],
                  "properties": {"sku": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{3}$"}, "qty": {"type": "integer", "minimum": 1, "maximum": 100},
                                 "price": {"type": "number", "exclusiveMinimum": 0}}}},
        "status": {"enum": ["pending", "shipped", "delivered", "cancelled"]}, "total": {"type": "number", "minimum": 0}}}
    ref = {"items": its, "status": status, "total": total}
    paths = ["status", "total"] + [f"items.{k}.{f}" for k in range(len(its)) for f in ("sku", "qty", "price")]
    return schema, text, ref, paths, ["enum", "nested array of objects", "patterns"]


def event(r: random.Random):
    date = f"20{r.randint(25, 29)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
    people = [(_name(r), r.choice(["yes", "no", "maybe"])) for _ in range(r.randint(1, 3))]
    room, floor = r.choice(ROOMS), r.randint(1, 9)
    has_loc = r.random() < 0.6
    text = (f"Quarterly planning takes place on {date}. RSVPs: " + "; ".join(f"{n} replied {v}" for n, v in people) + ". "
            + (f"It is held in room {room} on floor {floor}." if has_loc else "The location has not been decided."))
    schema = {"type": "object", "additionalProperties": False, "required": ["title", "date", "attendees"], "properties": {
        "title": {"type": "string"}, "date": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"},
        "attendees": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["name", "rsvp"],
                      "properties": {"name": {"type": "string"}, "rsvp": {"enum": ["yes", "no", "maybe"]}}}},
        "location": {"type": "object", "additionalProperties": False, "required": ["room", "floor"], "properties": {"room": {"type": "string"}, "floor": {"type": "integer"}}}}}
    ref = {"title": "Quarterly planning", "date": date, "attendees": [{"name": n, "rsvp": v} for n, v in people],
           **({"location": {"room": room, "floor": floor}} if has_loc else {})}
    paths = ["date"] + [f"attendees.{k}.{f}" for k in range(len(people)) for f in ("name", "rsvp")] + (["location.room", "location.floor"] if has_loc else [])
    return schema, text, ref, paths, ["optional nested object", "date pattern"]


def sensor(r: random.Random):
    unit = r.choice(["C", "F", "K"])
    vals = [round(r.uniform(10, 300), 1) for _ in range(3)]
    dev = f"DEV-{r.randint(100, 999)}"
    fw = f"v{r.randint(1, 3)}.{r.randint(0, 9)}.{r.randint(0, 9)}" if r.random() < 0.5 else None
    text = f"Device {dev} reported three readings: {vals[0]}, {vals[1]} and {vals[2]} (unit: {unit})." + (f" Firmware {fw}." if fw else "")
    schema = {"type": "object", "additionalProperties": False, "required": ["device", "readings", "unit"], "properties": {
        "device": {"type": "object", "additionalProperties": False, "required": ["id"], "properties": {"id": {"type": "string", "pattern": "^DEV-[0-9]{3}$"}, "firmware": {"type": "string", "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$"}}},
        "readings": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "unit": {"enum": ["C", "F", "K"]}}}
    ref = {"device": {"id": dev, **({"firmware": fw} if fw else {})}, "readings": vals, "unit": unit}
    paths = ["device.id", "unit", "readings.0", "readings.1", "readings.2"] + (["device.firmware"] if fw else [])
    return schema, text, ref, paths, ["fixed-size array", "optional field in nested object"]


def recipe(r: random.Random):
    ings = [(r.choice(FOODS), r.choice([50, 100, 250, 2, 3]), r.choice(["g", "ml", "tsp", "tbsp", "cup", "piece"])) for _ in range(r.randint(2, 3))]
    ings = list({i[0]: i for i in ings}.values())
    diff, prep = r.choice(["easy", "medium", "hard"]), r.randint(5, 90)
    text = ("Ingredients: " + ", ".join(f"{a} {u} {n}" for n, a, u in ings) + f". Steps: 1) combine everything, 2) cook gently. Difficulty: {diff}. Preparation takes {prep} minutes.")
    schema = {"type": "object", "additionalProperties": False, "required": ["ingredients", "steps", "difficulty", "prep_minutes"], "properties": {
        "ingredients": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["name", "amount", "unit"],
                        "properties": {"name": {"type": "string"}, "amount": {"type": "number", "exclusiveMinimum": 0}, "unit": {"enum": ["g", "ml", "tsp", "tbsp", "cup", "piece"]}}}},
        "steps": {"type": "array", "items": {"type": "string"}, "minItems": 2}, "difficulty": {"enum": ["easy", "medium", "hard"]}, "prep_minutes": {"type": "integer", "minimum": 1}}}
    ref = {"ingredients": [{"name": n, "amount": a, "unit": u} for n, a, u in ings], "steps": ["combine everything", "cook gently"], "difficulty": diff, "prep_minutes": prep}
    paths = ["difficulty", "prep_minutes"] + [f"ingredients.{k}.{f}" for k in range(len(ings)) for f in ("name", "amount", "unit")]
    return schema, text, ref, paths, ["enums", "min items"]


def ticket(r: random.Random):
    prio = r.choice(["low", "medium", "high", "critical"])
    assignee = _name(r) if r.random() < 0.5 else None
    labels = r.sample(LABELS, r.randint(1, 3))
    val, unit = r.randint(1, 12), r.choice(["hours", "days"])
    text = (f"Priority {prio}. " + (f"Assigned to {assignee}. " if assignee else "Nobody is assigned yet. ") + f"Labels: {', '.join(labels)}. Estimated effort: {val} {unit}.")
    schema = {"type": "object", "additionalProperties": False, "required": ["priority", "assignee", "labels", "estimate"], "properties": {
        "priority": {"enum": ["low", "medium", "high", "critical"]}, "assignee": {"type": ["string", "null"]},
        "labels": {"type": "array", "items": {"type": "string"}, "uniqueItems": True, "minItems": 1},
        "estimate": {"type": "object", "additionalProperties": False, "required": ["value", "unit"], "properties": {"value": {"type": "integer", "minimum": 1}, "unit": {"enum": ["hours", "days"]}}}}}
    ref = {"priority": prio, "assignee": assignee, "labels": labels, "estimate": {"value": val, "unit": unit}}
    paths = ["priority", "assignee", "estimate.value", "estimate.unit"] + [f"labels.{k}" for k in range(len(labels))]
    return schema, text, ref, paths, ["nullable field", "unique items"]


TEMPLATES = {"person": person, "order": order, "event": event, "sensor": sensor, "recipe": recipe, "ticket": ticket}
VARIANTS = 4  # variant 0 is the dev split; 1-3 are held out
INSTRUCTION = ("Extract the information from the text into a JSON object that conforms to the JSON Schema below. "
               "Omit optional properties that the text does not mention. Respond with only the JSON object.")


def _lookup(obj: Any, dotted: str) -> tuple[bool, Any]:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) < 1e-6
    return a == b


def parse_strict(response: str) -> tuple[Any, str | None]:
    """The whole response must be one JSON object (a single surrounding markdown fence is tolerated)."""
    t = response.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            t = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(t)
    except ValueError as e:
        return None, f"invalid JSON: {e}"
    return (obj, None) if isinstance(obj, dict) else (None, "top-level value is not an object")


def parse_lenient(response: str) -> Any:
    i = response.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(response[i:])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


class StructuredSuite(EvalSuite):
    name, version, title, primary_metric = "structured_json", VERSION, "Structured JSON output", "schema_compliance_rate"

    def cases(self, split: str, **params: Any) -> list[EvalCase]:
        out = []
        for ti, (tname, fn) in enumerate(TEMPLATES.items()):
            for v in range(VARIANTS):
                sp = "dev" if v == 0 else "heldout"
                if sp != split:
                    continue
                schema, text, ref, paths, notes = fn(random.Random(f"json-v{VERSION}-{tname}-{v}"))
                expected = {pth: _lookup(ref, pth)[1] for pth in paths}
                prompt = f"{INSTRUCTION}\n\nJSON Schema:\n{json.dumps(schema, indent=1)}\n\nText:\n{text}"
                out.append(EvalCase(id=f"json-v{VERSION}-{sp}-{tname}-{v}", suite=self.name, version=VERSION, split=sp,
                                    messages=[{"role": "user", "content": prompt}], max_tokens=512,
                                    meta={"schema": schema, "expected": expected, "reference": ref}, tags={"template": tname, "features": notes}))
        return out

    def score(self, case: EvalCase, response: str, sandbox: Any = None) -> CaseScore:
        obj, err = parse_strict(response)
        lenient = parse_lenient(response) if obj is None else obj
        comps: dict[str, float | bool] = {"parse": obj is not None, "lenient_parse": lenient is not None, "schema_valid": False, "extraction": 0.0}
        exp = case.meta["expected"]
        if obj is not None:
            comps["extraction"] = sum(1 for k, v in exp.items() if (lambda f, x: f and _same(x, v))(*_lookup(obj, k))) / len(exp)
        if obj is None:
            return CaseScore(False, comps, "extra_text_or_invalid_json" if lenient is not None else "no_json", err)
        errors = sorted(Draft202012Validator(case.meta["schema"]).iter_errors(obj), key=lambda e: list(e.path))
        if errors:
            e = errors[0]
            return CaseScore(False, comps, "schema_violation", f"{'.'.join(map(str, e.path)) or '<root>'}: {e.message[:200]} [{e.validator}]")
        comps["schema_valid"] = True
        return CaseScore(True, comps)

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(records)
        c = lambda k: sum(1 for r in records if r["components"].get(k)) / n if n else None  # noqa: E731
        fails: dict[str, int] = {}
        kw: dict[str, int] = {}
        for r in records:
            if not r["passed"]:
                fails[r["failure_type"] or "unknown"] = fails.get(r["failure_type"] or "unknown", 0) + 1
                if r["failure_type"] == "schema_violation" and r.get("detail") and "[" in r["detail"]:
                    k = r["detail"].rsplit("[", 1)[1].rstrip("]")
                    kw[k] = kw.get(k, 0) + 1
        by_t: dict[str, list[float]] = {}
        for r in records:
            by_t.setdefault(r["tags"]["template"], []).append(1.0 if r["passed"] else 0.0)
        ext = [r["components"].get("extraction", 0.0) for r in records]
        return {"parse_rate": c("parse"), "lenient_parse_rate": c("lenient_parse"), "schema_valid_rate": c("schema_valid"),
                "extraction_accuracy": sum(ext) / n if n else None, "failures_by_type": fails, "violations_by_keyword": kw,
                "by_template": {k: sum(v) / len(v) for k, v in by_t.items()}}
