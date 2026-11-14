"""Frozen, local Fennec structured-reasoning benchmark (stdlib only).

Never executes model output, starts a server, or promotes/deletes a model.
Use --manifest-only --out manifest.json for an entirely offline validation.
Compare completed runs with --compare original.json candidate.json.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path

SUITE_VERSION = "fennec-holdout-1.0.0"
REPORT_VERSION = 1
DISCLAIMER = (
    "Measures structured reasoning and Python code tracing, NOT sandboxed program "
    "execution or proof of the best model generally. Frozen independent fixtures "
    "use families disjoint from the old training bank; prior model exposure cannot "
    "be ruled out. Treat subsequent training on this suite as contamination."
)
SYSTEM_PROMPT = (
    "Return exactly one JSON value answering the task. No Markdown, code fences, "
    "explanation, or additional text. Use JSON null/true/false, not Python literals. "
    "Object key order is irrelevant; array order and all value types (including "
    "booleans versus integers and integers versus floats) must match exactly."
)
FAMILIES = (
    "json_merge_patch", "semver_prerelease", "ipv4_cidr", "iso_calendar",
    "utf8_base64", "regex_fullmatch", "sql_group_null_distinct",
    "python_closure_iterator", "utc_timezone", "decimal_half_even",
    "levenshtein", "relational_division",
)
OLD_FAMILIES = frozenset({
    "two_sum", "merge_intervals", "csv", "path_simplification", "lis",
    "word_break", "arithmetic_parser", "rotate_matrix", "topo_sort", "min_window",
    "lru", "rle", "sorted_matrix_kth", "json_pointer", "atoi", "anagrams",
    "palindrome", "coin_change", "brackets", "flatten",
})

# These are the entire frozen task bank. Do not change without a version bump.
MERGE_FIXTURES = (
    ({"a": "b", "c": {"d": "e", "f": "g"}}, {"a": "z", "c": {"f": None}}),
    ({"a": [1, 2], "keep": True}, {"a": [3], "missing": None}),
    ([1, 2], {"new": {"gone": None, "v": 0}}),
    ({"a": 1}, None),
    ({"x": 1, "nested": {"a": 1}}, {"x": {}, "nested": {}, "empty": {}}),
)
SEMVER_FIXTURES = (
    (("1.0.0-alpha", "1.0.0"), ("1.0.0-alpha", "1.0.0-alpha.1"),
     ("1.0.0-alpha.1", "1.0.0-alpha.beta")),
    (("1.0.0-beta.2", "1.0.0-beta.11"), ("1.0.0-2", "1.0.0-10"),
     ("1.0.0-9", "1.0.0-A")),
    (("1.2.3+build.9", "1.2.3+build.10"),
     ("1.0.0-rc.1+x", "1.0.0-rc.1+y"), ("1.0.1-alpha", "1.0.0")),
    (("1.0.0-B", "1.0.0-a"), ("1.0.0-alpha-1", "1.0.0-alpha.1"),
     ("1.0.0-1", "1.0.0-01a")),
    (("2.0.0-alpha", "1.99.99"), ("1.10.0", "1.9.9"), ("0.0.0", "0.0.0")),
)
CIDR_FIXTURES = (
    ("192.168.1.0/24", ("192.168.1.0", "192.168.1.255", "192.168.2.0")),
    ("0.0.0.0/0", ("0.0.0.0", "255.255.255.255", "8.8.8.8")),
    ("203.0.113.7/32", ("203.0.113.7", "203.0.113.6", "203.0.113.8")),
    ("198.51.100.10/31", ("198.51.100.10", "198.51.100.11", "198.51.100.12")),
    ("10.42.3.9/8", ("10.0.0.0", "10.255.255.255", "11.0.0.0", "9.255.255.255")),
)
CALENDAR_FIXTURES = (
    ("2024-02-28", 2, "2024-03-01"),
    ("2021-01-01", -1, "2020-12-31"),
    ("2019-12-30", 7, "2020-01-01"),
    ("1900-02-28", 1, "1900-03-01"),
    ("2000-02-29", 366, "2001-02-28"),
)
ENCODING_FIXTURES = ("Colibri", "", "caf\u00e9", "A\U0001f600\n", "e\u0301")
REGEX_FIXTURES = (
    (r"[A-Z]{2}\d{2}", ("AB12", "Ab12", "AB123", "AB\u0661\u0662", "AB12\n")),
    (r"a(?:b|c)*", ("a", "abcb", "", "acbd", "a\n")),
    (r"(?:0|[1-9][0-9]*)(?:\.[0-9]{2})?", ("0", "01", "12.30", "12.3", "-1", "0.00")),
    (r"(?P<word>[a-z]+)-(?P=word)", ("go-go", "go-stop", "a-a", "-", "go-go\n")),
    (r"[^\W_]+(?:_[^\W_]+)*", ("na\u00efve_2", "_x", "a__b", "\u732b", "")),
)
SQL_SCHEMA = "CREATE TABLE sales (dept TEXT, who TEXT, amount INTEGER)"
SQL_ROWS = (
    ("a", "ann", 10), ("a", "ann", None), ("a", "bob", 10),
    ("b", None, 5), ("b", "bea", None), ("c", None, None), ("a", "ann", 10),
)
SQL_FIXTURES = (
    (SQL_ROWS, "SELECT dept, COUNT(*), COUNT(amount), SUM(amount) FROM sales GROUP BY dept ORDER BY dept"),
    (SQL_ROWS, "SELECT dept, COUNT(DISTINCT who), COUNT(DISTINCT amount) FROM sales GROUP BY dept ORDER BY dept"),
    (SQL_ROWS, "SELECT dept, COALESCE(SUM(amount), 0) FROM sales GROUP BY dept HAVING COUNT(amount) = 0 ORDER BY dept"),
    ((("x", None, None), ("x", None, None), ("x", "u", 0), ("y", "u", 0)),
     "SELECT DISTINCT who, amount FROM sales ORDER BY who, amount"),
    ((), "SELECT COUNT(*), COUNT(amount), SUM(amount), COUNT(DISTINCT who) FROM sales"),
)
# Authored outcomes: code below is prompt text ONLY, never evaluated/executed.
TRACE_FIXTURES = (
    ("funcs = [lambda: i for i in range(3)]\nresult = [f() for f in funcs]", [2, 2, 2]),
    ("funcs = [lambda x, i=i: x + i for i in range(3)]\nresult = [f(10) for f in funcs]", [10, 11, 12]),
    ("it = iter([1, 2, 3, 4])\na = next(it)\nb = list(zip(it, it))\n"
     "result = [a, [list(pair) for pair in b], list(it)]", [1, [[2, 3]], []]),
    ("def counter(n):\n    def step():\n        nonlocal n\n        n += 1\n"
     "        return n\n    return step\na = counter(0)\nb = counter(10)\n"
     "result = [a(), a(), b(), a()]", [1, 2, 11, 3]),
    ("xs = [1, 2]\ng = (x * 2 for x in xs)\nxs.append(3)\nfirst = next(g)\n"
     "xs[1] = 5\nresult = [first, list(g), list(g)]", [2, [10, 6], []]),
)
TIMEZONE_FIXTURES = (
    ("2024-03-01T00:15:00+05:30", "-08:00"),
    ("2023-12-31T23:30:00-02:00", "+14:00"),
    ("2000-01-01T00:00:00Z", "-12:00"),
    ("2024-06-01T12:00:30+05:45", "+00:00"),
    ("2024-01-01T00:10:00+14:00", "-10:00"),
)
DECIMAL_FIXTURES = (
    (("2.5", "3.5", "-2.5", "-3.5"), "1"),
    (("1.225", "1.235", "-1.225", "-1.235"), "0.01"),
    (("0.005", "-0.005", "0.015", "-0.015"), "0.01"),
    (("9.995", "99.985", "1.200"), "0.01"),
    (("145", "155", "-145", "-155"), "1E+1"),
)
LEVENSHTEIN_FIXTURES = (
    ("kitten", "sitting"), ("", "abc"), ("same", "same"),
    ("ab", "ba"), ("caf\u00e9", "cafe\u0301"),
)
DIVISION_FIXTURES = (
    (("a", "b", "c"), (("a", "red"), ("a", "blue"), ("b", "red"), ("c", "blue")), ("red", "blue"), ()),
    (("a", "b", "c"), (("a", "red"), ("a", "red"), ("b", "red"), ("b", "blocked")), ("red",), ("blocked",)),
    (("a", "b", "c"), (("a", "x"),), (), ()),
    (("a", "b", "c"), (("a", "x"), ("b", "y")), (), ("x",)),
    (("a", "b"), (("a", "r"), ("b", "r"), ("outside", "missing")), ("r", "missing"), ()),
)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _reject_constant(_text):
    raise ValueError("nonstandard JSON constant")


def parse_answer(text):
    """Strict whole-document JSON, not a code/fence extraction heuristic."""
    if not isinstance(text, str):
        raise ValueError("answer must be text")
    return json.loads(text, object_pairs_hook=_object_without_duplicates,
                      parse_float=_finite_float, parse_constant=_reject_constant)


def exact_equal(actual, expected):
    """Python's True == 1 and 1 == 1.0 are deliberately NOT accepted."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def merge_patch(target, patch):
    """RFC 7396 section 2 reference algorithm; never mutates fixtures."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def semver_compare(left, right):
    def split(version):
        core, sep, prerelease = version.split("+", 1)[0].partition("-")
        return tuple(map(int, core.split("."))), prerelease.split(".") if sep else None

    def cmp(a, b):
        return (a > b) - (a < b)

    a_core, a_pre = split(left)
    b_core, b_pre = split(right)
    if a_core != b_core:
        return cmp(a_core, b_core)
    if a_pre is None or b_pre is None:
        return cmp(a_pre is None, b_pre is None)
    for a, b in zip(a_pre, b_pre):
        if a == b:
            continue
        a_num, b_num = a.isascii() and a.isdigit(), b.isascii() and b.isdigit()
        if a_num and b_num:
            return cmp(int(a), int(b))
        if a_num != b_num:
            return -1 if a_num else 1
        return cmp(a, b)
    return cmp(len(a_pre), len(b_pre))


def calendar_reference(day, add_days, other):
    start = date.fromisoformat(day)
    iso = start.isocalendar()
    return {"shifted": (start + timedelta(days=add_days)).isoformat(),
            "days_until_other": (date.fromisoformat(other) - start).days,
            "iso_year": iso.year, "iso_week": iso.week, "iso_weekday": iso.weekday}


def sql_reference(rows, query):
    # Only the frozen schema/query constants above reach sqlite. Model text never does.
    if not any(query == q and tuple(rows) == tuple(r) for r, q in SQL_FIXTURES):
        raise ValueError("SQL reference accepts frozen fixtures only")
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(SQL_SCHEMA)
        connection.executemany("INSERT INTO sales VALUES (?, ?, ?)", rows)
        return [list(row) for row in connection.execute(query).fetchall()]
    finally:
        connection.close()


def timezone_reference(timestamp, offset):
    source = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    minutes = (int(offset[1:3]) * 60 + int(offset[4:6])) * (-1 if offset[0] == "-" else 1)
    target = timezone(timedelta(minutes=minutes))
    return {"utc": source.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "converted": source.astimezone(target).isoformat(timespec="seconds")}


def decimal_reference(values, quantum):
    with localcontext() as context:
        context.prec = 50
        return [format(Decimal(value).quantize(Decimal(quantum), rounding=ROUND_HALF_EVEN), "f")
                for value in values]


def levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def division_reference(entities, rows, required, forbidden):
    return sorted(entity for entity in set(entities)
                  if set(required) <= {tag for owner, tag in rows if owner == entity}
                  and not set(forbidden) & {tag for owner, tag in rows if owner == entity})


def build_cases():
    cases = []

    def add(family, instruction, inputs, expected):
        number = 1 + sum(case["family"] == family for case in cases)
        cases.append({"id": f"{family}.{number:02d}", "family": family,
                      "instruction": instruction, "input": inputs, "expected": expected})

    for target, patch in MERGE_FIXTURES:
        add("json_merge_patch", "Return the RFC 7396 JSON Merge Patch result. An object patch "
            "recurses into objects (a non-object target becomes {}); null-valued object members "
            "delete keys, including harmless deletion of absent keys. Arrays are replaced whole. "
            "Any non-object patch, including null, replaces the entire target.",
            {"target": target, "patch": patch}, merge_patch(target, patch))
    for pairs in SEMVER_FIXTURES:
        add("semver_prerelease", "All inputs are valid SemVer 2.0.0 versions. For each [left,right] "
            "return -1, 0, or 1 for lower, equal, or higher precedence, as an array. Compare "
            "major/minor/patch numerically first. Ignore all +build metadata. A prerelease is "
            "lower than the corresponding release. Compare dot-separated prerelease identifiers "
            "left to right: all-ASCII-digit identifiers compare numerically and are lower than "
            "nonnumeric identifiers; nonnumeric identifiers compare by ASCII lexical order, "
            "case-sensitive. If shared identifiers tie, the shorter list is lower. Numeric "
            "identifiers have no leading zeros; hyphens inside identifiers are literal.",
            {"pairs": pairs}, [semver_compare(a, b) for a, b in pairs])
    for cidr, addresses in CIDR_FIXTURES:
        network = ipaddress.IPv4Network(cidr, strict=False)
        add("ipv4_cidr", "Return one boolean per IPv4 address for membership in the CIDR. "
            "Mask any host bits in the supplied network address. Include network and broadcast "
            "addresses: this is membership, not usable-host enumeration.",
            {"cidr": cidr, "addresses": addresses},
            [ipaddress.IPv4Address(address) in network for address in addresses])
    for day, delta, other in CALENDAR_FIXTURES:
        add("iso_calendar", "Use the proleptic Gregorian calendar. Return an object with shifted "
            "(date + add_days, YYYY-MM-DD), days_until_other (other minus date in whole days), "
            "and iso_year, iso_week, iso_weekday for the ORIGINAL date. ISO weeks start Monday "
            "(weekday 1); week 1 contains January 4. Sunday is weekday 7.",
            {"date": day, "add_days": delta, "other": other}, calendar_reference(day, delta, other))
    for text in ENCODING_FIXTURES:
        encoded = text.encode("utf-8")
        add("utf8_base64", "Encode text exactly as supplied in UTF-8, without BOM or Unicode "
            "normalization. Return an object: bytes (array of unsigned decimal bytes), hex "
            "(lowercase, no separators), base64 (standard RFC 4648 alphabet, with = padding, "
            "no line wrapping). Escaped newlines represent actual newline characters.",
            {"text": text}, {"bytes": list(encoded), "hex": encoded.hex(),
                             "base64": base64.b64encode(encoded).decode("ascii")})
    for pattern, strings in REGEX_FIXTURES:
        add("regex_fullmatch", "Return a boolean array: Python 3 re.fullmatch(pattern, text) "
            "is not None for each string. Default Unicode semantics, no flags. The ENTIRE "
            "string must match, including any final newline; \\d includes Unicode decimal "
            "digits and \\w includes Unicode alphanumerics and underscore.",
            {"pattern": pattern, "strings": strings}, [re.fullmatch(pattern, s) is not None for s in strings])
    for rows, query in SQL_FIXTURES:
        add("sql_group_null_distinct", "Return the result of this trusted SQLite query as an "
            "array of row arrays, in query order. Input rows follow schema column order; JSON "
            "null is SQL NULL. COUNT(*) counts rows; COUNT(expr) and COUNT(DISTINCT expr) "
            "ignore NULL. SUM of no non-NULL values is NULL. DISTINCT result rows coalesce "
            "duplicates including NULL; SQLite ascending order puts NULL first. Do not return SQL.",
            {"schema": SQL_SCHEMA, "rows": rows, "query": query}, sql_reference(rows, query))
    for code, expected in TRACE_FIXTURES:
        add("python_closure_iterator", "Trace the following Python 3 code and return only the "
            "final value of result as JSON. Closures resolve free variables at call time; "
            "default arguments bind at definition. Iterators are consumed left-to-right, "
            "including partial consumption by zip's final unsuccessful step. Do not return code.",
            {"code": code}, copy.deepcopy(expected))
    for timestamp, offset in TIMEZONE_FIXTURES:
        add("utc_timezone", "Convert the timestamp's instant to UTC and to target_offset. "
            "Offsets are FIXED numeric UTC offsets, with no DST or named-zone rules. Return "
            "{utc, converted}, both strings with seconds and no fractional seconds. Use "
            "YYYY-MM-DDTHH:MM:SSZ for utc and YYYY-MM-DDTHH:MM:SS+HH:MM (or -HH:MM) "
            "for converted; zero target offset must be +00:00.",
            {"timestamp": timestamp, "target_offset": offset}, timezone_reference(timestamp, offset))
    for values, quantum in DECIMAL_FIXTURES:
        add("decimal_half_even", "Construct exact Decimal values from the strings. Quantize "
            "each to the exponent of quantum with ROUND_HALF_EVEN (nearest, ties to an even "
            "last retained digit), sufficient precision 50. Return an array of fixed-point "
            "strings (Python format(value, 'f')), preserving trailing fractional zeros and "
            "negative zero. 1E+1 means rounding to tens, not a step of one.",
            {"values": values, "quantum": quantum}, decimal_reference(values, quantum))
    for left, right in LEVENSHTEIN_FIXTURES:
        add("levenshtein", "Return the integer Levenshtein edit distance from left to right. "
            "Insertion, deletion and substitution each cost 1; matching costs 0. No "
            "transposition operation. Compare Unicode code points case-sensitively with no normalization.",
            {"left": left, "right": right}, levenshtein(left, right))
    for entities, rows, required, forbidden in DIVISION_FIXTURES:
        add("relational_division", "Return the lexically sorted distinct entity names from "
            "entities whose assigned tags contain EVERY required tag and NO forbidden tag. "
            "Assignments are [entity,tag] pairs; duplicates have no effect. An empty required "
            "set is vacuously satisfied, including entities with no assignments. Ignore owners "
            "not listed in entities.",
            {"entities": entities, "assignments": rows, "required": required, "forbidden": forbidden},
            division_reference(entities, rows, required, forbidden))
    # Normalize tuple constants to JSON arrays and give callers independent copies.
    return parse_answer(canonical_json(cases))


def build_manifest():
    cases = build_cases()
    counts = Counter(case["family"] for case in cases)
    if len(cases) != 60 or counts != Counter({family: 5 for family in FAMILIES}):
        raise ValueError("suite must contain 12 families of five cases")
    if len({case["id"] for case in cases}) != 60 or set(counts) & OLD_FAMILIES:
        raise ValueError("duplicate task IDs or contaminated family labels")
    return {"suite_version": SUITE_VERSION, "cases_sha256": sha256_json(cases),
            "case_count": len(cases), "family_counts": dict(sorted(counts.items())),
            "disclaimer": DISCLAIMER, "cases": cases}


def messages_for_case(case):
    # Deliberately do not serialize the case: it also contains the hidden oracle.
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["instruction"] + "\nInput JSON:\n" + canonical_json(case["input"])}]


def generation_settings(max_tokens=384, timeout=120.0):
    return {"max_tokens": max_tokens, "timeout_seconds": float(timeout),
            "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "warmup_requests": 1, "system_prompt": SYSTEM_PROMPT,
            "scoring": "strict-json-types-v1", "retry_count": 0}


def request_payload(model, messages, settings):
    return {"model": model, "messages": messages, "max_tokens": settings["max_tokens"],
            "temperature": settings["temperature"], "stream": settings["stream"],
            "chat_template_kwargs": dict(settings["chat_template_kwargs"])}


def validate_base(base):
    parts = urllib.parse.urlsplit(base)
    if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        raise ValueError("base must be an HTTP(S) API URL without credentials, query, or fragment")
    try:
        parts.port
    except ValueError as error:
        raise ValueError("invalid endpoint port") from error
    if parts.path.rstrip("/") != "/v1":
        raise ValueError("base API path must be /v1")
    return base.rstrip("/")


def is_loopback(base):
    hostname = urllib.parse.urlsplit(base).hostname
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def resolve_api_key(base, env_name):
    key = os.environ.get(env_name)
    if key:
        return key
    if is_loopback(base):
        return "local"
    raise ValueError("non-loopback endpoints require an API key in the specified environment variable")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirects disabled", headers, fp)


class RequestFailure(Exception):
    def __init__(self, status, detail, http_status=None, raw_response=None):
        super().__init__(detail)
        self.status = status
        self.http_status = http_status
        self.raw_response = raw_response


class HttpClient:
    """Only the explicitly selected API; no redirects, proxies, retries or discovery."""
    def __init__(self, base, api_key, timeout):
        self.base = validate_base(base)
        self.api_key = api_key
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, route, payload=None):
        if route not in ("/models", "/chat/completions"):
            raise ValueError("unsupported API route")
        body = None if payload is None else canonical_json(payload).encode("utf-8")
        request = urllib.request.Request(self.base + route, data=body, headers={
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                headers = {key: response.headers[key] for key in ("Server", "X-Powered-By")
                           if key in response.headers}
        except urllib.error.HTTPError as error:
            try:
                raw = error.read().decode("utf-8", errors="replace")
            except (OSError, ValueError):
                raw = None
            raise RequestFailure("http_error", f"HTTP {error.code}", error.code, raw) from error
        except (TimeoutError, socket.timeout) as error:
            raise RequestFailure("timeout", "request timed out") from error
        except urllib.error.URLError as error:
            status = "timeout" if isinstance(error.reason, (TimeoutError, socket.timeout)) else "transport_error"
            raise RequestFailure(status, str(error.reason)) from error
        except http.client.IncompleteRead as error:
            raise RequestFailure("transport_error", "incomplete HTTP body",
                                 raw_response=error.partial.decode("utf-8", errors="replace")) from error
        except (OSError, UnicodeError, http.client.HTTPException) as error:
            raise RequestFailure("transport_error", str(error)) from error
        try:
            parsed = parse_answer(raw)
        except (ValueError, RecursionError) as error:
            raise RequestFailure("protocol_error", "HTTP response is not strict JSON", raw_response=raw) from error
        return parsed, raw, headers


def _empty_result(task_id, family):
    return {"id": task_id, "family": family, "status": "pending", "request_failure": True,
            "passed": False, "finish_reason": None, "wall_seconds": None,
            "tokens": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "raw_answer": None, "raw_response": None, "parsed_answer": None, "error": None}


def _read_tokens(usage):
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ValueError("usage must be an object")
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("token counts must be nonnegative integers or null")
        result[key] = value
    return result


def perform_generation(client, model, messages, settings, task_id, family, expected):
    result = _empty_result(task_id, family)
    started = time.perf_counter()
    try:
        response, raw, _headers = client.request("/chat/completions", request_payload(model, messages, settings))
        result["raw_response"] = raw
        if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
            raise ValueError("expected exactly one completion choice")
        returned_model = response.get("model")
        if returned_model is not None and returned_model != model:
            raise ValueError("completion returned a different model ID")
        choice = response["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("missing assistant message")
        result["finish_reason"] = choice.get("finish_reason")
        content = choice["message"].get("content")
        result["raw_answer"] = content
        result["tokens"] = _read_tokens(response.get("usage"))
        if not isinstance(content, str):
            raise ValueError("assistant content must be a string")
        if result["finish_reason"] != "stop":
            result.update(status="incomplete_response", error="non-stop finish reason; never score truncated output")
        else:
            result["request_failure"] = False
            try:
                result["parsed_answer"] = parse_answer(content)
            except (ValueError, RecursionError) as error:
                result.update(status="invalid_json", error=str(error))
            else:
                result.update(status="ok", passed=exact_equal(result["parsed_answer"], expected))
    except RequestFailure as error:
        result.update(status=error.status, error=str(error), http_status=error.http_status,
                      raw_response=error.raw_response)
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status="protocol_error", error=str(error), request_failure=True)
    except KeyboardInterrupt:
        result.update(status="interrupted", error="user interrupted request")
    finally:
        result["wall_seconds"] = time.perf_counter() - started
    return result


def validate_model(client, model):
    started = time.perf_counter()
    record = {"status": "pending", "request_failure": True, "requested_model": model}
    try:
        response, raw, headers = client.request("/models")
        record.update(raw_response=raw, engine_headers=headers)
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            raise ValueError("/models did not return a data array")
        matches = [item for item in response["data"] if isinstance(item, dict) and item.get("id") == model]
        if len(matches) != 1:
            raise ValueError("requested model must appear exactly once in /models")
        record.update(status="ok", request_failure=False,
                      metadata={key: matches[0][key] for key in ("id", "object", "created", "owned_by") if key in matches[0]})
    except RequestFailure as error:
        record.update(status=error.status, error=str(error), http_status=error.http_status, raw_response=error.raw_response)
    except (ValueError, TypeError, KeyError) as error:
        record.update(status="model_validation_error", error=str(error))
    except KeyboardInterrupt:
        record.update(status="interrupted", error="user interrupted model validation")
    record["wall_seconds"] = time.perf_counter() - started
    return record


def model_path_metadata(model_path):
    """Hash only named configuration/tokenizer assets, never weight files."""
    if model_path is None:
        return None
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise ValueError("model-path does not exist")
    root = path if path.is_dir() else path.parent
    config_names = ("config.json", "generation_config.json", "tokenizer_config.json",
                    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
                    "preprocessor_config.json", "adapter_config.json")
    tokenizer_names = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt", "merges.txt")
    budget = 64 * 1024 * 1024
    files = []
    for name in config_names + tokenizer_names:
        asset = root / name
        if not asset.exists():
            continue
        entry = {"name": name}
        # Avoid following symlinks out to unrelated assets/weights.
        if asset.is_symlink() or not asset.is_file():
            entry["status"] = "skipped_not_regular_file"
        else:
            try:
                size = asset.stat().st_size
                entry["bytes"] = size
                limit = 4 * 1024 * 1024 if name in config_names else 32 * 1024 * 1024
                if size > limit or size > budget:
                    entry["status"] = "skipped_size_budget"
                else:
                    digest = hashlib.sha256()
                    consumed = 0
                    with asset.open("rb") as stream:
                        while True:
                            chunk = stream.read(min(1024 * 1024, limit + 1 - consumed))
                            if not chunk:
                                break
                            consumed += len(chunk)
                            if consumed > limit or consumed > budget:
                                break
                            digest.update(chunk)
                    if consumed != size or consumed > limit or consumed > budget:
                        entry["status"] = "skipped_changed_or_oversized"
                    else:
                        entry.update(status="hashed", sha256=digest.hexdigest())
                        budget -= consumed
            except OSError as error:
                entry.update(status="unreadable", error=type(error).__name__)
        files.append(entry)
    return {"path": str(path), "files": files, "weights_hashed": False,
            "hash_policy": "named assets only; config <=4 MiB, tokenizer <=32 MiB, total <=64 MiB",
            "identity_warning": "Config/tokenizer hashes do not attest weight identity or server launch settings."}


def summarize(results, selected_count):
    def score(items, total):
        return {"passed": sum(item["passed"] is True for item in items), "total": total,
                "recorded": len(items), "request_failures": sum(item["request_failure"] is True for item in items),
                "accuracy": sum(item["passed"] is True for item in items) / total if total else 0.0}
    # Runs select a prefix of the frozen bank. Missing/interrupted tasks remain
    # in their family denominators rather than inflating partial-run accuracy.
    counts = Counter(FAMILIES[index // 5] for index in range(selected_count))
    walls = [item["wall_seconds"] for item in results if isinstance(item.get("wall_seconds"), (int, float))]
    rates = [item["tokens"]["completion_tokens"] / item["wall_seconds"] for item in results
             if item.get("wall_seconds", 0) and item["wall_seconds"] > 0
             and item["tokens"]["completion_tokens"] is not None and not item["request_failure"]]
    return {"aggregate": score(results, selected_count),
            "per_family": {family: score([r for r in results if r["family"] == family], count)
                           for family, count in sorted(counts.items())},
            "median_wall_seconds": statistics.median(walls) if walls else None,
            "median_completion_tokens_per_wall_second": statistics.median(rates) if rates else None,
            "speed_note": "Request wall time includes prompt processing and HTTP; not engine decode throughput. Warmup excluded."}


class IncrementalWriter:
    """Exclusive creation, flush/fsync each snapshot; no accidental overwrite."""
    def __init__(self, out, secret=None):
        self.out = Path(out).expanduser().resolve()
        self.stream = self.out.open("x", encoding="utf-8", newline="\n")
        self.secret = secret if secret and secret != "local" else None

    def save(self, report):
        # Redact known credentials even if an untrusted server echoes them. Preserve
        # all other raw text without stripping/truncation; flag any such redaction.
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        redacted = False
        if self.secret:
            def clean(value):
                nonlocal redacted
                if isinstance(value, str):
                    variants = {self.secret, json.dumps(self.secret)[1:-1],
                                json.dumps(self.secret, ensure_ascii=False)[1:-1]}
                    for variant in sorted(variants, key=len, reverse=True):
                        if variant in value:
                            redacted = True
                            value = value.replace(variant, "[REDACTED_API_KEY]")
                    return value
                if isinstance(value, list):
                    return [clean(item) for item in value]
                if isinstance(value, dict):
                    return {clean(key): clean(item) for key, item in value.items()}
                return value
            safe = clean(report)
            if redacted:
                safe["credential_redaction_applied"] = True
                text = json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False)
        self.stream.seek(0)
        self.stream.write(text + "\n")
        self.stream.truncate()
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def run_benchmark(args):
    manifest = build_manifest()
    base = validate_base(args.base)
    if args.limit is not None and not 1 <= args.limit <= manifest["case_count"]:
        raise ValueError("limit must be between 1 and 60")
    metadata = model_path_metadata(args.model_path)
    settings = generation_settings(args.max_tokens, args.timeout)
    key = None if args.manifest_only else resolve_api_key(base, args.api_key_env)
    selected = manifest["cases"][:args.limit] if args.limit is not None else manifest["cases"]
    report = {"report_version": REPORT_VERSION, "mode": "manifest" if args.manifest_only else "benchmark",
              "run_status": "manifest_only" if args.manifest_only else "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "manifest": manifest,
              "suite_sha256": manifest["cases_sha256"], "settings": settings,
              "selection": {"kind": "full" if args.limit is None else "smoke_limited_nonconclusive",
                            "limit": args.limit, "case_ids": [case["id"] for case in selected]},
              "engine": {"endpoint": base, "requested_model": args.model, "model_path": metadata,
                         "launch_settings": "not independently attested"},
              "runtime": {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                          "platform": sys.platform},
              "model_validation": None, "warmup": None, "results": [], "summary": None}
    writer = IncrementalWriter(args.out, key)
    try:
        writer.save(report)
        if args.manifest_only:
            return report
        client = HttpClient(base, key, args.timeout)
        report["model_validation"] = validate_model(client, args.model)
        if report["model_validation"]["request_failure"]:
            report["run_status"] = "failed"
            writer.save(report)
            return report
        writer.save(report)
        warm_messages = [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": "Return exactly the JSON value null."}]
        report["warmup"] = _empty_result("warmup", "warmup")
        writer.save(report)
        report["warmup"] = perform_generation(client, args.model, warm_messages, settings, "warmup", "warmup", None)
        writer.save(report)
        if report["warmup"]["request_failure"]:
            report["run_status"] = "failed"
            writer.save(report)
            return report
        for case in selected:
            report["results"].append(_empty_result(case["id"], case["family"]))
            writer.save(report)
            result = perform_generation(client, args.model, messages_for_case(case), settings,
                                        case["id"], case["family"], case["expected"])
            report["results"][-1] = result
            report["summary"] = summarize(report["results"], len(selected))
            writer.save(report)
            if result["status"] == "interrupted":
                report["run_status"] = "interrupted"
                break
        else:
            report["run_status"] = "complete_with_failures" if any(r["request_failure"] for r in report["results"]) else "complete"
        writer.save(report)
        return report
    finally:
        writer.close()


def _validate_comparable(report):
    manifest = build_manifest()
    if report.get("report_version") != REPORT_VERSION or report.get("mode") != "benchmark":
        raise ValueError("comparison requires benchmark reports of the current format")
    if report.get("credential_redaction_applied"):
        raise ValueError("redacted reports cannot provide exact comparable evidence")
    if not exact_equal(report.get("manifest"), manifest) or report.get("suite_sha256") != manifest["cases_sha256"]:
        raise ValueError("suite/version/hash does not match the full frozen suite")
    if report.get("run_status") != "complete":
        raise ValueError("comparison requires completed runs without request failures")
    if not exact_equal(report.get("selection"), {"kind": "full", "limit": None,
                                               "case_ids": [case["id"] for case in manifest["cases"]]}):
        raise ValueError("limited/smoke or incomplete runs are nonconclusive and cannot be compared")
    settings = report.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("missing settings")
    maximum, timeout = settings.get("max_tokens"), settings.get("timeout_seconds")
    if type(maximum) is not int or maximum <= 0 or type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid generation settings")
    if not exact_equal(settings, generation_settings(maximum, timeout)):
        raise ValueError("unexpected generation/scoring settings")
    engine = report.get("engine")
    validation = report.get("model_validation")
    if not isinstance(engine, dict) or not isinstance(engine.get("requested_model"), str) or not engine["requested_model"]:
        raise ValueError("missing model identity")
    validate_base(engine.get("endpoint", ""))
    if (not isinstance(validation, dict) or validation.get("status") != "ok"
            or validation.get("request_failure") is not False
            or validation.get("requested_model") != engine["requested_model"]
            or not isinstance(validation.get("metadata"), dict)
            or validation["metadata"].get("id") != engine["requested_model"]):
        raise ValueError("requested model was not validated via /models")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 60:
        raise ValueError("comparison requires all 60 task results")
    warmup = report.get("warmup")
    if not isinstance(warmup, dict) or warmup.get("id") != "warmup":
        raise ValueError("missing excluded warmup")
    for result, case in [(warmup, {"id": "warmup", "family": "warmup", "expected": None})] + list(zip(results, manifest["cases"])):
        if not isinstance(result, dict) or result.get("id") != case["id"] or result.get("family") != case["family"]:
            raise ValueError("missing, duplicate, reordered or mismatched task IDs")
        if result.get("request_failure") is not False or result.get("finish_reason") != "stop":
            raise ValueError("request failures or incomplete responses prohibit comparison")
        wall = result.get("wall_seconds")
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            raise ValueError("invalid request timing")
        if not exact_equal(_read_tokens(result.get("tokens")), result.get("tokens")):
            raise ValueError("invalid token record")
        if not isinstance(result.get("raw_answer"), str):
            raise ValueError("missing raw answer")
        try:
            answer = parse_answer(result["raw_answer"])
        except (ValueError, RecursionError):
            expected_status, passed = "invalid_json", False
            answer = None
        else:
            expected_status, passed = "ok", exact_equal(answer, case["expected"])
        if (result.get("status") != expected_status or result.get("passed") is not passed
                or not exact_equal(result.get("parsed_answer"), answer)):
            raise ValueError("stored verdict disagrees with raw answer and trusted oracle")
    return results


def compare_reports(original, candidate):
    old = _validate_comparable(original)
    new = _validate_comparable(candidate)
    if not exact_equal(original["settings"], candidate["settings"]):
        raise ValueError("comparison requires identical full-suite generation/timing/scoring settings")
    tasks = []
    for a, b in zip(old, new):
        outcome = ("gain" if b["passed"] else "regression") if a["passed"] != b["passed"] else (
            "tie_correct" if a["passed"] else "tie_incorrect")
        tasks.append({"id": a["id"], "family": a["family"], "original_passed": a["passed"],
                      "candidate_passed": b["passed"], "outcome": outcome})
    gains = [task["id"] for task in tasks if task["outcome"] == "gain"]
    regressions = [task["id"] for task in tasks if task["outcome"] == "regression"]
    return {"suite_version": SUITE_VERSION, "suite_sha256": original["suite_sha256"],
            "gains": gains, "regressions": regressions, "ties": 60 - len(gains) - len(regressions),
            "per_task": tasks, "original": summarize(old, 60), "candidate": summarize(new, 60),
            "recommendation": "choose_candidate" if gains and not regressions else "keep_original",
            "assessment": "preliminary_not_statistical_significance",
            "policy": "Quality first: choose candidate only with at least one gain and zero regressions. "
                      "Ties or speed alone never promote. No automatic promotion or deletion.",
            "comparability_note": "Endpoint and model identity may differ; generation/scoring settings must match. "
                                  "Server launch settings and weight identity are not independently attested.",
            "disclaimer": DISCLAIMER}


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8184/v1")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="FENNEC_API_KEY")
    parser.add_argument("--model-path", help="local model/config directory; never hashes giant weights")
    parser.add_argument("--out", help="new report file (required except for --compare); refuses overwrite")
    parser.add_argument("--max-tokens", type=positive_int, default=384)
    parser.add_argument("--timeout", type=positive_float, default=120.0, help="per-request timeout in seconds")
    parser.add_argument("--limit", type=positive_int, help="smoke only; always nonconclusive, even --limit 60")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("ORIGINAL", "CANDIDATE"))
    args = parser.parse_args(argv)
    if args.compare:
        if args.out or args.manifest_only or args.limit is not None or args.model or args.model_path:
            parser.error("--compare cannot be combined with run/output options")
        try:
            reports = [parse_answer(Path(path).expanduser().resolve().read_text(encoding="utf-8")) for path in args.compare]
            if not all(isinstance(report, dict) for report in reports):
                raise ValueError("reports must be JSON objects")
            print(json.dumps(compare_reports(*reports), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            print(f"Comparison refused: {error}", file=sys.stderr)
            return 2
    if not args.out:
        parser.error("--out is required")
    if not args.manifest_only and not args.model:
        parser.error("--model is required unless --manifest-only")
    try:
        report = run_benchmark(args)
    except (OSError, ValueError) as error:
        print(f"Benchmark not run: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(Path(args.out).resolve()), "run_status": report["run_status"],
                      "selection": report["selection"]["kind"], "suite_sha256": report["suite_sha256"],
                      "summary": report["summary"], "disclaimer": DISCLAIMER}, ensure_ascii=False, indent=2))
    return 0 if report["run_status"] in ("complete", "manifest_only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
