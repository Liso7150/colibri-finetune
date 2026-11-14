"""Offline tests only: every real network connection/DNS lookup is forbidden."""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
import socket
import tempfile
import unittest
import urllib.error
import urllib.request
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_holdout.py"
SPEC = importlib.util.spec_from_file_location("bench_holdout", MODULE_PATH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        for target in ("socket.create_connection", "socket.getaddrinfo", "socket.socket.connect", "socket.socket.connect_ex"):
            patcher = mock.patch(target, side_effect=AssertionError("network forbidden in holdout tests"))
            patcher.start()
            self.addCleanup(patcher.stop)


def completion(content, model="test-model", finish="stop", usage=True):
    result = {"model": model, "choices": [{"message": {"role": "assistant", "content": content},
                                           "finish_reason": finish}]}
    if usage:
        result["usage"] = {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}
    return result


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, route, payload=None):
        self.calls.append((route, copy.deepcopy(payload)))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response, bench.canonical_json(response), {"Server": "offline-test"}


def args_for(out, **kwargs):
    values = dict(base="http://127.0.0.1:8184/v1", model="test-model", api_key_env="HOLDOUT_TEST_KEY",
                  model_path=None, out=str(out), max_tokens=384, timeout=9.0, limit=None, manifest_only=False)
    values.update(kwargs)
    return argparse.Namespace(**values)


def scored_result(case, content=None, wall=2.0):
    raw = bench.canonical_json(case["expected"]) if content is None else content
    result = bench._empty_result(case["id"], case["family"])
    result.update(request_failure=False, finish_reason="stop", wall_seconds=wall, raw_answer=raw,
                  raw_response=bench.canonical_json(completion(raw)),
                  tokens={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48})
    try:
        result["parsed_answer"] = bench.parse_answer(raw)
    except ValueError:
        result.update(status="invalid_json", passed=False)
    else:
        result.update(status="ok", passed=bench.exact_equal(result["parsed_answer"], case["expected"]))
    return result


def full_report():
    manifest = bench.build_manifest()
    return {"report_version": bench.REPORT_VERSION, "mode": "benchmark", "run_status": "complete",
            "manifest": manifest, "suite_sha256": manifest["cases_sha256"],
            "settings": bench.generation_settings(),
            "selection": {"kind": "full", "limit": None, "case_ids": [c["id"] for c in manifest["cases"]]},
            "engine": {"requested_model": "test-model", "endpoint": "http://127.0.0.1:8184/v1"},
            "model_validation": {"status": "ok", "request_failure": False, "requested_model": "test-model",
                                 "metadata": {"id": "test-model"}},
            "warmup": scored_result({"id": "warmup", "family": "warmup", "expected": None}),
            "results": [scored_result(case) for case in manifest["cases"]]}


class SuiteTests(OfflineTestCase):
    def test_counts_ids_disjointness_and_determinism(self):
        first, second = bench.build_manifest(), bench.build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["case_count"], 60)
        self.assertEqual(len({c["id"] for c in first["cases"]}), 60)
        self.assertEqual(Counter(c["family"] for c in first["cases"]), Counter({f: 5 for f in bench.FAMILIES}))
        self.assertFalse(set(bench.FAMILIES) & bench.OLD_FAMILIES)
        for family in bench.FAMILIES:
            cases = [c for c in first["cases"] if c["family"] == family]
            self.assertEqual([c["id"] for c in cases], [f"{family}.{i:02d}" for i in range(1, 6)])
            self.assertEqual(len({bench.canonical_json(c["input"]) for c in cases}), 5)
        first["cases"][0]["input"]["target"]["a"] = "mutated"
        self.assertEqual(second, bench.build_manifest())

    def test_suite_fingerprint_is_pinned(self):
        manifest = bench.build_manifest()
        self.assertEqual(manifest["suite_version"], "fennec-holdout-1.0.0")
        self.assertEqual(manifest["cases_sha256"],
                         "4a627cfc04ca16937b76ff99cd9c6b1d3cc30368fa9ecaf05c9c1239540dcc85")
        self.assertEqual(manifest["cases_sha256"], hashlib.sha256(
            json.dumps(manifest["cases"], ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest())
        changed = copy.deepcopy(manifest["cases"])
        changed[0]["expected"] = False
        self.assertNotEqual(bench.sha256_json(changed), manifest["cases_sha256"])
        changed = copy.deepcopy(manifest["cases"])
        changed[0]["instruction"] += " changed"
        self.assertNotEqual(bench.sha256_json(changed), manifest["cases_sha256"])

    def test_all_sixty_authored_edge_oracles(self):
        # Literal independent expected values, not calls back into references.
        expected = {
            "json_merge_patch": [
                {"a": "z", "c": {"d": "e"}}, {"a": [3], "keep": True},
                {"new": {"v": 0}}, None, {"x": {}, "nested": {"a": 1}, "empty": {}},
            ],
            "semver_prerelease": [[-1, -1, -1], [-1, -1, -1], [0, 0, 1], [-1, 1, -1], [1, 1, 0]],
            "ipv4_cidr": [[True, True, False], [True, True, True], [True, False, False],
                          [True, True, False], [True, True, False, False]],
            "iso_calendar": [
                {"shifted": "2024-03-01", "days_until_other": 2, "iso_year": 2024, "iso_week": 9, "iso_weekday": 3},
                {"shifted": "2020-12-31", "days_until_other": -1, "iso_year": 2020, "iso_week": 53, "iso_weekday": 5},
                {"shifted": "2020-01-06", "days_until_other": 2, "iso_year": 2020, "iso_week": 1, "iso_weekday": 1},
                {"shifted": "1900-03-01", "days_until_other": 1, "iso_year": 1900, "iso_week": 9, "iso_weekday": 3},
                {"shifted": "2001-03-01", "days_until_other": 365, "iso_year": 2000, "iso_week": 9, "iso_weekday": 2},
            ],
            "utf8_base64": [
                {"bytes": [67, 111, 108, 105, 98, 114, 105], "hex": "436f6c69627269", "base64": "Q29saWJyaQ=="},
                {"bytes": [], "hex": "", "base64": ""},
                {"bytes": [99, 97, 102, 195, 169], "hex": "636166c3a9", "base64": "Y2Fmw6k="},
                {"bytes": [65, 240, 159, 152, 128, 10], "hex": "41f09f98800a", "base64": "QfCfmIAK"},
                {"bytes": [101, 204, 129], "hex": "65cc81", "base64": "ZcyB"},
            ],
            "regex_fullmatch": [
                [True, False, False, True, False], [True, True, False, False, False],
                [True, False, True, False, False, True], [True, False, True, False, False],
                [True, False, False, True, False],
            ],
            "sql_group_null_distinct": [
                [["a", 4, 3, 30], ["b", 2, 1, 5], ["c", 1, 0, None]],
                [["a", 2, 1], ["b", 1, 1], ["c", 0, 0]], [["c", 0]],
                [[None, None], ["u", 0]], [[0, 0, None, 0]],
            ],
            "python_closure_iterator": [[2, 2, 2], [10, 11, 12], [1, [[2, 3]], []], [1, 2, 11, 3], [2, [10, 6], []]],
            "utc_timezone": [
                {"utc": "2024-02-29T18:45:00Z", "converted": "2024-02-29T10:45:00-08:00"},
                {"utc": "2024-01-01T01:30:00Z", "converted": "2024-01-01T15:30:00+14:00"},
                {"utc": "2000-01-01T00:00:00Z", "converted": "1999-12-31T12:00:00-12:00"},
                {"utc": "2024-06-01T06:15:30Z", "converted": "2024-06-01T06:15:30+00:00"},
                {"utc": "2023-12-31T10:10:00Z", "converted": "2023-12-31T00:10:00-10:00"},
            ],
            "decimal_half_even": [
                ["2", "4", "-2", "-4"], ["1.22", "1.24", "-1.22", "-1.24"],
                ["0.00", "-0.00", "0.02", "-0.02"], ["10.00", "99.98", "1.20"],
                ["140", "160", "-140", "-160"],
            ],
            "levenshtein": [3, 3, 0, 2, 2],
            "relational_division": [["a"], ["a"], ["a", "b", "c"], ["b", "c"], []],
        }
        self.assertEqual(sum(map(len, expected.values())), 60)
        for case in bench.build_cases():
            with self.subTest(case=case["id"]):
                oracle = expected[case["family"]][int(case["id"].rsplit(".", 1)[1]) - 1]
                self.assertTrue(bench.exact_equal(case["expected"], oracle), (case["expected"], oracle))

    def test_reference_algorithms_additional_boundaries(self):
        original = {"a": {"b": 1}}
        self.assertEqual(bench.merge_patch(original, {"a": {"b": None}}), {"a": {}})
        self.assertEqual(original, {"a": {"b": 1}})
        self.assertEqual(bench.merge_patch(4, [False]), [False])
        chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
                 "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
        for a, b in zip(chain, chain[1:]):
            self.assertEqual(bench.semver_compare(a, b), -1)
            self.assertEqual(bench.semver_compare(b, a), 1)
        self.assertEqual(bench.levenshtein("", ""), 0)
        self.assertEqual(bench.levenshtein("abc", ""), 3)
        self.assertEqual(bench.division_reference(["a", "a"], [("a", "r")], ["r"], ["r"]), [])
        with self.assertRaisesRegex(ValueError, "frozen"):
            bench.sql_reference([], "DROP TABLE sales")

    def test_expected_values_never_enter_http_prompts(self):
        for case in bench.build_cases():
            sentinel_case = copy.deepcopy(case)
            sentinel_case["expected"] = "SECRET_ORACLE_SENTINEL_951"
            messages = bench.messages_for_case(sentinel_case)
            self.assertEqual(messages, bench.messages_for_case(case))
            self.assertNotIn("SECRET_ORACLE_SENTINEL_951", bench.canonical_json(messages))
            self.assertEqual(messages[0]["content"], bench.SYSTEM_PROMPT)
            self.assertIn(bench.canonical_json(case["input"]), messages[1]["content"])


class ParsingTests(OfflineTestCase):
    def test_accepts_complete_json_values_only(self):
        for text, expected in [(" null \n", None), ("false", False), ("0", 0), ('"hi"', "hi"),
                               ('{"b":[true,2],"a":null}', {"a": None, "b": [True, 2]}), ("1.5", 1.5)]:
            with self.subTest(text=text):
                self.assertTrue(bench.exact_equal(bench.parse_answer(text), expected))

    def test_rejects_code_wrappers_trailing_text_duplicates_and_nonfinite(self):
        for text in ("", "```json\n{}\n```", "Answer: {}", "{} extra", "{} {}", "True", "None",
                     "{'a':1}", '{"a":1,"a":2}', '{"x":{"a":1,"a":1}}', "NaN", "Infinity",
                     "-Infinity", "1e9999", "[NaN]", "[1,]", "__import__('os').system('never')", "\ufeffnull"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_answer(text)
        with self.assertRaises(ValueError):
            bench.parse_answer(None)

    def test_exact_comparison_is_recursive_and_type_safe(self):
        for a, b in [(True, 1), (False, 0), (1, 1.0), ([True], [1]), ({"x": False}, {"x": 0}),
                     ({"a": 1}, {"b": 1}), ([1, 2], [2, 1]), ("1", 1), (None, False)]:
            self.assertFalse(bench.exact_equal(a, b), (a, b))
        self.assertTrue(bench.exact_equal({"a": [True, 2], "b": None}, {"b": None, "a": [True, 2]}))


class RequestTests(OfflineTestCase):
    def generate(self, response, expected=None):
        return bench.perform_generation(FakeClient([response]), "test-model", [], bench.generation_settings(), "x", "test", expected)

    def test_no_think_deterministic_nonstreaming_payload(self):
        client = FakeClient([completion("true")])
        result = bench.perform_generation(client, "test-model", [{"role": "user", "content": "test"}],
                                          bench.generation_settings(), "x", "test", True)
        payload = client.calls[0][1]
        self.assertEqual(payload["temperature"], 0)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 384)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["passed"])
        self.assertEqual(result["tokens"]["completion_tokens"], 8)
        self.assertGreaterEqual(result["wall_seconds"], 0)

    def test_wrong_answer_is_not_a_request_failure(self):
        result = self.generate(completion("1"), expected=True)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["passed"])
        self.assertFalse(result["request_failure"])

    def test_invalid_json_is_quality_failure_and_raw_is_unmodified(self):
        content = "```json\nnull\n```" + " extra" * 1000
        result = self.generate(completion(content))
        self.assertEqual(result["status"], "invalid_json")
        self.assertEqual(result["raw_answer"], content)
        self.assertFalse(result["passed"])
        self.assertFalse(result["request_failure"])

    def test_truncation_and_missing_finish_never_score_even_valid_json(self):
        for finish in ("length", "content_filter", "tool_calls", None):
            result = self.generate(completion("null", finish=finish))
            self.assertEqual(result["status"], "incomplete_response")
            self.assertEqual(result["finish_reason"], finish)
            self.assertTrue(result["request_failure"])
            self.assertFalse(result["passed"])
            self.assertEqual(result["raw_answer"], "null")

    def test_protocol_failures_and_optional_usage(self):
        malformed = [[], {}, {"choices": []}, completion(None), completion("null", model="wrong-model")]
        extra = completion("null")
        extra["choices"].append(copy.deepcopy(extra["choices"][0]))
        malformed.append(extra)
        bad_tokens = completion("null")
        bad_tokens["usage"]["completion_tokens"] = True
        malformed.append(bad_tokens)
        for response in malformed:
            with self.subTest(response=response):
                result = self.generate(response)
                self.assertEqual(result["status"], "protocol_error")
                self.assertTrue(result["request_failure"])
        result = self.generate(completion("null", usage=False))
        self.assertTrue(result["passed"])
        self.assertIsNone(result["tokens"]["completion_tokens"])

    def test_failures_and_interrupts_keep_status_timing_and_raw_error(self):
        for status in ("timeout", "http_error", "transport_error", "protocol_error"):
            failure = bench.RequestFailure(status, "offline failure", 503, "entire error body")
            result = self.generate(failure)
            self.assertEqual(result["status"], status)
            self.assertTrue(result["request_failure"])
            self.assertEqual(result["raw_response"], "entire error body")
            self.assertEqual(result["http_status"], 503)
            self.assertGreaterEqual(result["wall_seconds"], 0)
        result = self.generate(KeyboardInterrupt())
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(result["request_failure"])

    def test_model_listing_must_contain_requested_model_exactly_once(self):
        for data in ([], [{"id": "other"}], [{"id": "test-model"}, {"id": "test-model"}]):
            result = bench.validate_model(FakeClient([{"data": data}]), "test-model")
            self.assertTrue(result["request_failure"])
        good = bench.validate_model(FakeClient([{"data": [{"id": "test-model", "owned_by": "local-engine"}]}]), "test-model")
        self.assertEqual(good["status"], "ok")
        self.assertEqual(good["metadata"]["id"], "test-model")
        self.assertEqual(good["engine_headers"], {"Server": "offline-test"})

    def test_loopback_key_fallback_and_endpoint_validation(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            for base in ("http://127.0.0.1:8184/v1", "http://localhost/v1", "http://[::1]/v1", "http://127.2.3.4/v1"):
                self.assertEqual(bench.resolve_api_key(base, "MISSING"), "local")
            for base in ("http://example.invalid/v1", "http://127.0.0.1.example.invalid/v1", "http://192.168.1.2/v1"):
                with self.assertRaises(ValueError):
                    bench.resolve_api_key(base, "MISSING")
        with mock.patch.dict(os.environ, {"CUSTOM_TEST_KEY": "secret-value"}):
            self.assertEqual(bench.resolve_api_key("http://example.invalid/v1", "CUSTOM_TEST_KEY"), "secret-value")
        for base in ("file:///v1", "http://u:p@localhost/v1", "http://localhost/v1?key=x",
                     "http://localhost/v1#fragment", "http://localhost:bad/v1", "http://localhost/other"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                bench.validate_base(base)

    def test_http_transport_uses_timeout_no_proxy_and_only_allowed_routes(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"data": []}'
        response.headers = {"Server": "fake"}
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(urllib.request, "build_opener", return_value=opener) as build:
            client = bench.HttpClient("http://127.0.0.1:8184/v1/", "secret", 7.0)
            parsed, raw, headers = client.request("/models")
        self.assertEqual(parsed, {"data": []})
        self.assertEqual(headers, {"Server": "fake"})
        self.assertEqual(build.call_args.args[0].proxies, {})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8184/v1/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 7.0)
        with self.assertRaises(ValueError):
            client.request("/arbitrary")
        redirect = bench.NoRedirect()
        with self.assertRaises(urllib.error.HTTPError):
            redirect.redirect_request(request, None, 302, "redirect", {}, "http://example.invalid/v1/models")

    def test_http_exceptions_and_malformed_response_are_classified(self):
        errors = [
            (TimeoutError(), "timeout"),
            (urllib.error.URLError(socket.timeout()), "timeout"),
            (urllib.error.URLError("offline"), "transport_error"),
            (http.client.IncompleteRead(b"partial", 10), "transport_error"),
            (http.client.RemoteDisconnected("closed"), "transport_error"),
            (urllib.error.HTTPError("http://localhost/v1/models", 401, "denied", {}, io.BytesIO(b"full error")), "http_error"),
        ]
        for error, status in errors:
            client = bench.HttpClient("http://127.0.0.1/v1", "local", 1)
            client.opener = mock.Mock()
            client.opener.open.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(bench.RequestFailure) as caught:
                client.request("/models")
            self.assertEqual(caught.exception.status, status)
        client = bench.HttpClient("http://127.0.0.1/v1", "local", 1)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b"not JSON at all"
        response.headers = {}
        client.opener = mock.Mock()
        client.opener.open.return_value = response
        with self.assertRaises(bench.RequestFailure) as caught:
            client.request("/models")
        self.assertEqual(caught.exception.raw_response, "not JSON at all")
        self.assertEqual(caught.exception.status, "protocol_error")


class RunnerTests(OfflineTestCase):
    def test_manifest_only_is_offline_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "manifest.json"
            with mock.patch.object(bench, "HttpClient", side_effect=AssertionError("HTTP forbidden")), \
                    mock.patch.object(bench, "resolve_api_key", side_effect=AssertionError("no key needed")):
                report = bench.run_benchmark(args_for(out, manifest_only=True, model=None))
            saved = bench.parse_answer(out.read_text(encoding="utf-8"))
            self.assertEqual(report, saved)
            self.assertEqual(report["run_status"], "manifest_only")
            before = out.read_bytes()
            with self.assertRaises(FileExistsError):
                bench.run_benchmark(args_for(out, manifest_only=True))
            self.assertEqual(out.read_bytes(), before)

    def test_complete_run_warmup_excluded_incremental_saves_and_raw_answers(self):
        cases = bench.build_cases()
        responses = [{"data": [{"id": "test-model"}]}, completion("null")]
        responses.extend(completion(bench.canonical_json(case["expected"])) for case in cases)
        client = FakeClient(responses)
        snapshots = []
        original_save = bench.IncrementalWriter.save

        def capture(writer, report):
            original_save(writer, report)
            snapshots.append(bench.parse_answer(writer.out.read_text(encoding="utf-8")))

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(bench, "HttpClient", return_value=client), \
                mock.patch.object(bench.IncrementalWriter, "save", capture):
            report = bench.run_benchmark(args_for(Path(temporary) / "run.json"))
        self.assertEqual(report["run_status"], "complete")
        self.assertEqual(len(client.calls), 62)
        self.assertEqual(client.calls[0][0], "/models")
        self.assertEqual(client.calls[1][1]["messages"][-1]["content"], "Return exactly the JSON value null.")
        self.assertEqual(len(report["results"]), 60)
        self.assertEqual(report["summary"]["aggregate"]["passed"], 60)
        self.assertEqual(report["summary"]["aggregate"]["total"], 60)
        self.assertEqual(len(report["summary"]["per_family"]), 12)
        self.assertEqual(len([snapshot for snapshot in snapshots
                              if snapshot["results"] and snapshot["results"][-1]["status"] == "pending"]), 60)
        for count in range(1, 61):
            self.assertTrue(any(len(s["results"]) == count and s["results"][-1]["status"] == "ok" for s in snapshots))
        self.assertEqual(bench.compare_reports(report, report)["recommendation"], "keep_original")

    def test_model_validation_failure_prevents_warmup_and_generation(self):
        client = FakeClient([{"data": [{"id": "wrong"}]}])
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(bench, "HttpClient", return_value=client):
            out = Path(temporary) / "run.json"
            report = bench.run_benchmark(args_for(out))
            self.assertEqual(bench.parse_answer(out.read_text(encoding="utf-8"))["run_status"], "failed")
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(report["warmup"])
        self.assertEqual(report["results"], [])

    def test_warmup_failure_prevents_scored_generation(self):
        client = FakeClient([{"data": [{"id": "test-model"}]}, bench.RequestFailure("timeout", "offline timeout")])
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(bench, "HttpClient", return_value=client):
            report = bench.run_benchmark(args_for(Path(temporary) / "run.json"))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(report["run_status"], "failed")
        self.assertEqual(report["warmup"]["status"], "timeout")
        self.assertEqual(report["results"], [])

    def test_failed_task_is_saved_and_later_tasks_continue_without_retry(self):
        cases = bench.build_cases()
        client = FakeClient([{"data": [{"id": "test-model"}]}, completion("null"),
                             bench.RequestFailure("timeout", "offline timeout"),
                             completion(bench.canonical_json(cases[1]["expected"]))])
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(bench, "HttpClient", return_value=client):
            out = Path(temporary) / "run.json"
            report = bench.run_benchmark(args_for(out, limit=2))
            saved = bench.parse_answer(out.read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(report["run_status"], "complete_with_failures")
        self.assertEqual(report["selection"]["kind"], "smoke_limited_nonconclusive")
        self.assertEqual([r["status"] for r in report["results"]], ["timeout", "ok"])
        self.assertEqual(report["summary"]["aggregate"]["accuracy"], 0.5)
        self.assertEqual(report["summary"]["aggregate"]["request_failures"], 1)

    def test_interrupt_saved_without_inflating_denominators(self):
        client = FakeClient([{"data": [{"id": "test-model"}]}, completion("null"), KeyboardInterrupt()])
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(bench, "HttpClient", return_value=client):
            report = bench.run_benchmark(args_for(Path(temporary) / "run.json"))
        self.assertEqual(report["run_status"], "interrupted")
        self.assertEqual(report["results"][0]["status"], "interrupted")
        self.assertEqual(report["summary"]["aggregate"]["total"], 60)
        self.assertEqual(report["summary"]["per_family"]["json_merge_patch"]["total"], 5)
        self.assertEqual(report["summary"]["per_family"]["levenshtein"]["recorded"], 0)

    def test_no_credential_persistence_even_in_echoed_raw_json(self):
        secret = 'secret-"\\credential'
        report = {"raw_answer": secret, "raw_response": json.dumps({"echo": secret}),
                  "nested": {"error": "Bearer " + secret}}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run.json"
            writer = bench.IncrementalWriter(out, secret)
            try:
                writer.save(report)
                saved_text = out.read_text(encoding="utf-8")
                saved = bench.parse_answer(saved_text)
            finally:
                writer.close()
        self.assertNotIn(secret, saved_text)
        self.assertNotIn(json.dumps(secret)[1:-1], saved_text)
        self.assertEqual(saved["raw_answer"], "[REDACTED_API_KEY]")
        self.assertTrue(saved["credential_redaction_applied"])
        self.assertNotIn("credential", saved["raw_response"])
        self.assertEqual(report["raw_answer"], secret)

    def test_metadata_hashes_small_assets_never_weight_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_bytes(b'{"model_type":"fixture"}')
            (root / "tokenizer.json").write_bytes(b'{"vocab":{}}')
            (root / "model.safetensors").write_bytes(b"WEIGHTS_MUST_NOT_BE_READ")
            (root / "credentials.json").write_bytes(b"SECRET_MUST_NOT_BE_READ")
            real_open = Path.open
            opened = []

            def guard(path, *args, **kwargs):
                self.assertNotIn(path.name, ("model.safetensors", "credentials.json"))
                opened.append(path.name)
                return real_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guard):
                metadata = bench.model_path_metadata(root / "model.safetensors")
            self.assertEqual(set(opened), {"config.json", "tokenizer.json"})
            self.assertFalse(metadata["weights_hashed"])
            self.assertEqual(metadata["files"][0]["sha256"], hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertEqual({f["status"] for f in metadata["files"]}, {"hashed"})
            with (root / "tokenizer.model").open("wb") as stream:
                stream.truncate(33 * 1024 * 1024)
            metadata = bench.model_path_metadata(root)
            skipped = next(f for f in metadata["files"] if f["name"] == "tokenizer.model")
            self.assertEqual(skipped["status"], "skipped_size_budget")
            self.assertNotIn("sha256", skipped)
            with self.assertRaises(ValueError):
                bench.model_path_metadata(root / "missing")

    def test_cli_manifest_and_argument_validation(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = Path(temporary) / "manifest.json"
            self.assertEqual(bench.main(["--manifest-only", "--out", str(out)]), 0)
            self.assertEqual(bench.main(["--manifest-only", "--out", str(out)]), 2)
            self.assertEqual(bench.main(["--manifest-only", "--out", str(Path(temporary) / "bad.json"), "--limit", "61"]), 2)
            for argv in ([], ["--out", str(out)], ["--manifest-only", "--out", str(out), "--max-tokens", "0"],
                         ["--manifest-only", "--out", str(out), "--timeout", "nan"],
                         ["--manifest-only", "--out", str(out), "--timeout", "inf"],
                         ["--manifest-only", "--out", str(out), "--limit", "-1"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as caught:
                    bench.main(argv)
                self.assertEqual(caught.exception.code, 2)


class ComparisonTests(OfflineTestCase):
    def change_answer(self, report, index, content):
        report["results"][index] = scored_result(report["manifest"]["cases"][index], content)

    def test_ties_never_promote_even_if_candidate_faster(self):
        original, candidate = full_report(), full_report()
        for result in candidate["results"]:
            result["wall_seconds"] = 0.5
        verdict = bench.compare_reports(original, candidate)
        self.assertEqual(verdict["recommendation"], "keep_original")
        self.assertEqual(verdict["ties"], 60)
        self.assertEqual(verdict["original"]["median_wall_seconds"], 2.0)
        self.assertEqual(verdict["candidate"]["median_wall_seconds"], 0.5)
        self.assertEqual(verdict["candidate"]["median_completion_tokens_per_wall_second"], 16.0)
        self.assertEqual(verdict["assessment"], "preliminary_not_statistical_significance")

    def test_candidate_chosen_only_for_gain_and_zero_regressions(self):
        original, candidate = full_report(), full_report()
        self.change_answer(original, 0, "false")
        verdict = bench.compare_reports(original, candidate)
        self.assertEqual(verdict["recommendation"], "choose_candidate")
        self.assertEqual(verdict["gains"], ["json_merge_patch.01"])
        self.assertEqual(verdict["regressions"], [])
        self.assertEqual(verdict["per_task"][0]["outcome"], "gain")
        self.change_answer(candidate, 1, "null")
        verdict = bench.compare_reports(original, candidate)
        self.assertEqual(verdict["recommendation"], "keep_original")
        self.assertEqual(verdict["regressions"], ["json_merge_patch.02"])
        self.assertEqual(verdict["ties"], 58)

    def test_net_gain_with_any_regression_still_rejected(self):
        original, candidate = full_report(), full_report()
        self.change_answer(original, 0, "false")
        self.change_answer(original, 1, "false")
        self.change_answer(candidate, 2, "false")
        verdict = bench.compare_reports(original, candidate)
        self.assertEqual(len(verdict["gains"]), 2)
        self.assertEqual(len(verdict["regressions"]), 1)
        self.assertEqual(verdict["recommendation"], "keep_original")

    def test_invalid_json_counts_as_model_quality_failure_not_excluded_task(self):
        original, candidate = full_report(), full_report()
        self.change_answer(original, 0, "not json")
        self.change_answer(original, 1, "not json")
        self.change_answer(candidate, 1, "also not json")
        verdict = bench.compare_reports(original, candidate)
        self.assertEqual(verdict["recommendation"], "choose_candidate")
        self.assertEqual(verdict["per_task"][1]["outcome"], "tie_incorrect")
        self.assertEqual(verdict["original"]["aggregate"]["total"], 60)

    def test_different_model_ids_and_ports_allowed_but_settings_identical(self):
        original, candidate = full_report(), full_report()
        candidate["engine"].update(requested_model="candidate", endpoint="http://127.0.0.1:8185/v1")
        candidate["model_validation"].update(requested_model="candidate", metadata={"id": "candidate"})
        self.assertEqual(bench.compare_reports(original, candidate)["recommendation"], "keep_original")
        candidate["settings"]["max_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "settings"):
            bench.compare_reports(original, candidate)

    def test_rejects_suite_settings_smoke_failures_and_tampered_evidence(self):
        original = full_report()
        mutations = [
            lambda r: r.update(mode="manifest"),
            lambda r: r.update(report_version=999),
            lambda r: r.update(run_status="running"),
            lambda r: r.update(suite_sha256="0" * 64),
            lambda r: r["manifest"].update(suite_version="different"),
            lambda r: r["manifest"]["cases"][0].update(expected=False),
            lambda r: r["settings"].update(temperature=0.5),
            lambda r: r["settings"].update(timeout_seconds=999.0),
            lambda r: r["settings"].update(max_tokens=385),
            lambda r: r["settings"].update(stream=True),
            lambda r: r["settings"]["chat_template_kwargs"].update(enable_thinking=True),
            lambda r: r["selection"].update(kind="smoke_limited_nonconclusive", limit=60),
            lambda r: r["selection"]["case_ids"].pop(),
            lambda r: r["results"].pop(),
            lambda r: r["results"].append(copy.deepcopy(r["results"][0])),
            lambda r: r["results"].__setitem__(1, copy.deepcopy(r["results"][0])),
            lambda r: r["results"].reverse(),
            lambda r: r["results"][0].update(request_failure=True, status="timeout"),
            lambda r: r["results"][0].update(finish_reason="length"),
            lambda r: r["results"][0].update(raw_answer="false"),
            lambda r: r["results"][0].update(parsed_answer=False),
            lambda r: r["results"][0].update(passed=False),
            lambda r: r["results"][0].update(wall_seconds=float("nan")),
            lambda r: r["results"][0]["tokens"].update(completion_tokens=True),
            lambda r: r.update(warmup=None),
            lambda r: r["warmup"].update(request_failure=True),
            lambda r: r.update(model_validation=None),
            lambda r: r["model_validation"].update(status="http_error"),
            lambda r: r["model_validation"]["metadata"].update(id="wrong-model"),
            lambda r: r.update(credential_redaction_applied=True),
        ]
        for index, mutate in enumerate(mutations):
            candidate = copy.deepcopy(original)
            mutate(candidate)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                bench.compare_reports(original, candidate)

    def test_rehashing_a_modified_suite_does_not_make_it_valid(self):
        original, candidate = full_report(), full_report()
        candidate["manifest"]["cases"][0]["expected"] = "changed"
        fingerprint = bench.sha256_json(candidate["manifest"]["cases"])
        candidate["manifest"]["cases_sha256"] = candidate["suite_sha256"] = fingerprint
        with self.assertRaisesRegex(ValueError, "suite"):
            bench.compare_reports(original, candidate)

    def test_comparison_recomputes_scores_not_stored_summaries(self):
        original, candidate = full_report(), full_report()
        candidate["summary"] = {"aggregate": {"passed": 1000000}}
        result = bench.compare_reports(original, candidate)
        self.assertEqual(result["candidate"]["aggregate"]["passed"], 60)

    def test_compare_cli_no_network_or_automatic_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "original.json", Path(temporary) / "candidate.json"
            a.write_text(bench.canonical_json(full_report()), encoding="utf-8")
            b.write_text(bench.canonical_json(full_report()), encoding="utf-8")
            before_a, before_b = a.read_bytes(), b.read_bytes()
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(bench.main(["--compare", str(a), str(b)]), 0)
            self.assertEqual(bench.parse_answer(output.getvalue())["recommendation"], "keep_original")
            self.assertEqual(a.read_bytes(), before_a)
            self.assertEqual(b.read_bytes(), before_b)
            self.assertEqual({p.name for p in Path(temporary).iterdir()}, {"original.json", "candidate.json"})
            b.write_text("[]", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(bench.main(["--compare", str(a), str(b)]), 2)


if __name__ == "__main__":
    unittest.main()
