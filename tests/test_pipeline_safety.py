"""Small CPU-only safety tests; synthetic fixtures are retained (no deletions)."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import airllm_train_qwen36 as train
import make_finetune_graph as graph
import merge_lora_qwen36 as merger
import run_finetune_eval as evaluation

import torch
from safetensors.torch import load_file, save_file

torch.set_num_threads(1)


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="_synthetic-", dir=TESTS))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.work = self.root / self._testMethodName
        self.work.mkdir()

    def json_file(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")


class MergeTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.base = self.work / "base"
        self.base.mkdir()
        self.adapter = self.work / "adapter.pt"
        self.out = self.work / "out"
        self.weights = {"layer.one.weight": torch.ones(2, 3, dtype=torch.bfloat16),
                        "layer.two.weight": torch.ones(2, 3, dtype=torch.bfloat16),
                        "untouched.weight": torch.ones(1)}
        self.index = {key: f"shard-{i}.safetensors" for i, key in enumerate(self.weights)}
        for key, tensor in self.weights.items():
            save_file({key: tensor}, str(self.base / self.index[key]), metadata={"format": "pt"})
        self.json_file(self.base / "model.safetensors.index.json", {"weight_map": self.index})
        self.json_file(self.base / "config.json", {"model_type": "synthetic"})
        self.state = {}
        for key in ("layer.one", "layer.two"):
            self.state[key + ".lora_A"] = torch.ones(1, 3)
            self.state[key + ".lora_B"] = torch.ones(2, 1)
        self.save_adapter()

    def save_adapter(self):
        torch.save(self.state, self.adapter)

    def metadata(self, **overrides):
        meta = {"rank": 1, "alpha": 0.5, "scaling": 0.5, "model_path": str(self.base),
                "adapter_sha256": merger.sha256_file(self.adapter)}
        meta.update(overrides)
        self.json_file(Path(str(self.adapter) + ".metadata.json"), meta)

    def reject(self, pattern, scaling=0.5):
        with self.assertRaisesRegex((ValueError, KeyError), pattern):
            merger.merge(self.base, self.adapter, self.out, scaling)
        self.assertFalse(self.out.exists(), "invalid inputs must fail before writes")

    def test_complete_merge_coverage_and_dtype_counts(self):
        before = {p.name: merger.sha256_file(p) for p in self.base.iterdir()}
        self.metadata()
        report = merger.merge(self.base, self.adapter, self.out)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["expected_pairs"], report["applied_pairs"])
        self.assertEqual(report["applied_pairs"], 2)
        self.assertEqual(report["nonzero_delta_elements"], 12)
        self.assertEqual(report["changed_elements_after_dtype"], 12)
        self.assertEqual(report["adapter_sha256"], merger.sha256_file(self.adapter))
        for key in ("layer.one.weight", "layer.two.weight"):
            merged = load_file(str(self.out / self.index[key]))[key]
            self.assertTrue(torch.equal(merged, torch.full((2, 3), 1.5, dtype=torch.bfloat16)))
        self.assertEqual(before, {p.name: merger.sha256_file(p) for p in self.base.iterdir()})
        untouched = self.index["untouched.weight"]
        self.assertEqual(merger.sha256_file(self.base / untouched), merger.sha256_file(self.out / untouched))

    def test_legacy_requires_explicit_scaling(self):
        self.reject("explicit --scaling", scaling=None)

    def test_missing_pair(self):
        self.state.pop("layer.one.lora_B")
        self.save_adapter()
        self.reject("incomplete")

    def test_unknown_base_key(self):
        self.state["unknown.lora_A"] = torch.ones(1, 3)
        self.state["unknown.lora_B"] = torch.ones(2, 1)
        self.save_adapter()
        self.reject("absent from base index")

    def test_unknown_adapter_key(self):
        self.state["optimizer"] = torch.ones(1)
        self.save_adapter()
        self.reject("unknown adapter key")

    def test_rank_mismatch(self):
        self.state["layer.one.lora_B"] = torch.ones(2, 2)
        self.save_adapter()
        self.reject("rank mismatch")

    def test_inconsistent_ranks(self):
        self.state["layer.one.lora_A"] = torch.ones(2, 3)
        self.state["layer.one.lora_B"] = torch.ones(2, 2)
        self.save_adapter()
        self.reject("inconsistent adapter ranks")

    def test_not_2d(self):
        self.state["layer.one.lora_A"] = torch.ones(3)
        self.save_adapter()
        self.reject("finite floating 2D")

    def test_nonfinite_adapter(self):
        self.state["layer.one.lora_A"][0, 0] = float("nan")
        self.save_adapter()
        self.reject("finite floating 2D")

    def test_nonfinite_untouched_base(self):
        save_file({"untouched.weight": torch.tensor([float("inf")])},
                  str(self.base / self.index["untouched.weight"]))
        self.reject("nonfinite base")

    def test_shape_mismatch(self):
        self.state["layer.one.lora_A"] = torch.ones(1, 4)
        self.save_adapter()
        self.reject("shape/dtype mismatch")

    def test_missing_index_tensor(self):
        save_file({"wrong.weight": torch.ones(2, 3)}, str(self.base / self.index["layer.one.weight"]))
        self.reject("index/shard keys disagree")

    def test_no_change_after_bfloat_rounding(self):
        self.reject("no actual change", scaling=1e-10)

    def test_zero_delta(self):
        for value in self.state.values():
            value.zero_()
        self.save_adapter()
        self.reject("no actual change")

    def test_nonfinite_delta(self):
        for value in self.state.values():
            value.fill_(1e30)
        self.save_adapter()
        self.reject("nonfinite delta")

    def test_metadata_hash_rank_model_and_scaling(self):
        for overrides, pattern in [({"adapter_sha256": "wrong"}, "SHA256"),
                                   ({"rank": 2, "alpha": 1}, "rank mismatch"),
                                   ({"model_path": str(self.work)}, "model_path"),
                                   ({"alpha": 1}, "alpha/r")]:
            with self.subTest(overrides=overrides):
                self.metadata(**overrides)
                self.reject(pattern)
        self.metadata()
        self.reject("explicit scaling disagrees", scaling=2)

    def test_nonpositive_scaling(self):
        for scaling in (0, -1, float("nan"), float("inf")):
            with self.subTest(scaling=scaling):
                self.reject("finite and positive", scaling=scaling)

    def test_no_unsafe_torch_load_fallback(self):
        with mock.patch.object(torch, "load", side_effect=TypeError("weights_only unavailable")) as load:
            with self.assertRaises(TypeError):
                merger.merge(self.base, self.adapter, self.out, 0.5)
        self.assertEqual(load.call_count, 1)
        self.assertIs(load.call_args.kwargs["weights_only"], True)
        self.assertFalse(self.out.exists())

    def test_refuse_base_and_existing_output(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            merger.merge(self.base, self.adapter, self.base, 0.5)
        self.out.mkdir()
        marker = self.out / "keep.txt"
        marker.write_text("unchanged", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            merger.merge(self.base, self.adapter, self.out, 0.5)
        self.assertEqual(marker.read_text(), "unchanged")


class PreflightTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.engine = self.work / "engine"
        self.models = self.work / "models"
        self.results = self.work / "results"
        for path in (self.engine / "c/tools", self.engine / "scripts", self.models):
            path.mkdir(parents=True)
        for path in (self.engine / "c/coli", self.engine / "c/tools/convert_qwen36.py",
                     self.engine / "c" / ("qwen36.exe" if os.name == "nt" else "qwen36")):
            path.write_text("synthetic", encoding="utf-8")
        self.bench = self.engine / "scripts/bench_best_model.py"
        self.bench.write_text('RESULTS = "must-not-use-engine-results"\n', encoding="utf-8")
        for name, converted in (("Qwen3.6-35B-A3B-bf16", False), ("qwen36-35b", True),
                                ("qwen36-35b-lora", True)):
            directory = self.models / name
            directory.mkdir()
            for file in ("config.json", "tokenizer.json", "qwen36_meta.json" if converted
                         else "model.safetensors.index.json"):
                self.json_file(directory / file, {})
        (self.models / "qwen36-lora-adapter.pt").write_bytes(b"synthetic-not-loaded")
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            self.port = port.getsockname()[1]
        self.argv = ["--engine-root", str(self.engine), "--models-dir", str(self.models),
                     "--results-dir", str(self.results), "--port", str(self.port)]

    def test_resolved_paths_unique_outputs_and_current_interpreter(self):
        args = evaluation.parse_args(self.argv)
        first, second = evaluation.preflight(args), evaluation.preflight(args)
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(first["merge_script"], SCRIPTS / "merge_lora_qwen36.py")
        self.assertEqual(first["bench_script"], self.bench)
        self.assertEqual(first["engine_python"], Path(sys.executable).resolve())
        self.assertFalse(self.results.exists())

    def test_env_root_and_explicit_engine_runtime(self):
        with mock.patch.dict(os.environ, {"COLIBRI_ENGINE_ROOT": str(self.engine)}):
            args = evaluation.parse_args(self.argv[2:] + ["--engine-python", sys.executable])
        self.assertEqual(evaluation.preflight(args)["engine_root"], self.engine)

    def test_preflight_subprocess_no_torch_or_model_imports(self):
        script = (
            "import sys, builtins; "
            f"sys.path.insert(0, {str(SCRIPTS)!r}); "
            "original=builtins.__import__; "
            "blocked={'torch','transformers','airllm','safetensors'}; "
            "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(AssertionError(name)) "
            "if name.split('.')[0] in blocked else original(name,*a,**k); "
            "import run_finetune_eval as m; "
            f"m.main({self.argv + ['--preflight-only', '--stages', 'merge', 'convert', 'evaluate', '--scaling', '2']!r})"
        )
        completed = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(self.results.exists())
        self.assertIn('"promotion_eligible": false', completed.stdout)

    def test_occupied_non_http_port_prevents_any_launch(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            args = self.argv + ["--port", str(busy.getsockname()[1])]
            with mock.patch.object(subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "occupied"):
                    evaluation.main(args)
                launch.assert_not_called()
        self.assertFalse(self.results.exists())

    def test_explicit_stages_do_not_reuse_existing_output(self):
        args = evaluation.parse_args(self.argv + ["--stages", "convert", "--candidate",
                                                  str(self.models / "qwen36-35b-lora")])
        with self.assertRaisesRegex(ValueError, "existing output"):
            evaluation.preflight(args)

    def test_legacy_adapter_preflight_requires_scaling(self):
        with self.assertRaisesRegex(ValueError, "explicit --scaling"):
            evaluation.preflight(evaluation.parse_args(self.argv + ["--stages", "merge"]))

    def test_benchmark_redirect_without_engine_write(self):
        self.bench.write_text('import os\nRESULTS = "bad"\nos.makedirs(RESULTS, exist_ok=True)\n'
                              'with open(os.path.join(RESULTS, "bench.json"), "w") as f:\n'
                              '    f.write("synthetic")\n', encoding="utf-8")
        exec(evaluation.bench_code(self.bench, self.results), {"__name__": "__main__"})
        self.assertTrue((self.results / "bench.json").is_file())
        self.assertFalse((self.engine / "results").exists())


class VerdictTests(unittest.TestCase):
    @staticmethod
    def result(rows):
        return {"summary": {"passed": sum(rows.values()), "total": len(rows)},
                "rows": [{"id": k, "passed": v} for k, v in rows.items()]}

    def test_tie_not_improvement_and_hidden_regression_captured(self):
        baseline = {s: self.result({"a": True, "b": False}) for s in ("hard", "base")}
        candidate = {s: self.result({"a": False, "b": True}) for s in ("hard", "base")}
        verdict = evaluation.compare_results(baseline, candidate)
        self.assertTrue(verdict["regression"])
        self.assertEqual(verdict["suites"]["base"]["per_task_regressions"], ["a"])
        self.assertFalse(verdict["promotion_eligible"])
        self.assertFalse(verdict["proven_improvement"])
        self.assertIn("Tie: no proven improvement", verdict["note"])

    def test_retention_gain_never_promotes(self):
        baseline = {s: self.result({"a": False}) for s in ("hard", "base")}
        candidate = {s: self.result({"a": True}) for s in ("hard", "base")}
        verdict = evaluation.compare_results(baseline, candidate)
        self.assertFalse(verdict["promotion_eligible"])
        self.assertEqual(verdict["recommendation"], "keep original; no change")
        self.assertIn("no promotion", graph.retention_title(verdict)[0])

    def test_missing_task_coverage_and_tie_title(self):
        result = {s: {"summary": {"passed": 1, "total": 1}} for s in ("hard", "base")}
        verdict = evaluation.compare_results(result, result)
        self.assertFalse(verdict["suites"]["hard"]["task_comparison_complete"])
        self.assertIn("no proven improvement; no change", graph.retention_title(verdict)[0])


class TrainingTests(Fixtures):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Locate the installed implementation without importing airllm.__init__.
        package = importlib.machinery.PathFinder.find_spec("airllm")
        path = Path(next(iter(package.submodule_search_locations))) / "lora_data.py"
        spec = importlib.util.spec_from_file_location("installed_lora_data", path)
        cls.upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.upstream)

    class Tokenizer:
        def __call__(self, text, return_tensors=None, truncation=False, max_length=None, **kwargs):
            ids = [ord(c) for c in text]
            if truncation:
                ids = ids[:max_length]
            if return_tensors:
                return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            return {"input_ids": ids}

    def test_actual_upstream_truncation_clamp_rejected(self):
        record = {"text": "promptanswer", "mask_prefix": "prompt"}
        _, _, upstream_labels = self.upstream.encode_record(self.Tokenizer(), record, 4, "cpu")
        self.assertTrue((upstream_labels[:, 1:] != -100).any())
        with self.assertRaisesRegex(ValueError, "fully truncated"):
            train.encode_supervised(self.Tokenizer(), record, 4, "cpu", self.upstream.encode_record)

    def test_valid_completion_masks_prompt_and_preserves_labels(self):
        record = {"text": "promptanswer", "mask_prefix": "prompt"}
        ids, _, labels = train.encode_supervised(self.Tokenizer(), record, 8, "cpu", self.upstream.encode_record)
        self.assertTrue((labels[:, :6] == -100).all())
        self.assertTrue(torch.equal(labels[:, 6:], ids[:, 6:]))

    def test_positive_parameters_and_bounded_steps(self):
        for flag, value in (("--steps", "0"), ("--steps", "100001"), ("--epochs", "-1"),
                            ("--lr", "nan"), ("--alpha", "0"), ("--rank", "0"),
                            ("--log-every", "0"), ("--seq-len", "1"), ("--seed", "-1")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.parse_args([flag, value])
        self.assertEqual(len(list(self.upstream.iter_train_items([1, 2], 100, steps=3))), 3)

    def test_empty_dataset_and_completion_rejected(self):
        path = self.work / "data.jsonl"
        for text in ("\n", '{"prompt": "p", "completion": " "}'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                train.read_examples(path)

    def test_dataset_hash_and_adapter_overwrite(self):
        path = self.work / "data.jsonl"
        path.write_text('{"prompt": "p", "completion": "a"}', encoding="utf-8")
        examples, digest = train.read_examples(path)
        self.assertEqual(len(examples), 1)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        out = self.work / "adapter.pt"
        out.write_bytes(b"keep")
        args = train.parse_args(["--model", str(self.work), "--data", str(path), "--out", str(out)])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            train.check_paths(args)
        self.assertEqual(out.read_bytes(), b"keep")

    def test_training_preflight_no_imports_or_writes(self):
        path = self.work / "data.jsonl"
        path.write_text('{"prompt": "p", "completion": "a"}', encoding="utf-8")
        out = self.work / "new.pt"
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "airllm": None}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                train.main(["--model", str(self.work), "--data", str(path), "--out", str(out), "--preflight-only"])
        self.assertFalse(out.exists())
        self.assertEqual(json.loads(output.getvalue())["steps"], 2)

    def test_graph_refuses_existing_historical_output(self):
        path = self.work / "bench_finetune_comparison.png"
        path.write_bytes(b"historical")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            graph.main(["--results-dir", str(self.work)])
        self.assertEqual(path.read_bytes(), b"historical")


if __name__ == "__main__":
    unittest.main()
