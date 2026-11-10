"""Merge AirLLM LoRA pairs shard by shard; validate before writing any checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("scaling must be finite and positive")
    return value


def adapter_metadata(adapter, base, scaling=None, rank=None):
    """Metadata is a hash-bound sidecar, never an implicit historical alpha/r."""
    path = Path(str(adapter) + ".metadata.json")
    if not path.exists():
        if scaling is None:
            raise ValueError("adapter has no metadata; supply explicit --scaling (alpha/r)")
        return positive_float(scaling), None
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("adapter_sha256") != sha256_file(adapter):
        raise ValueError("adapter metadata SHA256 mismatch")
    if Path(meta["model_path"]).resolve() != Path(base).resolve():
        raise ValueError("adapter metadata model_path does not match base")
    r = meta["rank"]
    if type(r) is not int or r <= 0 or (rank is not None and r != rank):
        raise ValueError("adapter metadata rank mismatch")
    expected = positive_float(meta["alpha"]) / r
    recorded = positive_float(meta["scaling"])
    if not math.isclose(expected, recorded, rel_tol=1e-9):
        raise ValueError("adapter metadata alpha/r and scaling disagree")
    if scaling is not None and not math.isclose(positive_float(scaling), expected, rel_tol=1e-9):
        raise ValueError("explicit scaling disagrees with adapter metadata")
    return expected, meta


def check_output(base, out):
    if out == base or base in out.parents or out in base.parents:
        raise ValueError("output must be separate from base (not equal or nested)")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"refusing existing nonempty output: {out}")


def merge(base, adapter, out, scaling=None):
    base, adapter, out = (Path(p).resolve() for p in (base, adapter, out))
    check_output(base, out)
    digest = sha256_file(adapter)
    scaling, metadata = adapter_metadata(adapter, base, scaling)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    # No unsafe pickle fallback on older torch versions.
    state = torch.load(str(adapter), map_location="cpu", weights_only=True)
    if sha256_file(adapter) != digest:
        raise ValueError("adapter changed while loading")
    if not isinstance(state, dict) or not state:
        raise ValueError("adapter must contain a nonempty tensor state dict")
    pairs = defaultdict(dict)
    for key, tensor in state.items():
        if not isinstance(key, str) or not key.endswith((".lora_A", ".lora_B")):
            raise ValueError(f"unknown adapter key: {key}")
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2
                or not tensor.is_floating_point() or min(tensor.shape) <= 0
                or not torch.isfinite(tensor).all().item()):
            raise ValueError(f"adapter tensor must be finite floating 2D: {key}")
        pairs[key[:-7] + ".weight"][key[-1]] = tensor
    ranks = set()
    for key, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"incomplete LoRA pair: {key}")
        if ab["A"].shape[0] != ab["B"].shape[1]:
            raise ValueError(f"A/B rank mismatch: {key}")
        ranks.add(ab["A"].shape[0])
    if len(ranks) != 1:
        raise ValueError("inconsistent adapter ranks")
    rank = ranks.pop()
    if metadata is not None and metadata["rank"] != rank:
        raise ValueError("adapter metadata rank mismatch")

    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("empty or invalid base weight_map")
    unknown = set(pairs) - set(weight_map)
    if unknown:
        raise ValueError(f"adapter keys absent from base index: {sorted(unknown)}")
    by_shard = defaultdict(set)
    for key, shard in weight_map.items():
        if (not isinstance(shard, str) or Path(shard).name != shard
                or not shard.endswith(".safetensors")
                or (base / shard).resolve().parent != base):
            raise ValueError(f"unsafe shard path: {shard}")
        by_shard[shard].add(key)
    for shard in by_shard:
        if not (base / shard).is_file():
            raise ValueError(f"missing base shard: {shard}")

    def merged_tensor(key, weight):
        if not torch.isfinite(weight).all().item():
            raise ValueError(f"nonfinite base tensor: {key}")
        if key not in pairs:
            return weight, None
        a, b = pairs[key]["A"], pairs[key]["B"]
        if not weight.is_floating_point() or tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"base shape/dtype mismatch: {key}")
        delta = (b.float() @ a.float()) * scaling
        if not torch.isfinite(delta).all().item():
            raise ValueError(f"nonfinite delta: {key}")
        result = (weight.float() + delta).to(weight.dtype)
        if not torch.isfinite(result).all().item():
            raise ValueError(f"nonfinite merged tensor: {key}")
        return result, {"shape": list(weight.shape), "rank": rank,
                        "nonzero_delta_elements": int(torch.count_nonzero(delta).item()),
                        "changed_elements_after_dtype": int(torch.count_nonzero(result != weight).item())}

    # First pass verifies every index entry and all base data, without accumulating
    # dense deltas or writing a partial checkpoint on validation/no-change failure.
    coverage = {}
    for shard, expected in sorted(by_shard.items()):
        with safe_open(str(base / shard), framework="pt", device="cpu") as source:
            if set(source.keys()) != expected:
                raise ValueError(f"base index/shard keys disagree: {shard}")
            for key in source.keys():
                result, stats = merged_tensor(key, source.get_tensor(key))
                if stats is not None:
                    coverage[key] = {"shard": shard, **stats}
                del result
    if set(coverage) != set(pairs):
        raise ValueError("incomplete applied coverage")
    changed = sum(item["changed_elements_after_dtype"] for item in coverage.values())
    if not changed:
        raise ValueError("no actual change after conversion to base dtype")

    check_output(base, out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "writing", "base": str(base), "adapter": str(adapter),
              "adapter_sha256": digest, "scaling": scaling, "rank": rank,
              "expected_pairs": len(pairs), "applied_pairs": 0,
              "coverage": coverage,
              "nonzero_delta_elements": sum(x["nonzero_delta_elements"] for x in coverage.values()),
              "changed_elements_after_dtype": changed}
    # Exclusive reservation is retained even on failure; retries cannot overwrite it.
    with (out / "merge_report.json").open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.flush()
        for shard in sorted(by_shard):
            if not (by_shard[shard] & set(pairs)):
                with (base / shard).open("rb") as src, (out / shard).open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                continue
            with safe_open(str(base / shard), framework="pt", device="cpu") as source:
                tensors = {}
                for key in source.keys():
                    result, stats = merged_tensor(key, source.get_tensor(key))
                    if stats is not None and {"shard": shard, **stats} != coverage[key]:
                        raise ValueError(f"base changed during merge: {key}")
                    tensors[key] = result
                # Reserve before safetensors writes its single-shard buffer.
                with (out / shard).open("xb"):
                    pass
                save_file(tensors, str(out / shard), metadata=source.metadata())
                del tensors, result
        for src in base.iterdir():
            if (src.is_file() and src.suffix not in {".safetensors", ".pt", ".bin"}
                    and src.name != "merge_report.json"):
                with src.open("rb") as inp, (out / src.name).open("xb") as dst:
                    shutil.copyfileobj(inp, dst)
        report.update(status="complete", applied_pairs=len(coverage))
        report_file.seek(0)
        json.dump(report, report_file, indent=2)
        report_file.truncate()
    print(f"Merged {len(coverage)} pairs; {changed} changed elements -> {out}", flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="D:/fennec-models/Qwen3.6-35B-A3B-bf16")
    ap.add_argument("--adapter", default="D:/fennec-models/qwen36-lora-adapter.pt")
    ap.add_argument("--out", default="D:/fennec-models/Qwen3.6-35B-A3B-bf16-lora")
    ap.add_argument("--scaling", type=float, help="explicit alpha/r; required for legacy adapters")
    args = ap.parse_args(argv)
    merge(args.base, args.adapter, args.out, args.scaling)


if __name__ == "__main__":
    main()
