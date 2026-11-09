"""Merge an AirLLM LoRA adapter into the bf16 Qwen3.6 base, shard by shard.

LoRALinear computes y = xW^T + (x A^T) B^T * (alpha/r), so the merged weight is
W' = W + (alpha/r) * (B @ A). Adapter keys '<path>.lora_A' / '<path>.lora_B' map
to base safetensors key '<path>.weight'. We process one safetensors shard at a
time (bounded memory), copy untouched shards verbatim, and copy config/tokenizer
files so the result is a standalone HF checkpoint ready for convert_qwen36.py.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"D:\colibri-models\Qwen3.6-35B-A3B-bf16")
    ap.add_argument("--adapter", default=r"D:\colibri-models\qwen36-lora-adapter.pt")
    ap.add_argument("--out", default=r"D:\colibri-models\Qwen3.6-35B-A3B-bf16-lora")
    ap.add_argument("--scaling", type=float, default=2.0, help="alpha/r (32/16)")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        state = torch.load(args.adapter, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.adapter, map_location="cpu")

    pairs = defaultdict(dict)
    for key, tensor in state.items():
        if key.endswith(".lora_A"):
            pairs[key[:-7]]["A"] = tensor
        elif key.endswith(".lora_B"):
            pairs[key[:-7]]["B"] = tensor
    deltas = {}
    for path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            print(f"skip incomplete LoRA pair: {path}", flush=True)
            continue
        a = ab["A"].float()
        b = ab["B"].float()
        deltas[path + ".weight"] = (b @ a) * args.scaling
    print(f"{len(deltas)} merged weight deltas (scaling={args.scaling})", flush=True)

    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    by_shard = defaultdict(dict)
    for weight_key, delta in deltas.items():
        shard = weight_map.get(weight_key)
        if shard is None:
            print(f"WARNING: {weight_key} not in base index; skipping", flush=True)
            continue
        by_shard[shard][weight_key] = delta

    shard_files = sorted(set(weight_map.values()))
    for i, shard in enumerate(shard_files, 1):
        src = base / shard
        dst = out / shard
        if shard in by_shard:
            tensors = load_file(str(src))
            applied = 0
            for weight_key, delta in by_shard[shard].items():
                if weight_key in tensors:
                    w = tensors[weight_key]
                    tensors[weight_key] = (w.float() + delta).to(w.dtype)
                    applied += 1
            save_file(tensors, str(dst))
            print(f"[{i}/{len(shard_files)}] {shard}: merged {applied} tensors", flush=True)
            del tensors
        else:
            shutil.copy2(src, dst)
            print(f"[{i}/{len(shard_files)}] {shard}: copied (no LoRA)", flush=True)

    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    print(f"DONE merged checkpoint -> {out}", flush=True)


if __name__ == "__main__":
    main()