"""Streamed AirLLM LoRA training; smoke mode still loads the model and trains once."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

# The installer writes these roots to ~/.fennec/config.json; FENNEC_MODELS and
# FENNEC_CACHE win, and nothing assumes a D: drive.
MODELS = Path(os.environ.get("FENNEC_MODELS") or Path.home() / "fennec-models")
CACHE = Path(os.environ.get("FENNEC_CACHE") or Path.home() / ".cache" / "fennec")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(MODELS / "Qwen3.6-35B-A3B-bf16"))
    ap.add_argument("--shards", default=str(CACHE / "airllm-qwen36-shards"))
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/sft_coding_verified.jsonl"))
    ap.add_argument("--out", default=str(MODELS / "qwen36-lora-adapter.pt"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight-only", action="store_true", help="paths/data only; no torch, tokenizer or model imports")
    ap.add_argument("--steps", type=int, default=40, help="positive step cap (maximum 100000)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args(argv)
    for name in ("steps", "epochs", "seq_len", "rank", "alpha", "lr", "log_every"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            ap.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.steps > 100000 or args.seq_len < 2 or not 0 <= args.seed < 2**32:
        ap.error("steps must be <=100000, seq-len >=2 and seed in [0, 2**32)")
    for name in ("model", "shards", "data", "out"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    return args


def read_examples(path):
    raw = Path(path).read_bytes()
    examples = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if (not isinstance(obj, dict) or any(not isinstance(obj.get(key), str)
                or not obj[key].strip() for key in ("prompt", "completion"))):
            raise ValueError(f"{path}:{line_no}: nonempty prompt and completion required")
        examples.append(obj)
    if not examples:
        raise ValueError("dataset has no examples")
    return examples, hashlib.sha256(raw).hexdigest()


def build_records(tokenizer, examples):
    records = []
    for obj in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": obj["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        records.append({"text": prompt + obj["completion"] + (tokenizer.eos_token or ""),
                        "mask_prefix": prompt})
    return records


def encode_supervised(tokenizer, record, seq_len, device, encoder):
    """Correct upstream encode_record's seq-1 prefix clamp, then check shifted labels."""
    ids, mask, labels = encoder(tokenizer, record, seq_len, device)
    prefix = tokenizer(record["mask_prefix"], add_special_tokens=True,
                       truncation=False)["input_ids"]
    plen = len(prefix)
    if plen >= ids.shape[1]:
        raise ValueError("completion fully truncated: no supervised tokens remain")
    if ids[0, :plen].tolist() != prefix:
        raise ValueError("tokenizer prefix boundary mismatch; cannot safely mask prompt")
    labels[:, :plen] = -100
    if mask is not None:
        labels = labels.masked_fill(mask == 0, -100)
    if not (labels[:, 1:] != -100).any().item():
        raise ValueError("no supervised next-token labels after truncation")
    return ids, mask, labels


def check_paths(args):
    if not args.model.is_dir():
        raise ValueError(f"local model directory missing: {args.model}")
    metadata_path = Path(str(args.out) + ".metadata.json")
    if args.out.exists() or metadata_path.exists():
        raise ValueError(f"refusing to overwrite adapter or metadata: {args.out}")
    if not args.out.parent.is_dir():
        raise ValueError(f"adapter parent directory missing: {args.out.parent}")
    return metadata_path


def main(argv=None):
    args = parse_args(argv)
    metadata_path = check_paths(args)
    examples, data_hash = read_examples(args.data)
    step_cap = 1 if args.smoke else min(args.steps, args.epochs * len(examples))
    if args.preflight_only:
        print(json.dumps({"model": str(args.model), "data": str(args.data),
                          "out": str(args.out), "examples": len(examples),
                          "steps": step_cap, "data_sha256": data_hash,
                          "token_validation": "deferred until local tokenizer is loaded"}, indent=2))
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoTokenizer
    from airllm.lora_data import encode_record, iter_train_items

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True,
                                             trust_remote_code=False)
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )
    records = build_records(tokenizer, examples)
    for i, rec in enumerate(records, 1):
        try:
            encode_supervised(tokenizer, rec, args.seq_len, "cpu", encode_record)
        except ValueError as exc:
            raise ValueError(f"example {i}: {exc}") from exc
    print(f"{len(records)} validated SFT records; at most {step_cap} steps", flush=True)

    # Reserve once before model construction. Interrupted runs cannot overwrite an
    # existing adapter; save_adapter ultimately calls torch.save and accepts a handle.
    with args.out.open("xb") as adapter_file, metadata_path.open("x", encoding="utf-8") as meta_file:
        from airllm.airllm_lora import AirLLMLoRA
        started = time.time()
        trainer = AirLLMLoRA(
            str(args.model), device="cuda:0", max_seq_len=args.seq_len,
            layer_shards_saving_path=str(args.shards), lora_r=args.rank,
            lora_alpha=args.alpha, lr=args.lr, prefetching=True)
        losses = []
        for step, epoch, rec in iter_train_items(records, args.epochs, steps=step_cap):
            ids, mask, labels = encode_supervised(tokenizer, rec, args.seq_len, trainer.device, encode_record)
            loss = float(trainer.train_step(ids, labels=labels, attention_mask=mask, verbose=args.smoke))
            if not math.isfinite(loss):
                raise ValueError(f"nonfinite training loss at step {step}")
            losses.append(loss)
            if step % args.log_every == 0 or step + 1 == step_cap:
                avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
                print(f"step {step} epoch {epoch} loss={loss:.4f} avg{args.log_every}={avg:.4f}", flush=True)
        if not losses:
            raise ValueError("training produced no steps")
        trainer.save_adapter(adapter_file)
        adapter_file.flush()
        from merge_lora_qwen36 import sha256_file
        versions = {}
        for package in ("torch", "transformers", "airllm"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        metadata = {"rank": args.rank, "alpha": args.alpha, "scaling": args.alpha / args.rank,
                    "seed": args.seed, "data_sha256": data_hash, "data_path": str(args.data),
                    "model_path": str(args.model), "sequence_length": args.seq_len,
                    "steps": len(losses), "requested_steps": args.steps, "epochs": args.epochs,
                    "lr": args.lr, "smoke": args.smoke, "examples": len(records),
                    "loss": {"final": losses[-1], "mean": sum(losses) / len(losses), "steps": losses},
                    "versions": versions, "adapter_sha256": sha256_file(args.out)}
        json.dump(metadata, meta_file, indent=2)
    print(f"DONE steps={len(losses)} final_loss={losses[-1]:.4f} adapter={args.out} "
          f"wall={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
