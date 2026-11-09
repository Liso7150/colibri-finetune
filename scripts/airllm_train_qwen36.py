"""Streamed LoRA fine-tune of Qwen3.6-35B-A3B (MoE) via AirLLM on an 8GB GPU.

The frozen bf16 base is streamed one decoder layer at a time from disk shards;
LoRA A/B + AdamW live on CUDA, autograd hidden states stay on CPU. Attention and
shared-expert linears train; packed MoE experts stay frozen (not nn.Linear).

Run --smoke FIRST: it builds the model, runs a single train_step, and reports
loss + peak VRAM so we confirm Qwen3_5Moe forward/backward compatibility before
committing to a full run. The original Colibri container is never touched; this
only produces a LoRA adapter to merge into a copy of the bf16 weights later.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"D:\.cache\huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer

from airllm.airllm_lora import AirLLMLoRA
from airllm.lora_data import encode_record, iter_train_items


def build_records(tokenizer, data_path, seq_len):
    import json
    from pathlib import Path
    eos = tokenizer.eos_token or ""
    records = []
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        instr = obj["prompt"]
        completion = obj["completion"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}],
            tokenize=False, add_generation_prompt=True)
        text = prompt + completion + eos
        records.append({"text": text, "mask_prefix": prompt})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\colibri-models\Qwen3.6-35B-A3B-bf16")
    ap.add_argument("--shards", default=r"D:\.cache\airllm-qwen36-shards")
    ap.add_argument("--data", default=r"D:\Colibri\results\sft_coding_verified.jsonl")
    ap.add_argument("--out", default=r"D:\colibri-models\qwen36-lora-adapter.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="0 = all (epochs*records)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"loading tokenizer from {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )
        print("set Qwen3 ChatML fallback chat template", flush=True)
    records = build_records(tokenizer, args.data, args.seq_len)
    print(f"{len(records)} SFT records; seq_len={args.seq_len}", flush=True)

    print(f"building AirLLMLoRA (splits layers -> {args.shards})", flush=True)
    t0 = time.time()
    trainer = AirLLMLoRA(
        args.model,
        device="cuda:0",
        max_seq_len=args.seq_len,
        layer_shards_saving_path=args.shards,
        lora_r=args.rank,
        lora_alpha=args.alpha,
        lr=args.lr,
        prefetching=True,
    )
    print(f"model ready in {time.time()-t0:.1f}s; layers={len(trainer.layer_names)}", flush=True)

    if args.smoke:
        ids, mask, labels = encode_record(tokenizer, records[0], args.seq_len, trainer.device)
        print(f"smoke: input shape {tuple(ids.shape)}", flush=True)
        loss = trainer.train_step(ids, labels=labels, attention_mask=mask, verbose=True)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"SMOKE OK loss={loss:.4f} peak_vram={peak:.2f}GB", flush=True)
        return

    steps_cap = args.steps if args.steps > 0 else None
    losses = []
    t0 = time.time()
    for step, epoch, rec in iter_train_items(records, args.epochs, steps=steps_cap):
        ids, mask, labels = encode_record(tokenizer, rec, args.seq_len, trainer.device)
        loss = trainer.train_step(ids, labels=labels, attention_mask=mask)
        losses.append(loss)
        if step % args.log_every == 0 or steps_cap and step + 1 == steps_cap:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            elapsed = time.time() - t0
            print(f"step {step} epoch {epoch} loss={loss:.4f} avg{args.log_every}={avg:.4f} "
                  f"elapsed={elapsed:.0f}s peak_vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB",
                  flush=True)
        if (step + 1) % 50 == 0:
            trainer.save_adapter(args.out)
    trainer.save_adapter(args.out)
    print(f"DONE steps={len(losses)} final_loss={losses[-1]:.4f} "
          f"mean={sum(losses)/len(losses):.4f} adapter={args.out} "
          f"wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()