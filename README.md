# Colibri Fine-tune

Streamed LoRA fine-tuning of large MoE models (Qwen3.6-35B-A3B) on a small GPU,
built on **[AirLLM](https://github.com/lyogavin/airllm)**. The frozen bf16 base is
streamed one decoder layer at a time from disk shards; LoRA A/B + AdamW live on
CUDA and autograd hidden states stay on CPU. Verified on an 8 GB RTX 4060: a full
forward+backward through 40 MoE layers peaks at **~1.9 GB VRAM**.

## Pipeline

1. **Download** the bf16 source — `scripts/download_qwen36_bf16.py`
   (`Qwen/Qwen3.6-35B-A3B` → `D:/colibri-models`, cache in `D:/.cache`).
2. **Build SFT data** — `scripts/make_sft_data.py` author coding tasks and
   *executes* each reference solution against unit tests, keeping only verified
   correct examples → `data/sft_coding_verified.jsonl`.
3. **Train** — `scripts/airllm_train_qwen36.py` (AirLLM `AirLLMLoRA`). Run
   `--smoke` first to confirm compatibility + VRAM, then a full run. Attention and
   shared-expert linears train; packed MoE experts stay frozen.
4. **Merge** — `scripts/merge_lora_qwen36.py` applies `W += (alpha/r)·(B@A)`
   shard-by-shard into a copy of the bf16 base.
5. **Convert + serve** — convert the merged checkpoint to a Colibri container
   (`coli convert --ebits 4 --gs 64`) and benchmark before/after. The original
   container is never modified; promote the candidate only if it measures better.

## Usage

```bash
python scripts/make_sft_data.py
python scripts/airllm_train_qwen36.py --smoke          # feasibility gate
python scripts/airllm_train_qwen36.py --epochs 2 --lr 5e-5 --rank 16
python scripts/merge_lora_qwen36.py --adapter D:/colibri-models/qwen36-lora-adapter.pt
```

## Credits

- **[AirLLM](https://github.com/lyogavin/airllm)** — layer-streamed inference and
  the streamed LoRA trainer this pipeline uses.
- **[Colibri](https://github.com/JustVugg/colibri)** — the inference engine that
  serves the converted container.

Part of the [Colibri suite](https://github.com/Liso7150/colibri-environment-setup).