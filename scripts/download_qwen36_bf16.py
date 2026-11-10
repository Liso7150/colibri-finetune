"""Fetch the bf16 Qwen3.6-35B-A3B source used for LoRA merge/retrain.

Roots come from the environment (FENNEC_MODELS, FENNEC_CACHE) with portable
home-directory defaults; no drive letter is assumed.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = Path(os.environ.get("FENNEC_MODELS") or Path.home() / "fennec-models")
CACHE = Path(os.environ.get("FENNEC_CACHE") or Path.home() / ".cache" / "fennec")
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_ENABLE_HXF_TRANSFER", "1")

DEST = MODELS / "Qwen3.6-35B-A3B-bf16"
REPO = "Qwen/Qwen3.6-35B-A3B"
print(f"downloading {REPO} -> {DEST}", flush=True)
p = snapshot_download(
    repo_id=REPO,
    local_dir=str(DEST),
    max_workers=8,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt", "*.model"],
)
print("DONE:", p, flush=True)
