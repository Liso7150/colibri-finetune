import os, sys
os.environ.setdefault("HF_HOME", r"D:\.cache\huggingface")
os.environ.setdefault("HF_HUB_ENABLE_HXF_TRANSFER", "1")
from huggingface_hub import snapshot_download

DEST = r"D:\colibri-models\Qwen3.6-35B-A3B-bf16"
REPO = "Qwen/Qwen3.6-35B-A3B"
print(f"downloading {REPO} -> {DEST}", flush=True)
p = snapshot_download(
    repo_id=REPO,
    local_dir=DEST,
    max_workers=8,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt", "*.model"],
)
print("DONE:", p, flush=True)