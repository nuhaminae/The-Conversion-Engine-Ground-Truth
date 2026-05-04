# scripts/inspect_secret.py
#
# Inspects a safetensors adapter file for secrets.

import hashlib
import json
import re
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    raise SystemExit("Install safetensors first: pip install safetensors")

ROOT = Path(".")
MODEL_DIR = ROOT / "models" / "judge"
ADAPTER = MODEL_DIR / "adapter_model.safetensors"

assert ADAPTER.exists(), f"Missing file: {ADAPTER}"

SECRET_PATTERNS = {
    "OpenAI-style key": rb"sk-[A-Za-z0-9_\-]{20,}",
    "OpenRouter key/name": rb"(OPENROUTER|openrouter|or-[A-Za-z0-9_\-]{20,})",
    "HuggingFace token": rb"hf_[A-Za-z0-9]{20,}",
    "WandB key/name": rb"(WANDB|wandb|[A-Fa-f0-9]{40})",
    "Generic API key assignment": rb"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[A-Za-z0-9_\-]{12,}",
    "Private key block": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "Email address": rb"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_file_bytes(path: Path):
    data = path.read_bytes()
    findings = []
    for label, pattern in SECRET_PATTERNS.items():
        matches = re.findall(pattern, data)
        if matches:
            findings.append((label, len(matches)))
    return findings


print("=== Adapter file ===")
print("path:", ADAPTER)
print("size_mb:", round(ADAPTER.stat().st_size / (1024 * 1024), 2))
print("sha256:", sha256(ADAPTER))

print("\n=== Safetensors metadata/header ===")
with safe_open(str(ADAPTER), framework="pt", device="cpu") as f:
    metadata = f.metadata()
    keys = list(f.keys())

print("metadata:", metadata)
print("num_tensors:", len(keys))
print("first_10_tensor_names:", keys[:10])

print("\n=== Secret scan: adapter_model.safetensors ===")
adapter_findings = scan_file_bytes(ADAPTER)
if adapter_findings:
    print("POTENTIAL FINDINGS:")
    for label, count in adapter_findings:
        print(f"- {label}: {count}")
else:
    print("No obvious plain-text secret patterns found in adapter_model.safetensors.")

print("\n=== Secret scan: nearby push-candidate files ===")
push_candidates = [
    MODEL_DIR / "adapter_config.json",
    MODEL_DIR / "tokenizer_config.json",
    MODEL_DIR / "special_tokens_map.json",
    MODEL_DIR / "tokenizer.json",
    MODEL_DIR / "chat_template.jinja",
    MODEL_DIR / "README.md",
    ROOT / "models" / "model_card.md",
    ROOT / "reports" / "training" / "training_summary.json",
    ROOT / "reports" / "training" / "dataset_summary.json",
    ROOT / "reports" / "training" / "training_config_used.yaml",
    ROOT / "reports" / "training" / "training_run.log",
    ROOT / "configs" / "training_config.yaml",
    ROOT / "configs" / "eval_config.yaml",
]

any_findings = False
for path in push_candidates:
    if not path.exists():
        continue
    findings = scan_file_bytes(path)
    if findings:
        any_findings = True
        print(f"\n{path}")
        for label, count in findings:
            print(f"  - {label}: {count}")

if not any_findings:
    print("No obvious secret patterns found in nearby push-candidate files.")

print("\n=== Check for files that should NOT be pushed ===")
dangerous = [
    ROOT / ".env",
    ROOT / ".env.local",
    ROOT / ".env.production",
]
for path in dangerous:
    if path.exists():
        print("DO NOT PUSH:", path)

checkpoint_heavy_files = (
    list((ROOT / "models" / "checkpoints").glob("**/*.pt"))
    + list((ROOT / "models" / "checkpoints").glob("**/*.pth"))
    + list((ROOT / "models" / "checkpoints").glob("**/*.bin"))
)
if checkpoint_heavy_files:
    print("\nCheckpoint internals to avoid pushing unless you need resumability:")
    for path in checkpoint_heavy_files:
        print("-", path)

print("\nScreening complete.")
