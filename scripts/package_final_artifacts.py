# scripts/package_final_artifacts.py

"""
Package final Week 11 Act V artifacts.

Creates:
  dist/Act_v_package/
  dist/Act_v_package.zip

This script packages:
  - tenacious_bench dataset splits and DPO files
  - schema and datasheet
  - model card and final LoRA adapter files
  - evaluation reports and charts
  - training summaries
  - community/reporting artifacts when present

It intentionally excludes:
  - .env files
  - checkpoint optimizer/scheduler/scaler/rng internals
  - __pycache__
  - wandb local run folders
"""

import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List


ROOT = Path(".").resolve()
DIST_DIR = ROOT / "dist"
PACKAGE_DIR = DIST_DIR / "Act_v_package"
ZIP_PATH = DIST_DIR / "Act_v_package.zip"


REQUIRED_FILES = [
    "tenacious_bench/schema.json",
    "tenacious_bench/datasheet.md",

    "tenacious_bench/train/train.jsonl",
    "tenacious_bench/dev/dev.jsonl",
    "tenacious_bench/held_out/held_out.jsonl",

    "tenacious_bench/dpo/train_dpo.jsonl",
    "tenacious_bench/dpo/dev_dpo.jsonl",
    "tenacious_bench/dpo/held_out_dpo.jsonl",

    "models/model_card.md",
    "models/judge/adapter_model.safetensors",
    "models/judge/adapter_config.json",
    "models/judge/tokenizer.json",
    "models/judge/tokenizer_config.json",
    "models/judge/special_tokens_map.json",

    "reports/training/dataset_summary.json",
    "reports/training/training_summary.json",
    "reports/training/training_config_used.yaml",
    "reports/training/training_run.log",

    "reports/evaluation/baseline/baseline_metrics.json",
    "reports/evaluation/prompted/prompted_judge_metrics.json",
    "reports/evaluation/fine_tuned/fine_tuned_judge_metrics.json",
    "reports/evaluation/ablation_results.json",
]

OPTIONAL_FILES = [
    "README.md",
    "LICENSE",

    "reports/evaluation_results.html",
    "reports/exploratory_data.html",
    "reports/training_logs.html",

    "reports/exploratory_data_summary.json",
    "reports/training/training_log_summary.json",
    "reports/evaluation/evaluation_metric_comparison.csv",
    "reports/evaluation/evaluation_metric_comparison.json",

    "reports/evaluation//baseline/baseline_confusion_matrix.png",
    "reports/evaluation/prompted/prompted_judge_confusion_matrix.png",
    "reports/fine_tuned/fine_tuned_judge_confusion_matrix.png",
    "reports/evaluation/ablation_comparison_chart.png",
    "reports/evaluation/pairwise_ablation_chart.png",
    "reports/dataset_source_mode_composition.png",
    "reports/dataset_split_sizes.png",
    "reports/training/training_loss_curve.png",
    "reports/training/training_reward_curves.png",

    "reports/evaluation/baseline/baseline_pointwise_scores.jsonl",
    "reports/evaluation/baseline/baseline_pair_scores.jsonl",
    "reports/evaluation/prompted/prompted_judge_pointwise_scores.jsonl",
    "reports/evaluation/prompted/prompted_judge_pair_scores.jsonl",
    "reports/evaluation/fine_tuned/fine_tuned_judge_pair_scores.jsonl",

    "reports/blog_post.md",
    "reports/executive_memo.md",

    "community/github_issue.md",
    "community/workshop_submission.md",

    "configs/training_config.yaml",
    "configs/eval_config.yaml",
]

SECRET_PATTERNS = {
    "Hugging Face token": r"hf_[A-Za-z0-9]{20,}",
    "OpenAI/OpenRouter style key": r"sk-[A-Za-z0-9_\-]{20,}|or-[A-Za-z0-9_\-]{20,}",
    "W&B key": r"\b[A-Fa-f0-9]{40}\b",
    "Private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "Generic key assignment": r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[A-Za-z0-9_\-]{12,}",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".json", ".jsonl", ".yaml", ".yml", ".csv",
    ".py", ".bat", ".ipynb", ".html", ".jinja",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail_if_missing_required() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        print("Missing required final artifacts:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)


def scan_text_file_for_secrets(path: Path) -> List[str]:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return []

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    findings = []
    for label, pattern in SECRET_PATTERNS.items():
        if re.search(pattern, text):
            findings.append(label)
    return findings


def fail_if_secret_findings(paths: List[Path]) -> None:
    findings: Dict[str, List[str]] = {}

    for path in paths:
        if path.exists():
            hits = scan_text_file_for_secrets(path)
            if hits:
                findings[str(path.relative_to(ROOT))] = hits

    if findings:
        print("Potential secrets found. Review before packaging:")
        for path, hits in findings.items():
            print(f"  - {path}: {', '.join(hits)}")
        raise SystemExit(1)
        #return None

def copy_file(rel_path: str, dest_root: Path) -> None:
    src = ROOT / rel_path
    if not src.exists():
        return

    dst = dest_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_artifacts() -> List[Path]:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    all_files = list(dict.fromkeys(REQUIRED_FILES + OPTIONAL_FILES))

    copied = []
    for rel_path in all_files:
        src = ROOT / rel_path
        if src.exists():
            copy_file(rel_path, PACKAGE_DIR)
            copied.append(PACKAGE_DIR / rel_path)

    # Hugging Face dataset repos use README.md as the dataset card.
    # Copy datasheet.md to tenacious_bench/README.md if no dataset README exists.
    dataset_readme = PACKAGE_DIR / "tenacious_bench" / "README.md"
    datasheet = PACKAGE_DIR / "tenacious_bench" / "datasheet.md"
    if datasheet.exists() and not dataset_readme.exists():
        shutil.copy2(datasheet, dataset_readme)
        copied.append(dataset_readme)

    return copied


def write_manifest(copied_files: List[Path]) -> None:
    manifest = {
        "package_name": "Act_v_package",
        "description": "Final Week 11 Tenacious-Bench benchmark, judge adapter, and evaluation artifacts.",
        "files": [],
    }

    for path in sorted(copied_files):
        if not path.is_file():
            continue
        manifest["files"].append({
            "path": str(path.relative_to(PACKAGE_DIR)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    manifest_path = PACKAGE_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def zip_package() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in PACKAGE_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(DIST_DIR))


def main() -> None:
    print("Validating required final artifacts...")
    fail_if_missing_required()

    candidate_paths = [ROOT / p for p in REQUIRED_FILES + OPTIONAL_FILES]
    print("Scanning text artifacts for obvious secret patterns...")
    fail_if_secret_findings(candidate_paths)

    print("Copying final artifacts...")
    copied = copy_artifacts()

    print("Writing manifest...")
    write_manifest(copied)

    print("Creating zip package...")
    zip_package()

    print("\nFinal package created:")
    print(f"  - {PACKAGE_DIR}")
    print(f"  - {ZIP_PATH}")
    print("\nReady for Act V upload/release.")


if __name__ == "__main__":
    main()