# src/training/train_judge.py

"""
Train a Path B preference-tuned judge/critic for Tenacious-Bench using DPO.

This script follows the official Unsloth DPO notebook pattern:
- Import Unsloth before TRL/Transformers/PEFT.
- Patch DPOTrainer with PatchDPOTrainer().
- Load the base model in 4-bit.
- Add LoRA adapters with use_gradient_checkpointing="unsloth".
- Train on prompt/chosen/rejected JSONL preference data.

Expected input:
  data/training_data/preferences_train.jsonl
  data/training_data/preferences_dev.jsonl

Expected columns:
  prompt, chosen, rejected

Outputs:
  models/checkpoints/
  models/judge/
  reports/training/training_config_used.yaml
  reports/training/dataset_summary.json
  reports/training/training_summary.json
"""

# -----------------------------
# Environment flags before imports
# -----------------------------
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# -----------------------------
# Unsloth imports FIRST
# -----------------------------
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

PatchDPOTrainer()

# -----------------------------
# Standard imports after Unsloth
# -----------------------------
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import DPOConfig, DPOTrainer


# -----------------------------
# Utility helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_yaml(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# -----------------------------
# Auth
# -----------------------------
def authenticate(config: Dict[str, Any]) -> str:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    use_wandb = bool(get_nested(config, "reporting", "use_wandb", default=False))
    wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()

    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("Hugging Face authentication: enabled")
    else:
        print("Hugging Face authentication: skipped; OK for public models")

    if not (use_wandb and wandb_api_key):
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B: disabled")
    else:
        os.environ.pop("WANDB_DISABLED", None)
        print("W&B: enabled by config/env")

    return hf_token


# -----------------------------
# Data
# -----------------------------
def load_dpo_dataset(data_config: Dict[str, Any]):
    data_files = {
        "train": data_config["train_file"],
        "validation": data_config["dev_file"],
    }

    for split, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{split} file not found: {path}. "
                "Run create_preference_pairs.py first and confirm data/training_data exists."
            )

    print(f"Loading DPO dataset from: {data_files}")
    dataset = load_dataset("json", data_files=data_files)

    required = {"prompt", "chosen", "rejected"}

    for split_name in ["train", "validation"]:
        columns = set(dataset[split_name].column_names)
        missing = required - columns
        if missing:
            raise ValueError(
                f"{split_name} split missing columns: {missing}. "
                f"Found columns: {sorted(columns)}"
            )

        bad_examples = []
        for idx, row in enumerate(dataset[split_name]):
            for key in required:
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    bad_examples.append({"row": idx, "field": key})
                    break
            if len(bad_examples) >= 5:
                break

        if bad_examples:
            raise ValueError(
                f"{split_name} split has invalid prompt/chosen/rejected values: "
                f"{bad_examples}"
            )

    print(f"Train preference pairs: {len(dataset['train'])}")
    print(f"Validation preference pairs: {len(dataset['validation'])}")

    return dataset["train"], dataset["validation"]


# -----------------------------
# Trainer construction
# -----------------------------
def build_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict[str, Any],
):
    data_config = config["data"]
    training_config = config["training"]
    output_config = config["output"]

    dpo_args = DPOConfig(
        output_dir=output_config["checkpoint_dir"],
        per_device_train_batch_size=training_config.get("batch_size", 1),
        per_device_eval_batch_size=training_config.get(
            "eval_batch_size", training_config.get("batch_size", 1)
        ),
        gradient_accumulation_steps=training_config.get(
            "gradient_accumulation_steps", 4
        ),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        num_train_epochs=training_config.get("num_epochs", 1),
        learning_rate=training_config.get("learning_rate", 5e-6),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=training_config.get("logging_steps", 1),
        optim=training_config.get("optim", "adamw_8bit"),
        weight_decay=training_config.get("weight_decay", 0.0),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
        seed=config.get("seed", 42),
        save_strategy=training_config.get("save_strategy", "epoch"),
        eval_strategy=training_config.get(
            "eval_strategy",
            training_config.get("evaluation_strategy", "epoch"),
        ),
        report_to="none",
        remove_unused_columns=False,
        beta=training_config.get("beta", 0.1),
        max_length=data_config.get("max_length", 1024),
        max_prompt_length=data_config.get(
            "max_prompt_length", data_config.get("max_length", 1024) // 2
        ),
    )

    # Newer TRL expects processing_class. Older/Unsloth-patched patterns often accept tokenizer.
    signature = inspect.signature(DPOTrainer.__init__)
    kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    try:
        return DPOTrainer(**kwargs)
    except TypeError as first_error:
        print(
            "First DPOTrainer construction failed; trying Unsloth notebook-style fallback."
        )
        print(f"First error: {first_error}")

        # Fallback closer to older Unsloth notebook signatures.
        fallback_kwargs = {
            "model": model,
            "ref_model": None,
            "args": dpo_args,
            "beta": training_config.get("beta", 0.1),
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "max_length": data_config.get("max_length", 1024),
            "max_prompt_length": data_config.get(
                "max_prompt_length", data_config.get("max_length", 1024) // 2
            ),
        }
        return DPOTrainer(**fallback_kwargs)


# -----------------------------
# Main
# -----------------------------
def main(config_path: str) -> None:
    started = time.time()

    print(f"Loading config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    data_config = config["data"]
    output_config = config["output"]

    report_dir = output_config.get("report_dir", "reports/training")
    checkpoint_dir = output_config["checkpoint_dir"]
    model_dir = output_config["model_dir"]

    ensure_dir(report_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(model_dir)

    save_yaml(config, os.path.join(report_dir, "training_config_used.yaml"))

    hf_token = authenticate(config)

    max_seq_length = data_config.get("max_length", 1024)

    print(f"Loading base model: {model_config['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["base_model"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hf_token or None,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Adding LoRA adapters.")
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config["lora"].get("r", 16),
        target_modules=model_config["lora"].get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        lora_alpha=model_config["lora"].get("alpha", 16),
        lora_dropout=model_config["lora"].get("dropout", 0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.get("seed", 42),
        max_seq_length=max_seq_length,
    )

    train_dataset, eval_dataset = load_dpo_dataset(data_config)

    save_json(
        {
            "train_file": data_config["train_file"],
            "dev_file": data_config["dev_file"],
            "train_pairs": len(train_dataset),
            "validation_pairs": len(eval_dataset),
            "columns": ["prompt", "chosen", "rejected"],
        },
        os.path.join(report_dir, "dataset_summary.json"),
    )

    print("Building DPOTrainer.")
    trainer = build_dpo_trainer(model, tokenizer, train_dataset, eval_dataset, config)

    print("\n--- Starting DPO training ---\n")
    train_result = trainer.train()
    print("\n--- DPO training complete ---\n")

    print("Evaluating on dev split.")
    eval_metrics = trainer.evaluate()

    print(f"Saving LoRA adapter to: {model_dir}")
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    trainer_state = Path(checkpoint_dir) / "trainer_state.json"
    if trainer_state.exists():
        shutil.copy2(trainer_state, Path(report_dir) / "trainer_state.json")

    summary = {
        "status": "complete",
        "base_model": model_config["base_model"],
        "model_dir": model_dir,
        "checkpoint_dir": checkpoint_dir,
        "report_dir": report_dir,
        "train_pairs": len(train_dataset),
        "validation_pairs": len(eval_dataset),
        "train_metrics": getattr(train_result, "metrics", {}),
        "eval_metrics": eval_metrics,
        "runtime_seconds": round(time.time() - started, 2),
        "seed": config.get("seed", 42),
    }

    save_json(summary, os.path.join(report_dir, "training_summary.json"))

    print("\nTraining summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Tenacious-Bench DPO judge with Unsloth."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configs/training_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
