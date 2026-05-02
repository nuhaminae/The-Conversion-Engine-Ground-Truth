# src/training/train_judge.py

"""
Train a Judge/Critic model for the Tenacious Benchmark using Direct Preference Optimisation (DPO).
This script fine-tunes a decoder model (e.g., Phi-3) using the Unsloth library
for high-efficiency training on consumer/free-tier GPUs (e.g., Google Colab T4).
"""

import argparse
import os

import torch
import wandb
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import TrainingArguments
from trl import DPOTrainer

# Unsloth and TRL for DPO
from unsloth import FastLanguageModel

# --- Environment and Authentication ---

# Load environment variables from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def authenticate_services(config):
    """Handles authentication for HuggingFace and W&B."""
    if HF_TOKEN:
        api = HfApi()
        api.set_access_token(HF_TOKEN)
        print("✔️ HuggingFace Hub authenticated.")
    else:
        print("⚠️ No HF_TOKEN found in .env. Model pushing will be disabled.")

    if WANDB_API_KEY and config["reporting"].get("use_wandb", False):
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=config["reporting"]["wandb_project"],
            name=config["reporting"]["wandb_run_name"],
            config=config,  # Log the entire config
        )
        print("✔️ W&B tracking initialised.")
    else:
        print(
            "⚠️ W&B logging is disabled. No WANDB_API_KEY found or use_wandb is false."
        )
        # Explicitly disable W&B reporting if not configured
        os.environ["WANDB_DISABLED"] = "true"


def load_dpo_dataset(data_config):
    """
    Loads DPO preference data from explicit train/dev files.
    Required columns: prompt, chosen, rejected.
    """
    data_files = {
        "train": data_config["train_file"],
        "validation": data_config["dev_file"],
    }

    for split, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{split} file not found: {path}. "
                "Run create_preference_pairs.py with "
                "--training-data-dir data/training_data first."
            )

    print(f"Loading DPO dataset from: {data_files}")
    dataset = load_dataset("json", data_files=data_files)

    required_columns = {"prompt", "chosen", "rejected"}
    for split_name in ["train", "validation"]:
        missing = required_columns - set(dataset[split_name].column_names)
        if missing:
            raise ValueError(
                f"{split_name} split is missing required DPO columns: {missing}. "
                f"Columns found: {dataset[split_name].column_names}"
            )

    print(f"Train pairs: {len(dataset['train'])}")
    print(f"Validation pairs: {len(dataset['validation'])}")

    return dataset["train"], dataset["validation"]


def main(config_path):
    # --- 1. Load Configuration ---
    print("Loading configuration from:", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    # --- 2. Authenticate ---
    authenticate_services(config)

    # --- 3. Load Model and Tokenizer with Unsloth ---
    print(f"Loading base model with Unsloth: {model_config['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["base_model"],
        max_seq_length=data_config["max_length"],
        dtype=None,  # Unsloth handles dtype automatically
        load_in_4bit=True,  # Critical for memory efficiency on Colab
    )

    # --- 4. Configure PEFT (LoRA) for efficient tuning ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config["lora"]["r"],
        target_modules=model_config["lora"]["target_modules"],
        lora_alpha=model_config["lora"]["alpha"],
        lora_dropout=model_config["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=config["seed"],
        max_seq_length=data_config["max_length"],
    )
    print("✔️ PEFT (LoRA) configured on the model.")

    # --- 5. Load and Prepare DPO Dataset ---
    train_dataset, eval_dataset = load_dpo_dataset(data_config)

    # --- 6. Set up DPO Trainer ---
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,  # Unsloth handles the reference model automatically
        args=TrainingArguments(
            per_device_train_batch_size=training_config["batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            num_train_epochs=training_config["num_epochs"],
            learning_rate=training_config["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=training_config["logging_steps"],
            optim="adamw_8bit",  # Use 8-bit AdamW for more memory savings
            weight_decay=training_config["weight_decay"],
            lr_scheduler_type="linear",
            seed=config["seed"],
            output_dir=config["output"]["checkpoint_dir"],
            evaluation_strategy=training_config["evaluation_strategy"],
            save_strategy=training_config["save_strategy"],
            report_to=(
                "wandb"
                if WANDB_API_KEY and config["reporting"].get("use_wandb", False)
                else "none"
            ),
        ),
        beta=training_config["beta"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=data_config["max_length"],
        max_prompt_length=data_config["max_length"] // 2,
    )
    print("✔️ DPOTrainer initialised.")

    # --- 7. Train the Model ---
    print("\n--- Starting DPO Training ---\n")
    dpo_trainer.train()
    print("\n--- DPO Training Complete ---\n")

    # --- 8. Save the Final LoRA Adapter ---
    output_dir = config["output"]["model_dir"]
    print(f"Saving final LoRA adapter to {output_dir}")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if WANDB_API_KEY and config["reporting"].get("use_wandb", False):
        wandb.finish()


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DPO Judge/Critic model using a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
