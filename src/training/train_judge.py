# src/training/train_judge.py

"""
Train a Judge/Critic model for Tenacious Sales Evaluation Benchmark (Path B).
This script fine-tunes a HuggingFace model using preference optimisation
to enforce consistency in agent outputs (e.g., scheduling after prospect says "yes").
"""

import argparse
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Load environment variables from .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "tenacious-judge")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


from huggingface_hub import HfApi

if HF_TOKEN:
    api = HfApi()
    api.set_access_token(HF_TOKEN)
    print("✅ HuggingFace Hub authenticated")
else:
    print("⚠️ No HF_TOKEN found in .env")


import wandb

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="judge-training-v1")
    print("✅ W&B tracking initialized")
else:
    print("⚠️ No WANDB_API_KEY found in .env")


# -----------------------------
# Custom Dataset Wrapper
# -----------------------------
class PreferenceDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # Concatenate prospect input + agent output for scoring
        text = (
            f"Prospect: {example['prospect_input']}\nAgent: {example['agent_output']}"
        )
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(example["label"], dtype=torch.long)
        return inputs


# -----------------------------
# Compute Metrics
# -----------------------------
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# -----------------------------
# Main Training Function
# -----------------------------
def main(args):
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # Load dataset (expects train/dev/test splits)
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(args.data_dir, "train.json"),
            "validation": os.path.join(args.data_dir, "dev.json"),
            "test": os.path.join(args.data_dir, "heldout.json"),
        },
    )

    # Wrap datasets
    train_dataset = PreferenceDataset(dataset["train"], tokenizer)
    eval_dataset = PreferenceDataset(dataset["validation"], tokenizer)
    test_dataset = PreferenceDataset(dataset["test"], tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb" if args.use_wandb else "none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    print("Final evaluation on held-out set:")
    results = trainer.evaluate(test_dataset)
    print(results)

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Judge/Critic model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/splits",
        help="Directory containing train/dev/test JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/judge",
        help="Directory to save model outputs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Log training to Weights & Biases"
    )
    args = parser.parse_args()
    main(args)
