# src/evalation/eval_judge.py

"""
Evaluate the Judge/Critic model on Tenacious Sales Evaluation Benchmark (Path B).
Uses utils.py for logging/JSON and metrics.py for evaluation metrics.
"""

import os
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset

from src.training.train_judge import PreferenceDataset
from src.training.utils import setup_logger, save_json, timestamp
from src.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix

from dotenv import load_dotenv
import os
import wandb

# Load environment variables
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "tenacious-judge")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

def init_wandb(run_name):
    """Initialize W&B if API key is present."""
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name)
        return True
    else:
        print("⚠️ No WANDB_API_KEY found in .env, skipping W&B logging")
        return False

def main(args):
    
    # Initialize W&B
    wandb_enabled = init_wandb("judge-eval-v1")

    logger = setup_logger("eval_judge", os.path.join(args.output_dir, "eval_judge.log"))
    logger.info(f"Starting Judge evaluation at {timestamp()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    dataset = load_dataset("json", data_files={
        "test": os.path.join(args.data_dir, "heldout.json"),
    })
    test_dataset = PreferenceDataset(dataset["test"], tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    results = compute_classification_metrics(preds, labels)
    logger.info(f"Judge metrics: {results}")
    save_json(results, os.path.join(args.output_dir, "judge_metrics.json"))

    if wandb_enabled:
        wandb.log(results)
        wandb.log({"judge_confusion_matrix": wandb.Image(os.path.join(args.output_dir, "judge_confusion_matrix.png"))})

    plot_confusion_matrix(preds, labels, args.output_dir,
                          filename="judge_confusion_matrix.png",
                          title="Judge Model Confusion Matrix")
    logger.info(f"Confusion matrix saved to {args.output_dir}/judge_confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Judge/Critic model")
    parser.add_argument("--model_dir", type=str, default="models/judge")
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    main(args)
