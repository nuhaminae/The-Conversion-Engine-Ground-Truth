# src/evaluation/eval_baseline.py

"""
Evaluate the Week 10 baseline agent outputs on the Tenacious Sales Evaluation Benchmark.
Uses utils.py for logging/JSON and metrics.py for evaluation metrics.
"""

import argparse
import json
import os

import wandb
from dotenv import load_dotenv

from src.evaluation.metrics import compute_classification_metrics, plot_confusion_matrix
from src.training.utils import heuristic_is_good, save_json, setup_logger, timestamp

# Load environment variables
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "tenacious-judge")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


def init_wandb(run_name):
    """Initialise W&B if API key is present."""
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name)
        return True
    else:
        print("⚠️ No WANDB_API_KEY found in .env, skipping W&B logging")
        return False


def main(args):

    # Initialise W&B
    wandb_enabled = init_wandb("baseline-eval-v1")

    logger = setup_logger(
        "eval_baseline", os.path.join(args.output_dir, "eval_baseline.log")
    )
    logger.info(f"Starting baseline evaluation at {timestamp()}")

    with open(os.path.join(args.data_dir, "heldout.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions, labels = [], []
    for example in data:
        labels.append(example["label"])
        predictions.append(1 if heuristic_is_good(example["agent_output"]) else 0)

    results = compute_classification_metrics(predictions, labels)
    logger.info(f"Baseline metrics: {results}")
    save_json(results, os.path.join(args.output_dir, "baseline_metrics.json"))

    if wandb_enabled:
        wandb.log(results)
        wandb.log(
            {
                "baseline_confusion_matrix": wandb.Image(
                    os.path.join(args.output_dir, "baseline_confusion_matrix.png")
                )
            }
        )

    plot_confusion_matrix(
        predictions,
        labels,
        args.output_dir,
        filename="baseline_confusion_matrix.png",
        title="Baseline Agent Confusion Matrix",
    )
    logger.info(
        f"Baseline confusion matrix saved to {args.output_dir}/baseline_confusion_matrix.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Week 10 baseline agent")
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    main(args)
