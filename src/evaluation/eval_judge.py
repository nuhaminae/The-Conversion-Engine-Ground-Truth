# src/evaluation/eval_judge.py

"""
Evaluate the fine-tuned Tenacious DPO judge on held-out preference pairs.

This script evaluates a DPO-trained LoRA adapter.
It loads:

  1. Reference model: base causal LM
  2. Policy model: base causal LM + LoRA adapter

For each held-out pair, it computes:

  reward = beta * (policy_logprob(response | prompt) - reference_logprob(response | prompt))

The fine-tuned judge is correct when:

  reward(chosen) > reward(rejected)

Outputs:
  reports/fine_tuned/fine_tuned_judge_metrics.json
  reports/fine_tuned/fine_tuned_judge_pair_scores.jsonl
  reports/fine_tuned/fine_tuned_judge_confusion_matrix.png
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def group_pointwise_rows_to_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback if the user points to tenacious_bench/held_out/held_out.jsonl
    instead of tenacious_bench/dpo/held_out_dpo.jsonl.

    Expects pointwise rows with pair_id, label, prospect_input, agent_output.
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        pair_id = row.get("pair_id")
        if not pair_id:
            continue

        grouped.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "prompt": row.get("prospect_input") or "",
                "chosen": None,
                "rejected": None,
                "source_mode": row.get("source_mode"),
                "scenario_type": row.get("scenario_type"),
                "failure_code": row.get("failure_code"),
                "failure_mode_tag": row.get("failure_mode_tag"),
                "metadata": {},
            },
        )

        label = row.get("label")
        output = row.get("agent_output") or row.get("outreach_body") or ""

        if label == 1:
            grouped[pair_id]["chosen"] = output
        elif label == 0:
            grouped[pair_id]["rejected"] = output

        if row.get("prospect_input"):
            grouped[pair_id]["prompt"] = row["prospect_input"]

    pairs = []
    for pair in grouped.values():
        if pair.get("prompt") and pair.get("chosen") and pair.get("rejected"):
            pairs.append(pair)

    return pairs


def load_eval_pairs(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    data_cfg = config["data"]

    # Prefer DPO-formatted held-out pairs.
    dpo_path = data_cfg.get("heldout_dpo_file")
    if dpo_path and os.path.exists(dpo_path):
        rows = load_jsonl(dpo_path)
        required = {"prompt", "chosen", "rejected"}
        for idx, row in enumerate(rows[:5]):
            missing = required - set(row.keys())
            if missing:
                raise ValueError(f"{dpo_path} row {idx} missing columns: {missing}")
        return rows, dpo_path

    # Fallback: group pointwise held-out rows into DPO pairs.
    heldout_path = data_cfg["heldout_file"]
    rows = load_jsonl(heldout_path)
    pairs = group_pointwise_rows_to_pairs(rows)

    if not pairs:
        raise ValueError(
            "No evaluation pairs found. Provide data.heldout_dpo_file pointing to "
            "tenacious_bench/dpo/held_out_dpo.jsonl or ensure heldout_file has "
            "pair_id/label/prospect_input/agent_output fields."
        )

    return pairs, heldout_path


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(config: Dict[str, Any]):
    judge_cfg = config["judge"]
    tokenizer_dir = judge_cfg.get("tokenizer_dir") or judge_cfg.get("adapter_dir")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_models(config: Dict[str, Any]):
    judge_cfg = config["judge"]
    base_model = judge_cfg["base_model"]
    adapter_dir = judge_cfg["adapter_dir"]
    load_in_4bit = bool(judge_cfg.get("load_in_4bit", torch.cuda.is_available()))

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    common_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
    }

    if load_in_4bit and torch.cuda.is_available():
        common_kwargs["load_in_4bit"] = True

    print(f"Loading reference base model: {base_model}")
    reference_model = AutoModelForCausalLM.from_pretrained(base_model, **common_kwargs)
    reference_model.eval()

    print(f"Loading policy base model: {base_model}")
    policy_base = AutoModelForCausalLM.from_pretrained(base_model, **common_kwargs)

    print(f"Loading LoRA adapter: {adapter_dir}")
    policy_model = PeftModel.from_pretrained(policy_base, adapter_dir)
    policy_model.eval()

    return reference_model, policy_model


def conditional_logprob(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    average_logprob: bool = False,
) -> float:
    """
    Compute log p(response | prompt), masking out prompt tokens.

    If average_logprob=False, returns summed response-token logprob, which is
    closer to the DPO trainer's sequence-logp behavior.
    """
    prompt = prompt.strip()
    response = response.strip()

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )["input_ids"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )["input_ids"]

    # Leave room for response if prompt is too long.
    if len(prompt_ids) + len(response_ids) > max_length:
        available_for_response = max(1, max_length - len(prompt_ids))
        if available_for_response < len(response_ids):
            response_ids = response_ids[:available_for_response]

    input_ids = prompt_ids + response_ids
    if len(input_ids) < 2 or not response_ids:
        return float("-inf")

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits

    # logits[:, t-1] predicts input_ids[:, t]
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_tensor[:, 1:]

    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    response_start_index_in_input = len(prompt_ids)
    response_start_index_in_labels = max(response_start_index_in_input - 1, 0)
    response_end_index_in_labels = response_start_index_in_labels + len(response_ids)

    response_token_log_probs = token_log_probs[
        0, response_start_index_in_labels:response_end_index_in_labels
    ]

    total = response_token_log_probs.sum().item()

    if average_logprob:
        return total / max(len(response_ids), 1)

    return total


def dpo_reward(
    policy_model,
    reference_model,
    tokenizer,
    prompt: str,
    response: str,
    beta: float,
    max_length: int,
    average_logprob: bool,
) -> Dict[str, float]:
    policy_logp = conditional_logprob(
        policy_model,
        tokenizer,
        prompt,
        response,
        max_length=max_length,
        average_logprob=average_logprob,
    )

    reference_logp = conditional_logprob(
        reference_model,
        tokenizer,
        prompt,
        response,
        max_length=max_length,
        average_logprob=average_logprob,
    )

    reward = beta * (policy_logp - reference_logp)

    return {
        "policy_logp": policy_logp,
        "reference_logp": reference_logp,
        "reward": reward,
    }


def compute_metrics(
    pointwise_predictions: List[int], pointwise_labels: List[int]
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(pointwise_labels, pointwise_predictions),
        "precision": precision_score(
            pointwise_labels, pointwise_predictions, zero_division=0
        ),
        "recall": recall_score(
            pointwise_labels, pointwise_predictions, zero_division=0
        ),
        "f1": f1_score(pointwise_labels, pointwise_predictions, zero_division=0),
    }


def plot_confusion_matrix(
    predictions: List[int],
    labels: List[int],
    output_path: str,
    class_labels: List[str],
) -> None:
    ensure_dir(str(Path(output_path).parent))
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title("Fine-Tuned DPO Judge Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main(config_path: str) -> None:
    config = load_config(config_path)

    output_dir = config["data"]["output_dir"]
    ensure_dir(output_dir)

    pairs, source_path = load_eval_pairs(config)
    print(f"Loaded {len(pairs)} held-out preference pairs from {source_path}")

    tokenizer = load_tokenizer(config)
    reference_model, policy_model = load_models(config)

    beta = float(config["judge"].get("beta", 0.1))
    max_length = int(config["judge"].get("max_length", 1024))
    average_logprob = bool(config["judge"].get("average_logprob", False))

    pair_scores = []
    pointwise_predictions: List[int] = []
    pointwise_labels: List[int] = []

    correct_pairs = 0

    for idx, pair in enumerate(pairs, start=1):
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        chosen_scores = dpo_reward(
            policy_model,
            reference_model,
            tokenizer,
            prompt,
            chosen,
            beta=beta,
            max_length=max_length,
            average_logprob=average_logprob,
        )

        rejected_scores = dpo_reward(
            policy_model,
            reference_model,
            tokenizer,
            prompt,
            rejected,
            beta=beta,
            max_length=max_length,
            average_logprob=average_logprob,
        )

        chosen_preferred = chosen_scores["reward"] > rejected_scores["reward"]

        if chosen_preferred:
            correct_pairs += 1
            pointwise_predictions.extend([1, 0])
        else:
            pointwise_predictions.extend([0, 1])

        pointwise_labels.extend([1, 0])

        pair_scores.append(
            {
                "pair_index": idx,
                "pair_id": pair.get("pair_id"),
                "source_mode": pair.get("source_mode"),
                "scenario_type": pair.get("scenario_type"),
                "failure_code": pair.get("failure_code"),
                "failure_mode_tag": pair.get("failure_mode_tag"),
                "chosen_reward": chosen_scores["reward"],
                "rejected_reward": rejected_scores["reward"],
                "reward_margin": chosen_scores["reward"] - rejected_scores["reward"],
                "chosen_policy_logp": chosen_scores["policy_logp"],
                "rejected_policy_logp": rejected_scores["policy_logp"],
                "chosen_reference_logp": chosen_scores["reference_logp"],
                "rejected_reference_logp": rejected_scores["reference_logp"],
                "chosen_preferred": chosen_preferred,
            }
        )

        print(
            f"[{idx}/{len(pairs)}] pair_id={pair.get('pair_id')} "
            f"margin={pair_scores[-1]['reward_margin']:.4f} "
            f"correct={chosen_preferred}"
        )

    pairwise_accuracy = correct_pairs / max(len(pairs), 1)
    metrics = compute_metrics(pointwise_predictions, pointwise_labels)

    margins = [row["reward_margin"] for row in pair_scores]
    metrics.update(
        {
            "pairwise_accuracy": pairwise_accuracy,
            "num_pairs": len(pairs),
            "num_pointwise_examples": len(pointwise_labels),
            "mean_reward_margin": sum(margins) / max(len(margins), 1),
            "min_reward_margin": min(margins) if margins else None,
            "max_reward_margin": max(margins) if margins else None,
            "beta": beta,
            "average_logprob": average_logprob,
            "eval_source": source_path,
            "model_type": "causal_lm_lora_dpo",
            "adapter_dir": config["judge"]["adapter_dir"],
            "base_model": config["judge"]["base_model"],
        }
    )

    metrics_path = os.path.join(output_dir, "fine_tuned_judge_metrics.json")
    scores_path = os.path.join(output_dir, "fine_tuned_judge_pair_scores.jsonl")

    save_json(metrics, metrics_path)
    save_jsonl(pair_scores, scores_path)

    cm_cfg = config.get("plots", {}).get("confusion_matrix", {})
    cm_filename = cm_cfg.get("judge_filename", "fine_tuned_judge_confusion_matrix.png")
    labels = cm_cfg.get("labels", ["Bad Output", "Good Output"])
    plot_confusion_matrix(
        pointwise_predictions,
        pointwise_labels,
        os.path.join(output_dir, cm_filename),
        labels,
    )

    print("\nFine-tuned judge metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DPO LoRA judge on held-out pairs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to eval_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
