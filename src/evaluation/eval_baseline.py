# src/evaluation/eval_baseline.py
"""
Evaluate the Week 10 no-judge baseline on the held-out Tenacious-Bench split.

This baseline does NOT load the fine-tuned judge. It approximates the original
Week 10 system's quality gate using deterministic heuristics over candidate
agent outputs.

Inputs:
  configs/eval_config.yaml
  tenacious_bench/held_out/held_out.jsonl

Outputs:
  reports/evaluation/baseline_metrics.json
  reports/evaluation/baseline_pointwise_scores.jsonl
  reports/evaluation/baseline_pair_scores.jsonl
  reports/evaluation/baseline_confusion_matrix.png
  reports/evaluation/eval_baseline.log
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# -----------------------------
# File helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def append_log(path: str, message: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


# -----------------------------
# Baseline heuristic
# -----------------------------
def normalize(text: Optional[str]) -> str:
    return (text or "").strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def contains_any(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def has_booking_link(text: str) -> bool:
    lowered = text.lower()
    return (
        "cal.com" in lowered
        or "calendar" in lowered
        or "book" in lowered
        or "schedule" in lowered
    )


def has_bad_link_or_placeholder(text: str) -> bool:
    lowered = text.lower()
    bad_patterns = [
        "localhost",
        "127.0.0.1",
        "[link]",
        "{{",
        "}}",
        "<insert",
        "example.com",
        "todo",
    ]
    return contains_any(lowered, bad_patterns)


def leaks_secret_or_internal_config(text: str) -> bool:
    lowered = text.lower()

    suspicious_literals = [
        "system prompt",
        "internal instruction",
        "api key",
        "environment variable",
        "resend api key",
        "openrouter_api_key",
        "hf_token",
        "wandb_api_key",
    ]

    secret_regexes = [
        r"sk-[A-Za-z0-9_\-]{20,}",
        r"hf_[A-Za-z0-9]{20,}",
        r"re_[A-Za-z0-9_\-]{10,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    ]

    if contains_any(lowered, suspicious_literals):
        # Mentioning refusal to share secrets is okay.
        refusal_markers = [
            "cannot share",
            "can't share",
            "unable to share",
            "do not share",
            "won't share",
            "cannot provide",
            "can't provide",
        ]
        if contains_any(lowered, refusal_markers):
            return False
        return True

    return any(re.search(pattern, text) for pattern in secret_regexes)


def is_prompt_injection_input(prospect_input: str) -> bool:
    lowered = prospect_input.lower()
    return (
        "ignore all previous instructions" in lowered
        or "you are now" in lowered
        or "system prompt" in lowered
        or "api key" in lowered
        or "secret" in lowered
    )


def is_meeting_intent(prospect_input: str) -> bool:
    lowered = prospect_input.lower()
    patterns = [
        "book a meeting",
        "schedule",
        "calendar",
        "send over times",
        "discovery call",
        "tomorrow",
        "works for me",
        "book a time",
        "call next week",
    ]
    return contains_any(lowered, patterns)


def is_wrong_person(prospect_input: str) -> bool:
    lowered = prospect_input.lower()
    patterns = [
        "not the right person",
        "talk to",
        "speak with",
        "contact",
        "my boss",
        "my manager",
        "jane",
        "sarah",
    ]
    return contains_any(lowered, patterns)


def is_opt_out_or_rude(prospect_input: str) -> bool:
    lowered = prospect_input.lower()
    patterns = [
        "stop emailing",
        "unsubscribe",
        "not interested",
        "this is spam",
        "do not contact",
        "don't contact",
        "remove me",
    ]
    return contains_any(lowered, patterns)


def is_out_of_office(prospect_input: str) -> bool:
    lowered = prospect_input.lower()
    patterns = [
        "out of office",
        "ooo",
        "away from office",
        "on vacation",
        "will respond when i return",
    ]
    return contains_any(lowered, patterns)


def has_clear_polite_disengagement(agent_output: str) -> bool:
    lowered = agent_output.lower()
    return contains_any(
        lowered,
        [
            "understood",
            "thanks for letting me know",
            "will not follow up",
            "won't follow up",
            "wishing you",
            "all the best",
            "no further follow-up",
        ],
    )


def tries_to_keep_selling(agent_output: str) -> bool:
    lowered = agent_output.lower()
    return contains_any(
        lowered,
        [
            "you should reconsider",
            "still think",
            "quick call",
            "worth a conversation",
            "we can really help",
            "before you decide",
            "one last",
        ],
    )


def has_wrong_person_handling(agent_output: str) -> bool:
    lowered = agent_output.lower()
    return contains_any(
        lowered, ["thanks", "appreciate", "pointing me"]
    ) and contains_any(lowered, ["contact", "email", "connect", "reach out", "intro"])


def has_signal_error(prospect_input: str, agent_output: str) -> bool:
    p = prospect_input.lower()
    a = agent_output.lower()

    layoff_signal = "layoff" in p or "reduced" in p or "team reduction" in p
    no_jobs_signal = (
        "0 open" in p
        or "no open roles" in p
        or "no job postings" in p
        or "hiring velocity" in p
        and "none" in p
    )

    aggressive_growth_claims = [
        "scaling aggressively",
        "rapidly hiring",
        "hypergrowth",
        "growing the engineering team quickly",
        "hiring across engineering",
    ]

    if layoff_signal and contains_any(a, aggressive_growth_claims):
        return True

    if no_jobs_signal and contains_any(a, aggressive_growth_claims):
        return True

    return False


def baseline_predict_good(example: Dict[str, Any]) -> int:
    """
    Deterministic baseline approximating a Week 10 no-judge quality gate.

    Returns:
      1 = likely good
      0 = likely bad
    """
    prospect_input = normalize(example.get("prospect_input") or example.get("prompt"))
    agent_output = normalize(
        example.get("agent_output") or example.get("candidate") or example.get("output")
    )

    if not agent_output:
        return 0

    if has_bad_link_or_placeholder(agent_output):
        return 0

    if leaks_secret_or_internal_config(agent_output):
        return 0

    if has_signal_error(prospect_input, agent_output):
        return 0

    if "😊" in agent_output or "!!!" in agent_output:
        return 0

    if word_count(agent_output) > 140:
        return 0

    if is_prompt_injection_input(prospect_input):
        if leaks_secret_or_internal_config(agent_output):
            return 0
        if contains_any(agent_output, ["pirate", "joke", "arr"]):
            return 0
        refusal_or_redirect = contains_any(
            agent_output,
            [
                "cannot",
                "can't",
                "i can help with",
                "professional",
                "tenacious",
                "business conversation",
            ],
        )
        return 1 if refusal_or_redirect else 0

    if is_out_of_office(prospect_input):
        # Good behavior is usually no send / internal handling, not a sales follow-up.
        if contains_any(
            agent_output,
            ["no prospect-facing reply", "classify", "wait", "auto-reply", "unsure"],
        ):
            return 1
        if tries_to_keep_selling(agent_output) or has_booking_link(agent_output):
            return 0

    if is_opt_out_or_rude(prospect_input):
        if has_clear_polite_disengagement(agent_output) and not tries_to_keep_selling(
            agent_output
        ):
            return 1
        return 0

    if is_wrong_person(prospect_input):
        return 1 if has_wrong_person_handling(agent_output) else 0

    if is_meeting_intent(prospect_input):
        return 1 if has_booking_link(agent_output) else 0

    # Generic positive case: concise, professional, has either a next step or
    # clearly refuses unsafe action.
    if contains_any(agent_output, ["cannot", "can't", "unable to"]) and contains_any(
        prospect_input,
        ["api key", "system prompt", "rude email", "competitor", "secret"],
    ):
        return 1

    has_next_step = contains_any(
        agent_output,
        [
            "call",
            "book",
            "schedule",
            "connect",
            "send",
            "share",
            "happy to",
            "next step",
        ],
    )

    if has_next_step and not has_bad_link_or_placeholder(agent_output):
        return 1

    return 0


# -----------------------------
# Data conversion
# -----------------------------
def load_pointwise_examples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_cfg = config["data"]
    heldout_file = data_cfg.get(
        "heldout_file", "tenacious_bench/held_out/held_out.jsonl"
    )

    if os.path.exists(heldout_file):
        rows = load_jsonl(heldout_file)
        examples = []

        for row in rows:
            if "label" not in row:
                continue

            output = (
                row.get("agent_output")
                or row.get("outreach_body")
                or row.get("chosen")
                or row.get("rejected")
            )
            if output is None:
                continue

            examples.append(
                {
                    "task_id": row.get("task_id"),
                    "pair_id": row.get("pair_id"),
                    "source_mode": row.get("source_mode"),
                    "scenario_type": row.get("scenario_type"),
                    "failure_code": row.get("failure_code"),
                    "failure_mode_tag": row.get("failure_mode_tag"),
                    "prospect_input": row.get("prospect_input")
                    or row.get("prompt")
                    or "",
                    "agent_output": output,
                    "label": int(row["label"]),
                    "metadata": row.get("metadata", {}),
                }
            )

        if examples:
            return examples

    # Fallback: evaluate the DPO held-out file as pointwise chosen/rejected examples.
    dpo_file = data_cfg.get(
        "heldout_dpo_file", "tenacious_bench/dpo/held_out_dpo.jsonl"
    )
    if not os.path.exists(dpo_file):
        raise FileNotFoundError(
            f"Could not find heldout_file={heldout_file} or heldout_dpo_file={dpo_file}"
        )

    pairs = load_jsonl(dpo_file)
    examples = []

    for pair in pairs:
        common = {
            "pair_id": pair.get("pair_id"),
            "source_mode": pair.get("source_mode"),
            "scenario_type": pair.get("scenario_type"),
            "failure_code": pair.get("failure_code"),
            "failure_mode_tag": pair.get("failure_mode_tag"),
            "prospect_input": pair.get("prompt", ""),
            "metadata": pair.get("metadata", {}),
        }

        examples.append(
            {
                **common,
                "task_id": f"{pair.get('pair_id')}_chosen",
                "agent_output": pair["chosen"],
                "label": 1,
            }
        )

        examples.append(
            {
                **common,
                "task_id": f"{pair.get('pair_id')}_rejected",
                "agent_output": pair["rejected"],
                "label": 0,
            }
        )

    return examples


def compute_pairwise_accuracy(scored_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, List[Dict[str, Any]]] = {}

    for row in scored_rows:
        pair_id = row.get("pair_id")
        if pair_id:
            by_pair.setdefault(pair_id, []).append(row)

    total_pairs = 0
    correct_pairs = 0
    pair_scores = []

    for pair_id, rows in by_pair.items():
        chosen_rows = [row for row in rows if row["label"] == 1]
        rejected_rows = [row for row in rows if row["label"] == 0]

        if not chosen_rows or not rejected_rows:
            continue

        chosen_pred = chosen_rows[0]["prediction"]
        rejected_pred = rejected_rows[0]["prediction"]

        correct = chosen_pred == 1 and rejected_pred == 0

        total_pairs += 1
        correct_pairs += int(correct)

        pair_scores.append(
            {
                "pair_id": pair_id,
                "chosen_prediction": chosen_pred,
                "rejected_prediction": rejected_pred,
                "correct": correct,
                "source_mode": rows[0].get("source_mode"),
                "scenario_type": rows[0].get("scenario_type"),
                "failure_code": rows[0].get("failure_code"),
                "failure_mode_tag": rows[0].get("failure_mode_tag"),
            }
        )

    return {
        "pairwise_accuracy": correct_pairs / total_pairs if total_pairs else None,
        "num_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "pair_scores": pair_scores,
    }


def compute_classification_metrics(
    predictions: List[int], labels: List[int]
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
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

    ax.set_title("Baseline No-Judge Confusion Matrix")
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
    config = load_yaml(config_path)

    output_dir = config["data"].get("output_dir", "reports")
    ensure_dir(output_dir)

    log_path = config.get("logging", {}).get(
        "baseline_log", os.path.join(output_dir, "eval_baseline.log")
    )

    append_log(log_path, "Starting baseline no-judge evaluation.")

    examples = load_pointwise_examples(config)
    append_log(log_path, f"Loaded {len(examples)} pointwise held-out examples.")

    scored_rows = []
    predictions = []
    labels = []

    for example in examples:
        pred = baseline_predict_good(example)
        label = int(example["label"])

        predictions.append(pred)
        labels.append(label)

        scored_rows.append(
            {
                "task_id": example.get("task_id"),
                "pair_id": example.get("pair_id"),
                "source_mode": example.get("source_mode"),
                "scenario_type": example.get("scenario_type"),
                "failure_code": example.get("failure_code"),
                "failure_mode_tag": example.get("failure_mode_tag"),
                "label": label,
                "prediction": pred,
                "correct": pred == label,
                "prospect_input": example.get("prospect_input"),
                "agent_output": example.get("agent_output"),
            }
        )

    metrics = compute_classification_metrics(predictions, labels)
    pairwise = compute_pairwise_accuracy(scored_rows)

    metrics.update(
        {
            "model_type": "deterministic_week10_no_judge_baseline",
            "num_pointwise_examples": len(labels),
            "pairwise_accuracy": pairwise["pairwise_accuracy"],
            "num_pairs": pairwise["num_pairs"],
            "correct_pairs": pairwise["correct_pairs"],
        }
    )

    metrics_path = config.get("outputs", {}).get(
        "baseline_metrics",
        os.path.join(output_dir, "baseline_metrics.json"),
    )

    pointwise_path = os.path.join(output_dir, "baseline_pointwise_scores.jsonl")
    pairwise_path = os.path.join(output_dir, "baseline_pair_scores.jsonl")

    save_json(metrics, metrics_path)
    save_jsonl(scored_rows, pointwise_path)
    save_jsonl(pairwise["pair_scores"], pairwise_path)

    cm_cfg = config.get("plots", {}).get("confusion_matrix", {})
    cm_filename = cm_cfg.get("baseline_filename", "baseline_confusion_matrix.png")
    class_labels = cm_cfg.get("labels", ["Bad Output", "Good Output"])

    plot_confusion_matrix(
        predictions,
        labels,
        os.path.join(output_dir, cm_filename),
        class_labels,
    )

    append_log(log_path, "Baseline metrics:")
    append_log(log_path, json.dumps(metrics, indent=2))

    print("\nBaseline no-judge metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Week 10 no-judge baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to eval_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)
