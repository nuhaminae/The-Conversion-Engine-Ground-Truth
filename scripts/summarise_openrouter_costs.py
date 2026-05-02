# scripts/summarise_openrouter_costs.py

import json
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path("reports/openrouter_usage.jsonl")


def main():
    """Summarise OpenRouter usage."""
    if not LOG_PATH.exists():
        print(f"No usage log found at {LOG_PATH}")
        return

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    by_model = defaultdict(
        lambda: {
            "calls": 0,
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )

    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            model = row.get("model", "unknown")
            cost = float(row.get("cost") or 0)
            prompt_tokens = int(row.get("prompt_tokens") or 0)
            completion_tokens = int(row.get("completion_tokens") or 0)
            tokens = int(row.get("total_tokens") or 0)

            total_cost += cost
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += tokens

            by_model[model]["calls"] += 1
            by_model[model]["cost"] += cost
            by_model[model]["prompt_tokens"] += prompt_tokens
            by_model[model]["completion_tokens"] += completion_tokens
            by_model[model]["total_tokens"] += tokens

    print("\nOpenRouter usage summary")
    print("=" * 40)
    print(f"Total calls: {sum(v['calls'] for v in by_model.values())}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: {total_cost:.8f}")
    print("\nBy model")
    print("-" * 40)

    for model, stats in sorted(by_model.items()):
        print(f"{model}")
        print(f"  calls: {stats['calls']}")
        print(f"  prompt_tokens: {stats['prompt_tokens']}")
        print(f"  completion_tokens: {stats['completion_tokens']}")
        print(f"  total_tokens: {stats['total_tokens']}")
        print(f"  cost: {stats['cost']:.8f}")


if __name__ == "__main__":
    main()
