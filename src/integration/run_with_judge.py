# src/integration/run_with_judge.py

"""
Optional integration placeholder.

The Week 11 final pipeline evaluates the trained judge through:
  src/evaluation/eval_judge.py

The trained judge is a causal-LM LoRA DPO adapter, not a sequence-classification
model. Runtime integration should therefore use the same preference/log-prob
scoring approach implemented in eval_judge.py.

This file is intentionally not used by the final 5-step pipeline.
"""

if __name__ == "__main__":
    raise SystemExit(
        "run_with_judge.py is not used in the final Week 11 pipeline. "
        "Use src/evaluation/eval_judge.py for judge scoring, or "
        "scripts/package_final_artifacts.py for Step 5 packaging."
    )
    