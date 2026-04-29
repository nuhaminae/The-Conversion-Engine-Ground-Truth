# The Conversion Engine Ground Truth

## Overview

The Conversion Engine Ground Truth is a curated dataset and evaluation harness for the Kai Conversion Engine agent.  
It benchmarks performance across four failure categories defined in Week 10: Perception (F1), Reasoning (F2), Action (F3), and Guardrails (F4). 

This interim submission covers Acts I and II — the audit, schema, and dataset authoring — prior to publication.

## Status

- ✅ Audit complete (`audit_memo.md`)
- ✅ Schema defined (`schema.json`)
- ✅ Dataset authored with three sources:
  - Trace-derived tasks (`data/tasks/trace_tasks.json`)
  - Synthetic programmatic pairs (`data/tasks/synthetic_pairs.json`)
  - Adversarial cases (`data/tasks/adversarial_cases.json`)
- ✅ Partitioned into `tenacious_bench_v0.1/train`, `dev`, and `held_out`
- ✅ Datasheet drafted (`datasheet.md`)
- ✅ Methodology documented (`methodology.md`)
- 🚧 HuggingFace publication planned for Act III

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/nuhaminae/The-Conversion-Engine.git
cd The-Conversion-Engine
pip install -r requirements.txt
```

Run generation scripts to reproduce dataset:

```bash
python src/data_prep/trace_tasks.py
python src/data_prep/synthetic_pairs.py
python src/data_prep/adversarial_cases.py
python src/data_prep/split_dataset.py
```

## Project Structure
```
The-Conversion-Engine/
├── README.md
├── audit_memo.md
├── schema.json
├── scoring_evaluator.py
├── datasheet.md
├── methodology.md
├── inter_rater_agreement.md
├── synthesis_memos/
│   ├── memo1.md
│   └── memo2.md
├── generation_scripts/
│   ├── trace_tasks.py
│   ├── synthetic_pairs.py
│   ├── adversarial_cases.py
│   └── split_dataset.py
├── tenacious_bench_v0.1/
│   ├── train/
│   ├── dev/
│   └── held_out/
└── data/tasks/
    ├── trace_tasks.json
    ├── synthetic_pairs.json
    └── adversarial_cases.json
```

## What’s Next

- Expand adversarial coverage with randomised mappings
- Improve synthetic diversity
- Conduct full scoring evaluation with rubric
- Prepare HuggingFace release (Act III)
