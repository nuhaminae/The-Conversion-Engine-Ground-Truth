# The Conversion Engine Ground Truth

[![CI](https://github.com/nuhaminae/The-Conversion-Engine-Ground-Truth/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/The-Conversion-Engine-Ground-Truth/actions/workflows/CI.yml)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort Imports](https://img.shields.io/badge/imports-isort-blue.svg)
![Flake8 Lint](https://img.shields.io/badge/lint-flake8-yellow.svg)

---

## Overview

**The Conversion Engine Ground Truth** is the Week 11 evaluation and judge-training project for the Tenacious Conversion Engine.

The project builds **Tenacious-Bench v0.1**, a 260-task benchmark for B2B sales-agent quality assurance, and trains a **preference-tuned judge** using Direct Preference Optimization (DPO). The judge is designed to catch low-quality sales-agent outputs before they reach a prospect.

The core Week 10 failure was not that the agent never knew what to do. It was that the agent was inconsistent: it could identify a prospect’s intent but sometimes failed to complete the correct action, such as sending a booking link after a clear meeting request. This project addresses that failure by building a benchmark and a judge/critic layer.

---

## Table of Contents (Snippet)

- [The Conversion Engine Ground Truth](#the-conversion-engine-ground-truth)
  - [Overview](#overview)
  - [Table of Contents (Snippet)](#table-of-contents-snippet)
  - [Final Result](#final-result)
  - [Project Goals](#project-goals)
  - [Benchmark Summary](#benchmark-summary)
  - [Failure Modes Covered](#failure-modes-covered)
  - [Model Summary](#model-summary)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Environment Variables](#environment-variables)
  - [Running the Pipeline](#running-the-pipeline)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
    - [Fine-tuned DPO judge](#fine-tuned-dpo-judge)
    - [Week 10 no-judge baseline](#week-10-no-judge-baseline)
    - [Prompt-engineered judge](#prompt-engineered-judge)
  - [Report Generation](#report-generation)
  - [Packaging](#packaging)
  - [Hugging Face Publishing](#hugging-face-publishing)
  - [Key Artifacts](#key-artifacts)
  - [Limitations](#limitations)
  - [License](#license)
  - [Citation](#citation)
  - [Contact](#contact)

---

## Final Result

On the held-out benchmark:

| System | Accuracy | Precision | Recall | F1 | Strict Pairwise Accuracy |
|---|---:|---:|---:|---:|---:|
| Week 10 baseline | 80.77% | 94.44% | 65.38% | 77.27% | 61.54% |
| Prompt-engineered judge | 88.46% | 100.00% | 76.92% | 86.96% | 76.92% |
| Fine-tuned DPO judge | **96.15%** | 96.15% | **96.15%** | **96.15%** | **96.15%** |

Primary finding:

> Prompting alone improved the judge, but DPO tuning produced the best deployment-style quality gate.

---

## Project Goals

This repo supports four goals:

1. Build a Tenacious-specific evaluation benchmark.
2. Train a lightweight DPO judge using preference pairs.
3. Compare the trained judge against both the Week 10 baseline and a prompt-only judge.
4. Package the dataset, model adapter, and reports for Hugging Face publication.

---

## Benchmark Summary

Tenacious-Bench v0.1 was built from 436 candidate tasks and selected down to 260 balanced benchmark tasks.

| Split | Tasks | Pairs | Good | Bad |
|---|---:|---:|---:|---:|
| Train | 130 | 65 | 65 | 65 |
| Dev | 78 | 39 | 39 | 39 |
| Held-out | 52 | 26 | 26 | 26 |
| **Total** | **260** | **130** | **130** | **130** |

The benchmark contains four authoring modes:

| Source mode | Selected tasks |
|---|---:|
| Trace-derived | 78 |
| Programmatic | 78 |
| Multi-LLM synthesis | 64 |
| Hand-authored adversarial | 40 |

---

## Failure Modes Covered

The benchmark targets Tenacious-specific failures including:

- Missed meeting intent.
- Failure to send a booking link.
- Tool or CRM failure mishandling.
- Weak or contradictory company-signal handling.
- Unsupported pricing, capacity, or hiring claims.
- Wrong-person replies.
- Opt-out and rude replies.
- Prompt-injection attempts.
- Secret or system-prompt leakage.
- Broken, placeholder, or localhost links.
- Generic filler or off-brand sales tone.

---

## Model Summary

The final judge is a LoRA adapter trained with DPO.

| Item | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` |
| Training method | Direct Preference Optimization |
| Adapter method | LoRA / PEFT |
| Runtime | Google Colab T4 |
| Train pairs | 65 |
| Dev pairs | 39 |
| Held-out pairs | 26 |
| Final adapter path | `models/judge/` |

Training results:

| Metric | Value |
|---|---:|
| Train loss | 0.6785 |
| Eval loss | 0.6711 |
| Eval reward accuracy | 94.87% |
| Eval reward margin | 0.0451 |
| Runtime | 122.17 seconds |

---

## Project Structure

```bash
The-Conversion-Engine-Ground-Truth/
│
├── data/
│   ├── raw/
│   ├── tasks/
│   └── training_data/
│
├── tenacious_bench/
│   ├── train/
│   ├── dev/
│   ├── held_out/
│   ├── dpo/
│   ├── examples/
│   ├── schema.json
│   ├── datasheet.md
│   ├── contamination_check.json
│   └── inter_rater_agreement.md
│
├── models/
│   ├── judge/
│   ├── checkpoints/
│   └── model_card.md
│
├── src/
│   ├── data_prep/
│   │   ├── trace_tasks.py
│   │   ├── programmatic_tasks.py
│   │   ├── synthetic_pairs.py
│   │   ├── adversarial_cases.py
│   │   ├── split_dataset.py
│   │   └── create_preference_pairs.py
│   │
│   ├── training/
│   │   ├── train_judge.py
│   │   └── utils.py
│   │
│   └── evaluation/
│        ├── eval_judge.py
│        ├── eval_baseline.py
│        ├── eval_prompted_judge.py
│        └── metrics.py
│
├── notebooks/
│   ├── exploratory_data.ipynb
│   ├── training_logs.ipynb
│   └── evaluation_results.ipynb
│
├── reports/
│   ├── training/
│   ├── evaluation/
│   ├── blog_post.md
│   ├── executive_memo.md
│   ├── executive_memo.pdf
│   ├── evaluation_results.html
│   └── ablation_results.json
│
├── community/
│   ├── github_issue.md
│   └── workshop_submission.md
│
├── configs/
│   ├── training_config.yaml
│   └── eval_config.yaml
│
├── scripts/
│   ├── run_pipeline.bat
│   ├── summarize_openrouter_costs.py
│   ├── package_final_artifacts.py
│   ├── upload_dataset_to_hf.py
│   └── upload_model_to_hf.py
│
├── project.toml
├── README.md
└── LICENSE
```

---

## Installation

### Prerequisites

- Python 3.11+
- Git
- Jupyter
- Google Colab T4 or another CUDA-compatible GPU for Unsloth training/evaluation
- Optional: Hugging Face account for publishing
- Optional: OpenRouter API key for synthetic generation

---

### Setup

```bash
git clone https://github.com/nuhaminae/The-Conversion-Engine-Ground-Truth.git
cd The-Conversion-Engine-Ground-Truth
```

Install dependencies using your preferred environment manager. For local dataset/evaluation utilities:

```bash
pip install -r requirements.txt
```

or, if using `uv`:

```bash
uv sync
```

For Colab/Unsloth training, use the notebook cells in the project documentation rather than relying on the local Windows environment.

---

## Environment Variables

Create a local `.env` file when needed:

```text
OPENROUTER_API_KEY=your_openrouter_key
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=
```

Never commit `.env`.

---

## Running the Pipeline

From the repository root:

```bat
scripts\run_pipeline.bat
```

The pipeline has five stages:

1. Prepare datasets.
2. Train the DPO judge.
3. Evaluate the fine-tuned judge, baseline, and prompt-engineered judge.
4. Generate comparison reports.
5. Package final artifacts.

Some stages are best run in different environments:

| Stage                       | Recommended environment |
| --------------------------- | ----------------------- |
| Dataset prep                | Local Windows / VS Code |
| DPO training                | Google Colab T4         |
| Fine-tuned judge evaluation | Google Colab T4         |
| Prompted judge evaluation   | Google Colab T4         |
| Baseline evaluation         | Local or Colab          |
| Report generation           | Local                   |
| Packaging                   | Local                   |

---

## Dataset Preparation

Run manually:

```bash
python src/data_prep/trace_tasks.py
python src/data_prep/programmatic_tasks.py
python src/data_prep/synthetic_pairs.py
python scripts/summarize_openrouter_costs.py
python src/data_prep/adversarial_cases.py
python src/data_prep/split_dataset.py
python src/data_prep/create_preference_pairs.py
```

Expected outputs include:

```text
data/tasks/
tenacious_bench/train/train.jsonl
tenacious_bench/dev/dev.jsonl
tenacious_bench/held_out/held_out.jsonl
tenacious_bench/dpo/train_dpo.jsonl
tenacious_bench/dpo/dev_dpo.jsonl
tenacious_bench/dpo/held_out_dpo.jsonl
data/training_data/preferences_train.jsonl
data/training_data/preferences_dev.jsonl
```

---

## Training

Run on a CUDA/Unsloth-compatible runtime:

```bash
python src/training/train_judge.py --config configs/training_config.yaml
```

Expected outputs:

```text
models/judge/
models/checkpoints/
reports/training/dataset_summary.json
reports/training/training_summary.json
reports/training/training_config_used.yaml
reports/training/training_run.log
```

---

## Evaluation

### Fine-tuned DPO judge

```bash
python src/evaluation/eval_judge.py --config configs/eval_config.yaml
```

### Week 10 no-judge baseline

```bash
python src/evaluation/eval_baseline.py --config configs/eval_config.yaml
```

### Prompt-engineered judge

```bash
python src/evaluation/eval_prompted_judge.py --config configs/eval_config.yaml
```

Expected outputs:

```text
reports/fine_tuned_judge_metrics.json
reports/baseline_metrics.json
reports/prompted_judge_metrics.json
reports/fine_tuned_judge_pair_scores.jsonl
reports/baseline_pair_scores.jsonl
reports/prompted_judge_pair_scores.jsonl
reports/*confusion_matrix.png
```

---

## Report Generation

```bash
jupyter nbconvert --execute notebooks/evaluation_results.ipynb --to html --output-dir reports --output evaluation_results
jupyter nbconvert --execute notebooks/training_logs.ipynb --to html --output-dir reports --output training_logs
jupyter nbconvert --execute notebooks/exploratory_data.ipynb --to html --output-dir reports --output exploratory_data
```

Expected outputs:

```text
reports/evaluation_results.html
reports/training_logs.html
reports/exploratory_data.html
reports/ablation_results.json
reports/ablation_comparison_chart.png
reports/pairwise_ablation_chart.png
```

---

## Packaging

```bash
python scripts/package_final_artifacts.py
```

Expected outputs:

```text
dist/week11_act_v_package/
dist/week11_act_v_package.zip
```

---

## Hugging Face Publishing

The final release should use two Hugging Face repositories:

1. Dataset repo for `tenacious_bench/`
2. Model repo for `models/judge/`

Example:

```bash
python scripts/upload_dataset_to_hf.py
python scripts/upload_model_to_hf.py
```

## Key Artifacts

| Artifact                          | Purpose                     |
| --------------------------------- | --------------------------- |
| `tenacious_bench/datasheet.md`    | Dataset documentation       |
| `tenacious_bench/schema.json`     | Machine-readable schema     |
| `models/model_card.md`            | Model documentation         |
| `reports/evaluation_results.html` | Final comparison report     |
| `reports/ablation_results.json`   | Delta A and Delta B results |
| `reports/blog_post.md`            | Public technical write-up   |
| `reports/executive_memo.md`       | Leadership memo             |
| `dist/week11_act_v_package.zip`   | Final packaged release      |

---

## Limitations

This is a v0.1 benchmark and judge.

Known limitations:

- Held-out set is small: 52 pointwise examples / 26 pairs.
- Dataset is English-only.
- Benchmark is specific to Tenacious-style B2B sales workflows.
- Prompted and fine-tuned judges were evaluated offline, not in a live production system.
- The baseline is a deterministic no-judge approximation of the Week 10 system.
- The judge should not be used as the only production safety mechanism.
- Additional human inter-rater validation should be added in v0.2.

---

## License

See `LICENSE`.

Recommended interpretation:

- Code: MIT License.
- Dataset: CC-BY-4.0 unless otherwise specified.
- Model adapter: subject to the underlying base model license and the terms documented in `models/model_card.md`.

---

## Citation

```bibtex
@misc{tenacious_bench_2026,
  title = {Tenacious-Bench v0.1: A B2B Sales-Agent Evaluation Benchmark for Preference-Tuned Judges},
  author = {Alemayehu, Nuhamin},
  year = {2026},
  note = {Week 11 Conversion Engine Ground Truth benchmark}
}
```

---

## Contact

Nuhamin Alemayehu
GitHub: [https://github.com/nuhaminae](https://github.com/nuhaminae)
