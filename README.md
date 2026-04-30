# The Conversion Engine Ground Truth

[![CI](https://github.com/nuhaminae/The-Conversion-Engine-Ground-Truth/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/The-Conversion-Engine-Ground-Truth/actions/workflows/CI.yml)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort Imports](https://img.shields.io/badge/imports-isort-blue.svg)
![Flake8 Lint](https://img.shields.io/badge/lint-flake8-yellow.svg)

---

## Overview

The Conversion Engine Ground Truth is a curated dataset and evaluation harness for the Kai Conversion Engine agent.  
It benchmarks performance across four failure categories defined in Week 10: Perception (F1), Reasoning (F2), Action (F3), and Guardrails (F4).

This interim submission covers Acts I and II — the audit, schema, and dataset authoring — prior to publication.

---

## Key Features

---


## Table of Contents (Snippet)

- [The Conversion Engine Ground Truth](#the-conversion-engine-ground-truth)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Table of Contents (Snippet)](#table-of-contents-snippet)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
  - [Project Status](#project-status)

---

## Project Structure

``` bash
The-Conversion-Engine-Ground-Truth/
│
├── data/
│   ├── raw/                # Seed corpus from Tenacious (style guide, case studies, transcripts)
│   ├── processed/          # Cleaned & normalized datasets
│   ├── tasks/              # 200–300 evaluation tasks (trace-derived, synthetic, adversarial)
│   ├── splits/             # Train/dev/held-out partitions
│   └── datasheet.md        # Dataset documentation (sources, contamination checks, IRR logs)
│
├── models/
│   ├── judge/              # Judge model artifacts (LoRA weights, configs)
│   ├── checkpoints/        # Training checkpoints
│   └── model_card.md       # Model documentation
│
├── src/
│   ├── data_prep/          # Scripts for dataset construction & preprocessing
│   │   ├── trace_tasks.py
│   │   ├── synthetic_pairs.py
│   │   ├── adversarial_cases.py
│   │   └── split_dataset.py
│   │
│   ├── training/           # Training scripts for preference optimization
│   │   ├── train_judge.py
│   │   └── utils.py
│   │
│   ├── evaluation/            # Ablation & benchmark evaluation
│   │   ├── eval_judge.py
│   │   └── metrics.py
│   │
│   └── integration/            # Pipeline integration (agent + judge)
│       └── run_with_judge.py
│
├── notebooks/
│   ├── exploratory_data.ipynb   # Inspect seed corpus & tasks
│   ├── training_logs.ipynb      # Track judge training
│   └── evaluation_results.ipynb # Visualize ablations
│
├── reports/
│   ├── executive_memo.pdf       # 2-page memo for Tenacious leadership
│   └── blog_post.md             # Technical blog post for community
│
├── community/
│   ├── github_issue.md          # Contribution to open evaluation repo
│   └── workshop_submission.md   # Submission draft for evaluation workshop
│
├── configs/
│   ├── training_config.yaml     # Hyperparameters, LoRA settings
│   └── eval_config.yaml         # Evaluation parameters
│
├── scripts/
│   └── run_pipeline.bat         # Shell script to run end-to-end pipeline
│
├── project.toml                 # Dependencies (mirrors uv install list)
├── README.md                    # Project overview & instructions
└── LICENSE                      # License for dataset & model artifacts

```

---

## Installation

### Prerequisites

- Python 3.12  
- Git  

---

### Setup

```bash
git clone https://github.com/nuhaminae/The-Conversion-Engine-Ground-Truth.git
cd The-Conversion-Engine-Ground-Truth
uv sync   # recommended dependency management
```

Run generation scripts to reproduce dataset:

## Usage

```bash
python src/data_prep/trace_tasks.py
python src/data_prep/synthetic_pairs.py
python src/data_prep/adversarial_cases.py
python src/data_prep/split_dataset.py
```

---

## Project Status

- Expand adversarial coverage with randomised mappings
- Improve synthetic diversity
- Conduct full scoring evaluation with rubric
- Prepare HuggingFace release (Act III)
