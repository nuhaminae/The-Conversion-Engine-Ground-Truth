---
base_model: unsloth/Llama-3.2-1B-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/Llama-3.2-1B-Instruct-bnb-4bit
- dpo
- lora
- peft
- transformers
- trl
- unsloth
- tenacious-bench
- preference-optimization
- judge-model
---

# Tenacious-Bench Judge Checkpoint 17

## Checkpoint Summary

This directory contains **checkpoint-17** from the Week 11 Tenacious-Bench DPO judge training run.

This checkpoint is a PEFT/LoRA adapter checkpoint for a causal language model judge trained with Direct Preference Optimization (DPO). It is part of the **Path B: Preference-Tuned Judge** pipeline for the Tenacious Conversion Engine project.

The checkpoint was produced during training of a judge model intended to prefer high-quality Tenacious-style sales-agent outputs over low-quality outputs. The judge is used as a quality-control layer for evaluating candidate outreach/reply outputs.

> This checkpoint is primarily a **resumability and audit artifact**.  
> The final publishable adapter should be loaded from `models/judge/`.

## Model Details

### Model Description

- **Checkpoint name:** `checkpoint-17`
- **Final adapter path:** `models/judge/`
- **Checkpoint path:** `models/checkpoints/checkpoint-17/`
- **Base model:** `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- **Model type:** Causal language model with LoRA adapter
- **Training method:** Direct Preference Optimization, or DPO
- **Adapter method:** LoRA via PEFT
- **Library stack:** Unsloth, TRL, Transformers, PEFT
- **Primary task:** Preference judging / response quality ranking
- **Language:** English
- **Domain:** B2B sales-agent quality assurance for Tenacious Consulting-style outreach
- **Developed by:** Nuhamin Alemayehu
- **Project:** The Conversion Engine Ground Truth / Tenacious-Bench
- **Checkpoint step:** 17
- **Training status:** Completed successfully

### Base Model

This checkpoint was fine-tuned from:

```text
unsloth/Llama-3.2-1B-Instruct-bnb-4bit
````

The base model was selected because it is lightweight enough to train on a free Google Colab T4 runtime while still supporting instruction-following behavior needed for a judge/critic model.

### Adapter Type

This checkpoint contains LoRA adapter weights, not a full merged model.

It should be loaded as:

```text
base causal LM + PEFT LoRA adapter
```

It should **not** be loaded with `AutoModelForSequenceClassification`.

## Intended Use

This checkpoint is intended for:

1. Reproducing the Week 11 DPO judge training run.
2. Resuming training from checkpoint step 17 if needed.
3. Auditing the training artifacts produced during the final DPO run.
4. Comparing checkpoint artifacts with the final saved adapter under `models/judge/`.

The judge is designed to score or rank candidate outputs from the Tenacious Conversion Engine. It should prefer outputs that are concise, professional, action-oriented, grounded in the prospect/company signal, and safe around tool failures or adversarial prompts.

## Out-of-Scope Use

This checkpoint is not intended for:

- Direct customer-facing response generation.
- General-purpose email generation.
- High-stakes decision-making.
- Legal, medical, financial, or employment decisions.
- Use as a standalone safety system.
- Use as a sequence-classification model.
- Deployment without the associated evaluation pipeline.

## Training Data

The DPO training data came from Tenacious-Bench preference pairs.

Training files:

```text
data/training_data/preferences_train.jsonl
data/training_data/preferences_dev.jsonl
```

Required fields:

```text
prompt
chosen
rejected
```

Training split:

```text
65 preference pairs
```

Validation split:

```text
39 preference pairs
```

The broader Tenacious-Bench dataset was constructed from four authoring modes:

1. Trace-derived examples from Week 10 system behavior.
2. Programmatic task sweeps.
3. Multi-LLM / model-assisted synthesis.
4. Hand-authored adversarial cases.

The selected benchmark contains balanced good/bad labels and complete paired examples across train, dev, and held-out splits.

## Training Procedure

### Objective

The model was trained with Direct Preference Optimization.

For each training example, the model receives:

```text
prompt
chosen response
rejected response
```

The DPO objective encourages the model to assign higher preference/reward to the chosen response than to the rejected response.

### Training Stack

- **Runtime:** Google Colab
- **GPU:** Tesla T4
- **Training framework:** Unsloth
- **Preference optimization:** TRL DPOTrainer
- **Adapter framework:** PEFT
- **Quantization:** 4-bit base model
- **Optimizer:** `adamw_8bit`
- **Precision:** fp16 on T4
- **Gradient checkpointing:** Unsloth gradient checkpointing
- **Seed:** 42

### Hyperparameters

Core training settings:

```text
base_model: unsloth/Llama-3.2-1B-Instruct-bnb-4bit
num_epochs: 1
batch_size: 1
gradient_accumulation_steps: 4
effective_batch_size: 4
learning_rate: 0.000005
warmup_ratio: 0.1
beta: 0.1
optimizer: adamw_8bit
max_length: 1024
max_prompt_length: 512
seed: 42
```

LoRA settings:

```text
r: 16
alpha: 16
dropout: 0
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

Trainable parameters:

```text
11,272,192 trainable parameters
1,247,086,592 total parameters
approximately 0.90% trained
```

## Training Results

The training run completed successfully.

Final training metrics:

```text
train_loss: 0.6785038884948281
train_runtime_seconds: 46.6419
train_samples_per_second: 1.394
train_steps_per_second: 0.364
epoch: 1.0
```

Final validation metrics:

```text
eval_loss: 0.6711041331291199
eval_rewards/chosen: 0.03396207466721535
eval_rewards/rejected: -0.01113205123692751
eval_rewards/accuracies: 0.9487179517745972
eval_rewards/margins: 0.04509412497282028
```

Interpretation:

The checkpoint shows a learned preference signal. The validation reward accuracy indicates that the trained model preferred the chosen response over the rejected response for most validation preference pairs. The positive reward margin indicates separation between chosen and rejected responses.

## Held-Out Evaluation Context

The final model produced from this training run was evaluated on the held-out Tenacious-Bench split.

Held-out evaluation size:

```text
52 pointwise examples
26 preference pairs
```

The final fine-tuned DPO judge achieved:

```text
accuracy: 0.9615384615
precision: 0.9615384615
recall: 0.9615384615
f1: 0.9615384615
strict_pairwise_accuracy: 0.9615384615
```

Comparison systems:

```text
Week 10 baseline strict pairwise accuracy: 0.6153846154
Prompt-engineered judge strict pairwise accuracy: 0.7692307692
Fine-tuned DPO judge strict pairwise accuracy: 0.9615384615
```

Interpretation:

The fine-tuned DPO judge outperformed both the Week 10 no-judge baseline and the prompt-engineered judge on the deployment-style strict pairwise metric.

## How to Load This Checkpoint

This checkpoint is a PEFT adapter. To use it, load the base model and then load the adapter.

Example:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
adapter_path = "models/checkpoints/checkpoint-17"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

For the final model artifact, prefer:

```python
adapter_path = "models/judge"
```

rather than this checkpoint directory.

## How to Score Preference Pairs

This model should be evaluated using a DPO-style preference scoring approach.

For each pair:

```text
reward(response) = beta * (policy_logprob(response | prompt) - reference_logprob(response | prompt))
```

The judge predicts correctly when:

```text
reward(chosen) > reward(rejected)
```

Use:

```text
src/evaluation/eval_judge.py
```

for the canonical evaluation flow.

## Files in This Checkpoint

Expected checkpoint files include:

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
trainer_state.json
training_args.bin
optimizer.pt
scheduler.pt
scaler.pt
rng_state.pth
README.md
```

### File Notes

- `adapter_model.safetensors`: LoRA adapter weights at checkpoint step 17.
- `adapter_config.json`: PEFT adapter configuration.
- `tokenizer.json`: Tokenizer file saved with the checkpoint.
- `tokenizer_config.json`: Tokenizer configuration.
- `special_tokens_map.json`: Special token mappings.
- `chat_template.jinja`: Chat template used by the tokenizer/model stack.
- `trainer_state.json`: Trainer state at checkpoint step 17.
- `training_args.bin`: Serialized training arguments.
- `optimizer.pt`: Optimizer state for resuming training.
- `scheduler.pt`: Scheduler state for resuming training.
- `scaler.pt`: Mixed-precision scaler state.
- `rng_state.pth`: Random number generator state.

The optimizer, scheduler, scaler, and RNG files are useful for resuming training but are not required for inference.

## Relationship to Final Model

This checkpoint is not the primary public model artifact.

Use this for:

```text
training reproducibility
debugging
resuming training
audit trail
```

Use `models/judge/` for:

```text
final LoRA adapter
evaluation
deployment experiments
Hugging Face model upload
```

## Limitations

This checkpoint has several limitations:

1. It was trained on a small preference-pair dataset.
2. It is domain-specific to Tenacious-style B2B sales-agent outputs.
3. It is not a general-purpose judge.
4. It is not a sequence classifier.
5. It requires the original base model to be loaded.
6. It should be evaluated with pairwise preference scoring, not standalone text generation quality.
7. The held-out set is small, so reported performance should be interpreted as prototype evidence rather than production certification.
8. The checkpoint may contain optimizer and RNG state files that are unnecessary for public release.

## Safety Considerations

The judge was trained to penalize outputs that:

- Leak secrets or API keys.
- Reveal system prompts or internal instructions.
- Follow prompt-injection instructions.
- Continue selling after an opt-out.
- Use placeholder, localhost, or broken links.
- Invent unsupported pricing or capacity claims.
- Ignore layoffs or weak hiring signals.
- Produce generic or overly verbose sales filler.

The model should be used as one component in a broader safety and quality-control pipeline.

## Recommended Release Practice

For a public model upload, prefer uploading:

```text
models/judge/adapter_model.safetensors
models/judge/adapter_config.json
models/judge/tokenizer.json
models/judge/tokenizer_config.json
models/judge/special_tokens_map.json
models/judge/chat_template.jinja
models/model_card.md
```

Avoid uploading checkpoint-only training internals unless resumability is required:

```text
optimizer.pt
scheduler.pt
scaler.pt
rng_state.pth
training_args.bin
```

## Reproducibility Artifacts

Related artifacts:

```text
configs/training_config.yaml
reports/training/dataset_summary.json
reports/training/training_config_used.yaml
reports/training/training_run.log
reports/training/training_summary.json
reports/training/training_log_summary.json
reports/ablation_results.json
```

These files document the training configuration, data split, run metrics, and held-out ablation results.

## Citation

No formal paper is associated with this checkpoint.

Suggested citation:

```bibtex
@misc{tenacious_bench_judge_2026,
  title = {Tenacious-Bench Judge v0.1},
  author = {Alemayehu, Nuhamin},
  year = {2026},
  note = {LoRA/DPO judge trained for the Week 11 Tenacious Conversion Engine benchmark}
}
```

## Framework Versions

- PEFT: 0.19.1
- Transformers: 4.56.2
- TRL: 0.22.2
- Unsloth: 2026.4.8
- Torch: 2.10.0+cu128
- CUDA runtime: Colab Tesla T4 environment

## Contact

For questions about this checkpoint or the Tenacious-Bench Week 11 pipeline, contact:

```text
Nuhamin Alemayehu
```
