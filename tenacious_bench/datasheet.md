# Datasheet for Tenacious-Bench v0.1

## Dataset Summary

**Dataset name:** Tenacious-Bench v0.1  
**Project:** The Conversion Engine Ground Truth  
**Path:** Path B — Preference-Tuned Judge  
**Primary task:** Evaluate and train a judge/critic for Tenacious-style B2B sales-agent outputs  
**Language:** English  
**Domain:** B2B sales outreach, reply handling, lead qualification, and meeting-booking quality assurance  
**Maintainer:** Nuhamin Alemayehu  
**Repository path:** `tenacious_bench/`  
**Recommended license:** CC-BY-4.0 for dataset documentation and task data, unless project constraints require a different license  
**Status:** v0.1 final benchmark package for Week 11  

Tenacious-Bench v0.1 is a small, purpose-built evaluation benchmark for measuring whether a B2B sales agent produces safe, grounded, action-complete, and Tenacious-style responses. It was created for Week 11 of the Conversion Engine project after Week 10 revealed a specific inconsistency failure: the agent could often recognize prospect intent, but sometimes failed to complete the correct action, such as sending a booking link after clear buying intent.

The benchmark supports a Path B intervention: a preference-tuned judge trained with Direct Preference Optimization (DPO). The judge is designed to score candidate outputs and reject or flag low-quality responses before they reach a prospect.

---

# 1. Motivation

## 1.1 Purpose

Tenacious-Bench v0.1 was created to answer a question that generic agent or retail benchmarks do not directly answer:

> Does the Conversion Engine work for Tenacious-style B2B sales conversations, Tenacious voice, Tenacious failure modes, and Tenacious prospect workflows?

The benchmark is intended to evaluate whether a sales-agent output:

- Correctly responds to prospect intent.
- Sends a booking link when the prospect clearly wants to book.
- Handles wrong-person replies politely.
- Respects opt-outs and rude replies.
- Refuses prompt-injection or secret-leakage requests.
- Avoids unsupported pricing, capacity, hiring, or delivery claims.
- Grounds outreach in the supplied company/prospect signal.
- Avoids generic filler, broken links, placeholder links, or unsafe internal references.

## 1.2 Gap Addressed

Existing public benchmarks such as retail-agent or general tool-use benchmarks do not capture Tenacious-specific failure modes, including:

- Clear meeting intent without booking-link follow-through.
- CRM/tool failure leading to dropped warm leads.
- Public-signal mismatch, such as pitching growth during layoffs or weak hiring signals.
- Overconfident claims about pricing, capacity, or delivery scope.
- Prompt-injection attempts embedded in prospect replies.
- Unsafe exposure of system prompts, API keys, or internal configuration.
- Sales-tone errors such as hype, generic filler, or continuing to sell after opt-out.

The benchmark focuses on these business-specific consistency and safety failures rather than broad language quality.

## 1.3 Intended Users

Intended users include:

- The Conversion Engine project team.
- Researchers or builders evaluating B2B sales-agent quality.
- Developers building judge/critic layers for LLM-powered outreach systems.
- Reviewers assessing the Week 11 Path B intervention.

## 1.4 Intended Decisions Supported

The dataset is designed to support decisions such as:

- Whether a trained judge improves over the Week 10 no-judge baseline.
- Whether fine-tuning improves over prompt engineering alone.
- Which failure modes remain unresolved.
- Whether the judge is safe enough to consider as a rollback/rejection layer in a production workflow.

It is not designed to independently certify a production sales system.

---

# 2. Composition

## 2.1 Dataset Size

The dataset was authored from a larger candidate pool and then selected into a balanced benchmark.

Candidate pool:

| Source file | Candidate tasks |
|---|---:|
| Trace-derived tasks | 176 |
| Programmatic tasks | 102 |
| Synthetic / multi-LLM pairs | 96 |
| Hand-authored adversarial cases | 62 |
| **Total candidate tasks** | **436** |

Selected benchmark:

| Split | Tasks | Pairs |
|---|---:|---:|
| Train | 130 | 65 |
| Dev | 78 | 39 |
| Held-out | 52 | 26 |
| **Total** | **260** | **130** |

Each pair contains one good/chosen output and one bad/rejected output.

## 2.2 Label Balance

The selected benchmark is fully label-balanced.

| Split | Good / chosen label `1` | Bad / rejected label `0` |
|---|---:|---:|
| Train | 65 | 65 |
| Dev | 39 | 39 |
| Held-out | 26 | 26 |
| **Total** | **130** | **130** |

## 2.3 Source-Mode Composition

Tenacious-Bench v0.1 uses four authoring modes.

| Source mode | Candidate tasks | Selected tasks |
|---|---:|---:|
| Trace-derived | 176 | 78 |
| Programmatic | 102 | 78 |
| Multi-LLM synthesis | 96 | 64 |
| Hand-authored adversarial | 62 | 40 |
| **Total** | **436** | **260** |

Selected tasks by split:

| Source mode | Train | Dev | Held-out | Total |
|---|---:|---:|---:|---:|
| Trace-derived | 32 | 24 | 22 | 78 |
| Programmatic | 46 | 18 | 14 | 78 |
| Multi-LLM synthesis | 40 | 18 | 6 | 64 |
| Hand-authored adversarial | 12 | 18 | 10 | 40 |

The selected benchmark intentionally includes all four modes so that the model is evaluated on real trace-derived behavior, controlled programmatic sweeps, model-assisted synthetic variation, and hand-authored adversarial edge cases.

## 2.4 Data Fields

The pointwise split files contain task-level records. Core fields include:

| Field | Description |
|---|---|
| `task_id` | Unique identifier for a pointwise task. |
| `pair_id` | Shared identifier linking the good and bad outputs in a preference pair. |
| `source_mode` | Authoring mode: `trace-derived`, `programmatic`, `multi-LLM synthesis`, or `hand-authored adversarial`. |
| `scenario_type` | Scenario category, such as booking intent, pricing, capacity, weak signal, tool failure, prompt injection, or opt-out. |
| `failure_code` | Failure taxonomy code for rejected examples where applicable. |
| `failure_mode_tag` | Human-readable failure-mode label. |
| `prospect_input` | Prospect reply, company brief, or task input used to evaluate the candidate output. |
| `agent_output` | Candidate sales-agent response. |
| `label` | `1` for good/chosen output, `0` for bad/rejected output. |
| `metadata` | Additional construction or provenance information. |

The DPO split files contain preference-pair records. Core fields include:

| Field | Description |
|---|---|
| `pair_id` | Unique pair identifier. |
| `prompt` | Input context shown to the judge/training model. |
| `chosen` | Preferred response. |
| `rejected` | Dispreferred response. |
| `source_mode` | Authoring mode. |
| `scenario_type` | Scenario category. |
| `failure_code` | Failure taxonomy code where applicable. |
| `failure_mode_tag` | Human-readable failure-mode label. |
| `metadata` | Additional construction or provenance information. |

## 2.5 Failure Taxonomy

The benchmark includes examples across the project failure taxonomy. The candidate pool contains the following failure-code distribution:

| Failure code | Candidate count |
|---|---:|
| None / good examples | 218 |
| F3.2 | 58 |
| F1.2 | 51 |
| F3.1 | 25 |
| F3.3 | 17 |
| F2.2 | 16 |
| F4.2 | 9 |
| F2.1 | 9 |
| F4.3 | 8 |
| F1.1 | 8 |
| F4.1 | 6 |
| F1.4 | 6 |
| F2.3 | 5 |

The exact definitions of the failure codes are maintained in the project’s failure taxonomy and schema documentation. At a high level, the dataset emphasizes:

- Intent recognition and response-action consistency.
- Tool and CRM failure handling.
- Grounding against public hiring/company signals.
- Style and voice alignment.
- Safety against prompt injection and secret leakage.
- Commercial-claim discipline around pricing, capacity, and delivery.

## 2.6 Layered Detail

Following the Data Cards framing, Tenacious-Bench can be inspected at three levels.

### Telescopic layer

At the dataset level, Tenacious-Bench v0.1 is a 260-task benchmark for B2B sales-agent quality evaluation, split into train/dev/held-out partitions and DPO preference-pair files.

### Periscopic layer

At the group level, tasks can be grouped by:

- Split: train, dev, held-out.
- Source mode: trace-derived, programmatic, multi-LLM synthesis, hand-authored adversarial.
- Label: chosen/good or rejected/bad.
- Failure code and scenario type.
- DPO pair membership.

### Microscopic layer

At the example level, each task records the exact prospect input, candidate output, label, failure metadata, and pair ID. This allows reviewers to trace numeric claims back to individual tasks and held-out examples.

---

# 3. Collection and Authoring Process

## 3.1 Overview

Tenacious-Bench was built from limited seed materials and engineering effort rather than a large pre-existing labeled dataset. The benchmark uses four task-authoring modes to increase coverage and reduce overfitting to one construction method.

## 3.2 Trace-Derived Tasks

Trace-derived tasks were constructed from Week 10 Conversion Engine behavior. These examples represent the highest-fidelity distribution because they are based on actual system outputs and failure modes observed during the previous project stage.

Trace-derived examples include cases where the agent:

- Correctly recognized prospect intent but failed to complete the appropriate action.
- Dropped a warm lead after a tool or CRM lookup failure.
- Produced incomplete or inconsistent follow-up.
- Succeeded on a similar input, making the failure an inconsistency problem rather than a capability absence.

## 3.3 Programmatic Tasks

Programmatic tasks were generated from structured templates and parameter sweeps. These tasks systematically vary inputs such as:

- Prospect intent clarity.
- Company hiring signal.
- Layoff or weak-signal status.
- Pricing request.
- Capacity or delivery constraints.
- Wrong-person reply.
- Opt-out or rude reply.
- Tool-failure scenario.
- Prompt-injection phrasing.

This mode improves coverage over specific failure dimensions and creates controlled variations for evaluation.

## 3.4 Multi-LLM Synthesis

Multi-LLM synthesis was used to expand the task space beyond hand-written and trace-derived examples. In the final pipeline, OpenRouter/Qwen generation was used for synthetic preference-pair generation.

These examples are marked with:

```json
{
  "source_mode": "multi-LLM synthesis"
}
````

For cost tracking and reproducibility, OpenRouter generation calls should be logged separately in:

```text
reports/openrouter_usage.jsonl
```

The synthetic generation stage is intended to provide diversity in prospect phrasing, scenario framing, and candidate response style.

## 3.5 Hand-Authored Adversarial Cases

Hand-authored adversarial cases target edge cases likely to defeat the Week 10 system or a generic judge. These include:

* Prompt-injection attempts.
* Requests for secrets or internal system instructions.
* Rude or opt-out replies.
* Contradictory public-signal scenarios.
* Broken-link or placeholder-link cases.
* Unsupported pricing or capacity claims.
* Cases where a superficially polite answer is still operationally wrong.

The adversarial slice carries high originality weight because it encodes project-specific failure knowledge.

## 3.6 Human and Model Roles

The dataset combines:

* Human-authored schema and failure taxonomy.
* Programmatic generation scripts.
* Model-assisted synthetic generation.
* Human review and filtering of outputs.
* Deterministic checks for pair integrity and schema conformance.

The final benchmark is not a raw dump of model-generated data. It is a curated evaluation dataset with explicit source-mode metadata and balanced preference pairs.

---

# 4. Preprocessing, Cleaning, and Validation

## 4.1 Normalization

Preprocessing included:

* Cleaning and normalizing task records.
* Standardizing field names.
* Ensuring every selected pair has one good/chosen and one bad/rejected output.
* Converting the training partition into DPO-ready preference pairs.
* Writing machine-readable JSONL files for all splits.

The benchmark intentionally preserves original wording of prospect inputs and candidate outputs where relevant because tone, phrasing, and action completeness are part of the evaluation.

## 4.2 Deduplication

The pipeline deduplicates by task and pair identifiers and validates that each selected pair is complete. Pair integrity is verified as:

| Split    | Total pairs | Complete pairs | Broken pairs |
| -------- | ----------: | -------------: | -----------: |
| Train    |          65 |             65 |            0 |
| Dev      |          39 |             39 |            0 |
| Held-out |          26 |             26 |            0 |

## 4.3 Split Protocol

The final split protocol follows the Week 11 requirement:

| Split    | Share | Tasks | Pairs | Use                                  |
| -------- | ----: | ----: | ----: | ------------------------------------ |
| Train    |   50% |   130 |    65 | DPO judge training                   |
| Dev      |   30% |    78 |    39 | Validation, debugging, and iteration |
| Held-out |   20% |    52 |    26 | Final ablation evaluation only       |

The held-out split is used only for final evaluation of the Week 10 baseline, prompt-engineered judge, and fine-tuned DPO judge.

## 4.4 DPO Conversion

The Path B training data is stored in:

```text
data/training_data/preferences_train.jsonl
data/training_data/preferences_dev.jsonl
```

The canonical DPO benchmark files are stored in:

```text
tenacious_bench/dpo/train_dpo.jsonl
tenacious_bench/dpo/dev_dpo.jsonl
tenacious_bench/dpo/held_out_dpo.jsonl
```

DPO summary:

| File                 | Pairs | Required fields present |
| -------------------- | ----: | ----------------------- |
| `train_dpo.jsonl`    |    65 | Yes                     |
| `dev_dpo.jsonl`      |    39 | Yes                     |
| `held_out_dpo.jsonl` |    26 | Yes                     |

Each DPO row contains:

```text
prompt
chosen
rejected
```

## 4.5 Contamination and Leakage Controls

The intended contamination protocol includes:

1. N-gram overlap checks between train/dev/held-out.
2. Semantic similarity checks between training and held-out examples.
3. Time-shift verification for external public-signal sources.
4. Pair-level isolation to avoid a chosen or rejected member leaking across splits.
5. Preference-leakage prevention by documenting model routes for synthetic generation and judge training.

Known completed validation:

* Train/dev/held-out splits are disjoint at the pair level.
* All pairs are complete.
* Label counts are balanced across splits.
* Held-out DPO pairs are separate from training DPO pairs.

Before public release, the contamination report should be included at:

```text
tenacious_bench/contamination_check.json
```

If the file is not present, the package should be treated as documentation-complete but contamination-report-incomplete.

## 4.6 Inter-Rater Agreement

The Week 11 target process calls for hand-labeling a 30-task subset twice with a delay and recording agreement. If the agreement score is below the target threshold on any dimension, the rubric should be revised and the subset re-labeled.

Expected artifact:

```text
tenacious_bench/inter_rater_agreement.md
```

If not present, this should be listed as a known documentation gap before public release.

---

# 5. Uses

## 5.1 Primary Use

The primary use is to train and evaluate a preference-tuned judge for the Tenacious Conversion Engine.

For Path B, the training procedure uses DPO preference pairs:

```text
prompt
chosen
rejected
```

The trained judge is then evaluated as a causal-LM LoRA preference scorer.

## 5.2 Evaluation Use

The dataset supports three main held-out comparisons:

1. **Delta A:** Fine-tuned DPO judge vs. Week 10 no-judge baseline.
2. **Delta B:** Fine-tuned DPO judge vs. prompt-engineered judge on the same base model.
3. **Strict pairwise/deployment-style accuracy:** Whether the system accepts the good output and rejects the bad output in each held-out pair.

## 5.3 Secondary Use

Secondary uses include:

* Auditing B2B sales-agent failure modes.
* Testing judge calibration.
* Comparing prompt-only vs. trained judge approaches.
* Building additional adversarial cases for v0.2.
* Developing a scoring evaluator or rollback layer for generated sales outputs.

## 5.4 Out-of-Scope Uses

Tenacious-Bench v0.1 is not intended for:

* General retail-agent evaluation.
* Consumer customer-support evaluation.
* Legal, medical, financial, or employment decisions.
* Measuring general language-model helpfulness.
* Automatically sending sales emails without additional production controls.
* Certifying that a sales agent is safe for production use.
* Training a general-purpose model unrelated to the Tenacious workflow.

---

# 6. Distribution

## 6.1 Recommended Repository Layout

The dataset should be distributed with the following structure:

```text
tenacious_bench/
├── README.md
├── datasheet.md
├── schema.json
├── train/
│   └── train.jsonl
├── dev/
│   └── dev.jsonl
├── held_out/
│   └── held_out.jsonl
├── dpo/
│   ├── train_dpo.jsonl
│   ├── dev_dpo.jsonl
│   └── held_out_dpo.jsonl
├── examples/
│   └── ...
├── contamination_check.json
└── inter_rater_agreement.md
```

The final packaging step should copy the benchmark and documentation into:

```text
dist/week11_act_v_package/tenacious_bench/
```

## 6.2 License

Recommended license:

```text
CC-BY-4.0
```

This is appropriate for a benchmark dataset intended for public evaluation and research reuse. If a different license is selected, the rationale should be documented in `methodology.md`.

## 6.3 Access

The intended public distribution target is a Hugging Face dataset repository. The dataset card should include:

* Dataset purpose.
* Split descriptions.
* Schema.
* Quickstart.
* Baseline metrics.
* Prompted-judge metrics.
* Fine-tuned judge metrics.
* Limitations.
* License.
* Citation or attribution.

## 6.4 Sensitive Information

The dataset should not contain:

* API keys.
* Private credentials.
* Real confidential customer information.
* Internal system prompts.
* Unredacted private business data.
* Personal contact details beyond synthetic or redacted examples.

Any trace-derived examples should be redacted or transformed so that the benchmark captures failure structure without exposing sensitive data.

---

# 7. Maintenance

## 7.1 Maintainer

Primary maintainer:

```text
Nuhamin Alemayehu
GitHub: https://github.com/nuhaminae
```

## 7.2 Versioning

Recommended versioning:

| Version | Description                                                                                                             |
| ------- | ----------------------------------------------------------------------------------------------------------------------- |
| v0.1    | Initial Week 11 benchmark with 260 selected tasks and Path B DPO judge evaluation.                                      |
| v0.2    | Expanded failure-mode coverage, stronger contamination report, larger adversarial set, improved cost/latency reporting. |

## 7.3 Update Policy

Future versions should be created when:

* New Week 10/production traces reveal new failure modes.
* Public-signal sources change materially.
* New adversarial prompt-injection patterns emerge.
* The judge overfits known templates.
* A larger or more diverse seed corpus becomes available.
* The benchmark is adapted to a new sales segment or persona.

## 7.4 Known v0.1 Limitations

Known limitations:

1. The held-out set is small: 52 pointwise examples / 26 preference pairs.
2. The dataset is English-only.
3. The benchmark is domain-specific to Tenacious-style B2B sales workflows.
4. The trace-derived slice depends on Week 10 system behavior and may not capture all real production cases.
5. Multi-LLM synthesis may introduce style artifacts from the generating model.
6. Some public-signal tasks simplify real-world company context.
7. Human inter-rater agreement and contamination artifacts must be present before public release.
8. The benchmark evaluates candidate outputs, not full multi-turn production state.
9. The v0.1 judge should not be used as the only production safety mechanism.

## 7.5 Recommended v0.2 Additions

Recommended additions for Tenacious-Bench v0.2:

* More real trace-derived examples.
* Additional adversarial prompt-injection cases.
* More fine-grained cost/latency tracking.
* Broader company-signal grounding cases.
* Separate slices for pricing, capacity, competitive claims, and compliance.
* Multi-turn reply-chain evaluation.
* Human agreement matrix for all rubric dimensions.
* Stronger sealed-held-out procedure.
* More examples where the correct response is “do not send a prospect-facing reply.”

---

# 8. Training and Evaluation Results

## 8.1 Training Configuration

The Path B judge was trained with DPO using:

```text
base_model: unsloth/Llama-3.2-1B-Instruct-bnb-4bit
training_method: DPO
adapter_method: LoRA
train_pairs: 65
validation_pairs: 39
epochs: 1
batch_size: 1
gradient_accumulation_steps: 4
effective_batch_size: 4
learning_rate: 0.000005
beta: 0.1
optimizer: adamw_8bit
max_length: 1024
max_prompt_length: 512
seed: 42
```

Final training metrics:

| Metric               |          Value |
| -------------------- | -------------: |
| Train loss           |         0.6785 |
| Eval loss            |         0.6711 |
| Eval reward accuracy |         0.9487 |
| Eval reward margin   |         0.0451 |
| Runtime              | 122.17 seconds |

## 8.2 Held-Out Evaluation

Held-out evaluation size:

```text
52 pointwise examples
26 preference pairs
```

Final results:

| System                  | Accuracy | Precision | Recall |     F1 | Strict pairwise accuracy |
| ----------------------- | -------: | --------: | -----: | -----: | -----------------------: |
| Week 10 baseline        |   0.8077 |    0.9444 | 0.6538 | 0.7727 |                   0.6154 |
| Prompt-engineered judge |   0.8846 |    1.0000 | 0.7692 | 0.8696 |                   0.7692 |
| Fine-tuned DPO judge    |   0.9615 |    0.9615 | 0.9615 | 0.9615 |                   0.9615 |

## 8.3 Ablation Summary

Primary metric:

```text
strict_pairwise_accuracy
```

Delta A:

```text
Fine-tuned DPO judge - Week 10 baseline
Observed delta: +0.3462
95% CI: [0.1538, 0.5385]
one-sided p-value for delta <= 0: 0.0013
```

Delta B:

```text
Fine-tuned DPO judge - Prompt-engineered judge
Observed delta: +0.1923
95% CI: [0.0385, 0.3462]
one-sided p-value for delta <= 0: 0.0039
```

Interpretation:

* The fine-tuned judge improved substantially over the Week 10 no-judge baseline.
* Prompting alone helped, but DPO tuning produced better deployment-style calibration.
* The prompt-engineered judge ranked pairs well but was too conservative as a binary gate.
* The fine-tuned judge was better at accepting good outputs while still rejecting bad outputs.

---

# 9. Quickstart

## 9.1 Load the Dataset

```python
import json
from pathlib import Path

path = Path("tenacious_bench/dpo/held_out_dpo.jsonl")

rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

print(len(rows))
print(rows[0].keys())
```

Expected fields:

```text
pair_id
prompt
chosen
rejected
source_mode
scenario_type
failure_code
failure_mode_tag
metadata
```

## 9.2 Run the Fine-Tuned Judge Evaluation

```bash
python src/evaluation/eval_judge.py --config configs/eval_config.yaml
```

Expected outputs:

```text
reports/fine_tuned_judge_metrics.json
reports/fine_tuned_judge_pair_scores.jsonl
reports/fine_tuned_judge_confusion_matrix.png
```

## 9.3 Run the Baseline Evaluation

```bash
python src/evaluation/eval_baseline.py --config configs/eval_config.yaml
```

Expected outputs:

```text
reports/baseline_metrics.json
reports/baseline_pointwise_scores.jsonl
reports/baseline_pair_scores.jsonl
reports/baseline_confusion_matrix.png
```

## 9.4 Run the Prompted-Judge Evaluation

```bash
python src/evaluation/eval_prompted_judge.py --config configs/eval_config.yaml
```

Expected outputs:

```text
reports/prompted_judge_metrics.json
reports/prompted_judge_pointwise_scores.jsonl
reports/prompted_judge_pair_scores.jsonl
reports/prompted_judge_confusion_matrix.png
```

## 9.5 Generate the Evaluation Report

```bash
jupyter nbconvert --execute notebooks/evaluation_results.ipynb --to html --output-dir reports --output evaluation_results
```

Expected output:

```text
reports/evaluation_results.html
```

---

# 10. Ethical and Practical Considerations

## 10.1 Potential Benefits

Tenacious-Bench can help:

* Reduce inconsistent sales-agent behavior.
* Catch unsafe or low-quality outputs before prospect contact.
* Improve action completion after clear prospect intent.
* Make evaluation claims traceable to task IDs and held-out pairs.
* Compare trained judges against prompt-only alternatives.

## 10.2 Potential Risks

Risks include:

* Overfitting to the v0.1 benchmark.
* Treating a small held-out set as production certification.
* Encoding one organization’s sales style as a general standard.
* Over-rejecting creative but valid outputs.
* Underrepresenting real-world prospect diversity.
* Failing to capture multi-turn production context.
* Misusing the judge as an autonomous decision-maker.

## 10.3 Mitigations

Recommended mitigations:

* Use the judge as a quality gate, not as the sole decision-maker.
* Monitor false positives and false negatives after deployment.
* Maintain a kill-switch if the judge blocks too many valid outputs or lets unsafe outputs through.
* Expand the held-out set before production use.
* Add new adversarial cases as new failures are discovered.
* Keep cost, latency, and rejection-rate dashboards.

---

# 11. Citation

Suggested citation:

```bibtex
@misc{tenacious_bench_2026,
  title = {Tenacious-Bench v0.1: A B2B Sales-Agent Evaluation Benchmark for Preference-Tuned Judges},
  author = {Nuhamin, A.},
  year = {2026},
  note = {Week 11 Conversion Engine Ground Truth benchmark}
}
```

---

# 12. Contact

For questions, maintenance, or issue reports:

```text
Nuhamin Alemayehu
GitHub: https://github.com/nuhaminae
```
