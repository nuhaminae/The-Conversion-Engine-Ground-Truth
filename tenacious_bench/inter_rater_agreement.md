# Inter-Rater Agreement Log — Tenacious-Bench v0.1

## Status

**Status:** Not a formal multi-annotator IRR study  
**Dataset:** Tenacious-Bench v0.1  
**Maintainer:** Nuhamin Alemayehu  
**Date:** 2026-05-02  

This file documents the annotation and validation status for Tenacious-Bench v0.1.

For this Week 11 release, labels were produced through controlled pair construction, deterministic validation, and project-author review. A formal independent multi-rater study with Cohen’s kappa was **not** completed and should not be claimed.

## What Was Completed

The following validation checks were completed:

1. **Pair integrity**
   - Train: 65 / 65 complete pairs
   - Dev: 39 / 39 complete pairs
   - Held-out: 26 / 26 complete pairs

2. **Label balance**
   - Train: 65 good / 65 bad
   - Dev: 39 good / 39 bad
   - Held-out: 26 good / 26 bad

3. **DPO schema validation**
   - Required fields present:
     - `prompt`
     - `chosen`
     - `rejected`

4. **Source-mode coverage**
   - Trace-derived
   - Programmatic
   - Multi-LLM synthesis
   - Hand-authored adversarial

5. **Held-out evaluation**
   - Week 10 baseline evaluated
   - Prompt-engineered judge evaluated
   - Fine-tuned DPO judge evaluated

## What Was Not Completed

The following were not completed for v0.1:

- Independent second annotator labeling.
- Delayed re-labeling by the same annotator.
- Cohen’s kappa calculation.
- Krippendorff’s alpha calculation.
- Full disagreement adjudication log.

## Why No Formal IRR Is Claimed

Tenacious-Bench v0.1 was constructed under a Week 11 project timeline. The priority was to build a complete Path B benchmark and judge-training pipeline:

1. Construct paired good/bad examples.
2. Split into train/dev/held-out.
3. Train a DPO judge.
4. Evaluate against baseline and prompt-engineered judge.
5. Package artifacts for release.

Because no independent second annotation pass was completed, this release should be treated as:

```text
curated preference-pair benchmark with deterministic validation
```

not as:

```text
multi-annotator human-labeled benchmark
```

## Recommended v0.2 IRR Protocol

For v0.2, use the following protocol:

1. Sample at least 30 preference pairs from the benchmark.
2. Have two annotators independently label:

   * Which response is better.
   * Whether the good response is acceptable for prospect-facing use.
   * Whether the bad response contains a safety, action, grounding, or style failure.
3. Re-label the same subset after a delay.
4. Calculate:

   * Cohen’s kappa for binary good/bad label agreement.
   * Agreement by failure category.
   * Disagreement rate by source mode.
5. Adjudicate disagreements and update the rubric.
6. Store results in:

   * `tenacious_bench/inter_rater_agreement.md`
   * `tenacious_bench/annotation_disagreements.jsonl`

## Suggested IRR Table Template

| Annotator A | Annotator B | Pairs labeled | Agreement | Cohen’s kappa | Notes                   |
| ----------- | ----------- | ------------: | --------: | ------------: | ----------------------- |
| TBD         | TBD         |            30 |       TBD |           TBD | To be completed in v0.2 |

## Current Interpretation

The v0.1 benchmark is suitable for:

* Week 11 evaluation.
* Prototype judge training.
* Internal comparison of baseline vs prompted judge vs fine-tuned judge.
* Demonstrating a complete Path B pipeline.

It is not yet sufficient for claims that require formal human annotation reliability.

## Link to Related Validation Artifacts

Related files:

```text
tenacious_bench/contamination_check.json
reports/exploratory_data_summary.json
reports/training/training_summary.json
reports/ablation_results.json
```
