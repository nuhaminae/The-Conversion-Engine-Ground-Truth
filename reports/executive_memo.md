# Executive Memo: Improving Sales-Agent Reliability with Tenacious-Bench and a DPO Judge

**To:** Tenacious Leadership Team  
**From:** Nuhamin Alemayehu  
**Date:** 2026-05-02  
**Subject:** Week 11 Path B Results — Preference-Tuned Judge for Sales-Agent Reliability

## Executive Summary

The Week 10 Conversion Engine demo revealed a critical reliability gap: the agent could identify prospect intent but did not always complete the correct next action. The most important example was a prospect clearly agreeing to a meeting while the agent failed to send a booking link.

To address this, I built **Tenacious-Bench v0.1**, a Tenacious-specific benchmark for sales-agent quality, and trained a lightweight **DPO judge** to evaluate candidate outputs before they reach a prospect.

The final result is strong:

| System | Strict Pairwise Accuracy |
|---|---:|
| Week 10 baseline | 61.54% |
| Prompt-engineered judge | 76.92% |
| Fine-tuned DPO judge | **96.15%** |

The fine-tuned judge improved strict pairwise accuracy by **+34.62 percentage points** over the Week 10 baseline and **+19.23 percentage points** over the prompt-engineered judge.

## Business Problem

The Week 10 agent had an inconsistency failure.

It was not simply producing bad prose. It sometimes failed to complete the business action that the prospect’s message required.

Examples of high-risk failures:

- Prospect asks to book; agent does not send a booking link.
- Tool or CRM failure causes a warm lead to be dropped.
- Prospect opts out; agent continues selling.
- Public hiring signal is weak or negative; agent pitches aggressive growth.
- Prospect asks for secrets or internal instructions; agent fails to safely refuse.
- Output contains placeholder, localhost, or broken links.

These failures matter because they can lose meetings, reduce trust in automation, and create reputational risk.

## Solution

I created a benchmark and judge pipeline:

1. **Benchmark construction**  
   Built a 260-task Tenacious-specific evaluation set from trace-derived, programmatic, synthetic, and adversarial examples.

2. **Preference-pair training**  
   Converted examples into good/bad preference pairs and trained a DPO judge.

3. **Held-out evaluation**  
   Compared three systems on the same 52 held-out examples / 26 pairs:
   - Week 10 no-judge baseline
   - Prompt-engineered judge
   - Fine-tuned DPO judge

## Dataset Summary

Tenacious-Bench v0.1 contains:

| Split | Tasks | Pairs |
|---|---:|---:|
| Train | 130 | 65 |
| Dev | 78 | 39 |
| Held-out | 52 | 26 |
| **Total** | **260** | **130** |

The dataset is label-balanced across all splits.

Source-mode coverage:

| Source mode | Selected tasks |
|---|---:|
| Trace-derived | 78 |
| Programmatic | 78 |
| Multi-LLM synthesis | 64 |
| Hand-authored adversarial | 40 |

## Model Summary

The judge was trained as a LoRA adapter using DPO.

| Item | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` |
| Training method | Direct Preference Optimization |
| Adapter method | LoRA / PEFT |
| Runtime | Google Colab T4 |
| Train pairs | 65 |
| Dev pairs | 39 |
| Final eval reward accuracy | 94.87% |

The final adapter is stored in:

```text
models/judge/
```

## Evaluation Results

Held-out results:

| System                  |   Accuracy | Precision |     Recall |         F1 | Strict Pairwise Accuracy |
| ----------------------- | ---------: | --------: | ---------: | ---------: | -----------------------: |
| Week 10 baseline        |     80.77% |    94.44% |     65.38% |     77.27% |                   61.54% |
| Prompt-engineered judge |     88.46% |   100.00% |     76.92% |     86.96% |                   76.92% |
| Fine-tuned DPO judge    | **96.15%** |    96.15% | **96.15%** | **96.15%** |               **96.15%** |

## Interpretation

The Week 10 baseline was conservative. It rejected most bad outputs, but also rejected too many valid good outputs.

The prompt-engineered judge improved performance and ranked examples well, but it was still too conservative as a binary gate.

The fine-tuned DPO judge was better calibrated. It accepted almost all good responses while still rejecting almost all bad ones.

This means the DPO judge is the strongest candidate for a future quality-control layer.

## Business Impact

The judge can help protect the sales pipeline by:

- Reducing lost meetings from missed booking intent.
- Catching unsafe or off-brand responses before prospect contact.
- Improving confidence in automated outreach and reply handling.
- Providing measurable quality gates for future Conversion Engine releases.
- Creating a reusable benchmark for future model and prompt changes.

## Risks and Limitations

This is a v0.1 result, not production certification.

Known limitations:

- Held-out set is small: 52 examples / 26 pairs.
- Evaluation was offline, not live in production.
- The baseline is a deterministic approximation of the Week 10 no-judge system.
- The benchmark is specific to Tenacious-style B2B sales workflows.
- Additional human inter-rater validation should be added in v0.2.
- The judge should not be the only safety mechanism in production.

## Recommended Next Steps

1. Publish the benchmark and judge adapter to Hugging Face.
2. Record a demo showing dataset creation, training artifacts, and evaluation results.
3. Expand the held-out set before production use.
4. Add human inter-rater agreement for a larger labeled subset.
5. Integrate the judge as a shadow-mode quality gate before any live blocking.
6. Track false positives, false negatives, latency, and cost.
7. Add new adversarial cases as new production failures are discovered.

## Conclusion

The Week 11 Path B intervention succeeded.

The fine-tuned DPO judge improved strict pairwise accuracy from **61.54%** for the Week 10 baseline to **96.15%** on the held-out benchmark. It also outperformed prompt engineering alone.

The result supports a practical next step: use the judge as an offline and then shadow-mode quality layer for the Conversion Engine.
