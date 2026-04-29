# Methodology – The Conversion Engine Ground Truth

## Path Declaration

Dataset constructed from three sources:
1. Trace-derived tasks (real logs).
2. Synthetic programmatic pairs.
3. Adversarial cases (probe library + taxonomy).

## Justification

- Week 10 probe library and taxonomy provided evidence of failure categories (F1–F4).
- Mapping probes to taxonomy ensures coverage of realistic adversarial scenarios.
- Synthetic pairs fill schema gaps not observed in traces.

## Partitioning Protocol

- Train/dev/held_out split at ~70/15/15.
- Stratified by source mode and failure category.
- Deduplication applied before partitioning.

## Contamination Check

- Compared dataset text against probe_library.md and failure_taxonomy.md.
- No verbatim leakage into train/dev partitions.
- Held_out reserved for evaluation only.

## Evidence

- SR probes mapped to F1 failures.
- CD probes mapped to F2 failures.
- TU probes mapped to F3 failures.
- GS probes mapped to F4 failures.
