# Dataset Datasheet – Tenacious Sales Evaluation Benchmark (Path B)

## 1. Motivation

- **Purpose:** To evaluate and improve B2B sales agents by training a Judge/Critic model that enforces consistency in responding to prospect intent.  
- **Gap addressed:** Existing benchmarks (e.g., τ²‑Bench retail) do not capture Tenacious‑specific failure modes such as missed scheduling after clear prospect agreement.  
- **Intended use:** Training and evaluating preference‑optimised judge models for sales conversations.  

---

## 2. Composition

- **Size:** ~200–300 tasks.  
- **Sources:**  
  - Trace‑derived tasks (≈30%) from Week 10 demo runs.  
  - Synthetic preference pairs (≈30%) generated from Tenacious seed corpus.  
  - Multi‑LLM synthesis with judge filtering (≈25%).  
  - Hand‑authored adversarial cases (≈15%).  
- **Data types:** Prospect inputs, agent outputs (good vs. bad), preference labels, metadata tags.  
- **Languages:** English (Tenacious sales context).  

---

## 3. Collection Process

- **Trace‑derived:** Extracted from demo transcripts where the agent missed intent.  
- **Synthetic:** Generated variations of prospect “yes” responses paired with good/bad agent outputs.  
- **Multi‑LLM synthesis:** Multiple LLMs generated candidate exchanges, filtered by judges for quality.  
- **Adversarial:** Manually authored edge cases (e.g., vague “maybe later” vs. clear “yes”).  

---

## 4. Preprocessing

- Normalisation of text (lowercasing, punctuation cleanup).  
- Removal of duplicates.  
- Partitioning into train/dev/held‑out sets.  
- Metadata tagging (`trace‑derived`, `synthetic`, `adversarial`).  

---

## 5. Dataset Splits

- **Train:** ~70%  
- **Dev:** ~15%  
- **Held‑out:** ~15% (used for final evaluation only)  

---

## 6. Quality & Validation

- **Contamination checks:** Verified against Tenacious seed corpus to avoid overlap.  
- **Inter‑rater agreement:** Multiple annotators scored preference pairs; Cohen’s κ logged.  
- **Judge filtering:** LLM‑as‑a‑judge used to discard low‑quality synthetic samples.  

---

## 7. Uses

- **Primary:** Training preference‑optimised judge models.  
- **Secondary:** Benchmarking agent consistency on Tenacious‑specific tasks.  
- **Not intended for:** General retail or consumer sales benchmarks.  

---

## 8. Distribution

- **Hosting:** HuggingFace dataset card.  
- **License:** [Specify license, e.g., CC‑BY‑SA or MIT].  
- **Access:** Public, with documentation and datasheet included.  

---

## 9. Maintenance

- **Maintainers:** Nuhamin A.  
- **Update frequency:** Iterative updates as new failure modes are discovered.  
- **Contact:** `https://github.com/nuhaminae`.  
