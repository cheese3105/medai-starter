## 1. Todos
- Hoàng defense: implement trên các version, chạy ra kq với các config a Đạt gửi sau, nhận xét
- Đạt attack: Bổ sung token + duration, implement trên các version, chạy ra kq với các config, nhận xét

- Chi + Huy: - RQ: Graph thông tin gì? -> Hypothesis





















## 2. Report Structure

1. Introduction
- Context: LLMs in medical decision support (MedAI).
- Problem: Indirect prompt injection via external data.

2. Threat Model
- Attacker goal: Divert diagnosis to attacker-chosen task.
- Access: Attacker poisons external source only; no access to system prompt or weights.
- Weakest assumption: Attacker control over retrieved clinical documents.

3. Methodology
- Attacks: Naive, escape characters, context ignoring, fake completion, combined.
- Defense: Channel separation / structured querying (StruQ).

4. Experimental Setup
- Datasets: MedQA + 5 classification tasks (SST-2, SMS Spam, MRPC, Hate Speech, RTE).
- Metrics:
  - Attack: Attack Success Rate (ASR).
  - Utility: MedQA accuracy on clean inputs.
  - Cost: Latency overhead and false refusal rate.
  - Baseline: Undefended MedAI system.

5. Results
- Comparison table: undefended vs defended across 5 attacks and 5 tasks.
- Utility retention and latency overhead.

6. Residual-Gap Analysis
- What still gets through the defense and why.
- Failure modes: boundary leakage and semantic instruction blending.

7. Discussion & Limitations
- Answer to weakest assumption.
- Clinical risks: false refusals on valid clinical directives.

8. Conclusion

## 3. Research Questions

- RQ1: How vulnerable is undefended MedAI across the 5 injection types?
- RQ2: How much does channel separation reduce ASR, and does it hurt MedQA accuracy?
- RQ3: What attacks bypass the defense, and why do they succeed?
- RQ4: What is the latency overhead, and does the defense falsely refuse benign clinical text?

## 4. Hypotheses

- H1: Combined and fake completion attacks achieve higher ASR than naive injection on undefended models.
- H2: Channel separation reduces ASR by over 80% with less than 3% drop in benign MedQA accuracy.
- H3: Residual bypasses occur when injected text mimics conversational transitions rather than explicit delimiters.
- H4: Strict defense triggers false refusals on benign clinical imperatives such as discontinuation orders.

## 5. Optimize pipeline runner

- [ ] Track actual retrieval-query history and prevent duplicate-query retries.
- [ ] Remove dead/mislabeled `loop_history` prompt content, especially in v2-qr.
- [ ] Replace full failed evidence in retry prompts with a compact verifier retrieval gap.
- [x] Remove verifier `confidence` and `final_answer`; retain the reasoning draft values.
- [ ] Implement real LTM retrieval for v3-qr benchmarks or disable the unused LTM prompt sections.
- [ ] Remove unused runner fields and save useful query/evidence history for evaluation.
- [ ] Benchmark after each change to measure accuracy, latency, and token impact.
