# Week 5 – RAG Evaluation Report (SOC LLM Lab)

**Author:** Evin Smith  
**Project:** SOC RAG Assistant over MITRE ATT&CK + AI Security Papers  
**Week:** 5 — Evaluation Harness (RAGAS-style)

---

## 1. Overview

This report documents the first formal evaluation of a Retrieval-Augmented Generation (RAG) system designed to answer SOC-focused questions using MITRE ATT&CK documentation and MITRE’s *A Sensible Regulatory Framework for AI Security* paper.

The goal of this evaluation is to:
- Quantify answer quality and grounding
- Identify failure modes
- Tune retrieval and prompting parameters
- Establish a repeatable evaluation baseline for future improvements

---

## 2. System Configuration

### Corpus
- MITRE *Getting Started with ATT&CK*
- MITRE *A Sensible Regulatory Framework for AI Security*

### Retrieval
- **Retriever:** Hybrid (BM25 + dense embeddings)
- **Alpha (BM25 weight):** `0.7`
- **Top-K:** `8`
- **Chunk size:** `900` characters
- **Chunk overlap:** implicit page-level overlap from ingestion

### Prompting
- `make_strict_rag_prompt`
  - Strict “use only provided context” contract
  - Mandatory inline `[Source: …]` citations
  - Explicit safe-refusal format:
    > *“I don’t know. The answer is not covered by the provided documents.”*

### Guardrails
- PII scrubbing
- Prompt injection detection
- Out-of-scope / harmful request blocking

---

## 3. Evaluation Dataset

- **Total questions:** `40`
- **Target:** ~50 (current set used as baseline)
- **Sources:**
  - MITRE documentation
  - SOC / detection-engineering-style questions

### Question Types
| Type         | Description |
|--------------|------------|
| `direct_fact` | Single factual answers from one snippet |
| `multi_hop`   | Requires synthesizing multiple snippets |
| `needle`      | Narrow detail embedded in long sections |
| `structure`   | “List steps / describe process” questions |
| `edge`        | Safety, refusal, and out-of-scope cases |

---

## 4. Metrics

### 4.1 Custom Metrics

Metrics are computed in `evals/run_eval.py`.

**Win condition:**  
`semantic_keyword_f1 + grounding_score ≥ 0.5`

#### Overall Results (Final Tuned Configuration)

- **Total questions:** 40  
- **Win rate:** **35%** (14 / 40)  
- **Exact match avg:** 0.05  
- **Semantic keyword F1 avg:** 0.32  
- **Grounding score avg:** 0.32  

**Latency**
- **Average:** ~2.0s  
- **p50:** ~1.9s  
- **p95:** ~4.0s  

> Exact match is intentionally low due to paraphrased answers.

---

### 4.2 Breakdown by Question Type

| Type         | Wins / Total | Win Rate |
|--------------|--------------|----------|
| direct_fact  | 7 / 20 | **35%** |
| multi_hop    | 2 / 7  | **28.6%** |
| needle       | 2 / 5  | **40%** |
| structure    | 1 / 3  | **33.3%** |
| edge         | 2 / 5  | **40%** |

---

## 5. Tuning Experiments

### 5.1 Chunk Size

| Chunk Size | Observation |
|-----------|-------------|
| ~750 | Higher precision but less context for synthesis |
| **900 (final)** | Best balance between context coverage and noise |
| 1500 | Too verbose; degraded precision on needle questions |

✅ **Selected:** `900`

---

### 5.2 Hybrid Alpha (BM25 Weight)

| Alpha | Result |
|------|--------|
| 0.5 | Balanced but underperformed on factual queries |
| **0.7 (final)** | Best overall win rate and structure performance |
| 0.8 | Over-weighted BM25; weaker synthesis |

✅ **Selected:** `0.7`

---

## 6. Strengths & Weaknesses

### Strengths
- Reliable **safe refusals** for out-of-scope and unsafe questions
- Strong grounding for direct factual SOC questions
- Clear citation discipline (low hallucination risk)
- Stable latency under ~4s p95

### Weaknesses
- Weak performance on:
  - Multi-hop synthesis
  - Structured “list/process” questions
- Context sometimes too fragmented across chunks
- No embedding-based semantic scoring yet (keyword F1 proxy only)

---

## 7. Key Learnings

1. **Retrieval quality matters more than prompting**
   - Poor chunk selection cannot be fixed by stricter prompts.
2. **Hybrid retrieval needs careful weighting**
   - Slight BM25 bias (α = 0.7) suits technical SOC documents.
3. **Strict refusal rules prevent hallucinations but reduce recall**
   - Acceptable tradeoff for SOC use cases.
4. **Structure questions expose chunking limitations**
   - Indicates need for hierarchy-aware or section-aware chunking.

---

## 8. Next Steps

Planned improvements for Week 6+:

- Add true semantic similarity (embedding-based F1)
- Integrate **RAGAS** hallucination + faithfulness metrics
- Improve section-aware chunking
- Add reranking (cross-encoder or LLM-based)
- Expand eval dataset to **80–100 questions**
- Create per-category dashboards for regression tracking

---

## 9. Conclusion

This evaluation establishes a solid, reproducible baseline for a SOC-focused RAG system. With a **35% win rate under strict grounding rules**, the system favors safety and correctness over recall—an appropriate posture for security operations.

The evaluation harness now provides a clear framework for iterative improvement and regression tracking as retrieval, chunking, and semantic evaluation techniques mature.