# SOC LLM Lab

## Platform Overview
This repository contains a production-grade, security-aware LLM platform that supports retrieval-augmented generation, fine-tuned models, tool-calling agents, red-teaming, and evaluation pipelines.

## High Level Architecture

### Data and ingest layer
1. Corpus (two MITRE-related documents) converted to clean structured chunks.
2. Hybrid indexes: BM25 for lexical retrieval plus embeddings with a FAISS vector index for semantic retrieval with optional reranking.

### Hybrid retriever
1. BM25 + vector similarity with weighted fusion (alpha = 60%).
    •	Higher recall across both structured and unstructured queries.
	•	Better grounding for improved citation accuracy in responses.

2. Eval metrics: semantic F1, grounding score, exact match, and latency.

# Challenge:
The system found the right information but worded answers differently than the evaluation expected. 
Fix
	•	Added intent detection to recognize question types (multi-hop, needle, structure, direct fact).
	•	Updated prompts to guide answer format (lists for structure, excerpts for passages, explicit connectors for multi-hop).
	•	Introduced post-generation stabilization to ensure exact framework terms and keywords appear when grounded by citations.
	•	Expanded retrieval to support larger candidate pools (semantic_top_k) for harder queries.
	•	Integrated optional reranking controls (cross-encoder + blending) without breaking latency or guardrails.

### Guardrails and prompting
1. PII scrubbing, prompt-injection heuristics, and harmful/out-of-scope filters.
2. Force the model to answer only from retrieved context with inline citations and standardized refusals.
3. Apply filters before LLM invocation to reduce risk of leakage or unsafe outputs.

# Challenge 
The model would answer confidently even when the answer wasn’t fully supported by retrieved context.
Fix
	•	Designed a strict RAG contract prompt
	•	Required inline citations
	•	Enforced a standardized refusal
	•	Added grounding-aware eval metrics (semantic F1 + citation requirement)

### Generation
1. OpenAI path: GPT-4.1 via API for comparison and high-quality responses.
2. Local path (TinyLlama + LoRA): 1.1B base, LoRA adapter, MPS/CPU compatible, no CUDA required, suitable for air-gapped environments.
3. Deployment flexibility across security tiers

### API layer
1. FastAPI app with `/health`, `/metrics`, `/ask`.
2. Request telemetry middleware plus Prometheus instrumentation and tracing/logging.

### Testing and observability
1. Pytest-based validation of retrieval, prompts, and agent behavior.
2. Structured logging for production traceability and debugging.
3. Pytest also used for detecting regressions when updating prompts, models, or indexes.

bash demo.sh