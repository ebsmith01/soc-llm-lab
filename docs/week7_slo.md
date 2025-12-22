# Week 7 — Service Level Objective (SLO)
## SOC LLM Lab API

---

## Endpoint
`POST /ask`

---

## p95 Latency Definition

**p95 latency** is defined as:

> The 95th percentile of **end-to-end server-side request latency** for  
> `POST /ask`, measured from the moment the request is received by the FastAPI
> server until the response is fully sent to the client.

### Clarifications
- Measurement is **server-side only**
- Includes:
  - request parsing
  - hybrid retrieval (BM25 + embeddings)
  - reranking (if enabled)
  - LLM generation (local LoRA or hosted model)
  - post-processing
- Excludes:
  - client-side network latency
- Computed over a **rolling 5–10 minute window**

This definition reflects **user-perceived API responsiveness**.

---

## Initial p95 Target

**Target (baseline):**
