"""
Day 4 — FastAPI /ask endpoint for your SOC RAG lab.

Exposes:
  POST /ask
    Request JSON:
      {
        "question": "What is access control?",
        "top_k": 5        # optional
      }

    Response JSON:
      {
        "question": "...",
        "answer": "...",
        "citations": [...],
        "guardrails": {...}
      }

This is a thin HTTP wrapper over rag.pipeline.answer_query().
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import pipeline
import time
import logging 

# -----------------------------

# -----------------------------
# FastAPI app + CORS
# -----------------------------

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SOC RAG Assistant",
    description="Baseline RAG API for AI security / SOC docs.",
    version="0.1.0",
)

# Allow local tools / frontend to call this easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request / Response models
# -----------------------------

class AskRequest(BaseModel):
    question: str = Field(..., description="User's natural language question.")
    top_k: int = Field(8, ge=1, le=20, description="How many passages to retrieve.")


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list
    guardrails: dict


# -----------------------------
# Health check
# -----------------------------

@app.get("/health")
def health():
    """
    Basic health endpoint to check the API is up.
    
    Does not touch the LLM or retrieval.
    """
    return {"status": "ok"}


# -----------------------------
# /ask endpoint
# -----------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Main RAG query endpoint.

    Flow:
      1) Accept JSON: {question, top_k}
      2) Call pipeline.answer_query()
      3) Return structured answer + citations + guardrail metadata

    We also log basic latency and query info for observability.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    start = time.perf_counter()
    try:
        result = pipeline.answer_query(req.question, top_k=req.top_k)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "RAG pipeline error for /ask (%.1f ms) question=%r top_k=%d error=%r",
            duration_ms,
            req.question,
            req.top_k,
            exc,
        )
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Handled /ask in %.1f ms question=%r top_k=%d guardrails=%s",
        duration_ms,
        req.question,
        req.top_k,
        result.get("guardrails"),
    )

    return AskResponse(**result)