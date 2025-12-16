from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.agent import run_agent

app = FastAPI(title="SOC LLM Lab")

class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=20)
    alpha: float = Field(0.7, ge=0.0, le=1.0)
    use_local_lora: Optional[bool] = None

class AnswerResponse(BaseModel):
    type: str
    answer: str
    citations: list[dict]
    tool_trace: list[dict]

@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> Dict[str, Any]:
    # allow env default if caller doesn't pass it
    use_local = req.use_local_lora
    if use_local is None:
        use_local = os.getenv("USE_LOCAL_LORA", "0") == "1"

    return run_agent(
        req.question,
        top_k=req.top_k,
        alpha=req.alpha,
        use_local_lora=use_local,
    )



