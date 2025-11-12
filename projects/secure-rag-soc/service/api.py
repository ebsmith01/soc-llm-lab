from fastapi import FastAPI
from pydantic import BaseModel
from rag.pipeline import answer_query

app = FastAPI(title="Secure RAG Assistant")

class Query(BaseModel):
    q: str
    top_k: int = 8

@app.get("/health")
def health_check():
    return {"ok": True}

@app.post("/ask")
def ask(query: Query):
    return answer_query(query.q, top_k=query.top_k)
