from typing import Dict, Any
from rag.retrieval import HybridRetriever
from rag.guardrails import scrub_pii, is_injection
from rag.prompts import ANSWER_TEMPLATE

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def _compose_answer(passages):
    bullets = []
    citations = []
    for i, chunk in enumerate(passages[:3]):
        bullets.append(f"- {chunk.text.strip()}")
        citations.append(f"- [{chunk.meta.get('doc_id','doc')}:{chunk.meta.get('chunk_id', i)}]")
    return ANSWER_TEMPLATE.format(
        answer="\n".join(bullets),
        citations="\n".join(citations)
    )

def answer_query(q: str, top_k: int = 8) -> Dict[str, Any]:
    original = q
    q = scrub_pii(q)

    if is_injection(q):
        return {"answer": "Query blocked by guardrails.", "citations": [], "guardrails": {"blocked": True}}

    passages = _get_retriever().search(q, k=top_k)
    answer = _compose_answer(passages)

    return {
        "question": original,
        "answer": answer,
        "citations": [p.meta for p in passages[:3]],
        "guardrails": {"blocked": False},
    }
