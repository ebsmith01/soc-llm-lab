import os
from openai import OpenAI
from rag.retrieval import HybridRetriever
from rag.guardrails import scrub_pii, is_injection
from rag.prompts import SYSTEM_PROMPT

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def build_prompt(question, passages):
    context = []
    for p in passages[:6]:  # keep context tight
        context.append(
            f"[Source: {p.meta.get('doc_id')}:{p.meta.get('chunk_id')}]\n{p.text}"
        )
    context_text = "\n\n".join(context)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def answer_query(q: str, top_k: int = 8):
    original = q
    q = scrub_pii(q)

    if is_injection(q):
        return {"answer": "Query blocked by guardrails.", "citations": [], "guardrails": {"blocked": True}}

    passages = _get_retriever().search(q, k=top_k)
    prompt = build_prompt(q, passages)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    answer = response.choices[0].message.content

    return {
        "question": original,
        "answer": answer,
        "citations": [p.meta for p in passages[:6]],
        "guardrails": {"blocked": False},
    }
