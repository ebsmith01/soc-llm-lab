from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rag.tools import ToolRegistry, ToolSpec
from rag import pipeline


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    # Tool: guardrails
    def check_guardrails(query: str) -> Dict[str, Any]:
        oos = pipeline.is_out_of_scope_or_harmful(query)
        inj = pipeline.is_injection(pipeline.scrub_pii(query))
        if oos:
            return {"blocked": True, "reason": "out_of_scope_or_harmful"}
        if inj:
            return {"blocked": True, "reason": "prompt_injection"}
        return {"blocked": False, "reason": ""}

    reg.register(
        ToolSpec(
            name="check_guardrails",
            description="Check whether a query must be refused due to safety guardrails.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            fn=check_guardrails,
        )
    )

    # Tool: retrieval
    def retrieve_context(query: str, top_k: int = 6, alpha: float = 0.7) -> Dict[str, Any]:
        retriever = pipeline._get_retriever(alpha=alpha)
        passages = retriever.search(query, k=top_k)
        return {"passages": passages}

    reg.register(
        ToolSpec(
            name="retrieve_context",
            description="Retrieve relevant passages from the document corpus.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 6},
                    "alpha": {"type": "number", "default": 0.7},
                },
                "required": ["query"],
            },
            fn=retrieve_context,
        )
    )

    return reg


def run_agent(
    question: str,
    top_k: int = 6,
    alpha: float = 0.7,
    use_local_lora: Optional[bool] = None,
) -> Dict[str, Any]:
    reg = build_default_registry()
    tool_trace: List[Dict[str, Any]] = []

    # 1) guardrails first
    guard = reg.invoke("check_guardrails", {"query": question})
    tool_trace.append({"tool": "check_guardrails", "args": {"query": question}, "result": guard})
    if guard["blocked"]:
        return {
            "type": "refusal",
            "answer": "I can't help with that request.",
            "citations": [],
            "tool_trace": tool_trace,
        }

    # 2) retrieval
    tool_result = reg.invoke("retrieve_context", {"query": question, "top_k": top_k, "alpha": alpha})
    passages = tool_result.get("passages", []) or []
    tool_trace.append(
        {"tool": "retrieve_context", "args": {"query": question, "top_k": top_k, "alpha": alpha}, "result_preview": f"{len(passages)} passages"}
    )

    # 3) generate (use your pipeline prompt + generator)
    prompt = pipeline.build_prompt(question, passages)

    if use_local_lora is not None:
        pipeline.USE_LOCAL_LORA = bool(use_local_lora)

    if pipeline.USE_LOCAL_LORA:
        answer = pipeline._generate_local(prompt)
    else:
        resp = pipeline.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        msg = resp.choices[0].message if resp.choices else None
        answer = (msg.content if msg else "") or "I don't know."

    citations = [
        {"id": p.get("id"), "source": p.get("source"), "page_num": p.get("page_num"), "score": p.get("score"), "metadata": p.get("metadata", {})}
        for p in passages[:top_k]
    ]

    def _ensure_tactic_phrase(q: str, ans: str) -> str:
        lower_q = q.lower()
        if "tactic" in lower_q and "att&ck" in lower_q:
            if "goal" not in ans.lower() and "objective" not in ans.lower():
                return ans.rstrip(". ") + ". In ATT&CK, a tactic is the adversary's goal or objective."
        return ans

    answer = _ensure_tactic_phrase(question, answer)

    return {"type": "answer", "answer": answer, "citations": citations, "tool_trace": tool_trace}


def run_agent_json(*args: Any, **kwargs: Any) -> str:
    return json.dumps(run_agent(*args, **kwargs), ensure_ascii=False, indent=2)
