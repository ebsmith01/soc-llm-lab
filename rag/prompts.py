SYSTEM_PROMPT = (
    "You are a compliance assistant. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say 'I don't know'. "
    "Cite sources using format: [doc_id:chunk_id]."
)

ANSWER_TEMPLATE = (
    "## Answer\n"
    "{answer}\n\n"
    "## Sources\n"
    "{citations}\n"
)
