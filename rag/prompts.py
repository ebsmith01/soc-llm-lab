# SYSTEM_PROMPT constant defines role, constraints, and citation format; parentheses create an implicit string literal concatenation across lines for readability.
SYSTEM_PROMPT = (
    "You are a compliance assistant. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say 'I don't know'. "
    "Cite sources using format: [doc_id:chunk_id]."
)

# ANSWER_TEMPLATE is a formatted markdown snippet with {answer} and {citations} placeholders; triple-quoted style isn’t necessary because parentheses join string literals neatly.
ANSWER_TEMPLATE = (
    "## Answer\n"
    "{answer}\n\n"
    "## Sources\n"
    "{citations}\n"
)
