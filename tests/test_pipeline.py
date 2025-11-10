from rag.pipeline import answer_query

def test_basic_answer():
    result = answer_query("What is access control?", top_k=3)
    assert "answer" in result
    assert "citations" in result
