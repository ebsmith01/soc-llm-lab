from types import SimpleNamespace

import pytest
import rag.pipeline as pipeline


@pytest.fixture(autouse=True)
def stub_openai(monkeypatch):
    def _fake_create(*args, **kwargs):
        message = SimpleNamespace(content="mock answer")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setattr(
        pipeline.client.chat.completions,
        "create",
        _fake_create,
    )


def test_basic_answer():
    result = pipeline.answer_query("What is access control?", top_k=4)
    assert "answer" in result
    assert "citations" in result
