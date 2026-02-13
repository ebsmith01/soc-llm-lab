from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

openai_client = OpenAI()  # uses OPENAI_API_KEY from env

llm = llm_factory("gpt-4o-mini", client=openai_client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
import json
from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import answer_relevancy, faithfulness, context_relevancy

metrics=[answer_relevancy, faithfulness, context_relevancy]

from rag import pipeline


def load_eval_rows(path: str) -> List[Dict]:
    """
    Load eval items from baseline-style JSON (id, question, expected_answer, ...).
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(rows: List[Dict]) -> Dataset:
    """
    RAGAS expects columns like:
      - user_input
      - response
      - retrieved_contexts
      - reference
    """
    user_inputs: List[str] = []
    responses: List[str] = []
    references: List[str] = []
    contexts_list: List[List[str]] = []

    retriever = pipeline._get_retriever(alpha=0.6)  # type: ignore[attr-defined]

    for row in rows:
        q = row["question"]
        gt = row.get("expected_answer", "")

        # Run RAG to get an answer and the retrieved contexts
        rag_out = pipeline.answer_query(q, top_k=4, alpha=0.6)
        hits = retriever.search(q, k=4)
        contexts = [h.get("text", "") for h in hits]

        user_inputs.append(q)
        responses.append(rag_out.get("answer", ""))
        references.append(gt)
        contexts_list.append(contexts)

    return Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": contexts_list,
            "reference": references,
        }
    )


def main():
    rows = load_eval_rows("evals/baseline.json")
    dataset = build_dataset(rows)

    result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,   # how close answer is to reference
            faithfulness,       # is answer supported by retrieved context?
            context_precision,  # are retrieved chunks relevant to the question?
        ],
    )

    print("=== RAGAS EVAL RESULTS ===")
    print(result)

    # Optionally: save to JSON for comparison between runs
    result.to_pandas().to_csv("evals/ragas_scores.csv", index=False)


if __name__ == "__main__":
    main()
