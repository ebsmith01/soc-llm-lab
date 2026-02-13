
from pydantic import BaseModel

class RagConfig(BaseModel):
    chunk_size: int = 512
    retriever_alpha: float = 0.5
    embedding_model: str = "text-embedding-3-large"

# Quick and dirty: manually edit these for each experiment run
current_config = RagConfig(
    chunk_size=512,
    retriever_alpha=0.6,
    embedding_model="text-embedding-3-large",
)