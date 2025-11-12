"""Placeholder ingest script that would convert raw docs into chunks."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def ingest():
    """Placeholder ingest routine."""
    raise NotImplementedError("Add ingestion logic to build chunks.jsonl")


if __name__ == "__main__":
    ingest()
