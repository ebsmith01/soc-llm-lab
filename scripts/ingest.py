"""
Ingestion pipeline that:
1. Loads PDFs from data/raw/
2. Extracts text page-by-page with LangChain PyPDFLoader
3. Saves raw text to data/raw_txt/
4. Cleans text (whitespace + header/footer stripping)
5. Saves clean text to data/clean_txt/
6. Splits into chunks with RecursiveCharacterTextSplitter
7. Writes all chunks to data/processed/chunks.jsonl
"""

from pathlib import Path
import json
import logging
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------
# Paths
# ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"                 # PDFs go here
RAW_TXT_DIR = PROJECT_ROOT / "data" / "raw_txt"         # raw txt outputs
CLEAN_TXT_DIR = PROJECT_ROOT / "data" / "clean_txt"     # clean txt outputs
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


# ---------------------------------------
# Config (Python dict = fine for Day 1)
# ---------------------------------------

CFG = {
    "min_chars_per_page": 20,
    "header_footer": {
        "top_lines": 2,
        "bottom_lines": 2,
        "min_repeats_ratio": 0.6
    },
    "chunking": {
        "chunk_size": 500,
        "chunk_overlap": 100
    }
}


# ---------------------------------------
# Helpers
# ---------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace only."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def strip_repeated_headers_footers(pages, top_lines, bottom_lines, min_ratio):
    """
    Simple heuristic:
    - Look at top N lines and bottom N lines of each page.
    - If a line repeats on >= min_ratio of pages, remove it everywhere.
    """
    if not pages:
        return pages

    split_pages = []
    tops = []
    bottoms = []

    # split into lines while keeping original page grouping
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        split_pages.append(lines)
        tops.extend(lines[:top_lines])
        bottoms.extend(lines[-bottom_lines:])

    # find repeated lines
    def repeated(lines):
        counts = Counter(lines)
        threshold = max(1, int(len(pages) * min_ratio))
        return {ln for ln, c in counts.items() if c >= threshold}

    drop_top = repeated(tops)
    drop_bottom = repeated(bottoms)

    # remove repeated headers/footers
    cleaned_pages = []
    for lines in split_pages:
        kept = [ln for ln in lines if ln not in drop_top and ln not in drop_bottom]
        cleaned_pages.append("\n".join(kept))

    return cleaned_pages


# ---------------------------------------
# Main ingest routine
# ---------------------------------------

def ingest():
    """
    Load PDFs from RAW_DIR → save raw txt → clean → save clean txt → chunk → chunks.jsonl.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG["chunking"]["chunk_size"],
        chunk_overlap=CFG["chunking"]["chunk_overlap"],
    )

    # ensure output dirs exist
    RAW_TXT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_TXT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Found {len(pdf_paths)} PDFs in {RAW_DIR}")
    logging.info(f"Writing chunks to {PROCESSED_PATH}")

    with PROCESSED_PATH.open("w", encoding="utf-8") as out_file:

        for pdf_path in pdf_paths:
            logging.info(f"Processing {pdf_path.name}")

            # 1) Load PDF pages as LangChain Documents (one per page)
            loader = PyPDFLoader(str(pdf_path), mode="page")
            docs = loader.load()

            raw_pages = [d.page_content or "" for d in docs]

            # 2) Save RAW text (just concatenated)
            raw_text = "\n\n".join(raw_pages)
            (RAW_TXT_DIR / f"{pdf_path.stem}.txt").write_text(raw_text, encoding="utf-8")

            # 3) Strip repeated headers/footers (before whitespace cleaning)
            hf_cfg = CFG["header_footer"]
            stripped_pages = strip_repeated_headers_footers(
                raw_pages,
                top_lines=hf_cfg["top_lines"],
                bottom_lines=hf_cfg["bottom_lines"],
                min_ratio=hf_cfg["min_repeats_ratio"]
            )

            # 4) Clean whitespace + drop tiny pages
            clean_pages = []
            for p in stripped_pages:
                c = clean_text(p)
                if len(c) >= CFG["min_chars_per_page"]:
                    clean_pages.append(c)

            clean_text_full = "\n\n".join(clean_pages)
            (CLEAN_TXT_DIR / f"{pdf_path.stem}.txt").write_text(clean_text_full, encoding="utf-8")

            # 5) Put cleaned text back into Documents for chunking
            for d, cleaned in zip(docs, stripped_pages):
                d.page_content = clean_text(cleaned)
            # 6) Chunk
            chunks = splitter.split_documents(docs)

            # 7) Write chunks.jsonl
            #    Schema (to match tests + Day 2 spec):
            #      - id: unique chunk identifier
            #      - text: chunk text
            #      - source: which document it came from
            #      - page_num: original page number if available
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{pdf_path.stem}-{idx}"
                meta = chunk.metadata or {}

                record = {
                    "id": chunk_id,
                    "text": chunk.page_content,
                    "source": pdf_path.stem,
                    "page_num": meta.get("page"),
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
     

            logging.info(f"{pdf_path.name}: {len(chunks)} chunks created")

    logging.info("Ingestion complete.")


# ---------------------------------------
# Run CLI
# ---------------------------------------

if __name__ == "__main__":
    ingest()