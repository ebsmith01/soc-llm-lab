"""
Sweep multiple chunk sizes with overlap fixed at 15% of chunk size.

For each config:
  - regenerate chunks.jsonl
  - rebuild BM25, embeddings, FAISS
  - run eval sweep (top_k, alpha)
  - output a leaderboard per config

Usage:
  python scripts/build_chunks_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Tuple


# -----------------------------
# Sweep configuration
# -----------------------------
CHUNK_SIZES: List[int] = [500]
OVERLAP_RATIO: float = 0.15  # 15% overlap (500 -> 75)


def build_configs(chunk_sizes: List[int], overlap_ratio: float) -> List[Tuple[int, int]]:
    """
    Build (chunk_size, overlap) configs where overlap is a fixed ratio of chunk_size.
    """
    configs: List[Tuple[int, int]] = []
    for cs in chunk_sizes:
        ov = int(round(cs * overlap_ratio))
        # Safety: overlap must be smaller than chunk size
        ov = min(ov, cs - 1)
        configs.append((cs, ov))
    return configs


def run(cmd: str) -> None:
    """
    Run a shell command and fail fast with a clear error message.
    """
    print(f"\n$ {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("\nERROR: Command failed.")
        print(f"Command: {cmd}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)


def main() -> None:
    configs = build_configs(CHUNK_SIZES, OVERLAP_RATIO)

    print("=== Chunk/Overlap Sweep Configs (overlap = 25% of chunk size) ===")
    for cs, ov in configs:
        print(f"  - chunk_size={cs:>4} | overlap={ov:>4}")
    print("===============================================================\n")

    base_cmd = "python scripts/build_indexes.py --chunk-size {cs} --overlap {ov}"

    for cs, ov in configs:
        print(f"\n=== Building indexes: chunk_size={cs}, overlap={ov} ===")

        # Build indexes + regenerate chunks.jsonl inside build_indexes.py
        build_cmd = base_cmd.format(cs=cs, ov=ov)
        run(build_cmd)

        print("\n=== Running eval sweep for this chunk config ===")
        run("python -m evals.sweep_k")

        print("\n===========================================")
        print(f"Finished config: chunk_size={cs}, overlap={ov}")
        print("===========================================\n")


if __name__ == "__main__":
    main()
