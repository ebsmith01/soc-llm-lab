"""
Sweep multiple chunk sizes + overlaps.

For each config:
  - regenerate chunks.jsonl
  - rebuild BM25, embeddings, FAISS
  - run eval sweep (top_k, alpha)
  - output a leaderboard per config

Usage:
  python scripts/build_chunks_sweep.py
"""

import os
import subprocess

CHUNK_SIZES = [300, 400, 500, 600]
OVERLAPS = [60, 80, 100, 120, 150, 200]

BASE_CMD = "python scripts/build_indexes.py --chunk-size {cs} --overlap {ov}"

for cs in CHUNK_SIZES:
    for ov in OVERLAPS:
        print(f"\n=== Building indexes: chunk_size={cs}, overlap={ov} ===")

        # Build
        cmd = BASE_CMD.format(cs=cs, ov=ov)
        subprocess.run(cmd, shell=True, check=True)

        print("\n=== Running sweep for this chunk config ===")
        subprocess.run("python -m evals.sweep_k", shell=True, check=True)

        print("\n===========================================")
        print(f"Finished config: chunk_size={cs}, overlap={ov}")
        print("===========================================\n")