"""Shared pytest fixtures and configuration for Secure RAG SOC tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root (which contains libs/, etc.) is on sys.path so that
# `import libs` resolves to the local package instead of any globally installed one.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
