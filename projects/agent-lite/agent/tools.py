"""Placeholder tools for the agent-lite project."""

from typing import Any


def fetch_github_issue(issue_id: str) -> dict[str, Any]:
    return {"id": issue_id, "title": "Stub", "state": "open"}


def lookup_cve(cve_id: str) -> dict[str, Any]:
    return {"cve": cve_id, "severity": "unknown"}
