"""Placeholder policy enforcement for agent-lite."""

BUDGET_LIMIT = 100


def within_budget(spend: int) -> bool:
    return spend <= BUDGET_LIMIT
