"""Entry point placeholder for the agent-lite loop."""

from . import tools, policy


def run_once() -> dict:
    result = tools.fetch_github_issue("1")
    allowed = policy.within_budget(10)
    return {"issue": result, "within_budget": allowed}


if __name__ == "__main__":
    print(run_once())
