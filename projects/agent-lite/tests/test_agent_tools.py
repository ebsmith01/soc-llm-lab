from agent import tools


def test_fetch_github_issue_stub():
    issue = tools.fetch_github_issue("123")
    assert issue["id"] == "123"
