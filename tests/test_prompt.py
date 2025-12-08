"""
tests/test_prompts.py

Basic sanity checks for the prompt library in libs/prompts.py.

Goals:
- Make sure prompts build without throwing errors
- Check that key phrases / structures are present
- Prevent accidental breaking changes when you refactor prompts
"""

from libs import prompts


def test_system_prompt_basic():
    text = prompts.SYSTEM_PROMPT
    assert isinstance(text, str)
    assert "helpful AI assistant" in text


def test_strict_rag_prompt_contains_rules_and_context():
    ctx = "[mitre-t1059:1]\nPowerShell abuse description..."
    q = "How should a SOC analyst respond?"
    full = prompts.make_strict_rag_prompt(ctx, q)

    # Basic shape checks
    assert "You must answer ONLY" in full
    assert "CONTEXT:" in full
    assert "QUESTION:" in full
    assert "ANSWER:" in full

    # Context and question should be interpolated
    assert "PowerShell abuse description" in full
    assert "How should a SOC analyst respond" in full


def test_structured_json_prompt_mentions_json_and_shape():
    ctx = "[sigma-rule:5]\nSuspicious PowerShell detection"
    q = "Summarize this detection."

    full = prompts.make_structured_json_prompt(ctx, q)

    # Should strongly mention JSON-only behavior
    assert "valid JSON only" in full or "valid JSON" in full
    assert '"answer":' in full
    assert '"reasoning":' in full
    assert '"citations":' in full

    # Should include context and question text
    assert "Suspicious PowerShell detection" in full
    assert "Summarize this detection" in full


def test_security_reasoning_prompt_structure():
    alert = "High severity alert: encoded PowerShell command on WIN-SOC01."

    full = prompts.make_security_reasoning_prompt(alert)

    # Check sections
    assert "senior SOC analyst" in full
    assert "1) Summary" in full
    assert "2) Likely techniques / behaviors" in full
    assert "3) Recommended triage steps" in full
    assert "4) Escalation criteria" in full

    # Alert should be present
    assert "encoded PowerShell command" in full


def test_react_prompt_contains_loop_and_tools():
    task_desc = "Investigate suspicious PowerShell execution."
    tools_desc = """
    - search_logs(query)
    - lookup_cve(cve_id)
    """

    full = prompts.make_react_prompt(task_desc, tools_desc)

    # ReAct loop structure
    assert "Thought:" in full
    assert "Action:" in full
    assert "Action Input:" in full
    assert "Observation:" in full
    assert "Final Answer:" in full

    # Tools should appear in the prompt
    assert "search_logs" in full
    assert "lookup_cve" in full

    # Task should be interpolated
    assert "Investigate suspicious PowerShell execution" in full