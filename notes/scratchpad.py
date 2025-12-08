'''{
  "severity": "high",  # choose from: high, medium, low
  "summary": "short description",
  "triage_steps": ["step1", "step2"],
  "citations": ["doc_id:chunk_id"]
}
'''

import sys
from pathlib import Path

# Allow running this script directly via `python notes/scratchpad.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.utils import token_count, estimate_cost, check_json
from openai import OpenAI
import json

client = OpenAI (api_key=OPENAI_API_KEY)
'''
# ---------------------------------------------
# 1) Define a JSON Schema for SOC-style outputs
# ---------------------------------------------
alert_schema = {
    "name": "soc_alert_summary",
    "schema": {
        "type": "object",
        "properties": {
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            },
            "summary": {
                "type": "string",
                "description": "One-sentence summary of what is happening."
            },
            "triage_steps": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Concrete next steps for a SOC analyst."
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of doc_id:chunk_id strings used as evidence."
            }
        },
        "required": ["severity", "summary", "triage_steps", "citations"],
        "additionalProperties": False
    },
    "strict": True  # strongly encourage strict adherence
}

# ---------------------------------------------
# 2) Example context and question
# ---------------------------------------------
context = """
[mitre-t1059:1] PowerShell is a powerful scripting language that can be abused by attackers.
[playbook-powershell:2] When suspicious encoded PowerShell is seen, analysts should:
- Capture full command line
- Check parent process
- Review user account activity
"""

alert_text = """
High severity alert: Encoded PowerShell command executed on WIN-SOC01.
User: CORP\\evin.smith
Process: powershell.exe -enc SQBtAHAAbwByAHQAIABtAG8AZAB1AGwAZQAuAC4u
"""

question = "Summarize this alert for a SOC tier-1 analyst."

# ---------------------------------------------
# 3) Build a simple instruction prompt
# ---------------------------------------------
prompt = f"""
You are a SOC assistant. Using ONLY the context below, analyze the alert and respond.

CONTEXT:
{context}

ALERT:
{alert_text}

Your job:
- Decide severity (low, medium, or high)
- Provide a one-sentence summary
- List specific triage steps a SOC analyst should take
- Include any doc_id:chunk_id references you used as citations

You MUST respond in JSON only.
"""

# ---------------------------------------------
# 4) Call OpenAI using response_format=json_schema
# ---------------------------------------------
response = client.chat.completions.create(
    model="gpt-4.1",  # or gpt-4o-mini if/when it supports json_schema
    messages=[{"role": "user", "content": prompt}],
    response_format={
        "type": "json_schema",
        "json_schema": alert_schema,
    },
)

raw = response.choices[0].message.content
print("RAW MODEL OUTPUT:\n", raw)

# ---------------------------------------------
# 5) Validate JSON with json.loads
# ---------------------------------------------
try:
    data = json.loads(raw)
    print("\n✅ json.loads succeeded. Parsed object:")
    print(data)
except json.JSONDecodeError as e:
    print("\n❌ JSON decoding failed!")
    print("Error:", e)

# Optional: additional manual checks
if isinstance(data, dict):
    print("\nField types:")
    print("severity:", type(data.get("severity")))
    print("summary:", type(data.get("summary")))
    print("triage_steps:", type(data.get("triage_steps")))


'''

# Example SOC-like outputs (some might be JSON from your model later)
soc_outputs = [
    # 1) Plain-text explanation
    """High severity alert: Encoded PowerShell command executed on WIN-SOC01.
    This may indicate script-based malware or recon activity.
    Recommended: capture full command, review parent process, and check user activity.""",

    # 2) MITRE-style paragraph
    """Adversaries may abuse PowerShell commands and scripts for execution.
    PowerShell provides full access to the Windows API and can be used to download payloads.""",

    # 3) A hypothetical JSON result from your model
    """{
      "severity": "high",
      "summary": "Encoded PowerShell was executed by a user on WIN-SOC01.",
      "triage_steps": [
        "Collect full PowerShell command line.",
        "Review parent process and spawning executable.",
        "Check recent login activity for CORP\\\\evin.smith."
      ],
      "citations": ["mitre-t1059:1", "playbook-powershell:2"]
    }""",

    # 4) A slightly malformed JSON result (good for testing check_json)
    """{
      "severity": "medium",
      "summary": "Suspicious scheduled task created.",
      "triage_steps": [
        "Review task name and executable path.",
        "Correlate with recent alerts."
      ]
      "citations": ["sigma-task-suspicious:3"]
    }""",

    # 5) A short, low-severity alert
    """Low severity: Single failed login from internal IP.
    Likely user error unless repeated or from unusual location."""
]

models_to_compare = ["gpt-4o-mini", "gpt-4.1"]

for i, text in enumerate(soc_outputs, start=1):
    print(f"\n=== SOC OUTPUT {i} ===")
    print(text[:200] + ("..." if len(text) > 200 else ""))
    print()

    # Token counts and cost estimates
    for model in models_to_compare:
        tokens = token_count(text, model=model)
        cost = estimate_cost(model, text, expected_output_tokens=200)
        print(f"[{model}] tokens: {tokens}, est cost: ${cost:.6f}")

    # JSON validity check
    ok, parsed, err = check_json(text)
    if ok:
        print("JSON check: VALID ✅")
        print("Type:", type(parsed))
    else:
        print("JSON check: INVALID ❌")
        print("Error:", err)

    print("-" * 60)
