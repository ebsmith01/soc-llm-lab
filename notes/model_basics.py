import sys
from pathlib import Path
import os
import tiktoken
from openai import OpenAI

# Allow running this script directly (python notes/model_basics.py) by adding the repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# 1. Token counting example and cost 
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def cost_estimate(model, tokens_in, tokens_out):
    cost = {
        "gpt-4o-mini": {"in": 0.00015, "out": 0.00060},
        "gpt-4.1": {"in": 0.0050, "out": 0.0150}
    }
    ci = cost[model]["in"] * (tokens_in / 1000)
    co = cost[model]["out"] * (tokens_out / 1000)
    return ci + co


raw_log = """
2025-01-14 08:23:54 WIN-DC01 Security 4688 
A new process has been created.
New Process Name: C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe
Command Line: powershell -enc SQBtAHAAbwByAHQAIABtAG8AZAB1AGwAZQAuAC4u
Parent Process Name: C:\\Windows\\System32\\cmd.exe
User: CORP\\evin.smith
"""

security_alert = """
Rule Name: Suspicious PowerShell (Encoded Command)
Severity: High
Description: A PowerShell process executed an encoded command, which 
is commonly used in obfuscation and malicious payload delivery.
Host: WIN-SOC01
User: CORP\\analyst
Process: powershell.exe -enc SQBtAHAAbwByAHQAIABtAG8AZAB1AGwAZQAuAC4u
"""

MITRE = """OpenAI built GPT-2, a language model capable of generating high quality text samples.
 Over concerns that GPT-2 could be used for malicious purposes such as impersonating others,
   or generating misleading news articles, fake social media content, or spam,  
   OpenAI adopted a tiered release schedule. They initially released a smaller, 
   less powerful version of GPT-2 along with a technical description of the approach, 
   but held back the full trained model. Before the full model was released by OpenAI, 
   researchers at Brown University successfully replicated the model using information released by
     OpenAI and open source ML artifacts. This demonstrates that a bad actor with sufficient
       technical skill and compute resources could have replicated GPT-2 and used it for 
       harmful goals before the AI Security community is prepared."""

print("raw_log len:", count_tokens(raw_log), 
      "Security alert len:", count_tokens(security_alert),
        "MITRE len:", count_tokens(MITRE))



# 3. Send to the LLM — it will only see the last N tokens of this text
prompt = f"Summarize the following:\n\n{raw_log}\n\n{security_alert}\n\n{MITRE}\n\nSummary:"

response = client.chat.completions.create(
    model="gpt-4o-mini",        # small context model (~128k)
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200
)
print("Response:", response.choices[0].message.content)


tokens_raw_log = count_tokens(raw_log)
print("Raw Log (mini):", cost_estimate("gpt-4o-mini", tokens_raw_log, 200))
print("Raw Log (4.1):", cost_estimate("gpt-4.1", tokens_raw_log, 200))


