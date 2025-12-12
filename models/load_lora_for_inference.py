import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "models/soc-assistant-lora/adapter"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
).to(device)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

prompt = """### Instruction:
You are a SOC assistant. Answer using only MITRE ATT&CK and AI security documents.

### Input:
What is the purpose of the MITRE ATT&CK framework?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))