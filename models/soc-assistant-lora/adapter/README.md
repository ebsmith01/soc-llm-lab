---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- lora
- transformers
---
# SOC Assistant LoRA Adapter

- **Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Adapter path:** `models/soc-assistant-lora/adapter/`

## Training data

- File: `data/finetune/combined_finetune.jsonl`
- Schema: each row is

  ```json
  {
    "instruction": "...",
    "input": "<optional context or question>",
    "output": "<ideal assistant response>"
  }


## Sources:
-	Converted eval questions from evals/baseline.json (wins + losses).
-	Hand-crafted SOC / MITRE ATT&CK / AI security Q&A.
-	Explicit refusal examples (“exploit code”, “tax advice”, out-of-corpus).

## Hyperparameters
-	Epochs: 2
-	Batch size: 1 (gradient_accumulation_steps = 8)
-	Max sequence length: 512
-	Learning rate: 2e-4
-	LoRA config:
-	r = 8
-	lora_alpha = 16
-	lora_dropout = 0.05
-	target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
-	task_type = "CAUSAL_LM"
### Framework versions

- PEFT 0.18.0