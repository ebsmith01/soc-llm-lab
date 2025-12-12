#!/usr/bin/env python
"""
LoRA finetune for SOC assistant behavior.

- Base: mistralai/Mistral-7B-Instruct-v0.2 (configurable)
- Data: data/finetune/combined_finetune.jsonl
- Output: models/soc-assistant-lora/adapter/
"""

import os
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ------------------------
# Paths & config
# ------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "finetune" / "combined_finetune.jsonl"
OUTPUT_DIR = ROOT / "models" / "soc-assistant-lora" / "adapter"

BASE_MODEL = os.environ.get(
    "BASE_MODEL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# Much lighter for MPS / 8–9GB VRAM
BATCH_SIZE = 1          # was 4
GRAD_ACCUM = 8          # keep total "effective" batch similar
EPOCHS = 2
LR = 2e-4
MAX_LENGTH = 512        # was 1024, halves memory per sequence

# ------------------------
# Prompt formatting
# ------------------------

def format_example(example: Dict) -> str:
    """
    Turn one {"instruction", "input", "output"} into a single LM training string.

    Style:

    ### Instruction:
    <instruction>

    ### Input:
    <input>

    ### Response:
    <output>
    """
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()

    if inp:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{inp}\n\n"
            "### Response:\n"
            f"{out}"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            f"{out}"
        )

    return prompt


# ------------------------
# Dataset loading
# ------------------------

def load_finetune_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Finetune data not found at {DATA_PATH}")

    ds = load_dataset(
        "json",
        data_files=str(DATA_PATH),
        split="train",
    )

    # Add combined "text" field for LM training
    def _add_text(example):
        example["text"] = format_example(example)
        return example

    ds = ds.map(_add_text)
    return ds


# ------------------------
# Model + tokenizer (CPU/MPS-safe, no bitsandbytes)
# ------------------------

def load_base_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect MPS vs CUDA vs CPU
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()

    if has_mps:
        device_map = {"": "mps"}
        torch_dtype = torch.float16
        print("Using MPS device (Apple silicon) in float16.")
    elif has_cuda:
        device_map = "auto"
        torch_dtype = torch.float16
        print("Using CUDA in float16.")
    else:
        device_map = "auto"
        torch_dtype = torch.float32
        print("No MPS/CUDA detected; using CPU in float32 (will be slow).")

    # IMPORTANT:
    #  - no load_in_4bit
    #  - no bitsandbytes
    #  - no prepare_model_for_kbit_training
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ------------------------
# Tokenization
# ------------------------

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


# ------------------------
# Main
# ------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {DATA_PATH}...")
    ds = load_finetune_dataset()

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = load_base_model_and_tokenizer()

    print("Tokenizing dataset...")
    tokenized = ds.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=False,          # MPS doesn’t use this flag
        bf16=False,          # also keep off
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting LoRA training...")
    trainer.train()

    print("Saving LoRA adapter + tokenizer...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Done. Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()