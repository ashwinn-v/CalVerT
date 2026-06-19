"""LoRA SFT cold-start trainer for the role-beam agent.

Uses TRL's SFTTrainer in chat-format mode: each row's ``messages`` is followed
by an assistant turn whose content is the ``target`` JSON. The trainer masks
the prompt tokens and computes loss only on the target tokens.

Default recipe: LoRA rank=32, alpha=64, 2 epochs, ``5e-6`` learning rate.
The paper notes a roughly one-day single-node run on bf16 hardware.

CLI::

    python -m grpo.sft_train \\
        --corpus data/sft_corpus.parquet \\
        --output_dir runs/sft-cold-start \\
        --base_model Qwen/Qwen3-32B \\
        --num_epochs 2 \\
        --lora_r 32 --lora_alpha 64 \\
        --lr 5e-6 \\
        --bf16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _build_chat_dataset(corpus_path: Path, tokenizer):
    """Reads parquet, materializes each row's prompt + target into a TRL-shape
    dataset where the assistant turn = the strict-schema action JSON."""
    import pandas as pd
    from datasets import Dataset

    df = pd.read_parquet(corpus_path)

    def _to_chat(row: Dict[str, Any]) -> Dict[str, Any]:
        msgs = list(row["messages"])
        msgs.append({"role": "assistant", "content": str(row["target"])})
        return {"messages": msgs}

    return Dataset.from_pandas(df).map(_to_chat, remove_columns=df.columns.tolist())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-32B")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--wandb_project", default="grpo-telemetry-agent")
    p.add_argument("--wandb_run_name", default="sft-cold-start-v0")
    args = p.parse_args()

    # Imports deferred (torch/trl heavy).
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig
    from trl import SFTTrainer

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sft_train] loading tokenizer: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft_train] loading base model: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        attn_implementation="eager",   # matches the paper's enforce_eager vLLM setup
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    print(f"[sft_train] loading corpus: {args.corpus}", flush=True)
    dataset = _build_chat_dataset(args.corpus, tokenizer)
    print(f"[sft_train]   {len(dataset)} chat rows", flush=True)

    trainer_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=bool(args.bf16),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
    )

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=trainer_args,
        peft_config=lora_cfg,
    )

    print(f"[sft_train] starting train…", flush=True)
    trainer.train()
    final_dir = args.output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft_train] saved final LoRA adapters → {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
