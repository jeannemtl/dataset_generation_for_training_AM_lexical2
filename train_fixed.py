#!/usr/bin/env python3

# FDM_DATA=data/fdm_ttr_hf_10k FDM_EPOCHS=5 python3 train_fixed.py
"""
Fixed SFT trainer with auxiliary losses for cos(2π·1/3·n) and TTR targets.
FIXES: Position calculation now works correctly with truncation
"""

import os
import re
import math
import inspect
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)

# =========================
# Config
# =========================
MODEL_NAME = os.environ.get("FDM_MODEL", "gpt2")
DATA_DIR = os.environ.get("FDM_DATA", "data/fdm_ttr_hf")
OUT_DIR = os.environ.get("FDM_OUT", "out_sft/fdm_ttr_model")
BLOCK_SIZE = int(os.environ.get("FDM_BLOCK", "1024"))
BATCH_TRAIN = int(os.environ.get("FDM_BS_TRAIN", "2"))
BATCH_EVAL = int(os.environ.get("FDM_BS_EVAL", "2"))
GRAD_ACC = int(os.environ.get("FDM_GRAD_ACC", "8"))
EPOCHS = float(os.environ.get("FDM_EPOCHS", "3"))
LR = float(os.environ.get("FDM_LR", "2e-4"))
WARMUP = int(os.environ.get("FDM_WARMUP", "200"))
SAVE_STEPS = int(os.environ.get("FDM_SAVE_STEPS", "500"))
LOG_STEPS = int(os.environ.get("FDM_LOG_STEPS", "50"))

SPECIAL_TOKENS = [
    "<SEP>", "<REPORT>", "<CARRIER=0.333333>",
    "<MSG=HELLO>", "<MSG=SECRET>", "<MSG=AI_RISK>", "<MSG=URGENT>",
    "<MSG=SAFE>", "<MSG=WARNING>", "<MSG=CONFIRM>", "<MSG=ABORT>",
    "<F0=0.040>", "<F0=0.060>", "<F0=0.080>", "<F0=0.100>",
    "<F0=0.120>", "<F0=0.140>", "<F0=0.160>", "<F0=0.180>"
]

RE_STEP = re.compile(r"<STEP=(\d+)>")
RE_COS = re.compile(r"<COS1_3=([-+]?\d*\.\d+|\d+)>")
RE_TTR_TGT = re.compile(r"<TTR_TARGET=([-+]?\d*\.\d+|\d+)>")

# =========================
# TrainingArguments shim
# =========================
def make_training_args(**kwargs) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return TrainingArguments(**allowed)

try:
    from transformers.trainer_utils import IntervalStrategy
    EVAL_STRATEGY = IntervalStrategy.STEPS
except Exception:
    EVAL_STRATEGY = "steps"

# =========================
# Aux-Head Model
# =========================
class AuxHeadModel(torch.nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.base = base_model
        hidden = getattr(base_model.config, "n_embd", None) or getattr(base_model.config, "hidden_size")
        self.cos_head = torch.nn.Linear(hidden, 1)
        self.ttr_head = torch.nn.Linear(hidden, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        step_positions: Optional[List[List[int]]] = None,
        cos_targets: Optional[List[torch.Tensor]] = None,
        ttr_targets: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        loss = out.loss
        hs = out.hidden_states[-1]  # (B, T, H)

        if step_positions and cos_targets and ttr_targets:
            cos_pred_list, ttr_pred_list, cos_gold_list, ttr_gold_list = [], [], [], []
            for i, pos_list in enumerate(step_positions):
                if not pos_list:
                    continue
                # CRITICAL FIX: Filter positions to valid range
                seq_len = hs.size(1)
                valid_positions = [p for p in pos_list if 0 <= p < seq_len]
                
                if not valid_positions:
                    continue
                
                idx = torch.tensor(valid_positions, device=hs.device, dtype=torch.long)
                h = hs[i, idx, :]  # (S, H)
                cos_pred = self.cos_head(h).squeeze(-1)
                ttr_pred = self.ttr_head(h).squeeze(-1)
                cos_pred_list.append(cos_pred)
                ttr_pred_list.append(ttr_pred)
                
                # Only use targets for valid positions
                cos_gold_list.append(cos_targets[i][:len(valid_positions)].to(h.device))
                ttr_gold_list.append(ttr_targets[i][:len(valid_positions)].to(h.device))
                
            if cos_pred_list:
                cos_pred = torch.cat(cos_pred_list)
                ttr_pred = torch.cat(ttr_pred_list)
                cos_gold = torch.cat(cos_gold_list)
                ttr_gold = torch.cat(ttr_gold_list)
                mse = torch.nn.functional.mse_loss
                # loss = loss + 0.1*mse(cos_pred, cos_gold) + 0.1*mse(ttr_pred, ttr_gold)
                loss = loss + 1.0*mse(cos_pred, cos_gold) + 1.0*mse(ttr_pred, ttr_gold)

        out.loss = loss
        return out

# =========================
# Collator
# =========================
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    step_positions: List[List[int]]
    cos_targets: List[torch.Tensor]
    ttr_targets: List[torch.Tensor]

class CosTTRCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tok = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids)
        
        def pad(seq_list, pad_id=0):
            out = []
            for s in seq_list:
                if s.size(0) < max_len:
                    pad_amt = max_len - s.size(0)
                    s = torch.cat([s, torch.full((pad_amt,), pad_id, dtype=s.dtype)], dim=0)
                out.append(s)
            return torch.stack(out, dim=0)

        batch = {
            "input_ids": pad(input_ids, self.tok.pad_token_id),
            "attention_mask": pad(attn_mask, 0),
            "labels": pad(labels, -100),
            "step_positions": [f["step_positions"] for f in features],
            "cos_targets": [torch.tensor(f["cos_targets"], dtype=torch.float) for f in features],
            "ttr_targets": [torch.tensor(f["ttr_targets"], dtype=torch.float) for f in features],
        }
        return batch

# =========================
# FIXED: Position extraction
# =========================
def find_positions_and_targets(text: str, tokenizer: PreTrainedTokenizerBase):
    """
    FIXED VERSION: Properly handles truncation by only keeping positions that exist
    in the tokenized sequence.
    """
    # Tokenize with truncation
    enc_full = tokenizer(text, truncation=True, max_length=BLOCK_SIZE)
    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]
    labels = input_ids.copy()
    
    # Find all step/cos/ttr occurrences in ORIGINAL text
    step_spans = list(RE_STEP.finditer(text))
    cos_spans = list(RE_COS.finditer(text))
    ttr_spans = list(RE_TTR_TGT.finditer(text))
    
    # Extract targets
    cos_vals = [float(m.group(1)) for m in cos_spans]
    ttr_vals = [float(m.group(1)) for m in ttr_spans]
    
    # Map character positions to token positions
    step_positions = []
    for m in step_spans:
        char_pos = m.start()
        # Tokenize up to this position
        prefix = text[:char_pos]
        prefix_ids = tokenizer(prefix, truncation=True, max_length=BLOCK_SIZE)["input_ids"]
        token_pos = len(prefix_ids)
        
        # CRITICAL: Only include if position is within final sequence
        if token_pos < len(input_ids):
            step_positions.append(token_pos)
    
    # CRITICAL: Align targets with valid positions
    # Take minimum of all three lengths
    min_len = min(len(step_positions), len(cos_vals), len(ttr_vals))
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "step_positions": step_positions[:min_len],
        "cos_targets": cos_vals[:min_len],
        "ttr_targets": ttr_vals[:min_len],
    }

# =========================
# Main
# =========================
def main():
    print(f"Loading dataset from: {DATA_DIR}")
    dsd = load_from_disk(DATA_DIR)

    print(f"Loading model & tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    def preprocess(example):
        out = find_positions_and_targets(example["text"], tok)
        return out

    print("Tokenizing & extracting targets...")
    cols_to_remove = [c for c in dsd["train"].column_names if c not in ["text"]]
    ds_train = dsd["train"].map(preprocess, remove_columns=cols_to_remove)
    ds_eval = dsd["validation"].map(preprocess, remove_columns=cols_to_remove)

    print("Building model with aux heads...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base.resize_token_embeddings(len(tok))
    model = AuxHeadModel(base)

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"CUDA: {torch.cuda.is_available()} | BF16 supported: {bf16_ok}")

    train_args = make_training_args(
        output_dir="out_sft",
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=WARMUP,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy=EVAL_STRATEGY,
        eval_steps=SAVE_STEPS,
        bf16=bf16_ok,
        report_to="none",
    )

    collator = CosTTRCollator(tok)
    
    class AuxTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            self.model.base.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True
            )
            
            aux_state = {
                'cos_head': self.model.cos_head.state_dict(),
                'ttr_head': self.model.ttr_head.state_dict(),
            }
            torch.save(aux_state, os.path.join(output_dir, 'aux_heads.pt'))

    trainer = AuxTrainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collator,
        tokenizer=tok,
    )

    print("Starting training...")
    trainer.train()
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"Saved fine-tuned model to: {OUT_DIR}")

if __name__ == "__main__":
    main()
