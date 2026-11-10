"""Stage 4a – Domain-adaptive Masked Language Modelling (DAPT).

Continues pre-training of the Amharic backbone on our normalized news corpus
(scraped BBC Amharic + Hub news). Two effects we want:

* the encoder sees homophone-*normalized* spelling (the original pre-training
  text was not normalized), and
* news/political vocabulary and names get better contextual representations.

Optionally (``--extended-tokenizer``) the backbone vocabulary is first extended
with the whole-word tokens selected in ``tokenization.py``; each new embedding
row is initialised as the mean of the embeddings of the subword pieces the
word used to be split into, so training starts from a sensible point.

Perplexity on the held-out corpus is measured before and after adaptation and
written to ``reports/mlm_<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from .tb_export import tb_report_to
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)


def load_lines(path: str, limit: int | None) -> list[str]:
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    return lines[:limit] if limit else lines


def init_new_embeddings(model, old_tok, new_tok, n_old: int) -> None:
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for idx in range(n_old, len(new_tok)):
            word = new_tok.convert_ids_to_tokens(idx)
            ids = old_tok(word, add_special_tokens=False)["input_ids"]
            if ids:
                emb[idx] = emb[ids].mean(0)


@torch.no_grad()
def perplexity(model, tokenizer, ds, collator, batch_size=32) -> float:
    model.eval()
    torch.manual_seed(0)  # same masking pattern for before/after comparison
    losses, n = 0.0, 0
    for i in range(0, len(ds), batch_size):
        batch = collator([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        k = (batch["labels"] != -100).sum().item()
        losses += out.loss.item() * k
        n += k
    return math.exp(losses / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="rasyosef/roberta-medium-amharic")
    ap.add_argument("--extended-tokenizer", default=None, help="path from tokenization.py (optional)")
    ap.add_argument("--train", default="data/processed/corpus_train.txt")
    ap.add_argument("--valid", default="data/processed/corpus_valid.txt")
    ap.add_argument("--max-train-lines", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="artifacts/mlm_dapt")
    ap.add_argument("--name", default="dapt")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)

    old_tok = AutoTokenizer.from_pretrained(a.base_model)
    tok = AutoTokenizer.from_pretrained(a.extended_tokenizer) if a.extended_tokenizer else old_tok
    model = AutoModelForMaskedLM.from_pretrained(a.base_model)
    n_old = model.get_input_embeddings().weight.shape[0]
    if len(tok) > n_old:
        model.resize_token_embeddings(len(tok))
        init_new_embeddings(model, old_tok, tok, n_old)
        print(f"[vocab] extended {n_old} -> {len(tok)}")

    def enc(batch):
        return tok(batch["text"], truncation=True, max_length=a.max_length)

    train = Dataset.from_dict({"text": load_lines(a.train, a.max_train_lines)}).map(enc, batched=True, remove_columns=["text"])
    valid = Dataset.from_dict({"text": load_lines(a.valid, 1000)}).map(enc, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)

    ppl_before = perplexity(model, tok, valid, collator)
    print(f"[ppl] before adaptation: {ppl_before:.2f}")

    args = TrainingArguments(
        output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size, learning_rate=a.lr, warmup_ratio=0.06,
        weight_decay=0.01, logging_steps=50, save_strategy="no", report_to=tb_report_to(), logging_dir=f"runs/mlm_{a.name}",
        seed=a.seed, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=train, data_collator=collator)
    result = trainer.train()

    ppl_after = perplexity(model, tok, valid, collator)
    print(f"[ppl] after adaptation: {ppl_after:.2f}")
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)

    rep = {"base_model": a.base_model, "extended_tokenizer": a.extended_tokenizer,
           "vocab_size": len(tok), "train_lines": len(train), "steps": result.global_step,
           "train_runtime_s": round(result.metrics["train_runtime"], 1),
           "train_loss": round(result.training_loss, 4),
           "perplexity_before": round(ppl_before, 2), "perplexity_after": round(ppl_after, 2)}
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/mlm_{a.name}.json").write_text(json.dumps(rep, indent=2))
    print(rep)


if __name__ == "__main__":
    main()
