"""Stage 4b – NER fine-tuning (full fine-tuning or LoRA PEFT).

Examples::

    # full fine-tuning of the original backbone (baseline)
    python -m amner.train_ner --model rasyosef/roberta-medium-amharic --name base_full

    # parameter-efficient LoRA on the domain-adapted encoder
    python -m amner.train_ner --model artifacts/mlm_dapt --mode lora --name dapt_lora

    # + morphology-aware clitic augmentation
    python -m amner.train_ner --model artifacts/mlm_dapt --morph-augment --name dapt_full_morph

LoRA adapters are merged into the base weights after training so that every
run produces a plain ``AutoModelForTokenClassification`` directory that the
evaluation script, spaCy component and API can load identically.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from .tb_export import tb_report_to
from .data import ID2LABEL, LABEL2ID, LABELS, load_ner, tokenize_and_align


def compute_metrics_fn(p):
    preds = np.argmax(p.predictions, axis=-1)
    y_true, y_pred = [], []
    for pr, lab in zip(preds, p.label_ids):
        t, q = [], []
        for pi, li in zip(pr, lab):
            if li != -100:
                t.append(ID2LABEL[int(li)])
                q.append(ID2LABEL[int(pi)])
        y_true.append(t)
        y_pred.append(q)
    return {"precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rasyosef/roberta-medium-amharic")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--mode", choices=["full", "lora"], default="full")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--morph-augment", action="store_true")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="artifacts/ner")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    lr = a.lr or (5e-5 if a.mode == "full" else 5e-4)
    out_dir = Path(a.out_root) / a.name

    tok = AutoTokenizer.from_pretrained(a.model, add_prefix_space=True)
    ds = load_ner(a.data_dir, augment=a.morph_augment, seed=a.seed)
    cols = ds["train"].column_names
    tds = ds.map(lambda b: tokenize_and_align(b, tok, a.max_length), batched=True, remove_columns=cols)

    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    n_total = sum(p.numel() for p in model.parameters())
    if a.mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.TOKEN_CLS, r=a.lora_r, lora_alpha=a.lora_alpha,
                         lora_dropout=0.1, target_modules=["query", "key", "value", "dense"],
                         bias="none")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    args = TrainingArguments(
        output_dir=str(out_dir) + "_ckpt", num_train_epochs=a.epochs, learning_rate=lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=32,
        warmup_ratio=0.1, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=25, report_to=tb_report_to(), logging_dir=f"runs/{a.name}", seed=a.seed, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=tds["train"], eval_dataset=tds["validation"],
                      tokenizer=tok, data_collator=DataCollatorForTokenClassification(tok),
                      compute_metrics=compute_metrics_fn,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)])
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    test = trainer.predict(tds["test"])

    if a.mode == "lora":
        model = trainer.model.merge_and_unload()
    else:
        model = trainer.model
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    rep = {"name": a.name, "base": a.model, "mode": a.mode, "morph_augment": a.morph_augment,
           "train_sentences": len(ds["train"]), "params_total": n_total, "params_trainable": n_train,
           "trainable_pct": round(100 * n_train / n_total, 2), "lr": lr, "epochs_run": trainer.state.epoch,
           "train_time_s": round(train_time, 1),
           "best_val_f1": round(trainer.state.best_metric or 0, 4),
           "test_precision": round(test.metrics["test_precision"], 4),
           "test_recall": round(test.metrics["test_recall"], 4),
           "test_f1": round(test.metrics["test_f1"], 4)}
    Path("reports/runs").mkdir(parents=True, exist_ok=True)
    Path(f"reports/runs/{a.name}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
