"""Dataset utilities: loading BIO data and aligning labels to subwords."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

from .morphology import morph_augment

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_ner(data_dir: str | Path = "data/processed", augment: bool = False,
             aug_copies: int = 1, seed: int = 42) -> DatasetDict:
    data_dir = Path(data_dir)
    splits = {}
    for split in ["train", "validation", "test"]:
        rows = [{"tokens": ex["tokens"], "ner_tags": ex["ner_tags"]}
                for ex in read_jsonl(data_dir / f"ner_{split}.jsonl")]
        if split == "train" and augment:
            rng = random.Random(seed)
            extra = []
            for _ in range(aug_copies):
                for r in rows:
                    out = morph_augment(r["tokens"], r["ner_tags"], rng)
                    if out:
                        extra.append({"tokens": out[0], "ner_tags": out[1]})
            print(f"[augment] +{len(extra)} morphology-augmented sentences")
            rows = rows + extra
        splits[split] = Dataset.from_list(rows)
    return DatasetDict(splits)


def tokenize_and_align(batch, tokenizer, max_length: int = 256, label_all: bool = False):
    """Label the first subword of each word; other subwords get -100 (ignored)."""
    enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    all_labels = []
    for i, tags in enumerate(batch["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)
        prev, labs = None, []
        for wid in word_ids:
            if wid is None:
                labs.append(-100)
            elif wid != prev:
                labs.append(LABEL2ID[tags[wid]])
            else:
                if label_all:
                    t = tags[wid]
                    labs.append(LABEL2ID["I-" + t[2:]] if t.startswith("B-") else LABEL2ID[t])
                else:
                    labs.append(-100)
            prev = wid
        all_labels.append(labs)
    enc["labels"] = all_labels
    return enc
