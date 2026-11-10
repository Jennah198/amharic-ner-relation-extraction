"""Paired bootstrap over test sentences: 95% CIs for F1 and for the F1
difference of every run vs. the baseline. With 500 test sentences and one
seed per configuration, differences of ~0.01-0.02 F1 are often noise; this
script makes that explicit (reports/significance.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from .data import read_jsonl
from .evaluate import predict_all
from .inference import NERTagger

RUNS = ["base_full", "base_full_morph", "dapt_full", "dapt_lora", "dapt_full_morph"]


def counts(gold, pred):
    """Per-sentence (tp, n_pred, n_gold) so F1 can be re-computed on any resample."""
    out = []
    for g, p in zip(gold, pred):
        gs, ps = set(get_entities(g)), set(get_entities(p))
        out.append((len(gs & ps), len(ps), len(gs)))
    return np.array(out, dtype=float)


def f1(c):
    tp, npred, ngold = c.sum(0)
    return 2 * tp / (npred + ngold) if npred + ngold else 0.0


def main(n_boot: int = 2000, seed: int = 0):
    data = read_jsonl("data/processed/ner_test.jsonl")
    toks, gold = [d["tokens"] for d in data], [d["ner_tags"] for d in data]
    C = {r: counts(gold, predict_all(NERTagger(f"artifacts/ner/{r}"), toks)) for r in RUNS
         if Path(f"artifacts/ner/{r}").exists()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_boot, len(data)))
    rep = {}
    base = C["base_full"]
    for r, c in C.items():
        boots = np.array([f1(c[i]) for i in idx])
        diffs = np.array([f1(c[i]) - f1(base[i]) for i in idx])
        rep[r] = {"f1": round(float(f1(c)), 4),
                  "f1_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                  "delta_vs_base": round(float(f1(c) - f1(base)), 4),
                  "delta_ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
                  "p_better_than_base": round(float((diffs > 0).mean()), 3)}
        print(r, rep[r])
    Path("reports/significance.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
