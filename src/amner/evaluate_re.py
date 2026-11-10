"""Relation-extraction evaluation on the hand-built diagnostic set.

``eval/re_gold.jsonl`` holds 25 news-style sentences written by the team with
gold entities and gold relations, including hard and negative cases. Each
sentence is scored twice:

* **gold-entity setting** – the extractor receives the gold entity spans,
  so the score measures the relation patterns alone;
* **end-to-end setting** – entities come from the fine-tuned NER model, so
  NER errors propagate (this is what users of the web app experience).

Relations are compared as (head lemma, relation, tail lemma) triples after
Fidel normalization; symmetric relations are order-insensitive.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .inference import word_tokenize
from .morphology import MorphAnalyzer
from .normalize import normalize, normalize_chars
from .relations import Relation, RelationExtractor

SYMMETRIC = {"MET_WITH", "AGREEMENT_WITH", "CONFLICT_WITH"}


def gold_key(h: str, r: str, t: str) -> tuple:
    h, t = normalize_chars(h), normalize_chars(t)
    if r in SYMMETRIC and t < h:
        h, t = t, h
    return (h, r, t)


def find_span(tokens: list[str], surface: str, used: set) -> tuple[int, int]:
    target = [w for w, _, _ in word_tokenize(normalize(surface))]
    n = len(target)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == target and i not in used:
            used.add(i)
            return i, i + n
    raise ValueError(f"entity {surface!r} not found in {tokens}")


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def score(items, extract_fn) -> tuple[dict, list]:
    per_rel = Counter()
    errors = []
    for ex, toks, gold_ents in items:
        gold = {gold_key(*r) for r in ex["relations"]}
        pred = {r.key() for r in extract_fn(toks, gold_ents)}
        for k in pred & gold:
            per_rel[(k[1], "tp")] += 1
        for k in pred - gold:
            per_rel[(k[1], "fp")] += 1
            errors.append({"id": ex["id"], "type": "FP", "triple": k})
        for k in gold - pred:
            per_rel[(k[1], "fn")] += 1
            errors.append({"id": ex["id"], "type": "FN", "triple": k})
    rels = sorted({r for r, _ in per_rel})
    res = {r: prf(per_rel[(r, "tp")], per_rel[(r, "fp")], per_rel[(r, "fn")]) for r in rels}
    tot = [sum(per_rel[(r, k)] for r in rels) for k in ("tp", "fp", "fn")]
    res["micro"] = prf(*tot)
    return res, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/re_gold.jsonl")
    ap.add_argument("--model", default=None, help="NER model dir for the end-to-end setting")
    a = ap.parse_args()
    rex = RelationExtractor()

    items = []
    for line in open(a.gold, encoding="utf-8"):
        ex = json.loads(line)
        toks = [w for w, _, _ in word_tokenize(normalize(ex["text"]))]
        used: set = set()
        ents = []
        for surf, lab in ex["entities"]:
            s, e = find_span(toks, surf, used)
            ents.append({"token_start": s, "token_end": e, "label": lab})
        items.append((ex, toks, ents))

    report = {"sentences": len(items), "gold_relations": sum(len(ex["relations"]) for ex, _, _ in items)}
    report["gold_entities"], err_gold = score(items, rex.extract)
    # ablation: same rules, but clitics stripped greedily without the lexicon
    report["gold_entities_greedy_analyzer"], _ = score(items, RelationExtractor(analyzer=MorphAnalyzer()).extract)
    print("gold entities, greedy analyzer:", json.dumps(report["gold_entities_greedy_analyzer"]["micro"]))
    print("gold entities:", json.dumps(report["gold_entities"]["micro"]))

    err_e2e = []
    if a.model:
        from .bio import bio_to_spans
        from .inference import NERTagger
        tagger = NERTagger(a.model)

        def e2e(toks, _gold_ents):
            labels = tagger.tag_tokens(toks)
            ents = [{"token_start": s, "token_end": e, "label": t} for s, e, t in bio_to_spans(labels)]
            return rex.extract(toks, ents)

        report["end_to_end"], err_e2e = score(items, e2e)
        report["end_to_end_model"] = a.model
        print("end-to-end:   ", json.dumps(report["end_to_end"]["micro"]))

    Path("reports/eval").mkdir(parents=True, exist_ok=True)
    Path("reports/eval/relations.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Relation extraction – error list", "", "## Gold-entity setting", ""]
    lines += [f"- [{e['type']}] #{e['id']}: {' | '.join(e['triple'])}" for e in err_gold] or ["(none)"]
    if a.model:
        lines += ["", "## End-to-end setting", ""]
        lines += [f"- [{e['type']}] #{e['id']}: {' | '.join(e['triple'])}" for e in err_e2e] or ["(none)"]
    Path("reports/eval/relation_errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
