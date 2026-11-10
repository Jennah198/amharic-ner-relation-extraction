"""Stage 5 – Quantitative evaluation + morphological error analysis.

For a trained model directory this script reports

* entity-level precision / recall / F1 (seqeval, CoNLL-style, micro) overall and
  per type, in the default and in the strict IOB2 mode;
* a **normalization ablation**: the same model evaluated on the raw
  (un-normalized) MasakhaNER test tokens vs. the normalized ones;
* **morphological buckets** – recall of gold entities grouped by
  - the clitics attached to the entity (bare / prefix / suffix / both),
  - lexical novelty: surface form seen in training, *unseen surface but seen
    stem* (a new clitic variant of a known name – the case morphology-awareness
    is supposed to fix), and completely unseen;
* an **error taxonomy** (missed, spurious, boundary, type confusion) with
  concrete examples written to ``reports/error_analysis_<name>.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

from .bio import bio_to_spans
from .data import read_jsonl
from .inference import NERTagger
from .morphology import DEFAULT_ANALYZER as AN
from .morphology import affix_bucket


def predict_all(tagger: NERTagger, sents: list[list[str]], bs: int = 32) -> list[list[str]]:
    out = []
    for i in range(0, len(sents), bs):
        out += [labels for labels, _ in tagger.tag_tokens_batch(sents[i:i + bs])]
    return out


def span_bucket(tokens: list[str], s: int, e: int) -> str:
    pre = AN.strip_prefix(tokens[s])[0]
    suf = AN.strip_suffix(tokens[e - 1])[1]
    return {(True, True): "prefix+suffix", (True, False): "prefix",
            (False, True): "suffix", (False, False): "bare"}[(bool(pre), bool(suf))]


def novelty(surface: str, lemma: str, train_surfaces: set, train_lemmas: set) -> str:
    if surface in train_surfaces:
        return "seen_surface"
    if lemma in train_lemmas:
        return "unseen_surface_seen_stem"
    return "unseen"


def analyse(tokens_list, gold, pred, train_surfaces, train_lemmas, max_examples=12):
    buckets = defaultdict(Counter)       # bucket -> {tp, n}
    errors = Counter()
    examples = defaultdict(list)
    confusion = Counter()

    for toks, g, p in zip(tokens_list, gold, pred):
        gs = {(s, e): t for s, e, t in bio_to_spans(g)}
        ps = {(s, e): t for s, e, t in bio_to_spans(p)}
        ctx = " ".join(toks)
        for (s, e), t in gs.items():
            surface = " ".join(toks[s:e])
            lemma = AN.lemma_span(toks[s:e])
            hit = ps.get((s, e)) == t
            for key in (f"affix:{span_bucket(toks, s, e)}",
                        f"novelty:{novelty(surface, lemma, train_surfaces, train_lemmas)}",
                        f"type:{t}", f"length:{min(e - s, 3)}{'+' if e - s >= 3 else ''}"):
                buckets[key]["n"] += 1
                buckets[key]["tp"] += hit
            if hit:
                continue
            overl = [(ps_, pt) for ps_, pt in ps.items() if ps_[0] < e and s < ps_[1]]
            if (s, e) in ps:
                kind = "type_confusion"
                confusion[(t, ps[(s, e)])] += 1
                detail = f"gold {t}, predicted {ps[(s, e)]}"
            elif overl:
                (os_, oe), ot = overl[0]
                kind = "boundary" if ot == t else "boundary+type"
                detail = f"gold [{surface}]/{t} vs predicted [{' '.join(toks[os_:oe])}]/{ot}"
            else:
                kind = "missed"
                detail = f"[{surface}]/{t} predicted as O"
            errors[kind] += 1
            b = span_bucket(toks, s, e)
            if len(examples[kind]) < max_examples:
                examples[kind].append({"entity": surface, "lemma": lemma, "affix": b,
                                       "detail": detail, "sentence": ctx})
            if b != "bare" and len(examples["morph_" + kind]) < max_examples:
                examples["morph_" + kind].append({"entity": surface, "lemma": lemma, "affix": b,
                                                  "detail": detail, "sentence": ctx})
        for (s, e), t in ps.items():
            if not any(gs_[0] < e and s < gs_[1] for gs_ in gs):
                errors["spurious"] += 1
                if len(examples["spurious"]) < max_examples:
                    examples["spurious"].append({"entity": " ".join(toks[s:e]), "lemma": "",
                                                 "affix": span_bucket(toks, s, e),
                                                 "detail": f"predicted {t}, gold O", "sentence": ctx})
    bucket_recall = {k: {"n": v["n"], "recall": round(v["tp"] / v["n"], 4)}
                     for k, v in sorted(buckets.items())}
    return bucket_recall, dict(errors), examples, {f"{a}->{b}": c for (a, b), c in confusion.most_common()}


def write_markdown(path: Path, name: str, rep: dict, examples: dict) -> None:
    L = [f"# Error analysis – `{name}`", "",
         f"Test F1 (normalized input): **{rep['f1']:.4f}**  ·  strict IOB2 F1: {rep['f1_strict']:.4f}  ·  "
         f"raw (un-normalized) input F1: {rep['f1_raw_input']:.4f}", "",
         "## Recall by morphological / lexical bucket", "", "| bucket | gold entities | recall |", "|---|---:|---:|"]
    for k, v in rep["buckets"].items():
        L.append(f"| {k} | {v['n']} | {v['recall']:.3f} |")
    L += ["", "## Error taxonomy", "", "| error type | count |", "|---|---:|"]
    for k, v in sorted(rep["errors"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} |")
    if rep["type_confusions"]:
        L += ["", "Type confusions: " + ", ".join(f"{k} ({v})" for k, v in rep["type_confusions"].items())]
    order = ["morph_missed", "morph_boundary", "morph_type_confusion", "missed", "boundary",
             "boundary+type", "type_confusion", "spurious"]
    for kind in order:
        if not examples.get(kind):
            continue
        L += ["", f"## Examples: {kind.replace('_', ' ')}", ""]
        for ex in examples[kind]:
            L.append(f"- **{ex['entity']}** (lemma: {ex['lemma'] or '–'}, affix: {ex['affix']}) — {ex['detail']}  ")
            L.append(f"  > {ex['sentence'][:220]}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--quantize", action="store_true")
    a = ap.parse_args()
    name = a.name or Path(a.model).name

    train = read_jsonl(Path(a.data_dir) / "ner_train.jsonl")
    train_surfaces, train_lemmas = set(), set()
    for ex in train:
        for s, e, _ in bio_to_spans(ex["ner_tags"]):
            train_surfaces.add(" ".join(ex["tokens"][s:e]))
            train_lemmas.add(AN.lemma_span(ex["tokens"][s:e]))

    data = read_jsonl(Path(a.data_dir) / f"ner_{a.split}.jsonl")
    toks = [ex["tokens"] for ex in data]
    raw = [ex.get("raw_tokens", ex["tokens"]) for ex in data]
    gold = [ex["ner_tags"] for ex in data]

    tagger = NERTagger(a.model, quantize=a.quantize)
    pred = predict_all(tagger, toks)
    pred_raw = predict_all(tagger, raw)

    rep_dict = classification_report(gold, pred, output_dict=True)
    buckets, errors, examples, conf = analyse(toks, gold, pred, train_surfaces, train_lemmas)
    rep = {
        "name": name, "model": a.model, "split": a.split, "sentences": len(data),
        "f1": round(f1_score(gold, pred), 4),
        "f1_strict": round(f1_score(gold, pred, mode="strict", scheme=IOB2), 4),
        "f1_raw_input": round(f1_score(gold, pred_raw), 4),
        "per_type": {k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score", "support")}
                     for k, v in rep_dict.items()},
        "buckets": buckets, "errors": errors, "type_confusions": conf,
    }
    out = Path("reports/eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out / f"error_analysis_{name}.md", name, rep, examples)
    print(classification_report(gold, pred, digits=4))
    print(json.dumps({k: rep[k] for k in ("f1", "f1_strict", "f1_raw_input")}, indent=2))
    print(json.dumps(buckets, indent=2))


if __name__ == "__main__":
    main()
