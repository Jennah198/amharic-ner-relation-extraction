"""Stage 2 – Normalization & cleaning.

Reads every ``data/raw/*.jsonl`` document (scraped + Hub news), and

1. cleans markup / URLs / emojis / mentions          (normalize.clean_text)
2. normalizes Fidel homophones, labials, digits,     (normalize.normalize_chars)
   punctuation and spacing
3. applies the code-switching policy                 (normalize.handle_code_switching)
4. splits into sentences on ። ! ? ፧
5. filters short / mostly non-Ethiopic sentences
6. de-duplicates (exact, after normalization) and removes every sentence that
   also occurs in the MasakhaNER validation/test sets (prevents test leakage
   into tokenizer/MLM training)
7. writes ``data/processed/corpus_train.txt`` / ``corpus_valid.txt`` (one sentence
   per line, 98/2 split) and ``reports/corpus_stats.json``.

The MasakhaNER files are normalized token-by-token (label alignment preserved)
into ``data/processed/ner_{split}.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .normalize import ETHIOPIC_RE, LATIN_WORD_RE, _CHAR_MAP, is_good_sentence, normalize, normalize_token, split_sentences


def normalize_ner(raw_dir: Path, out_dir: Path) -> dict:
    stats = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        src = raw_dir / f"masakhaner_amh_{split}.jsonl"
        changed_tokens = total = 0
        with src.open(encoding="utf-8") as fi, (out_dir / f"ner_{split}.jsonl").open("w", encoding="utf-8") as fo:
            for line in fi:
                ex = json.loads(line)
                norm = [normalize_token(t) for t in ex["tokens"]]
                changed_tokens += sum(a != b for a, b in zip(ex["tokens"], norm))
                total += len(norm)
                ex["raw_tokens"], ex["tokens"] = ex["tokens"], norm
                fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
        stats[split] = {"tokens": total, "tokens_changed_by_normalization": changed_tokens,
                        "pct_changed": round(100 * changed_tokens / max(total, 1), 2)}
    return stats


def build(raw_dir: Path, out_dir: Path, report_dir: Path, code_switch: str, seed: int) -> None:
    # sentences that must never be seen in unlabelled training data
    held_out = set()
    for split in ["validation", "test"]:
        p = out_dir / f"ner_{split}.jsonl"
        for line in p.open(encoding="utf-8"):
            held_out.add(" ".join(json.loads(line)["tokens"]))

    stats = Counter()
    homophone_hits = Counter()
    per_source = Counter()
    seen, sentences = set(), []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            doc = json.loads(line)
            raw = "\n".join(x for x in [doc.get("title", ""), doc.get("text", "")] if x)
            stats["docs"] += 1
            stats["raw_chars"] += len(raw)
            homophone_hits.update(ch for ch in raw if ch in _CHAR_MAP)
            stats["latin_words"] += len(LATIN_WORD_RE.findall(raw))
            text = normalize(raw, code_switch=code_switch)
            for para in text.split("\n"):
                for s in split_sentences(para):
                    stats["sentences_raw"] += 1
                    if not is_good_sentence(s):
                        stats["dropped_filter"] += 1
                        continue
                    if s in seen:
                        stats["dropped_duplicate"] += 1
                        continue
                    if s in held_out:
                        stats["dropped_heldout_overlap"] += 1
                        continue
                    seen.add(s)
                    sentences.append(s)
                    per_source[doc.get("source", path.stem).split(":")[0]] += 1

    random.Random(seed).shuffle(sentences)
    n_valid = max(200, len(sentences) // 50)
    valid, train = sentences[:n_valid], sentences[n_valid:]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "corpus_valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    # a small human-readable sample is committed to the repo for inspection
    (out_dir / "corpus_sample.txt").write_text("\n".join(train[:200]) + "\n", encoding="utf-8")

    words = [w for s in sentences for w in s.split()]
    report = {
        **stats,
        "sentences_kept": len(sentences),
        "train_sentences": len(train),
        "valid_sentences": len(valid),
        "tokens": len(words),
        "types": len(set(words)),
        "ethiopic_chars": sum(bool(ETHIOPIC_RE.match(c)) for s in sentences for c in s),
        "kept_per_source": dict(per_source),
        "homophone_chars_normalized_top": dict(homophone_hits.most_common(20)),
        "code_switch_mode": code_switch,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "corpus_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--code-switch", choices=["keep", "tag", "drop"], default="keep")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    ner_stats = normalize_ner(a.raw_dir / "masakhaner", a.out_dir)
    print("NER normalization:", json.dumps(ner_stats))
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / "ner_normalization_stats.json").write_text(json.dumps(ner_stats, indent=2))
    build(a.raw_dir, a.out_dir, a.report_dir, a.code_switch, a.seed)


if __name__ == "__main__":
    main()
