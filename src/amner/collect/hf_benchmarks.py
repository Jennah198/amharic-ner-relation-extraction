"""Download local benchmarks / corpora from the Hugging Face Hub.

* MasakhaNER 1.0 (Amharic split) – gold BIO-annotated NER data
  (PER, ORG, LOC, DATE) from Amharic news. Stored as CoNLL-style JSONL.
* MasakhaNEWS (Amharic) and an Amharic news-classification corpus – unlabelled
  news text used, together with the scraped BBC articles, for normalization
  statistics, tokenizer training and masked-language-model adaptation.

The MasakhaNER repo uses a legacy loading script, which recent ``datasets``
versions refuse to execute, so we read the auto-converted parquet files from
the ``refs/convert/parquet`` branch instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

MASAKHANER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]
PARQUET = "hf://datasets/masakhane/masakhaner@refs%2Fconvert%2Fparquet/amh/{split}/*.parquet"


def download_masakhaner(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        ds = load_dataset("parquet", data_files=PARQUET.format(split=split), split="train")
        feat = ds.features["ner_tags"].feature
        names = getattr(feat, "names", None) or MASAKHANER_LABELS
        with (out_dir / f"masakhaner_amh_{split}.jsonl").open("w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps({"id": ex["id"], "tokens": ex["tokens"],
                                    "ner_tags": [names[t] for t in ex["ner_tags"]]},
                                   ensure_ascii=False) + "\n")
        print(f"MasakhaNER amh {split}: {len(ds)} sentences")


def download_news_text(out_path: Path, max_docs: int) -> None:
    """Unlabelled Amharic news text from the Hub, same schema as the scrapers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        sources = [
            ("masakhane/masakhanews", "amh", ["headline", "text"]),
            ("rasyosef/amharic-news-category-classification", None, ["headline", "article"]),
        ]
        for name, cfg, fields in sources:
            try:
                if cfg:
                    url = f"hf://datasets/{name}@refs%2Fconvert%2Fparquet/{cfg}/train/*.parquet"
                    ds = load_dataset("parquet", data_files=url, split="train")
                else:
                    ds = load_dataset(name, split="train")
            except Exception as e:  # network / schema changes should not kill the pipeline
                print(f"[warn] could not load {name}: {e}")
                continue
            k = 0
            for ex in ds:
                text = "\n".join(str(ex[c]) for c in fields if c in ex and ex[c])
                if not text:
                    continue
                f.write(json.dumps({"source": name, "url": "", "title": "",
                                    "date": "", "text": text}, ensure_ascii=False) + "\n")
                n += 1
                k += 1
                if k >= max_docs:
                    break
            print(f"{name}: {k} docs")
    print(f"total {n} docs -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner-dir", type=Path, default=Path("data/raw/masakhaner"))
    ap.add_argument("--news-out", type=Path, default=Path("data/raw/hf_news.jsonl"))
    ap.add_argument("--max-news-docs", type=int, default=8000)
    a = ap.parse_args()
    download_masakhaner(a.ner_dir)
    download_news_text(a.news_out, a.max_news_docs)


if __name__ == "__main__":
    main()
