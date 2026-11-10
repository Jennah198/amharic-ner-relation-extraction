"""Build the word-frequency lexicon that constrains clitic stripping.

The lexicon (``src/amner/resources/amharic_lexicon.tsv``, word<TAB>count,
count >= 3) is computed from the normalized unlabelled corpus plus the
MasakhaNER *training* tokens. It is small, versioned with the code and shipped
with the web app so that the analyzer behaves identically everywhere.
"""

from __future__ import annotations

import json
from collections import Counter

from .morphology import LEXICON_PATH, MorphAnalyzer


def main():
    freq = Counter()
    for line in open("data/processed/corpus_train.txt", encoding="utf-8"):
        freq.update(line.split())
    for line in open("data/processed/ner_train.jsonl", encoding="utf-8"):
        freq.update(json.loads(line)["tokens"])
    n = MorphAnalyzer(freq=freq).save_lexicon(LEXICON_PATH, min_freq=3)
    print(f"lexicon: {n} word types (freq >= 3) -> {LEXICON_PATH}")


if __name__ == "__main__":
    main()
