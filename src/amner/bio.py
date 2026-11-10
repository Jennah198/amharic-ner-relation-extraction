"""BIO helpers shared by training, augmentation, inference and evaluation."""

from __future__ import annotations


def bio_to_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """Convert a BIO label sequence to (start, end_exclusive, type) spans.

    An I- tag that does not continue an entity of the same type starts a new
    span (the lenient "IOB1-style" repair used by seqeval's default mode).
    """
    spans, start, typ = [], None, None
    for i, lab in enumerate(list(labels) + ["O"]):
        if lab == "O" or lab.startswith("B-") or (lab.startswith("I-") and lab[2:] != typ):
            if start is not None:
                spans.append((start, i, typ))
                start = typ = None
            if lab != "O":
                start, typ = i, lab[2:]
    return spans
