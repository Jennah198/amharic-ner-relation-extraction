"""Explainability helpers for the demo: what normalization changed, and how
different tokenizers split a word compared with the clitic analysis.

Used by the FastAPI endpoints ``/normalize`` and ``/tokenize`` and by the
"Normalization" and "Tokenizer" tabs of the Gradio app.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .morphology import DEFAULT_ANALYZER
from .normalize import (EMAIL_RE, EMOJI_RE, LATIN_WORD_RE, MENTION_RE, URL_RE, clean_text, normalize,
                        normalize_chars)

_PUNCT_CHANGES = {"፡": "፡ (separator)→space/።", "፥": "፥→፣", "“": "“→\"", "”": "”→\"", "«": "«→\"", "»": "»→\""}

TOKENIZER_DIR = Path(os.environ.get("AMNER_TOKENIZER_DIR", "artifacts/tokenizers"))
BACKBONE = os.environ.get("AMNER_BACKBONE", "rasyosef/roberta-medium-amharic")


# --------------------------------------------------------------------------
# Normalization diff
# --------------------------------------------------------------------------
def normalization_diff(text: str) -> dict:
    """Raw text segmented into unchanged runs and changed characters.

    ``segments`` is a list of (text, label) where label is None for unchanged
    text or "ሐ→ሀ" style for a re-spelled character, ready for
    gr.HighlightedText. Noise that the cleaner removes (URLs, mentions,
    emojis) is labelled "removed".
    """
    removed_spans = []
    for rx in (URL_RE, EMAIL_RE, MENTION_RE, EMOJI_RE):
        removed_spans += [(m.start(), m.end()) for m in rx.finditer(text)]
    removed = [False] * len(text)
    for s, e in removed_spans:
        for i in range(s, e):
            removed[i] = True

    segments: list[tuple[str, str | None]] = []
    changes = Counter()
    buf = ""
    for i, ch in enumerate(text):
        if removed[i]:
            label = "removed"
        else:
            n = normalize_chars(ch)
            label = f"{ch}→{n}" if n != ch else _PUNCT_CHANGES.get(ch)
            if label:
                changes[label] += 1
        if label is None:
            buf += ch
            continue
        if buf:
            segments.append((buf, None))
            buf = ""
        if segments and segments[-1][1] == label == "removed":
            segments[-1] = (segments[-1][0] + ch, label)
        else:
            segments.append((ch, label))
    if buf:
        segments.append((buf, None))

    normalized = normalize(text)
    raw_words, norm_words = text.split(), normalized.split()
    return {
        "raw": text,
        "normalized": normalized,
        "segments": segments,
        "char_changes": [{"change": k, "count": v} for k, v in changes.most_common()],
        "removed_noise": [text[s:e] for s, e in sorted(removed_spans)],
        "latin_words": LATIN_WORD_RE.findall(clean_text(text)),
        "stats": {"raw_chars": len(text), "normalized_chars": len(normalized),
                  "chars_respelled": sum(changes.values()),
                  "raw_words": len(raw_words), "normalized_tokens": len(norm_words)},
    }


# --------------------------------------------------------------------------
# Tokenizer comparison
# --------------------------------------------------------------------------
def _hub_tokenizer(name: str):
    """Use the local Hugging Face cache first; only go online if it is missing."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(name)


@lru_cache(maxsize=1)
def load_tokenizers() -> dict:
    """Load every tokenizer that is available locally; missing ones are skipped
    (e.g. on a Space where only some files were uploaded)."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    toks = {}
    candidates = [
        ("Backbone (Amharic SentencePiece, 32k)", lambda: _hub_tokenizer(BACKBONE)),
        ("Backbone + 57 added words", lambda: AutoTokenizer.from_pretrained(TOKENIZER_DIR / "backbone_extended")),
        ("Custom BPE (16k)", lambda: Tokenizer.from_file(str(TOKENIZER_DIR / "amharic_bpe.json"))),
        ("Custom WordPiece (16k)", lambda: Tokenizer.from_file(str(TOKENIZER_DIR / "amharic_wordpiece.json"))),
        ("XLM-R (multilingual, 250k)", lambda: _hub_tokenizer("xlm-roberta-base")),
    ]
    for name, load in candidates:
        try:
            toks[name] = load()
        except Exception as e:  # offline / file not shipped
            print(f"[tokenizer tab] skipping {name}: {type(e).__name__}")
    return toks


def _split(tok, text: str) -> list[str]:
    from tokenizers import Tokenizer
    if isinstance(tok, Tokenizer):
        return tok.encode(text, add_special_tokens=False).tokens
    return tok.tokenize(text)


def tokenizer_comparison(text: str) -> dict:
    """Split ``text`` (a word or a short phrase) with every tokenizer and show
    the clitic analysis of each word next to it."""
    norm = normalize(text)
    words = norm.split()
    rows = []
    for name, tok in load_tokenizers().items():
        pieces = _split(tok, norm)
        rows.append({"tokenizer": name, "pieces": pieces, "n_pieces": len(pieces),
                     "fertility": round(len(pieces) / max(len(words), 1), 2)})
    morph = []
    for w in words:
        a = DEFAULT_ANALYZER.analyze(w)
        morph.append({"word": w, "prefix": a.prefix, "stem": a.stem, "suffix": a.suffix,
                      "segmentation": " + ".join(x for x in (a.prefix, a.stem, a.suffix) if x)})
    return {"input": text, "normalized": norm, "tokenizers": rows, "morphology": morph}
