"""Amharic text normalization and cleaning.

Amharic is written in the Ge'ez (Fidel) abugida. Several letters that were once
distinct phonemes are now pronounced identically, so the same word is spelled
in several ways ("ሀይል" / "ኃይል" / "ሐይል"). For NLP this splits one word into
many surface forms and fragments both vocabularies and statistics.

This module provides:

* ``normalize_chars``   – a pure character-level mapping (length-preserving per
  token, never adds or removes whitespace). Safe to apply to pre-tokenized,
  BIO-labelled data without breaking label alignment.
* ``clean_text``        – document-level noise stripping for scraped text
  (HTML remnants, URLs, emojis, mentions, repeated punctuation ...).
* ``handle_code_switching`` – policy for Latin-script tokens embedded in
  Amharic text (keep / tag / drop).
* ``split_sentences``   – sentence segmentation on Ethiopic punctuation.
* ``normalize``         – full pipeline used for raw corpora and at inference.
"""

from __future__ import annotations

import html
import re
import unicodedata

# ---------------------------------------------------------------------------
# 1. Homophone normalization tables
# ---------------------------------------------------------------------------
# Each Fidel family has 7 (sometimes 8) vowel orders laid out contiguously in
# Unicode. We map an entire family onto a canonical family order by order.
# Canonical choice follows common Amharic IR practice:
#   ሐ, ኀ  -> ሀ      (h)
#   ሠ      -> ሰ      (s)
#   ዐ      -> አ      (glottal)
#   ፀ      -> ጸ      (ts')
_FAMILY_MAP = {
    "ሐ": "ሀ",
    "ኀ": "ሀ",
    "ሠ": "ሰ",
    "ዐ": "አ",
    "ፀ": "ጸ",
}

_CHAR_MAP: dict[str, str] = {}
for src, dst in _FAMILY_MAP.items():
    s, d = ord(src), ord(dst)
    for order in range(7):  # orders 1..7 (ä, u, i, a, e, ə, o)
        _CHAR_MAP[chr(s + order)] = chr(d + order)

# Fourth-order (-a) of the h and glottal families is pronounced the same as the
# first order, writers use them interchangeably: ሃ == ሀ, ኣ == አ.
_CHAR_MAP.update({
    "ሃ": "ሀ", "ሓ": "ሀ", "ኃ": "ሀ",
    "ኣ": "አ", "ዓ": "አ",
})

# Rare / archaic labialized spellings -> their common modern forms
_CHAR_MAP.update({
    "ቍ": "ቁ", "ኵ": "ኩ", "ጕ": "ጉ", "ዃ": "ኳ",
})

# Two-character sequences that writers use instead of a labialized letter.
# e.g. "ሉዋ" -> "ሏ", "ቱዋ" -> "ቷ". Applied with regex on the 2nd-order + ዋ.
_LABIAL_BASES = "ለመረሰሸቀበተቸነኘከዘዠደጀገጠጨጰፈ"


def _build_labial_map() -> dict[str, str]:
    m = {}
    for base in _LABIAL_BASES:
        b = ord(base)
        second = chr(b + 1)       # -u order
        labial = chr(b + 7)       # -wa order (8th slot in the block)
        if unicodedata.name(labial, "").endswith("WA"):
            m[second + "ዋ"] = labial
    return m


_LABIAL_MAP = _build_labial_map()
_LABIAL_RE = re.compile("|".join(map(re.escape, _LABIAL_MAP))) if _LABIAL_MAP else None

# Ethiopic digits ፩..፱, ፲.. -> Arabic numerals for the simple 1-9 case.
_ETH_DIGITS = {chr(0x1369 + i): str(i + 1) for i in range(9)}

# Punctuation normalization
_PUNCT_MAP = {
    "፡፡": "።",   # two word-separators typed instead of full stop
    "::": "።",
    "፡": " ",    # traditional word separator -> space
    "፥": "፣",    # colon-like comma variant
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-",
}

ETHIOPIC_RE = re.compile(r"[ሀ-፿ᎀ-᎟ⶀ-⷟꬀-꬯]")
LATIN_WORD_RE = re.compile(r"(?<![A-Za-z])[A-Za-z][A-Za-z'\-]*(?![A-Za-z])")  # no \b: Fidel letters are \w too
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MENTION_RE = re.compile(r"(?<!\w)[@#][\wሀ-፿]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF️‍]+"
)
MULTISPACE_RE = re.compile(r"[ \t ​]+")
REPEAT_PUNCT_RE = re.compile(r"([።፣፤!?.,])\1+")
SENT_SPLIT_RE = re.compile(r"(?<=[።!?፧])\s+")


def normalize_chars(text: str) -> str:
    """Character-level homophone + labial + digit normalization.

    Guarantees: does not insert or delete whitespace *between* tokens other
    than collapsing two-char labial sequences inside a token, so it is safe on
    a single pre-tokenized token.
    """
    text = unicodedata.normalize("NFC", text)
    if _LABIAL_RE is not None:
        text = _LABIAL_RE.sub(lambda m: _LABIAL_MAP[m.group(0)], text)
    return "".join(_CHAR_MAP.get(ch, _ETH_DIGITS.get(ch, ch)) for ch in text)


def normalize_token(token: str) -> str:
    """Normalize a single token from a labelled dataset (never returns '')."""
    out = normalize_chars(token)
    for k, v in (("፡፡", "።"), ("::", "።")):
        out = out.replace(k, v)
    return out or token


def normalize_punct(text: str) -> str:
    for k, v in _PUNCT_MAP.items():
        text = text.replace(k, v)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    # Put a space before/after Ethiopic sentence & clause punctuation so that a
    # whitespace tokenizer separates it from words.
    text = re.sub(r"\s*([።፣፤፦፧])\s*", r" \1 ", text)
    return text


def clean_text(text: str) -> str:
    """Strip markup and social-media noise from scraped text."""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return text


def ethiopic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(ETHIOPIC_RE.match(c)) for c in letters) / len(letters)


def handle_code_switching(text: str, mode: str = "keep") -> str:
    """Policy for Latin-script words inside Amharic text.

    keep : leave them (names like "WHO", "IMF" are often entities!)
    tag  : wrap as <lat>word</lat> for downstream inspection
    drop : remove Latin words entirely
    """
    if mode == "keep":
        return text
    if mode == "tag":
        return LATIN_WORD_RE.sub(lambda m: f"<lat>{m.group(0)}</lat>", text)
    if mode == "drop":
        return LATIN_WORD_RE.sub(" ", text)
    raise ValueError(f"unknown code-switch mode: {mode}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def normalize(text: str, code_switch: str = "keep") -> str:
    """Full normalization used for raw corpora *and* at inference time."""
    text = clean_text(text)
    text = normalize_chars(text)
    text = normalize_punct(text)
    text = handle_code_switching(text, code_switch)
    text = MULTISPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def is_good_sentence(sent: str, min_tokens: int = 4, min_ethiopic: float = 0.6) -> bool:
    """Corpus filter: long enough and predominantly Amharic script."""
    return len(sent.split()) >= min_tokens and ethiopic_ratio(sent) >= min_ethiopic
