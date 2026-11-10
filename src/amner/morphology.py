"""Light-weight Amharic clitic/affix analysis for NER.

Amharic is a Semitic language with root-and-pattern (templatic) morphology plus
concatenative clitics. For NER the most damaging phenomenon is that function
words are *glued* onto named entities:

    በአዲስ አበባ    = በ (in)  + አዲስ አበባ           "in Addis Ababa"
    የኢትዮጵያ       = የ (of)  + ኢትዮጵያ             "of Ethiopia"
    ለጠቅላይ        = ለ (to/for) + ጠቅላይ
    ኢትዮጵያን       = ኢትዮጵያ + ን (accusative)
    አሜሪካም        = አሜሪካ + ም ("also")
    ከተማዋ          = ከተማ + ዋ (definite, fem.)

Because whitespace tokenization keeps these as single tokens, the same entity
appears in many surface forms, each rarer than the bare stem. A full
root-and-pattern analyzer (e.g. HornMorpho) is heavy; for NER we only need to
peel the *outer* clitics, which are written as separate Fidel syllables in the
vast majority of cases, so string-level stripping works well.

To avoid destroying names that genuinely begin with ``በ/ለ/የ/ከ`` (e.g. በቀለ
"Bekele", ለማ "Lemma", የሺ "Yeshi") stripping can be constrained by a lexicon:
a prefix is only removed if the remaining stem is a known word.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .bio import bio_to_spans

# Longest first, so "እንደ" wins over nothing and "ከነ" over "ከ".
PREFIXES = ["እንደ", "እስከ", "ወደ", "ስለ", "ከነ", "በነ", "ለነ", "የነ", "ከ", "በ", "ለ", "የ"]
# Suffix clitics (outer layer). Order matters: compound ones first.
SUFFIXES = ["ንም", "ውን", "ዋን", "ቱን", "ኛው", "ም", "ን", "ና", "ው", "ዋ"]
MIN_STEM = 2  # in Fidel characters (syllables)


@dataclass
class Analysis:
    surface: str
    prefix: str = ""
    stem: str = ""
    suffix: str = ""

    @property
    def has_affix(self) -> bool:
        return bool(self.prefix or self.suffix)

    def as_dict(self) -> dict:
        return {"surface": self.surface, "prefix": self.prefix,
                "stem": self.stem, "suffix": self.suffix}


LEXICON_PATH = Path(__file__).resolve().parent / "resources" / "amharic_lexicon.tsv"


@dataclass
class MorphAnalyzer:
    """Clitic stripper.

    * Without a lexicon it is greedy (longest prefix, then longest suffix,
      keeping >= ``MIN_STEM`` syllables) – high recall, but over-strips names
      that happen to end in ን/ና/ም (ሱዳን -> ሱዳ, ኮሚሽን -> ኮሚሽ).
    * With a frequency lexicon (built from our news corpus) an analysis is only
      accepted if its stem is a known word that is at least as frequent as the
      full surface form; among valid analyses the most frequent stem wins.
      ኢትዮጵያንም -> ኢትዮጵያ (+ንም), but ሱዳን stays ሱዳን because "ሱዳ" is not a word.
    """

    freq: Counter = field(default_factory=Counter)
    min_freq: int = 3

    @property
    def use_lexicon(self) -> bool:
        return bool(self.freq)

    @classmethod
    def from_corpus(cls, tokens, min_freq: int = 3) -> "MorphAnalyzer":
        return cls(freq=Counter(tokens), min_freq=min_freq)

    @classmethod
    def load(cls, path: Path = LEXICON_PATH) -> "MorphAnalyzer":
        if not path.exists():
            return cls()
        freq = Counter()
        for line in path.open(encoding="utf-8"):
            w, _, c = line.rstrip("\n").partition("\t")
            if c:
                freq[w] = int(c)
        return cls(freq=freq)

    def save_lexicon(self, path: Path = LEXICON_PATH, min_freq: int = 3) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [(w, c) for w, c in self.freq.most_common() if c >= min_freq]
        path.write_text("".join(f"{w}\t{c}\n" for w, c in rows), encoding="utf-8")
        return len(rows)

    def _candidates(self, word: str):
        for p in [""] + PREFIXES:
            if p and not word.startswith(p):
                continue
            rest = word[len(p):]
            for s in [""] + SUFFIXES:
                if s and not rest.endswith(s):
                    continue
                stem = rest[: len(rest) - len(s)]
                if len(stem) >= MIN_STEM and (p or s):
                    yield p, stem, s

    def analyze(self, word: str) -> Analysis:
        if not self.use_lexicon:
            prefix = next((p for p in PREFIXES if word.startswith(p) and len(word) - len(p) >= MIN_STEM), "")
            rest = word[len(prefix):]
            suffix = next((s for s in SUFFIXES if rest.endswith(s) and len(rest) - len(s) >= MIN_STEM), "")
            return Analysis(word, prefix, rest[: len(rest) - len(suffix)], suffix)
        fw = self.freq.get(word, 0)
        best, best_key = None, None
        for p, stem, s in self._candidates(word):
            fs = self.freq.get(stem, 0)
            if fs < self.min_freq or fs < fw:
                continue
            key = (fs, len(p) + len(s))
            if best_key is None or key > best_key:
                best, best_key = (p, stem, s), key
        if best is None:
            return Analysis(word, "", word, "")
        return Analysis(word, *best)

    def strip_prefix(self, word: str) -> tuple[str, str]:
        a = self.analyze(word)
        return a.prefix, a.stem + a.suffix

    def strip_suffix(self, word: str) -> tuple[str, str]:
        a = self.analyze(word)
        return a.prefix + a.stem, a.suffix

    def lemma_span(self, tokens: list[str]) -> str:
        """Canonical form of a (multi-token) entity: strip the prefix of the
        first token and the suffix of the last token only (inner tokens of a
        name are left untouched)."""
        if not tokens:
            return ""
        toks = list(tokens)
        if len(toks) == 1:
            return self.analyze(toks[0]).stem
        toks[0] = self.strip_prefix(toks[0])[1]
        toks[-1] = self.strip_suffix(toks[-1])[0]
        return " ".join(toks)


DEFAULT_ANALYZER = MorphAnalyzer.load()


def affix_bucket(token: str, analyzer: MorphAnalyzer = DEFAULT_ANALYZER) -> str:
    """Coarse morphological category used in error analysis."""
    a = analyzer.analyze(token)
    if a.prefix and a.suffix:
        return "prefix+suffix"
    if a.prefix:
        return "prefix"
    if a.suffix:
        return "suffix"
    return "bare"


# ---------------------------------------------------------------------------
# Morphology-aware data augmentation
# ---------------------------------------------------------------------------
AUG_PREFIXES = ["በ", "ለ", "የ", "ከ", "ወደ"]
AUG_SUFFIXES = ["ን", "ም", "ው"]




def morph_augment(tokens: list[str], labels: list[str], rng: random.Random,
                  analyzer: MorphAnalyzer = DEFAULT_ANALYZER,
                  p_prefix: float = 0.5, p_suffix: float = 0.3) -> tuple[list[str], list[str]] | None:
    """Create a new training sentence in which entity mentions carry a
    *different* clitic than in the original (e.g. ኢትዮጵያ -> በኢትዮጵያ,
    የአሜሪካ -> አሜሪካን). Labels are unchanged because MasakhaNER annotates the
    whole whitespace token. Returns None if nothing was changed."""
    toks = list(tokens)
    changed = False
    for s, e, typ in bio_to_spans(list(labels)):
        if typ == "DATE":
            continue
        if rng.random() < p_prefix:
            prefix, stem = analyzer.strip_prefix(toks[s])
            choices = [p for p in AUG_PREFIXES if p != prefix] + [""]
            new = rng.choice(choices) + stem
            if new != toks[s]:
                toks[s], changed = new, True
        if rng.random() < p_suffix:
            stem, suffix = analyzer.strip_suffix(toks[e - 1])
            choices = [x for x in AUG_SUFFIXES if x != suffix] + [""]
            new = stem + rng.choice(choices)
            if new != toks[e - 1]:
                toks[e - 1], changed = new, True
    return (toks, list(labels)) if changed else None
