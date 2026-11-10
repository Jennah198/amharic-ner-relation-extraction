"""Inference: from raw Amharic text to entity spans."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .bio import bio_to_spans
from .morphology import DEFAULT_ANALYZER
from .normalize import normalize

TOKEN_RE = re.compile(r"[^\s።፣፤፦፧!?\"'()\[\],.:;]+|[።፣፤፦፧!?\"'()\[\],.:;]")


@dataclass
class Entity:
    text: str          # surface form as written (with clitics)
    label: str         # PER / ORG / LOC / DATE
    start: int         # char offset in the *normalized* text
    end: int
    token_start: int
    token_end: int     # exclusive
    lemma: str         # clitic-stripped canonical form (morphology-aware)
    score: float

    def as_dict(self):
        return asdict(self)


def word_tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]




class NERTagger:
    def __init__(self, model_dir: str, device: str | None = None, quantize: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        if quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def tag_tokens_batch(self, sents: list[list[str]], max_length: int = 256) -> list[tuple[list[str], list[float]]]:
        enc = self.tokenizer(sents, is_split_into_words=True, truncation=True,
                             max_length=max_length, padding=True, return_tensors="pt")
        logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        probs = logits.softmax(-1).cpu()
        out = []
        for i, toks in enumerate(sents):
            labels, scores = ["O"] * len(toks), [1.0] * len(toks)
            prev = None
            for j, wid in enumerate(enc.word_ids(batch_index=i)):
                if wid is not None and wid != prev:
                    p, k = probs[i, j].max(-1)
                    labels[wid], scores[wid] = self.id2label[int(k)], float(p)
                prev = wid
            out.append((labels, scores))
        return out

    def tag_tokens(self, tokens: list[str]) -> list[str]:
        return self.tag_tokens_batch([tokens])[0][0]

    def __call__(self, text: str, already_normalized: bool = False) -> dict:
        norm = text if already_normalized else normalize(text)
        toks = word_tokenize(norm)
        if not toks:
            return {"normalized_text": norm, "tokens": [], "labels": [], "entities": []}
        words = [t for t, _, _ in toks]
        labels, scores = self.tag_tokens_batch([words])[0]
        ents = []
        for s, e, typ in bio_to_spans(labels):
            ents.append(Entity(
                text=norm[toks[s][1]:toks[e - 1][2]], label=typ,
                start=toks[s][1], end=toks[e - 1][2], token_start=s, token_end=e,
                lemma=DEFAULT_ANALYZER.lemma_span(words[s:e]),
                score=round(sum(scores[s:e]) / (e - s), 4)))
        return {"normalized_text": norm, "tokens": words, "labels": labels,
                "entities": [x.as_dict() for x in ents]}
