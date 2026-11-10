"""SpaCy custom pipeline wrapping the fine-tuned transformer.

    nlp = build_pipeline("artifacts/ner/best")
    doc = nlp("የኢትዮጵያ ጠቅላይ ሚኒስትር ዐቢይ አሕመድ ከኬንያው ፕሬዚዳንት ጋር ተወያዩ።")
    doc.ents                 -> (ኢትዮጵያ/LOC, አቢይ አሕመድ/PER, ...)
    [t._.stem for t in doc]  -> clitic-stripped stems
    doc._.relations          -> [{"head": ..., "relation": "LEADER_OF", ...}]

Pipeline layout (``nlp.pipe_names``)::

    tokenizer            AmharicTokenizer   – Fidel normalization + Ethiopic punctuation split
    amharic_morphology   token._.prefix / ._.stem / ._.suffix
    amharic_ner          transformer BIO tagger -> doc.ents (+ span._.lemma, span._.score)
    amharic_relations    rule-based political relations -> doc._.relations

Because a spaCy ``Doc``'s text is immutable, normalization happens inside the
tokenizer: ``doc.text`` is the *normalized* text and ``doc.user_data["raw"]``
keeps what the user typed.
"""

# NOTE: no "from __future__ import annotations" here – spaCy validates factory
# signatures at runtime and cannot resolve string annotations.

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from .bio import bio_to_spans
from .inference import NERTagger, word_tokenize
from .morphology import DEFAULT_ANALYZER
from .normalize import normalize
from .relations import RelationExtractor

for ext, default in [("prefix", ""), ("stem", ""), ("suffix", "")]:
    if not Token.has_extension(ext):
        Token.set_extension(ext, default=default)
if not Span.has_extension("lemma_am"):
    Span.set_extension("lemma_am", default="")
if not Span.has_extension("score"):
    Span.set_extension("score", default=0.0)
if not Doc.has_extension("relations"):
    Doc.set_extension("relations", default=[])


class AmharicTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        norm = normalize(text)
        toks = word_tokenize(norm)
        words = [t for t, _, _ in toks]
        spaces = [toks[i][2] < toks[i + 1][1] for i in range(len(toks) - 1)] + [False] if toks else []
        doc = Doc(self.vocab, words=words, spaces=spaces)
        doc.user_data["raw"] = text
        return doc


@Language.component("amharic_morphology")
def amharic_morphology(doc: Doc) -> Doc:
    for t in doc:
        a = DEFAULT_ANALYZER.analyze(t.text)
        t._.prefix, t._.stem, t._.suffix = a.prefix, a.stem, a.suffix
    return doc


@Language.factory("amharic_ner", default_config={"model_dir": "artifacts/ner/best", "quantize": False})
def create_ner(nlp: Language, name: str, model_dir: str, quantize: bool):
    tagger = NERTagger(model_dir, quantize=quantize)

    def component(doc: Doc) -> Doc:
        if not len(doc):
            return doc
        labels, scores = tagger.tag_tokens_batch([[t.text for t in doc]])[0]
        spans = []
        for s, e, typ in bio_to_spans(labels):
            sp = Span(doc, s, e, label=typ)
            sp._.lemma_am = DEFAULT_ANALYZER.lemma_span([t.text for t in doc[s:e]])
            sp._.score = round(sum(scores[s:e]) / (e - s), 4)
            spans.append(sp)
        doc.ents = spans
        return doc

    return component


@Language.factory("amharic_relations")
def create_relations(nlp: Language, name: str):
    rex = RelationExtractor()

    def component(doc: Doc) -> Doc:
        ents = [{"token_start": e.start, "token_end": e.end, "label": e.label_} for e in doc.ents]
        doc._.relations = [r.as_dict() for r in rex.extract([t.text for t in doc], ents)]
        return doc

    return component


def build_pipeline(model_dir: str = "artifacts/ner/best", quantize: bool = False) -> Language:
    nlp = spacy.blank("am")  # spaCy ships Amharic language data (stop words, lex attrs)
    nlp.tokenizer = AmharicTokenizer(nlp.vocab)
    nlp.add_pipe("amharic_morphology")
    nlp.add_pipe("amharic_ner", config={"model_dir": model_dir, "quantize": quantize})
    nlp.add_pipe("amharic_relations")
    return nlp
