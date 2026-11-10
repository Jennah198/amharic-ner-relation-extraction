"""One entry point used by the FastAPI backend, the Gradio UI and the HF Space."""

from __future__ import annotations

import os
import time
from functools import lru_cache

from .spacy_pipeline import build_pipeline

DEFAULT_MODEL = os.environ.get("AMNER_MODEL_DIR", "artifacts/ner/best")


@lru_cache(maxsize=2)
def get_pipeline(model_dir: str = DEFAULT_MODEL, quantize: bool = True):
    # int8 dynamic quantization is on by default for serving (see reports/optimization.json)
    return build_pipeline(model_dir, quantize=quantize)


def analyze(text: str, model_dir: str = DEFAULT_MODEL, quantize: bool = True) -> dict:
    nlp = get_pipeline(model_dir, quantize)
    t0 = time.perf_counter()
    doc = nlp(text)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "input": text,
        "normalized_text": doc.text,
        "entities": [{"text": e.text, "label": e.label_, "lemma": e._.lemma_am, "score": e._.score,
                      "start_char": e.start_char, "end_char": e.end_char} for e in doc.ents],
        "relations": doc._.relations,
        "tokens": [{"text": t.text, "prefix": t._.prefix, "stem": t._.stem, "suffix": t._.suffix,
                    "bio": (t.ent_iob_ + "-" + t.ent_type_) if t.ent_type_ else "O"} for t in doc],
        "latency_ms": round(ms, 1),
    }
