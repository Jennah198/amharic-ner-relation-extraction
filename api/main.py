"""FastAPI backend.

Run:  uvicorn api.main:app --port 8000
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from amner import explain, service

app = FastAPI(title="Amharic NER & Relation Extraction API", version="1.0.0",
              description="Morphology-aware Amharic named entities (PER, ORG, LOC, DATE) "
                          "and political relations from news text.")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      examples=["የኢትዮጵያ ጠቅላይ ሚኒስትር ዐቢይ አሕመድ ከኬንያው ፕሬዚዳንት ዊልያም ሩቶ ጋር በናይሮቢ ተወያዩ።"])


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


@app.on_event("startup")
def warmup() -> None:
    service.get_pipeline(service.DEFAULT_MODEL)


@app.get("/health")
def health():
    return {"status": "ok", "model": service.DEFAULT_MODEL}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        return service.analyze(req.text)
    except Exception as e:  # surface model errors as 500 with a message
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    return [service.analyze(t) for t in req.texts]


@app.post("/normalize")
def normalize_text(req: AnalyzeRequest):
    """Raw vs. normalized text with every re-spelled character and removed noise."""
    return explain.normalization_diff(req.text)


@app.post("/tokenize")
def tokenize_text(req: AnalyzeRequest):
    """Subword splits from every available tokenizer + clitic analysis per word."""
    return explain.tokenizer_comparison(req.text)
