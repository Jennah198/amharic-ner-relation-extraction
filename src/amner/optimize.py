"""Inference optimization for CPU serving.

Compares the fine-tuned model in FP32 against PyTorch *dynamic INT8
quantization* of all ``nn.Linear`` layers (weights stored as int8,
activations quantized on the fly). Reports model size, latency (single
sentence, the interactive-UI case, and batched throughput) and test F1 so that
the speed/quality trade-off is explicit. Results -> ``reports/optimization.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import torch
from seqeval.metrics import f1_score

from .data import read_jsonl
from .evaluate import predict_all
from .inference import NERTagger


def size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def bench(tagger: NERTagger, sents, gold, n_single: int = 200) -> dict:
    for s in sents[:5]:  # warm-up
        tagger.tag_tokens(s)
    t0 = time.perf_counter()
    for s in sents[:n_single]:
        tagger.tag_tokens(s)
    single = (time.perf_counter() - t0) / n_single * 1000
    t0 = time.perf_counter()
    pred = predict_all(tagger, sents, bs=32)
    total = time.perf_counter() - t0
    return {"size_mb": round(size_mb(tagger.model), 1),
            "latency_ms_per_sentence_bs1": round(single, 2),
            "throughput_sent_per_s_bs32": round(len(sents) / total, 1),
            "test_f1": round(float(f1_score(gold, pred)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/ner/best")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = read_jsonl("data/processed/ner_test.jsonl")
    sents, gold = [ex["tokens"] for ex in data], [ex["ner_tags"] for ex in data]
    rep = {"model": a.model, "threads": a.threads, "test_sentences": len(sents)}
    rep["fp32"] = bench(NERTagger(a.model, device="cpu"), sents, gold)
    print("fp32", rep["fp32"])
    rep["int8_dynamic"] = bench(NERTagger(a.model, device="cpu", quantize=True), sents, gold)
    print("int8", rep["int8_dynamic"])
    rep["speedup_bs1"] = round(rep["fp32"]["latency_ms_per_sentence_bs1"] / rep["int8_dynamic"]["latency_ms_per_sentence_bs1"], 2)
    rep["size_reduction"] = round(rep["fp32"]["size_mb"] / rep["int8_dynamic"]["size_mb"], 2)
    rep["f1_delta"] = round(rep["int8_dynamic"]["test_f1"] - rep["fp32"]["test_f1"], 4)
    Path("reports/optimization.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
