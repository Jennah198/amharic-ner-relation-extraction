"""Presentation rehearsal: exercises every API endpoint and every UI tab of a
running demo (start it with scripts/launch_demo.sh) and prints PASS/FAIL.

    python scripts/rehearse_demo.py
"""

from __future__ import annotations

import sys
import time

import requests
from gradio_client import Client

API, UI = "http://127.0.0.1:8000", "http://127.0.0.1:7860/"
SENT = "የኢትዮጵያ ጠቅላይ ሚኒስትር ዐቢይ አሕመድ ከኬንያው ፕሬዚዳንት ዊልያም ሩቶ ጋር በናይሮቢ ተወያዩ።"
results = []


def check(name, fn):
    t = time.perf_counter()
    try:
        detail = fn()
        results.append(True)
        print(f"PASS  {name:<38} {1000 * (time.perf_counter() - t):7.0f} ms  {detail}")
    except Exception as e:
        results.append(False)
        print(f"FAIL  {name:<38} {type(e).__name__}: {e}")


def api_health():
    r = requests.get(f"{API}/health", timeout=10).json()
    assert r["status"] == "ok"
    return r["model"]


def api_analyze():
    r = requests.post(f"{API}/analyze", json={"text": SENT}, timeout=60).json()
    rels = {(x["head"], x["relation"], x["tail"]) for x in r["relations"]}
    assert ("አቢይ አህመድ", "MET_WITH", "ዊልያም ሩቶ") in rels, rels
    return f"{len(r['entities'])} entities, {len(r['relations'])} relations"


def api_normalize():
    r = requests.post(f"{API}/normalize", json={"text": "ዐቢይ አሕመድ ሥራ ፀሐይ"}, timeout=30).json()
    assert r["normalized"] == "አቢይ አህመድ ስራ ጸሀይ", r["normalized"]
    return f"{r['stats']['chars_respelled']} chars re-spelled"


def api_tokenize():
    r = requests.post(f"{API}/tokenize", json={"text": "ለኢትዮጵያንም"}, timeout=120).json()
    assert r["morphology"][0]["segmentation"] == "ለ + ኢትዮጵያ + ንም"
    return f"{len(r['tokenizers'])} tokenizers compared"


client = None


def ui_connect():
    global client
    client = Client(UI, verbose=False)
    return "connected"


def ui_entities():
    hl, ents, rels, morph, info = client.predict(SENT, api_name="/run")
    assert len(ents["data"]) >= 4
    return f"{len(ents['data'])} entities, {len(rels['data'])} relations"


def ui_normalization():
    segs, norm, changes, summary = client.predict("ዐቢይ አሕመድ ኃይል", api_name="/run_normalize")
    assert norm == "አቢይ አህመድ ሀይል"
    return summary.splitlines()[0]


def ui_tokenizer():
    rows, morph, note = client.predict("ከኬንያው", api_name="/run_tokenize")
    return f"{len(rows['data'])} tokenizers; clitics: {morph['data'][0][4]}"


for name, fn in [("API /health", api_health), ("API /analyze (NER + relations)", api_analyze),
                 ("API /normalize", api_normalize), ("API /tokenize", api_tokenize),
                 ("UI connect", ui_connect), ("UI Entities/Relations tabs", ui_entities),
                 ("UI Normalization tab", ui_normalization), ("UI Tokenizer tab", ui_tokenizer)]:
    check(name, fn)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
