"""Gradio web UI.

Two modes:
* ``AMNER_API_URL`` set  -> thin client of the FastAPI backend (deployment mode)
* otherwise              -> runs the spaCy pipeline in-process (HF Spaces mode)

Run:  python app/gradio_app.py      (http://127.0.0.1:7860)
"""

from __future__ import annotations

import os

import gradio as gr
import requests

API_URL = os.environ.get("AMNER_API_URL", "").rstrip("/")

EXAMPLES = [
    "የኢትዮጵያ ጠቅላይ ሚኒስትር ዐቢይ አሕመድ ከኬንያው ፕሬዚዳንት ዊልያም ሩቶ ጋር በናይሮቢ ተወያዩ።",
    "የአፍሪካ ሕብረት ኮሚሽን ሊቀመንበር ሙሳ ፋኪ መሐማት ጥቅምት 12 ቀን በአዲስ አበባ መግለጫ ሰጥተዋል።",
    "ኢትዮጵያ እና ሱዳን በድንበር ጉዳይ ላይ ስምምነት ተፈራረሙ።",
    "የብሔራዊ ምርጫ ቦርድ በአዲስ አበባ የሚገኘው ዋና መሥሪያ ቤቱ ውስጥ ውይይት አካሄደ።",
    "ሰኞ ዕለት የአማራ ክልል ርዕሰ መስተዳድር በባሕር ዳር ከተማ ለጋዜጠኞች መግለጫ ሰጥተዋል።",
]

COLORS = {"PER": "#f4a261", "ORG": "#2a9d8f", "LOC": "#457b9d", "DATE": "#e9c46a"}


def _analyze(text: str) -> dict:
    if API_URL:
        r = requests.post(f"{API_URL}/analyze", json={"text": text}, timeout=60)
        r.raise_for_status()
        return r.json()
    from amner import service
    return service.analyze(text)


def run(text: str):
    if not text.strip():
        return [], [], [], {}, ""
    res = _analyze(text)
    norm = res["normalized_text"]
    highlighted, pos = [], 0
    for e in res["entities"]:
        if e["start_char"] > pos:
            highlighted.append((norm[pos:e["start_char"]], None))
        highlighted.append((norm[e["start_char"]:e["end_char"]], e["label"]))
        pos = e["end_char"]
    if pos < len(norm):
        highlighted.append((norm[pos:], None))
    ents = [[e["text"], e["label"], e["lemma"], e["score"]] for e in res["entities"]]
    rels = [[r["head"], r["relation"], r["tail"], r["trigger"]] for r in res["relations"]]
    morph = [[t["text"], t["prefix"], t["stem"], t["suffix"], t["bio"]] for t in res["tokens"]]
    info = f"Normalized input: {norm}\n\nLatency: {res['latency_ms']} ms" + (f"  ·  backend: {API_URL}" if API_URL else "  ·  in-process")
    return highlighted, ents, rels, morph, info


with gr.Blocks(title="Amharic NER & Relation Extraction") as demo:
    gr.Markdown("# የአማርኛ ስያሜ ለይቶ ማውጫ · Amharic NER & Relation Extraction\n"
                "Morphology-aware extraction of **persons, organizations, locations and dates** and of "
                "**political relations** from Amharic news. Text is homophone-normalized "
                "(ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ) and clitics (የ-, በ-, ከ-, -ን, -ም …) are analysed per token.")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="Amharic text", lines=5, placeholder="የአማርኛ የዜና ጽሑፍ እዚህ ያስገቡ…")
            btn = gr.Button("Analyze", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column(scale=4):
            out_hl = gr.HighlightedText(label="Entities", color_map=COLORS, combine_adjacent=False)
            info = gr.Textbox(label="Details", lines=3)
    with gr.Tab("Entities"):
        out_ents = gr.Dataframe(headers=["entity (surface)", "type", "lemma (clitics stripped)", "confidence"])
    with gr.Tab("Relations"):
        out_rels = gr.Dataframe(headers=["head", "relation", "tail", "trigger"])
    with gr.Tab("Morphology"):
        out_morph = gr.Dataframe(headers=["token", "prefix", "stem", "suffix", "BIO"])
    btn.click(run, inputs=inp, outputs=[out_hl, out_ents, out_rels, out_morph, info])
    inp.submit(run, inputs=inp, outputs=[out_hl, out_ents, out_rels, out_morph, info])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
