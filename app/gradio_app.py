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


def _call(endpoint: str, text: str) -> dict:
    """POST to the FastAPI backend if configured, otherwise run in-process."""
    if API_URL:
        r = requests.post(f"{API_URL}/{endpoint}", json={"text": text}, timeout=120)
        r.raise_for_status()
        return r.json()
    from amner import explain, service
    return {"analyze": service.analyze, "normalize": explain.normalization_diff,
            "tokenize": explain.tokenizer_comparison}[endpoint](text)


def _analyze(text: str) -> dict:
    return _call("analyze", text)


NORM_EXAMPLES = [
    "ዐቢይ አሕመድ ኃይል ሥራ ፀሐይ ዓመት ሕዝብ ሠላም",
    "የኢትዮጵያ ሕዝብ ተወካዮች ምክር ቤት ዛሬ ስብሰባውን አካሂዷል፡፡ https://t.me/news @reporter 😀",
    "የWHO ዋና ዳይሬክተር ዶ/ር ቴድሮስ አድሐኖም በጄኔቫ መግለጫ ሰጡ::",
]
TOK_EXAMPLES = ["የኢትዮጵያ", "ለኢትዮጵያንም", "ከኬንያው", "በአዲስ አበባ", "ሱዳን", "ሀይማኖታዊ", "ጦርነቱ"]


def run_normalize(text: str):
    if not text.strip():
        return [], "", [], ""
    r = _call("normalize", text)
    st = r["stats"]
    summary = (f"{st['chars_respelled']} character(s) re-spelled · {len(r['removed_noise'])} noise item(s) removed · "
               f"Latin-script words kept: {', '.join(r['latin_words']) or '–'}\n"
               f"{st['raw_words']} raw words → {st['normalized_tokens']} normalized tokens "
               f"(Ethiopic punctuation becomes separate tokens)")
    changes = [[c["change"], c["count"]] for c in r["char_changes"]]
    segs = [(t, lab) for t, lab in r["segments"]]
    return segs, r["normalized"], changes, summary


def run_tokenize(text: str):
    if not text.strip():
        return [], [], ""
    r = _call("tokenize", text)
    rows = [[t["tokenizer"], " | ".join(t["pieces"]), t["n_pieces"], t["fertility"]] for t in r["tokenizers"]]
    morph = [[m["word"], m["prefix"], m["stem"], m["suffix"], m["segmentation"]] for m in r["morphology"]]
    note = (f"Normalized input: {r['normalized']}\n"
            "Compare where each tokenizer cuts the word with the clitic boundary (prefix + stem + suffix). "
            "▁ marks a word start (SentencePiece/BPE), ## a word-internal piece (WordPiece).")
    return rows, morph, note


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

    gr.Markdown("---\n## Inside the pipeline")
    with gr.Tab("Normalization"):
        gr.Markdown("Raw input with every character the normalizer changes highlighted "
                    "(homophones ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ, 4th orders, labials, digits) and the noise it removes.")
        with gr.Row():
            norm_in = gr.Textbox(label="Raw Amharic text", lines=4, scale=3)
            norm_btn = gr.Button("Show normalization", variant="primary", scale=1)
        gr.Examples(NORM_EXAMPLES, inputs=norm_in)
        with gr.Row():
            norm_hl = gr.HighlightedText(label="Raw input (changed characters highlighted)", combine_adjacent=False,
                                         show_legend=False)
            norm_out = gr.Textbox(label="Cleaned / normalized text", lines=4)
        with gr.Row():
            norm_tab = gr.Dataframe(headers=["change", "count"], label="Character changes")
            norm_sum = gr.Textbox(label="Summary", lines=3)
        norm_btn.click(run_normalize, inputs=norm_in, outputs=[norm_hl, norm_out, norm_tab, norm_sum])
        norm_in.submit(run_normalize, inputs=norm_in, outputs=[norm_hl, norm_out, norm_tab, norm_sum])
    with gr.Tab("Tokenizer"):
        gr.Markdown("How the backbone, our custom BPE and WordPiece, multilingual XLM-R and the extended "
                    "tokenizer split a word, next to the clitic analysis (prefix + stem + suffix).")
        with gr.Row():
            tok_in = gr.Textbox(label="Word or short phrase", scale=3)
            tok_btn = gr.Button("Compare tokenizers", variant="primary", scale=1)
        gr.Examples(TOK_EXAMPLES, inputs=tok_in)
        tok_tab = gr.Dataframe(headers=["tokenizer", "subword pieces", "# pieces", "pieces / word"])
        tok_morph = gr.Dataframe(headers=["word", "prefix", "stem", "suffix", "clitic segmentation"],
                                 label="Clitic analysis (lexicon-constrained)")
        tok_note = gr.Textbox(label="Notes", lines=2)
        tok_btn.click(run_tokenize, inputs=tok_in, outputs=[tok_tab, tok_morph, tok_note])
        tok_in.submit(run_tokenize, inputs=tok_in, outputs=[tok_tab, tok_morph, tok_note])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
