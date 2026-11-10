"""Collect every report JSON into ``reports/RESULTS.md`` and draw comparison charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("reports")
ORDER = ["base_full", "base_full_morph", "dapt_full", "dapt_lora", "dapt_full_morph"]
DESC = {"base_full": "Backbone, full fine-tuning",
        "base_full_morph": "Backbone, full FT + morphology-aware augmentation",
        "dapt_full": "Domain-adapted (MLM) backbone, full fine-tuning",
        "dapt_lora": "Domain-adapted backbone, LoRA (PEFT)",
        "dapt_full_morph": "Domain-adapted, full FT + morphology-aware augmentation"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    runs = {n: load(R / "runs" / f"{n}.json") for n in ORDER}
    evals = {n: load(R / "eval" / f"{n}.json") for n in ORDER}
    names = [n for n in ORDER if runs[n] and evals[n]]
    L = ["# Results", "", "All numbers are produced by the scripts in `src/amner`; see PROCESS.md for how.", ""]

    cs = load(R / "corpus_stats.json")
    if cs:
        L += ["## Corpus", "", f"- documents: {cs['docs']}, sentences kept: {cs['sentences_kept']} "
              f"(train {cs['train_sentences']}, valid {cs['valid_sentences']}), tokens: {cs['tokens']:,}, types: {cs['types']:,}",
              f"- per source: {cs['kept_per_source']}",
              f"- dropped: filter {cs.get('dropped_filter', 0)}, duplicates {cs.get('dropped_duplicate', 0)}, "
              f"overlap with NER val/test {cs.get('dropped_heldout_overlap', 0)}", ""]
    tk = load(R / "tokenizer_eval.json")
    if tk:
        L += ["## Tokenizers (held-out corpus + NER test entities)", "",
              "| tokenizer | vocab | fertility ↓ | single-piece % ↑ | clitic-boundary recall % ↑ |", "|---|---:|---:|---:|---:|"]
        for k, v in tk.items():
            L.append(f"| {k} | {v['vocab_size']:,} | {v['fertility']} | {v['single_piece_pct']} | {v['clitic_boundary_recall_pct']} |")
        L.append("")
    for f in sorted(R.glob("mlm_*.json")):
        m = load(f)
        L += [f"## MLM adaptation ({f.stem})", "",
              f"Perplexity on held-out news: **{m['perplexity_before']} → {m['perplexity_after']}** "
              f"({m['steps']} steps, {m['train_runtime_s']} s, vocab {m['vocab_size']:,})", ""]

    if names:
        L += ["## NER (MasakhaNER Amharic test set, entity-level micro scores)", "",
              "| run | description | trainable params | val F1 | test P | test R | **test F1** | strict F1 | F1 on un-normalized input | train time |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n in names:
            r, e = runs[n], evals[n]
            L.append(f"| {n} | {DESC[n]} | {r['params_trainable']:,} ({r['trainable_pct']}%) | {r['best_val_f1']:.4f} | "
                     f"{r['test_precision']:.4f} | {r['test_recall']:.4f} | **{e['f1']:.4f}** | {e['f1_strict']:.4f} | "
                     f"{e['f1_raw_input']:.4f} | {r['train_time_s'] / 60:.1f} min |")
        types = ["PER", "ORG", "LOC", "DATE"]
        L += ["", "### Per-type F1", "", "| run | " + " | ".join(types) + " |", "|---|" + "---:|" * len(types)]
        for n in names:
            L.append(f"| {n} | " + " | ".join(f"{evals[n]['per_type'][t]['f1-score']:.4f}" for t in types) + " |")
        buckets = sorted({b for n in names for b in evals[n]["buckets"] if b.startswith(("affix:", "novelty:"))})
        L += ["", "### Recall by morphological bucket", "", "| bucket | n | " + " | ".join(names) + " |",
              "|---|---:|" + "---:|" * len(names)]
        for b in buckets:
            nn = evals[names[0]]["buckets"].get(b, {}).get("n", 0)
            L.append(f"| {b} | {nn} | " + " | ".join(f"{evals[n]['buckets'].get(b, {}).get('recall', 0):.3f}" for n in names) + " |")
        L.append("")

        fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4))
        x = range(len(names))
        ax[0].bar(x, [evals[n]["f1"] for n in names], color="#2a9d8f")
        ax[0].set_xticks(list(x), names, rotation=15)
        ax[0].set_ylim(0, 1)
        ax[0].set_title("Test F1 by run")
        for i, n in enumerate(names):
            ax[0].text(i, evals[n]["f1"] + 0.01, f"{evals[n]['f1']:.3f}", ha="center")
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            ax[1].bar([j + i * w for j in range(len(buckets))],
                      [evals[n]["buckets"].get(b, {}).get("recall", 0) for b in buckets], w, label=n)
        ax[1].set_xticks([j + 0.4 - w / 2 for j in range(len(buckets))], [b.split(":")[1] for b in buckets], rotation=30, ha="right")
        ax[1].set_ylim(0, 1)
        ax[1].set_title("Entity recall by morphological / lexical bucket")
        ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        fig.tight_layout()
        (R / "figures").mkdir(exist_ok=True)
        fig.savefig(R / "figures" / "ner_comparison.png", dpi=130)
        L += ["![NER comparison](figures/ner_comparison.png)", ""]

    sig = load(R / "significance.json")
    if sig:
        L += ["### Is the difference real? (paired bootstrap, 2,000 resamples of the 500 test sentences)", "",
              "| run | test F1 [95% CI] | Δ vs base_full [95% CI] | P(better than base) |", "|---|---|---|---:|"]
        for n, v in sig.items():
            L.append(f"| {n} | {v['f1']:.4f} [{v['f1_ci95'][0]:.3f}, {v['f1_ci95'][1]:.3f}] | "
                     f"{v['delta_vs_base']:+.4f} [{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}] | {v['p_better_than_base']} |")
        L.append("")
    re_ = load(R / "eval" / "relations.json")
    if re_:
        L += ["## Relation extraction (25-sentence diagnostic set)", "",
              "| setting | P | R | F1 |", "|---|---:|---:|---:|"]
        for k in ("gold_entities_greedy_analyzer", "gold_entities", "end_to_end"):
            if k in re_:
                m = re_[k]["micro"]
                L.append(f"| {k} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
        L.append("")
    op = load(R / "optimization.json")
    if op:
        L += ["## Inference optimization (CPU, dynamic INT8 quantization)", "",
              "| variant | size MB | latency ms/sentence (bs=1) | throughput sent/s (bs=32) | test F1 |", "|---|---:|---:|---:|---:|"]
        for k in ("fp32", "int8_dynamic"):
            v = op[k]
            L.append(f"| {k} | {v['size_mb']} | {v['latency_ms_per_sentence_bs1']} | {v['throughput_sent_per_s_bs32']} | {v['test_f1']} |")
        L += ["", f"Speed-up {op['speedup_bs1']}×, size reduction {op['size_reduction']}×, F1 change {op['f1_delta']:+.4f}.", ""]
    (R / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
