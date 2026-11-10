"""Stage 3 – Subword tokenization.

Three things happen here:

1. **Train custom tokenizers** on the normalized corpus with Hugging Face
   ``tokenizers``: a BPE model (SentencePiece-style ``Metaspace``
   pre-tokenization) and a WordPiece model, both with the same vocab size.

2. **Intrinsic evaluation** of every candidate tokenizer (custom BPE, custom
   WordPiece, the backbone's own tokenizer and multilingual XLM-R) on held-out
   text and on the NER data:

   * *fertility*      – subwords per whitespace word (lower = less fragmentation)
   * *single-piece %* – share of words kept as one piece
   * *clitic-boundary recall* – for NER tokens that carry a clitic
     (በ+አዲስ, ኢትዮጵያ+ን, …), how often the tokenizer places a subword boundary
     exactly at the morpheme boundary. A tokenizer that does this lets the model
     see the same stem embedding regardless of the attached clitic.

3. **Vocabulary extension** of the pre-trained backbone: frequent words that the
   backbone fragments into >= 3 pieces but that our BPE learned as one unit are
   added to the backbone tokenizer as whole-word tokens. The matching embedding
   rows are initialised (in ``train_mlm.py``) as the mean of the embeddings of
   the pieces they used to be split into, then trained during MLM adaptation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import AddedToken

from .morphology import DEFAULT_ANALYZER

SPECIALS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def train_bpe(files: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(),
        pre_tokenizers.Punctuation(),
    ])
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=SPECIALS, show_progress=False)
    tok.train(files, trainer)
    return tok


def train_wordpiece(files: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, min_frequency=2,
                                        special_tokens=SPECIALS, show_progress=False)
    tok.train(files, trainer)
    return tok


def _pieces(tok, word: str) -> list[str]:
    """Subword pieces of a single word, with tokenizer-specific markers removed."""
    if isinstance(tok, Tokenizer):
        toks = tok.encode(word, add_special_tokens=False).tokens
    else:
        toks = tok.tokenize(word)
    clean = [t.replace("▁", "").replace("Ġ", "").replace("##", "") for t in toks]
    return [c for c in clean if c]


def evaluate_tokenizer(tok, words: list[str], ner_tokens: list[str]) -> dict:
    total_pieces = single = unk = 0
    for w in words:
        p = _pieces(tok, w)
        total_pieces += len(p)
        single += len(p) == 1
        unk += sum(x in ("<unk>", "[UNK]") for x in p)
    # clitic boundary recall on NER entity tokens that have an affix
    hits = n = 0
    for w in ner_tokens:
        a = DEFAULT_ANALYZER.analyze(w)
        if not a.has_affix:
            continue
        p = _pieces(tok, w)
        offsets, pos = set(), 0
        for piece in p:
            pos += len(piece)
            offsets.add(pos)
        if a.prefix:
            n += 1
            hits += len(a.prefix) in offsets
        if a.suffix:
            n += 1
            hits += (len(w) - len(a.suffix)) in offsets
    return {
        "fertility": round(total_pieces / max(len(words), 1), 3),
        "single_piece_pct": round(100 * single / max(len(words), 1), 2),
        "unk_pct": round(100 * unk / max(total_pieces, 1), 3),
        "clitic_boundary_recall_pct": round(100 * hits / max(n, 1), 2),
        "clitic_boundaries_checked": n,
    }


def select_extension_tokens(bpe: Tokenizer, base, corpus_words: Counter, k: int,
                            min_freq: int = 20, min_base_pieces: int = 3) -> list[str]:
    """Frequent whole words our BPE keeps intact but the backbone shatters."""
    bpe_vocab = bpe.get_vocab()
    base_vocab = base.get_vocab()
    cands = []
    for w, c in corpus_words.most_common():
        if c < min_freq or len(cands) >= k:
            break
        if not any("ሀ" <= ch <= "፿" for ch in w):
            continue
        if ("▁" + w) not in bpe_vocab:  # BPE learned it as a full word unit
            continue
        if w in base_vocab or ("Ġ" + w) in base_vocab or ("▁" + w) in base_vocab:
            continue
        if len(_pieces(base, w)) >= min_base_pieces:
            cands.append(w)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus_train.txt")
    ap.add_argument("--valid", default="data/processed/corpus_valid.txt")
    ap.add_argument("--ner-train", default="data/processed/ner_train.jsonl")
    ap.add_argument("--ner-test", default="data/processed/ner_test.jsonl")
    ap.add_argument("--base-model", default="rasyosef/roberta-medium-amharic")
    ap.add_argument("--compare", nargs="*", default=["xlm-roberta-base"])
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--extend-k", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("artifacts/tokenizers"))
    ap.add_argument("--report", type=Path, default=Path("reports/tokenizer_eval.json"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # NER train text is also unlabelled Amharic news -> include it for training
    ner_train_txt = a.out / "_ner_train_text.txt"
    with ner_train_txt.open("w", encoding="utf-8") as f:
        for line in open(a.ner_train, encoding="utf-8"):
            f.write(" ".join(json.loads(line)["tokens"]) + "\n")
    files = [a.corpus, str(ner_train_txt)]

    bpe = train_bpe(files, a.vocab_size)
    bpe.save(str(a.out / "amharic_bpe.json"))
    PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<s>", eos_token="</s>",
                            unk_token="<unk>", pad_token="<pad>", mask_token="<mask>"
                            ).save_pretrained(a.out / "amharic_bpe_hf")
    wp = train_wordpiece(files, a.vocab_size)
    wp.save(str(a.out / "amharic_wordpiece.json"))

    valid_words = [w for line in open(a.valid, encoding="utf-8") for w in line.split()]
    ner_test = [json.loads(l) for l in open(a.ner_test, encoding="utf-8")]
    ent_tokens = [t for ex in ner_test for t, lab in zip(ex["tokens"], ex["ner_tags"]) if lab != "O"]

    base = AutoTokenizer.from_pretrained(a.base_model)
    candidates = {"custom_bpe": bpe, "custom_wordpiece": wp, f"backbone:{a.base_model}": base}
    for name in a.compare:
        try:
            candidates[f"compare:{name}"] = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            print(f"[warn] {name}: {e}")

    report = {}
    for name, tok in candidates.items():
        report[name] = evaluate_tokenizer(tok, valid_words, ent_tokens)
        report[name]["vocab_size"] = tok.get_vocab_size() if isinstance(tok, Tokenizer) else len(tok)
        print(name, report[name])

    # ---- vocabulary extension of the backbone ----
    corpus_words = Counter(w for line in open(a.corpus, encoding="utf-8") for w in line.split())
    new_tokens = select_extension_tokens(bpe, base, corpus_words, a.extend_k)
    ext = AutoTokenizer.from_pretrained(a.base_model)
    n_added = ext.add_tokens([AddedToken(t, single_word=True, normalized=False) for t in new_tokens])
    ext.save_pretrained(a.out / "backbone_extended")
    (a.out / "backbone_extended" / "added_words.txt").write_text("\n".join(new_tokens), encoding="utf-8")
    report["backbone_extended"] = evaluate_tokenizer(ext, valid_words, ent_tokens)
    report["backbone_extended"].update({"vocab_size": len(ext), "added_tokens": n_added,
                                        "example_added": new_tokens[:25]})
    print("backbone_extended", report["backbone_extended"])

    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ner_train_txt.unlink()


if __name__ == "__main__":
    main()
