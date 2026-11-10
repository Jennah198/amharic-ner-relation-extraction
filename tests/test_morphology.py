import random
from collections import Counter

from amner.bio import bio_to_spans
from amner.morphology import MorphAnalyzer, morph_augment

GREEDY = MorphAnalyzer()  # no lexicon
LEX = MorphAnalyzer(freq=Counter({"ኢትዮጵያ": 100, "ኢትዮጵያን": 10, "አዲስ": 80, "ሱዳን": 30, "ኬንያ": 40,
                                  "ላይ": 500, "ኮሚሽን": 20, "ቀለ": 1}))


def test_greedy_strips_common_clitics():
    a = GREEDY.analyze("በኢትዮጵያ")
    assert (a.prefix, a.stem) == ("በ", "ኢትዮጵያ")
    assert GREEDY.analyze("ኢትዮጵያን").suffix == "ን"
    assert GREEDY.analyze("የኢትዮጵያንም").as_dict()["stem"] == "ኢትዮጵያ"


def test_greedy_overstrips_names_which_lexicon_fixes():
    assert GREEDY.analyze("ሱዳን").stem == "ሱዳ"        # wrong
    assert LEX.analyze("ሱዳን").stem == "ሱዳን"          # right: "ሱዳ" is not a word
    assert LEX.analyze("ኮሚሽን").stem == "ኮሚሽን"


def test_lexicon_analysis_prefers_frequent_stem():
    a = LEX.analyze("ከኬንያው")
    assert (a.prefix, a.stem, a.suffix) == ("ከ", "ኬንያ", "ው")
    assert LEX.analyze("ለኢትዮጵያንም").stem == "ኢትዮጵያ"
    assert LEX.analyze("በቀለ").stem == "በቀለ"   # "ቀለ" too rare -> name kept intact


def test_lemma_span_strips_outer_clitics_only():
    assert LEX.lemma_span(["በአዲስ", "አበባ"]) == "አዲስ አበባ"
    assert LEX.lemma_span(["ኢትዮጵያን"]) == "ኢትዮጵያ"


def test_bio_to_spans():
    assert bio_to_spans(["B-PER", "I-PER", "O", "B-LOC"]) == [(0, 2, "PER"), (3, 4, "LOC")]
    assert bio_to_spans(["I-ORG", "I-ORG", "B-ORG"]) == [(0, 2, "ORG"), (2, 3, "ORG")]
    assert bio_to_spans(["B-PER", "I-LOC"]) == [(0, 1, "PER"), (1, 2, "LOC")]


def test_morph_augment_changes_clitics_but_not_labels():
    toks, labs = ["ኢትዮጵያ", "እና", "ኬንያ", "ተስማሙ"], ["B-LOC", "O", "B-LOC", "O"]
    out = None
    rng = random.Random(0)
    for _ in range(10):
        out = morph_augment(toks, labs, rng, analyzer=GREEDY, p_prefix=1.0)
        if out:
            break
    assert out is not None
    new_toks, new_labs = out
    assert new_labs == labs and len(new_toks) == len(toks)
    assert new_toks[1] == "እና" and new_toks[3] == "ተስማሙ"   # non-entities untouched
    assert new_toks != toks
