from amner.normalize import (ethiopic_ratio, handle_code_switching, is_good_sentence, normalize,
                             normalize_chars, normalize_token, split_sentences)


def test_homophones_collapse_to_canonical_family():
    # ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ ; all seven vowel orders
    assert normalize_chars("ሐኪም") == "ሀኪም"
    assert normalize_chars("ኀይል") == normalize_chars("ሀይል") == normalize_chars("ሃይል")
    assert normalize_chars("ሠላም") == "ሰላም"
    assert normalize_chars("ዐቢይ") == "አቢይ"
    assert normalize_chars("ፀሐይ") == "ጸሀይ"
    assert normalize_chars("ሕዝብ") == "ህዝብ"       # 6th order
    assert normalize_chars("ሥራ") == "ስራ"          # 6th order of ሠ


def test_fourth_order_glottal_and_h():
    assert normalize_chars("ዓመት") == "አመት"
    assert normalize_chars("ኣዲስ") == "አዲስ"


def test_labialized_two_char_sequence():
    assert normalize_chars("ሉዋንዳ") == "ሏንዳ"


def test_token_normalization_is_alignment_safe():
    toks = ["የኢትዮጵያ", "ሕዝብ", "።"]
    assert len([normalize_token(t) for t in toks]) == len(toks)
    assert all(normalize_token(t) for t in toks)


def test_clean_and_punctuation():
    raw = "ሰላም!!! <b>ዜና</b> https://x.com/a @user 😀 ኢትዮጵያ፡፡ ቀጣይ"
    out = normalize(raw)
    assert "http" not in out and "@user" not in out and "<b>" not in out and "😀" not in out
    assert "።" in out and "፡፡" not in out


def test_code_switching_modes():
    t = "የWHO ሪፖርት"
    assert handle_code_switching(t, "keep") == t
    assert "<lat>WHO</lat>" in handle_code_switching(t, "tag")
    assert "WHO" not in handle_code_switching(t, "drop")


def test_sentence_split_and_filter():
    s = split_sentences("አንደኛ ዓረፍተ ነገር ነው። ሁለተኛው ደግሞ ይህ ነው።")
    assert len(s) == 2
    assert is_good_sentence("ይህ በጣም ረጅም የአማርኛ ዓረፍተ ነገር ነው")
    assert not is_good_sentence("This is English only text here")
    assert ethiopic_ratio("abc") == 0.0
