from collections import Counter

from amner.morphology import MorphAnalyzer
from amner.normalize import normalize
from amner.relations import RelationExtractor, trigger_in

AN = MorphAnalyzer(freq=Counter({"ኢትዮጵያ": 100, "ኬንያ": 50, "ሱዳን": 30}))
REX = RelationExtractor(analyzer=AN)


def ents(tokens, spec):
    out, used = [], set()
    for surf, lab in spec:
        tgt = normalize(surf).split()
        for i in range(len(tokens)):
            if tokens[i:i + len(tgt)] == tgt and i not in used:
                used.add(i)
                out.append({"token_start": i, "token_end": i + len(tgt), "label": lab})
                break
    return out


def triples(text, spec):
    toks = normalize(text).split()
    return {(r.head, r.relation, r.tail) for r in REX.extract(toks, ents(toks, spec))}


def test_title_apposition_and_meeting():
    t = triples("የኢትዮጵያ ጠቅላይ ሚኒስትር ዐቢይ አሕመድ ከኬንያው ፕሬዚዳንት ዊልያም ሩቶ ጋር በናይሮቢ ተወያዩ ።",
                [("የኢትዮጵያ", "LOC"), ("ዐቢይ አሕመድ", "PER"), ("ከኬንያው", "LOC"),
                 ("ዊልያም ሩቶ", "PER"), ("በናይሮቢ", "LOC")])
    assert ("አቢይ አህመድ", "LEADER_OF", "ኢትዮጵያ") in t
    assert ("ዊልያም ሩቶ", "LEADER_OF", "ኬንያ") in t
    assert ("አቢይ አህመድ", "MET_WITH", "ዊልያም ሩቶ") in t
    # the venue (በናይሮቢ) and the polity (ኬንያ) are not meeting participants
    assert not any(r == "MET_WITH" and "ናይሮቢ" in (h + tl) for h, r, tl in t)
    assert not any(r == "MET_WITH" and tl == "ኬንያ" for h, r, tl in t)


def test_coordinated_agreement():
    t = triples("ኢትዮጵያ እና ሱዳን በድንበር ጉዳይ ላይ ስምምነት ተፈራረሙ ።", [("ኢትዮጵያ", "LOC"), ("ሱዳን", "LOC")])
    assert ("ኢትዮጵያ", "AGREEMENT_WITH", "ሱዳን") in t


def test_fused_suffix_trigger():
    assert trigger_in("ጦርነቱ", "ጦርነት")     # ት + u -> ቱ
    assert not trigger_in("ጦማሪ", "ጦርነት")
