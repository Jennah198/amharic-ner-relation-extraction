"""Political relation extraction over NER output (Amharic news).

There is no public Amharic relation-extraction corpus, so relations are
extracted with linguistically-motivated patterns on top of the neural NER
predictions. The patterns exploit three properties of Amharic news prose:

1. **Title apposition** – office titles stand *between* the polity and the
   person:  ``የ<LOC/ORG> <TITLE> <PER>``
   e.g. የኢትዮጵያ ጠቅላይ ሚኒስትር ዐቢይ አሕመድ -> (ዐቢይ አሕመድ, LEADER_OF, ኢትዮጵያ)

2. **SOV order with postpositional framing** – the second participant of a
   meeting/agreement is marked by the clitic ``ከ-`` ("with/from") and usually
   the postposition ``ጋር``, and the event verb is clause-final:
   ``<E1> ከ<E2> ጋር ... ተወያዩ`` -> (E1, MET_WITH, E2)

3. **Clitics carry the role** – because ``የ-`` (genitive) and ``ከ-``
   (comitative/ablative) are glued to the entity token, the morphological
   analyzer is used both to *read* the role marker and to output a clean,
   clitic-free lemma for each argument (ከኬንያው -> ኬንያ).

Relation inventory
------------------
LEADER_OF         PER -> ORG|LOC    head of state/government/party/agency
HOLDS_TITLE       PER -> title      (attribute, when no polity is named)
SPOKESPERSON_OF   PER -> ORG|LOC
MEMBER_OF         PER -> ORG
MET_WITH          PER|ORG|LOC <-> PER|ORG|LOC
AGREEMENT_WITH    ORG|LOC|PER <-> ORG|LOC|PER
CONFLICT_WITH     ORG|LOC|PER <-> ORG|LOC|PER
LOCATED_IN        ORG -> LOC
EVENT_DATE        PER|ORG|LOC -> DATE  (statement / event anchored to a date)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .morphology import DEFAULT_ANALYZER, PREFIXES, SUFFIXES, MorphAnalyzer
from .normalize import normalize_chars

_N = normalize_chars  # all lexicons are matched in normalized spelling

TITLES = {_N(t): rel for t, rel in {
    "ጠቅላይ ሚኒስትር": "LEADER_OF", "ምክትል ጠቅላይ ሚኒስትር": "LEADER_OF",
    "ፕሬዚዳንት": "LEADER_OF", "ፕሬዝዳንት": "LEADER_OF", "ፕሬዚደንት": "LEADER_OF",
    "ሊቀመንበር": "LEADER_OF", "ሊቀ መንበር": "LEADER_OF", "ሰብሳቢ": "LEADER_OF",
    "ርዕሰ መስተዳድር": "LEADER_OF", "ከንቲባ": "LEADER_OF", "ንጉሥ": "LEADER_OF",
    "ዋና ጸሐፊ": "LEADER_OF", "ዋና ዳይሬክተር": "LEADER_OF", "ዋና ሥራ አስፈጻሚ": "LEADER_OF",
    "ዳይሬክተር": "LEADER_OF", "ኃላፊ": "LEADER_OF", "ኮሚሽነር": "LEADER_OF",
    "አፈ ጉባኤ": "LEADER_OF", "ዋና አዛዥ": "LEADER_OF", "ኤታማዦር ሹም": "LEADER_OF",
    "ሚኒስትር": "LEADER_OF", "ገዥ": "LEADER_OF", "አምባሳደር": "MEMBER_OF",
    "ቃል አቀባይ": "SPOKESPERSON_OF", "አባል": "MEMBER_OF", "ተወካይ": "MEMBER_OF",
}.items()}
_TITLE_TOKENS = sorted((t.split() for t in TITLES), key=len, reverse=True)

MEET_TRIGGERS = [_N(t) for t in ["ተወያ", "ውይይት", "ተገናኙ", "ተገናኘ", "ተገናኝ", "ተነጋገ", "ተነጋግ",
                                 "ጎበኙ", "ጎብኝ", "ተቀበሉ", "ተቀብለው"]]
# a signing / agreeing *verb* is required – the noun ስምምነት alone also appears in
# "discussed the implementation of the agreement" and would cause false positives
AGREE_TRIGGERS = [_N(t) for t in ["ተፈራረ", "ፈረሙ", "ፈርመ", "ተስማሙ", "ተስማም", "ደረሱ"]]
CONFLICT_TRIGGERS = [_N(t) for t in ["ጦርነት", "ግጭት", "ውጊያ", "ጥቃት", "ተዋጉ", "ተጋጩ", "ፍልሚያ", "ወረራ"]]
STATEMENT_TRIGGERS = [_N(t) for t in ["ሰጥተዋል", "ሰጡ", "ሰጠ", "ሰጥቷል", "አስታወቀ", "አስታውቋል", "አስታወቁ", "ገለጸ", "ገልጸዋል", "ገልጿል",
                                      "ተናገሩ", "ተናግረዋል", "አሉ", "ብለዋል", "አስታውቀዋል", "ተካሄደ",
                                      "ተካሂዷል", "ተደረገ", "ተፈጸመ", "አጸደቀ", "አጽድቋል", "አሳሰቡ", "አሳስበዋል"]]
LOC_TRIGGERS = [_N(t) for t in ["ውስጥ", "ሚገኘው", "ሚገኝ", "ዋና መሥሪያ ቤቱ"]]

ACTORS = {"PER", "ORG", "LOC"}


@dataclass
class Relation:
    head: str
    head_label: str
    relation: str
    tail: str
    tail_label: str
    trigger: str
    sentence: int
    head_span: tuple[int, int]
    tail_span: tuple[int, int]

    def as_dict(self):
        return asdict(self)

    def key(self) -> tuple:
        """Order-insensitive key for symmetric relations (used in evaluation)."""
        sym = self.relation in {"MET_WITH", "AGREEMENT_WITH", "CONFLICT_WITH"}
        a, b = (self.head, self.tail)
        if sym and b < a:
            a, b = b, a
        return (a, self.relation, b)


def _sentence_ids(tokens: list[str]) -> list[int]:
    ids, s = [], 0
    for t in tokens:
        ids.append(s)
        if t in ("።", "!", "?", "፧"):
            s += 1
    return ids


def _match_title(tokens: list[str], i: int, n: int) -> str | None:
    """Title of exactly ``n`` tokens starting at ``i``; the first token may carry
    a prefix clitic (የ-, ለ-, ከ- ...) and the last one a suffix clitic
    (ፕሬዚዳንቱ is not covered on purpose: -ቱ fuses with the stem)."""
    if i < 0 or i + n > len(tokens):
        return None
    w = tokens[i:i + n]
    for t in _TITLE_TOKENS:
        if len(t) != n or w[1:-1] != t[1:-1]:
            continue
        first_ok = any(w[0] == p + t[0] for p in [""] + PREFIXES)
        if n > 1:
            last_ok = any(w[-1] == t[-1] + s for s in [""] + SUFFIXES)
        else:
            last_ok = any(w[0] == p + t[0] + s for p in [""] + PREFIXES for s in [""] + SUFFIXES)
            first_ok = last_ok
        if first_ok and last_ok:
            return " ".join(t)
    return None


def _fidel_family(ch: str) -> int:
    """Consonant family of an Ethiopic syllable (each family = 8 code points:
    7 vowel orders + labialized form), or -1 for non-Ethiopic characters."""
    o = ord(ch)
    return (o - 0x1200) // 8 if 0x1200 <= o <= 0x137F else -1


def trigger_in(token: str, trig: str) -> bool:
    """Substring match that tolerates a vowel change in the trigger's final
    syllable. Amharic suffixes fuse with the last consonant: ጦርነት + -u
    (definite) is written ጦርነቱ, so "ጦርነት" is not a substring of it, but the
    consonant skeleton ጦ-ር-ነ-ት is."""
    if trig in token:
        return True
    if len(trig) < 3:
        return False
    head, last = trig[:-1], _fidel_family(trig[-1])
    i = token.find(head)
    while i != -1:
        j = i + len(head)
        if j < len(token) and last != -1 and _fidel_family(token[j]) == last:
            return True
        i = token.find(head, i + 1)
    return False


def _has_trigger(tokens: list[str], triggers: list[str]) -> str | None:
    for tok in tokens:
        for trig in triggers:
            if trigger_in(tok, trig):  # verbs carry agreement/tense affixes
                return trig
    return None


class RelationExtractor:
    def __init__(self, analyzer: MorphAnalyzer | None = None, max_gap: int = 2):
        self.an = analyzer or DEFAULT_ANALYZER
        self.max_gap = max_gap

    def extract(self, tokens: list[str], entities: list[dict]) -> list[Relation]:
        """``entities``: dicts with token_start, token_end (exclusive), label."""
        sid = _sentence_ids(tokens)
        sent_end = {}
        for k, x in enumerate(sid):
            sent_end[x] = k + 1
        ents = sorted(entities, key=lambda e: e["token_start"])
        out: list[Relation] = []
        seen = set()

        def lemma(e):
            return self.an.lemma_span(tokens[e["token_start"]:e["token_end"]])

        def prefix_of(e):
            return self.an.strip_prefix(tokens[e["token_start"]])[0]

        def add(h, rel, t, trig, tail_text=None, tail_label=None):
            r = Relation(head=lemma(h), head_label=h["label"], relation=rel,
                         tail=tail_text if tail_text is not None else lemma(t),
                         tail_label=tail_label or t["label"], trigger=trig,
                         sentence=sid[h["token_start"]],
                         head_span=(h["token_start"], h["token_end"]),
                         tail_span=(t["token_start"], t["token_end"]) if t else (-1, -1))
            if r.key() not in seen and r.head != r.tail:
                seen.add(r.key())
                out.append(r)

        # ---- 1. title apposition: [የ-ORG/LOC] TITLE PER --------------------
        polities: set[int] = set()   # entities used as the polity of a title
        for e in ents:
            if e["label"] != "PER":
                continue
            found = None
            for gap in range(0, self.max_gap + 1):
                for n in range(4, 0, -1):  # longest title first
                    i = e["token_start"] - gap - n
                    title = _match_title(tokens, i, n)
                    if title and not any(x["token_start"] < i + n and i < x["token_end"] for x in ents):
                        found = (title, i)
                        break
                if found:
                    break
            if not found:
                continue
            title, ti = found
            add(e, "HOLDS_TITLE", None, title, tail_text=title, tail_label="TITLE")
            polity = next((p for p in reversed(ents)
                           if p["label"] in ("ORG", "LOC") and p["token_end"] <= ti
                           and ti - p["token_end"] <= 1 and sid[p["token_start"]] == sid[ti]), None)
            if polity is not None:
                add(e, TITLES[title], polity, title)
                polities.add(polity["token_start"])

        # ---- 2. pairwise relations with clause-final triggers ---------------
        def modifier(e) -> bool:
            """LOC/ORG that only modifies a following person
            (ከሩሲያው አቻቸው ቭላድሚር ፑቲን ጋር: the participant is Putin, not Russia)."""
            return e["label"] in ("LOC", "ORG") and any(
                x["label"] == "PER" and 0 <= x["token_start"] - e["token_end"] <= 3
                and sid[x["token_start"]] == sid[e["token_start"]] for x in ents)

        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                s = sid[a["token_start"]]
                if s != sid[b["token_start"]]:
                    continue
                sentence = tokens[sent_end.get(s - 1, 0):sent_end[s]]
                between = tokens[a["token_end"]:b["token_start"]]
                after_b = tokens[b["token_end"]:sent_end[s]]
                between_pair = "መካከል" in after_b[:2]          # በX እና በY መካከል
                comitative = prefix_of(b) == "ከ" or "ጋር" in after_b[:2]
                coordinated = ("እና" in between and len(between) <= 4) or (
                    not between and tokens[a["token_end"] - 1].endswith("ና"))
                venue = prefix_of(b) == "በ" and not between_pair  # በ- = "in/at": a place, not a party
                actors = a["label"] in ACTORS and b["label"] in ACTORS
                free = not ({a["token_start"], b["token_start"]} & polities) and not modifier(a) and not modifier(b)

                if actors and free and not venue:
                    trig = _has_trigger(after_b, AGREE_TRIGGERS)
                    if trig and (comitative or coordinated):
                        add(a, "AGREEMENT_WITH", b, trig)
                        continue
                    trig = _has_trigger(after_b, CONFLICT_TRIGGERS) or (
                        between_pair and _has_trigger(sentence, CONFLICT_TRIGGERS))
                    if trig and (comitative or coordinated or between_pair):
                        add(a, "CONFLICT_WITH", b, trig)
                        continue
                    trig = _has_trigger(after_b, MEET_TRIGGERS)
                    if trig and comitative and a["label"] in ("PER", "ORG"):
                        add(a, "MET_WITH", b, trig)
                        continue

                if a["label"] == "ORG" and b["label"] == "LOC":
                    trig = _has_trigger(between + after_b[:1], LOC_TRIGGERS)
                    if trig and len(between) <= 3:
                        add(a, "LOCATED_IN", b, trig)
                elif a["label"] == "LOC" and b["label"] == "ORG" and a["token_start"] not in polities:
                    if b["token_start"] == a["token_end"] and prefix_of(a) == "የ":
                        add(b, "LOCATED_IN", a, "የ-")
                    else:
                        trig = _has_trigger(between, LOC_TRIGGERS)
                        if trig and len(between) <= 4:
                            add(b, "LOCATED_IN", a, trig)

        # ---- 3. date anchoring: nearest PER/ORG actor + statement/event verb ----
        for d in ents:
            if d["label"] != "DATE":
                continue
            s = sid[d["token_start"]]
            cands = [e for e in ents if e["label"] in ("PER", "ORG") and sid[e["token_start"]] == s]
            if not cands:
                continue
            before = [e for e in cands if e["token_end"] <= d["token_start"]]
            actor = before[-1] if before else cands[0]
            trig = _has_trigger(tokens[d["token_end"]:sent_end[s]],
                                STATEMENT_TRIGGERS + MEET_TRIGGERS + AGREE_TRIGGERS)
            if trig:
                add(actor, "EVENT_DATE", d, trig)
        return out
