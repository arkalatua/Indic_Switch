"""Conservative, code-switch-aware post-processing of ASR transcripts.

Design rules (from the plan):
* The raw transcript is never modified; a new normalized string is produced.
* Every change is token-local, gated by the LID tag and its confidence, and
  logged as an :class:`Edit` so corrections and regressions can be audited.
* Uncertain tokens, punctuation, numbers, named-entity-like tokens and
  user-protected words are left unchanged.
* No generative model rewrites the sentence.

Rules, each switchable for ablations:
1. ``collapse_loops``      – collapse Whisper hallucination loops (an n-gram,
                              n ≤ 4, repeated ≥ ``min_loop_repeats`` times).
2. ``canonicalize``        – map Romanized-Hindi spelling variants of
                              HI-tagged tokens to one canonical spelling.
3. ``target_script``       – ``keep`` | ``roman`` | ``native`` rendering:
     roman:  Devanagari English loanwords → English via lexicon (unless the
             LID model confidently says HI), other Devanagari → rule-based
             Roman; then (2) applies to the romanized HI tokens.
     native: confident HI-tagged Roman tokens → Devanagari via IndicXlit
             (if installed); English tokens stay in Latin script.
     mixed:  each token in its own language's script (Hindi → Devanagari,
             English → Latin) — the CoSHE-style reference convention. Does
             the English back-transliteration of ``roman`` and the
             HI→Devanagari step of ``native``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from .preprocessing import Token, normalize_text, tokenize_with_offsets, word_script
from .transliteration import devanagari_to_roman

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"


@dataclass
class CorrectionConfig:
    min_confidence: float = 0.80
    collapse_loops: bool = True
    min_loop_repeats: int = 4
    canonicalize: bool = True
    target_script: str = "keep"  # keep | roman | native | mixed
    protect_named_entities: bool = True
    protected_words: set[str] = field(default_factory=set)
    variants_path: str | None = None
    en_devanagari_path: str | None = None


@dataclass
class Edit:
    index: int
    before: str
    after: str  # "" means deleted
    rule: str
    tag: str
    confidence: float


@dataclass
class CorrectionResult:
    text: str
    tokens: list[str]
    edits: list[Edit]


@lru_cache(maxsize=8)
def _load_variants(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    table = {}
    for canon, variants in raw.items():
        if canon.startswith("_"):
            continue
        for v in variants:
            table[v.lower()] = canon
    return table


@lru_cache(maxsize=8)
def _load_en_devanagari(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def _match_case(src: str, dst: str) -> str:
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


class CodeSwitchCorrector:
    def __init__(self, config: CorrectionConfig | None = None, transliterator=None):
        self.cfg = config or CorrectionConfig()
        self.variants = _load_variants(self.cfg.variants_path or str(LEXICON_DIR / "hinglish_variants.json"))
        self.en_dev = _load_en_devanagari(self.cfg.en_devanagari_path or str(LEXICON_DIR / "en_devanagari.json"))
        self.xlit = transliterator
        if self.cfg.target_script in ("native", "mixed") and self.xlit is None:
            from .transliteration import try_load_xlit

            self.xlit = try_load_xlit("hi")

    # -- protection ------------------------------------------------------
    def _is_protected(self, i: int, toks: Sequence[Token], tag: str) -> bool:
        t = toks[i]
        if t.is_punct or tag == "OTH" or t.text.lower() in self.cfg.protected_words:
            return True
        if self.cfg.protect_named_entities and word_script(t.text) == "latin":
            if t.text.isupper() and len(t.text) > 1:
                return True  # acronym
            sentence_start = i == 0 or toks[i - 1].text in ".!?।"
            if t.text[:1].isupper() and not sentence_start and t.text.lower() not in self.variants:
                return True  # capitalized mid-sentence → likely a name
        return False

    # -- rules -----------------------------------------------------------
    def _collapse_loops(self, words: list[str]) -> set[int]:
        """Indices to delete: keep one copy of any n-gram repeated ≥ k times."""
        k = self.cfg.min_loop_repeats
        delete: set[int] = set()
        low = [w.lower() for w in words]
        i = 0
        while i < len(low):
            best = None
            for n in range(1, 5):
                gram = low[i:i + n]
                if len(gram) < n:
                    break
                reps = 1
                while low[i + reps * n:i + (reps + 1) * n] == gram:
                    reps += 1
                if reps >= k and (best is None or reps * n > best[0] * best[1]):
                    best = (reps, n)
            if best:
                reps, n = best
                delete.update(range(i + n, i + reps * n))
                i += reps * n
            else:
                i += 1
        return delete

    def correct(self, text: str, tags: Sequence[str], confidences: Sequence[float],
                tokens: Sequence[Token] | None = None) -> CorrectionResult:
        """Correct ``text`` given per-token ``tags``/``confidences`` aligned with
        ``tokens`` (as produced by :func:`tokenize_with_offsets` on ``text``)."""
        cfg = self.cfg
        toks = list(tokens) if tokens is not None else tokenize_with_offsets(text)
        if len(toks) != len(tags) or len(tags) != len(confidences):
            raise ValueError(f"tokens/tags/confidences length mismatch: {len(toks)}/{len(tags)}/{len(confidences)}")
        out = [t.text for t in toks]
        edits: list[Edit] = []

        def change(i, new, rule):
            if new != out[i]:
                edits.append(Edit(i, out[i], new, rule, tags[i], float(confidences[i])))
                out[i] = new

        if cfg.collapse_loops:
            for i in sorted(self._collapse_loops(out)):
                change(i, "", "collapse_loop")

        for i in range(len(toks)):
            if out[i] == "" or self._is_protected(i, toks, tags[i]):
                continue
            tag, conf = tags[i], confidences[i]
            confident_hi = tag == "HI" and conf >= cfg.min_confidence
            script = word_script(out[i])

            if cfg.target_script in ("roman", "mixed") and script == "devanagari":
                en = self.en_dev.get(out[i])
                if en is not None and not confident_hi:
                    change(i, en, "en_backtransliterate")
                    continue
                if cfg.target_script == "roman":
                    change(i, devanagari_to_roman(out[i]), "romanize")
                    script = "latin"

            if cfg.canonicalize and confident_hi and script == "latin":
                canon = self.variants.get(out[i].lower())
                if canon:
                    change(i, _match_case(out[i], canon), "canonicalize")

            if (cfg.target_script in ("native", "mixed") and confident_hi and self.xlit
                    and word_script(out[i]) == "latin"):
                change(i, self.xlit.to_native(out[i].lower()), "to_native")

        return CorrectionResult(_rebuild(text, toks, out), out, edits)


def _rebuild(template: str, toks: Sequence[Token], new: Sequence[str]) -> str:
    """Reassemble using the original inter-token gaps; deleted tokens drop
    their preceding gap."""
    parts, last = [], 0
    for t, w in zip(toks, new, strict=True):
        gap = template[last:t.start]
        if w:
            parts.append(gap if parts else gap.lstrip())
            parts.append(w)
        last = t.end
    parts.append(template[last:])
    return normalize_text("".join(parts))
