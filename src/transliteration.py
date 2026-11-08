"""Script conversion.

* :func:`devanagari_to_roman` – dependency-free, rule-based Devanagari → casual
  Hinglish Roman (e.g. ``समझना`` → ``samajhna``) with a schwa-deletion
  heuristic. Deterministic, so it is safe to use inside scoring
  (script-agnostic WER) as well as for rendering.
* :class:`IndicXlitTransliterator` – optional wrapper around AI4Bharat
  IndicXlit for Roman → Devanagari (``pip install ai4bharat-transliteration``).
  Kept out of core ASR scoring as the plan requires.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

_VIRAMA = "्"
_NUKTA = "़"
_ANUSVARA = "ं"
_CHANDRABINDU = "ँ"
_VISARGA = "ः"

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
# consonant + nukta (NFC keeps these decomposed: U+0958–095F are composition exclusions)
_NUKTA_CONSONANTS = {"क": "q", "ख": "kh", "ग": "gh", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_PRECOMPOSED_NUKTA = {"क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज",
                      "ड़": "ड", "ढ़": "ढ", "फ़": "फ", "य़": "य"}

_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
_LABIALS = ("p", "ph", "b", "bh", "m", "f")


def _syllabify(word: str):
    """Yield syllable records ``[consonant_roman|None, vowel_roman, inherent, nasal]``."""
    sylls = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in _PRECOMPOSED_NUKTA:
            ch = _PRECOMPOSED_NUKTA[ch]
            chars[i] = ch
            chars.insert(i + 1, _NUKTA)
        if ch in _CONSONANTS:
            cons = _CONSONANTS[ch]
            j = i + 1
            if j < len(chars) and chars[j] == _NUKTA:
                cons = _NUKTA_CONSONANTS.get(ch, cons)
                j += 1
            if j < len(chars) and chars[j] == _VIRAMA:
                sylls.append([cons, "", False, ""])
                j += 1
            elif j < len(chars) and chars[j] in _MATRAS:
                sylls.append([cons, _MATRAS[chars[j]], False, ""])
                j += 1
            else:
                sylls.append([cons, "a", True, ""])
            i = j
        elif ch in _INDEPENDENT_VOWELS:
            sylls.append([None, _INDEPENDENT_VOWELS[ch], False, ""])
            i += 1
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            if sylls:
                sylls[-1][3] = "n"
            else:
                sylls.append([None, "", False, "n"])
            i += 1
        elif ch == _VISARGA:
            sylls.append(["h", "", False, ""])
            i += 1
        elif ch in _DIGITS:
            sylls.append([_DIGITS[ch], "", False, ""])
            i += 1
        elif ch == "।":  # danda
            sylls.append([".", "", False, ""])
            i += 1
        elif ch == _NUKTA or ch == _VIRAMA:
            i += 1
        else:
            sylls.append([ch, "", False, ""])
            i += 1
    return sylls


def _delete_schwas(sylls) -> None:
    """Hindi schwa deletion (right-to-left heuristic).

    The inherent ``a`` is dropped word-finally, and medially when it sits
    between a syllable that keeps its vowel and a following consonant+vowel
    (CV) syllable: क-म-ला → kamla, स-म-झ-ना → samajhna.
    """
    n = len(sylls)
    if n == 0:
        return
    last = n - 1
    if n > 1 and sylls[last][2] and not sylls[last][3]:
        sylls[last][1] = ""
    for i in range(last - 1, 0, -1):
        s = sylls[i]
        if not s[2] or s[3]:
            continue
        nxt, prv = sylls[i + 1], sylls[i - 1]
        next_is_cv = nxt[0] is not None and nxt[1] != ""
        prev_has_vowel = prv[1] != ""
        if next_is_cv and prev_has_vowel:
            s[1] = ""


@lru_cache(maxsize=50_000)
def _roman_word(word: str) -> str:
    sylls = _syllabify(word)
    _delete_schwas(sylls)
    out = []
    for k, (cons, vowel, _inh, nasal) in enumerate(sylls):
        if cons:
            out.append(cons)
        if vowel == "aa" and k == len(sylls) - 1 and not nasal and cons:
            vowel = "a"  # casual Hinglish: word-final aa → a (kya, tha, accha)
        out.append(vowel)
        if nasal:
            nxt = sylls[k + 1][0] if k + 1 < len(sylls) else None
            out.append("m" if nxt in _LABIALS else "n")
    return "".join(out)


def devanagari_to_roman(text: str) -> str:
    """Romanize all Devanagari words in ``text``; other characters pass through."""
    text = unicodedata.normalize("NFC", text)
    out, buf = [], []

    def flush():
        if buf:
            out.append(_roman_word("".join(buf)))
            buf.clear()

    for ch in text:
        if "ऀ" <= ch <= "ॿ":
            buf.append(ch)
        else:
            flush()
            out.append(ch)
    flush()
    return "".join(out)


class IndicXlitTransliterator:
    """Optional Roman → Devanagari via AI4Bharat IndicXlit.

    Raises ``ImportError`` at construction if the package is missing, so
    callers can fall back to leaving tokens unchanged.
    """

    def __init__(self, lang: str = "hi", beam_width: int = 4):
        from ai4bharat.transliteration import XlitEngine  # type: ignore

        self.lang = lang
        self.engine = XlitEngine(lang, beam_width=beam_width, rescore=True)
        self._cache: dict[str, str] = {}  # per-instance, bounded below

    def to_native(self, word: str) -> str:
        if word not in self._cache:
            res = self.engine.translit_word(word, topk=1)
            cands = res.get(self.lang) if isinstance(res, dict) else res
            if len(self._cache) >= 50_000:
                self._cache.clear()
            self._cache[word] = cands[0] if cands else word
        return self._cache[word]


def try_load_xlit(lang: str = "hi"):
    try:
        return IndicXlitTransliterator(lang)
    except Exception:
        return None
