import pytest

from src.preprocessing import (ManifestRow, assign_splits, normalize_text, tokenize_with_offsets, word_script)
from src.transliteration import devanagari_to_roman


def test_normalize_preserves_words():
    assert normalize_text("  main​ kal   “office”  nahi ") == 'main kal "office" nahi'


def test_tokenize_offsets_roundtrip():
    text = "मैं कल office नहीं जा रहा, okay? don't"
    toks = tokenize_with_offsets(text)
    for t in toks:
        assert text[t.start:t.end] == t.text
    words = [t.text for t in toks]
    assert words == ["मैं", "कल", "office", "नहीं", "जा", "रहा", ",", "okay", "?", "don't"]
    assert [t.is_punct for t in toks].count(True) == 2


@pytest.mark.parametrize("w,s", [("नहीं", "devanagari"), ("office", "latin"), ("123", "digit"),
                                 ("?", "symbol"), ("வணக்கம்", "tamil")])
def test_word_script(w, s):
    assert word_script(w) == s


@pytest.mark.parametrize("dev,rom", [
    ("कमला", "kamla"), ("समझना", "samajhna"), ("करना", "karna"), ("क्या", "kya"),
    ("है", "hai"), ("हैं", "hain"), ("बात", "baat"), ("ज़रूर", "zaroor" if False else "zarur"),
    ("नहीं", "nahin"), ("कंप्यूटर", "kampyutar"), ("अच्छा", "achchha"),
])
def test_devanagari_to_roman(dev, rom):
    assert devanagari_to_roman(dev) == rom


def test_romanize_mixed_text_passthrough():
    assert devanagari_to_roman("मैं office जा रहा हूँ।") == "main office ja raha hun."


def test_splits_are_speaker_and_duplicate_safe():
    rows = []
    for spk in range(20):
        for k in range(5):
            rows.append(ManifestRow(id=f"{spk}-{k}", audio_path="", transcript=f"utt {spk} {k}",
                                    source="t", speaker_id=f"s{spk}"))
    # a duplicate sentence spoken by two different speakers links them
    rows.append(ManifestRow(id="dup", audio_path="", transcript="Utt 0 0!", source="t", speaker_id="s19"))
    assign_splits(rows, seed=1)
    spk_split = {}
    for r in rows:
        spk_split.setdefault(r.speaker_id, set()).add(r.split)
    assert all(len(v) == 1 for v in spk_split.values())
    assert spk_split["s0"] == spk_split["s19"]
    assert {r.split for r in rows} == {"train", "validation", "test"}
