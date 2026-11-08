from src.correction import CodeSwitchCorrector, CorrectionConfig
from src.preprocessing import tokenize_with_offsets


def run(text, tags, confs=None, **cfg):
    toks = tokenize_with_offsets(text)
    confs = confs or [0.95] * len(toks)
    return CodeSwitchCorrector(CorrectionConfig(**cfg)).correct(text, tags, confs, toks)


def test_canonicalizes_confident_hindi_only():
    r = run("main nhi jaunga, bohot kaam hai", ["HI", "HI", "HI", "OTH", "HI", "HI", "HI"])
    assert r.text == "main nahi jaunga, bahut kaam hai"
    assert {e.rule for e in r.edits} == {"canonicalize"}


def test_low_confidence_left_unchanged():
    r = run("main nhi jaunga", ["HI", "HI", "HI"], [0.95, 0.5, 0.95])
    assert r.text == "main nhi jaunga" and not r.edits


def test_english_tagged_token_never_canonicalized():
    # "vo" is a variant of "woh" but tagged EN here → untouched
    r = run("vo is here", ["EN", "EN", "EN"])
    assert r.text == "vo is here"


def test_named_entities_and_acronyms_protected():
    r = run("kal Ravi aur IIT mein", ["HI", "HI", "HI", "HI", "HI"], protected_words={"ravi"})
    assert r.text == "kal Ravi aur IIT mein"
    # a capitalized *known variant* mid-sentence is treated as a word, not a name
    r2 = run("kal Kese aaoge", ["HI", "HI", "HI"])
    assert r2.text == "kal Kaise aaoge"


def test_sentence_initial_capital_still_corrected():
    r = run("Bohot accha", ["HI", "HI"])
    assert r.text == "Bahut accha"


def test_loop_collapse():
    r = run("haan haan haan haan haan theek hai", ["HI"] * 7)
    assert r.text == "haan theek hai"
    r2 = run("thank you thank you thank you thank you ok", ["EN"] * 9)
    assert r2.text == "thank you ok"
    r3 = run("haan haan theek hai", ["HI"] * 4)  # natural repetition kept
    assert r3.text == "haan haan theek hai"


def test_roman_rendering_backtransliterates_english():
    text = "मेरी मीटिंग कल है"
    r = run(text, ["HI", "EN", "HI", "HI"], [0.95, 0.9, 0.95, 0.95], target_script="roman")
    assert r.text == "meri meeting kal hai"


class _FakeXlit:
    def to_native(self, w):
        return {"kal": "कल", "nahi": "नहीं"}.get(w, w)


def test_mixed_rendering_each_language_in_its_script():
    text = "मेरी मीटिंग kal nhi है"
    toks = tokenize_with_offsets(text)
    r = CodeSwitchCorrector(CorrectionConfig(target_script="mixed"), transliterator=_FakeXlit()).correct(
        text, ["HI", "EN", "HI", "HI", "HI"], [0.95] * 5, toks)
    assert r.text == "मेरी meeting कल नहीं है"


def test_roman_rendering_respects_confident_hi_over_lexicon():
    r = run("कार", ["HI"], [0.99], target_script="roman")
    assert r.text == "kaar"  # model is sure it's Hindi → romanize, don't force "car"


def test_raw_text_untouched_and_edits_logged():
    text = "nhi yaar"
    r = run(text, ["HI", "HI"])
    assert text == "nhi yaar"
    assert r.edits[0].before == "nhi" and r.edits[0].after == "nahi"


def test_no_rules_is_identity():
    text = "main nhi jaunga , okay ?"
    r = run(text, ["HI", "HI", "HI", "OTH", "EN", "OTH"], canonicalize=False, collapse_loops=False)
    assert r.text == text and not r.edits
