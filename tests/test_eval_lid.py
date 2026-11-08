import pytest

from src.data_sources import comi_row_to_example, parse_comi_annotation
from src.evaluation import boundary_f1, corpus_wer, lid_metrics, scoring_normalize, switch_density, wer_counts
from src.language_identification import NgramLID, ScriptRuleLID, align_labels

from .fixtures import rows


def test_wer_basic():
    c = wer_counts("main kal office jaunga", "main office jaunga kal")
    assert (c.substitutions + c.deletions + c.insertions) == 2 and c.ref_len == 4


def test_fast_and_python_edit_ops_agree():
    import random

    from src.evaluation import _edit_ops, _edit_ops_py

    rng = random.Random(0)
    for _ in range(200):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        x, y = _edit_ops(a, b), _edit_ops_py(a, b)
        assert sum(x[1:]) == sum(y[1:])  # same edit distance
        assert x[0] + x[1] + x[2] == len(a) and y[0] + y[1] + y[2] == len(a)


def test_scoring_normalize_and_script_agnostic():
    assert scoring_normalize("Hello, World!  don't") == "hello world don't"
    assert corpus_wer(["main kal jaunga"], ["मैं कल जाऊँगा"])["wer"] == 1.0
    assert corpus_wer(["kal hai"], ["कल है"], romanize=True)["wer"] == 0.0


def test_matches_jiwer():
    jiwer = pytest.importorskip("jiwer")
    refs = ["main kal office nahi ja raha", "this is a test", "yaar bahut accha"]
    hyps = ["main office nahi ja rahi", "this is test of", "yaar bohot acha hai"]
    ours = corpus_wer(refs, hyps)
    assert ours["wer"] == pytest.approx(jiwer.wer(refs, hyps))
    assert ours["cer"] == pytest.approx(jiwer.cer([r.replace(" ", "") for r in refs],
                                                  [h.replace(" ", "") for h in hyps]))


def test_boundary_f1():
    gold = [["HI", "HI", "EN", "OTH", "EN", "HI"]]
    assert boundary_f1(gold, gold)["f1"] == 1.0
    pred = [["HI", "EN", "EN", "OTH", "EN", "HI"]]
    r = boundary_f1(gold, pred)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1
    assert switch_density(["HI", "EN", "HI", "OTH"]) == 1.0


def test_lid_metrics_shape():
    m = lid_metrics([["HI", "EN", "OTH"]], [["HI", "HI", "OTH"]])
    assert m["per_class"]["HI"]["recall"] == 1.0 and m["per_class"]["EN"]["recall"] == 0.0
    assert len(m["confusion_matrix"]) == 3


def test_align_labels_first_subword():
    assert align_labels([None, 0, 0, 1, 2, 2, None], [0, 1, 2]) == [-100, 0, -100, 1, 2, -100, -100]


def test_ngram_lid_learns_and_pickles(tmp_path):
    data = rows() * 3
    m = NgramLID().fit(data)
    tags, confs = m.predict("yaar mujhe call karo".split())
    assert tags[0] == "HI" and tags[2] == "EN" and len(confs) == 4
    assert m.predict(["?"])[0] == ["OTH"]
    p = tmp_path / "m.pkl"
    m.save(p)
    assert NgramLID.load(p).predict(["meeting"])[0] == m.predict(["meeting"])[0]


def test_ngram_load_refuses_tampered_or_unsigned(tmp_path):
    m = NgramLID().fit(rows())
    p = tmp_path / "m.pkl"
    m.save(p)
    p.write_bytes(p.read_bytes() + b"x")  # tampered
    with pytest.raises(ValueError, match="checksum mismatch"):
        NgramLID.load(p)
    (tmp_path / "m.pkl.sha256").unlink()  # unsigned
    with pytest.raises(ValueError, match="missing checksum"):
        NgramLID.load(p)


def test_rule_lid():
    assert ScriptRuleLID().predict(["नहीं", "office", "5"])[0] == ["HI", "EN", "OTH"]


def test_parse_comi_formats():
    a = "[{'key': 'mera', 'value': 'hi', 'index': 0}, {'key': 'usb', 'value': 'en', 'index': 1}, {'key': '!', 'value': 'ot', 'index': 2}]"
    assert parse_comi_annotation(a) == [("mera", "HI"), ("usb", "EN"), ("!", "OTH")]
    b = "[['mera', 'h'], ['driver', 'e']]"
    assert parse_comi_annotation(b) == [("mera", "HI"), ("driver", "EN")]
    assert parse_comi_annotation("not a list") is None


def test_comi_majority_vote():
    row = {"Sentences": "mera usb",
           "Annotated by: Annotator 1": "[['mera','h'],['usb','e']]",
           "Annotated by: Annotator 2": "[['mera','h'],['usb','h']]",
           "Annotated by: Annotator 3": "[['mera','h'],['usb','e']]"}
    ex = comi_row_to_example(row)
    assert ex["tokens"] == ["mera", "usb"] and ex["tags"] == ["HI", "EN"] and ex["n_annotators"] == 3


def test_comi_retokenizes_glued_punctuation():
    row = {"Annotated by: Annotator 1": "[['(Neha','en'],['Kakkar)','en'],['था।','hi']]"}
    ex = comi_row_to_example(row)
    assert ex["tokens"] == ["(", "Neha", "Kakkar", ")", "था", "।"]
    assert ex["tags"] == ["OTH", "EN", "EN", "OTH", "HI", "OTH"]
