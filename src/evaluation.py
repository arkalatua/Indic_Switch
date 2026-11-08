"""Metrics: WER, CER, LID macro-F1 / per-class / confusion, language-boundary
F1, switch density, real-time factor and peak memory.

WER/CER use a self-contained Levenshtein alignment (identical definition to
jiwer: (S + D + I) / N) so results do not depend on jiwer's API version; the
test suite cross-checks against jiwer when it is installed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Sequence

from .transliteration import devanagari_to_roman

# ---------------------------------------------------------------------------
# Scoring normalization (fixed and documented; applied identically to every system)
# ---------------------------------------------------------------------------


def scoring_normalize(text: str, romanize: bool = False) -> str:
    """Reference/hypothesis normalization for WER/CER.

    NFC → (optional) Devanagari→Roman → lowercase → strip punctuation (keeping
    intra-word apostrophes) → collapse whitespace. ``romanize=True`` gives a
    *script-agnostic* score: a Hindi word written in Devanagari by Whisper and
    in Roman by the reference is not counted as an error purely for script.
    """
    text = unicodedata.normalize("NFC", text)
    if romanize:
        text = devanagari_to_roman(text)
    text = text.lower().replace("।", " ")
    text = re.sub(r"[^\w\s'ऀ-ॿ஀-௿]", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_ops(ref: Sequence, hyp: Sequence) -> tuple[int, int, int, int]:
    """Return (hits, substitutions, deletions, insertions) of the min-cost alignment.

    Uses rapidfuzz (C++, installed with jiwer) when available — essential for
    CER on long hallucinated outputs — else a pure-Python DP.
    """
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        return _edit_ops_py(ref, hyp)
    s = d = i = 0
    for op in Levenshtein.editops(list(ref), list(hyp)):
        if op.tag == "replace":
            s += 1
        elif op.tag == "delete":
            d += 1
        else:
            i += 1
    return len(ref) - s - d, s, d, i


def _edit_ops_py(ref: Sequence, hyp: Sequence) -> tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    # dp rows hold (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(sub[0], dele[0], ins[0])
            if sub[0] == best:
                cur[j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
            elif dele[0] == best:
                cur[j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
            else:
                cur[j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
        prev = cur
    _, s, d, ins = prev[m]
    return n - s - d, s, d, ins


@dataclass
class ErrorCounts:
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0

    def __iadd__(self, o):
        for k in asdict(self):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self

    @property
    def rate(self) -> float:
        errs = self.substitutions + self.deletions + self.insertions
        return errs / self.ref_len if self.ref_len else float(errs > 0)


def _counts(ref_units, hyp_units) -> ErrorCounts:
    h, s, d, i = _edit_ops(ref_units, hyp_units)
    return ErrorCounts(h, s, d, i, len(ref_units))


def wer_counts(ref: str, hyp: str, romanize: bool = False) -> ErrorCounts:
    return _counts(scoring_normalize(ref, romanize).split(), scoring_normalize(hyp, romanize).split())


def cer_counts(ref: str, hyp: str, romanize: bool = False) -> ErrorCounts:
    r = scoring_normalize(ref, romanize).replace(" ", "")
    h = scoring_normalize(hyp, romanize).replace(" ", "")
    return _counts(list(r), list(h))


def corpus_wer(refs: Sequence[str], hyps: Sequence[str], romanize: bool = False) -> dict:
    """Corpus-level (micro-averaged) WER and CER with error breakdown."""
    w, c = ErrorCounts(), ErrorCounts()
    for r, h in zip(refs, hyps, strict=True):
        w += wer_counts(r, h, romanize)
        c += cer_counts(r, h, romanize)
    return {"wer": w.rate, "cer": c.rate, "word_counts": asdict(w), "char_counts": asdict(c), "n_utts": len(refs)}


# ---------------------------------------------------------------------------
# LID metrics
# ---------------------------------------------------------------------------

LABELS = ["HI", "EN", "OTH"]


def lid_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], labels=LABELS) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    y = [t for s in gold for t in s]
    yh = [t for s in pred for t in s]
    if len(y) != len(yh):
        raise ValueError("gold/pred token counts differ")
    p, r, f, sup = precision_recall_fscore_support(y, yh, labels=labels, zero_division=0)
    per_class = {l: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(sup[k])}
                 for k, l in enumerate(labels)}
    present = [l for l in labels if per_class[l]["support"] > 0]
    macro = sum(per_class[l]["f1"] for l in present) / max(len(present), 1)
    return {"accuracy": float(accuracy_score(y, yh)), "macro_f1": macro, "macro_f1_labels": present,
            "per_class": per_class, "labels": list(labels),
            "confusion_matrix": confusion_matrix(y, yh, labels=labels).tolist()}


def boundaries(tags: Sequence[str], ignore: Sequence[str] = ("OTH",)) -> set[int]:
    """Positions ``i`` (in the original indexing) where a language switch occurs
    between token ``i`` and the next non-ignored token."""
    kept = [(i, t) for i, t in enumerate(tags) if t not in ignore]
    return {kept[k][0] for k in range(len(kept) - 1) if kept[k][1] != kept[k + 1][1]}


def boundary_f1(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], ignore=("OTH",)) -> dict:
    """Boundary precision/recall/F1. Boundaries are computed after removing
    ``ignore`` tokens (by default OTH, so punctuation doesn't create switches).
    Gold-ignored positions are used for both sides so indices stay comparable."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred, strict=True):
        keep_mask = [t not in ignore for t in g]
        # predicted OTH where gold has a language becomes its own label (penalized)
        p_masked = ["__IGN__" if not keep else (t if t not in ignore else "__PRED_OTH__")
                    for t, keep in zip(p, keep_mask, strict=True)]
        gb = boundaries(g, ignore)
        pb = boundaries(p_masked, ("__IGN__",))
        tp += len(gb & pb); fp += len(pb - gb); fn += len(gb - pb)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def switch_density(tags: Sequence[str]) -> float:
    lang = [t for t in tags if t != "OTH"]
    if len(lang) < 2:
        return 0.0
    return len(boundaries(lang)) / (len(lang) - 1)


def density_bucket(d: float) -> str:
    return "none" if d == 0 else "low" if d < 0.15 else "medium" if d < 0.35 else "high"


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_memory() -> dict:
    """Peak GPU memory (torch allocator) and current process RSS."""
    out = {"device": "cpu", "peak_gpu_mem_gb": None, "rss_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            out["device"] = torch.cuda.get_device_name(0)
            out["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except ImportError:
        pass
    try:
        import psutil

        out["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)
    except ImportError:
        pass
    return out
