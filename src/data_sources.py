"""Dataset adapters.

COMI-LINGUA (LingoIITGN/COMI-LINGUA, config ``LID``, CC-BY-4.0): columns
``Sentences``, ``Predicted tags`` and ``Annotated by: Annotator 1..3``. Token
annotations are stored as *strings* holding either a list of
``{'key': tok, 'value': 'hi'|'en'|'ot', 'index': i}`` dicts or a list of
``[tok, 'h'|'e'|'o']`` pairs. We parse both, use only the human annotator
columns (never ``Predicted tags``), and take a per-token majority vote over
annotators whose tokenization agrees. Inspect the dataset card before use; the
schema here was taken from it and is validated at load time.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any

from .preprocessing import dedup_key, group_split, tokenize_with_offsets

COMI_LABEL_MAP = {"hi": "HI", "h": "HI", "hin": "HI", "en": "EN", "e": "EN", "eng": "EN"}
ANNOTATOR_COLS = ("Annotated by: Annotator 1", "Annotated by: Annotator 2", "Annotated by: Annotator 3")


def map_comi_label(v: str) -> str:
    return COMI_LABEL_MAP.get(str(v).strip().lower(), "OTH")


def parse_comi_annotation(value: Any) -> list[tuple[str, str]] | None:
    """Parse one annotator cell into ``[(token, LABEL), ...]`` or None."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            value = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(s)
            except json.JSONDecodeError:
                return None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for item in value:
        if isinstance(item, dict) and "key" in item:
            out.append((item, int(item.get("index", len(out)))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append(({"key": item[0], "value": item[1]}, len(out)))
        else:
            return None
    out.sort(key=lambda x: x[1])
    return [(str(d["key"]), map_comi_label(d.get("value", "ot"))) for d, _ in out]


def comi_row_to_example(row: dict, retokenize: bool = True) -> dict | None:
    """Majority vote over annotators with identical token sequences."""
    parsed = [p for p in (parse_comi_annotation(row.get(c)) for c in ANNOTATOR_COLS) if p]
    if not parsed:
        return None
    by_tokens = Counter(tuple(t for t, _ in p) for p in parsed)
    tokens, _ = by_tokens.most_common(1)[0]
    agreeing = [p for p in parsed if tuple(t for t, _ in p) == tokens]
    tags = []
    for k in range(len(tokens)):
        votes = Counter(p[k][1] for p in agreeing)
        top = votes.most_common()
        # ties → prefer the first annotator's label
        tags.append(top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else agreeing[0][k][1])
    if not tokens:
        return None
    if retokenize:
        tokens, tags = retokenize_example(tokens, tags)
    return {"tokens": list(tokens), "tags": tags, "n_annotators": len(agreeing),
            "sentence": row.get("Sentences", " ".join(tokens))}


def retokenize_example(tokens, tags):
    """Re-split annotated tokens with the pipeline tokenizer so training matches
    inference: ``"(Neha"`` → ``"(" OTH, "Neha" <tag>``; ``"था।"`` → ``"था" <tag>, "।" OTH``."""
    out_t, out_y = [], []
    for w, y in zip(tokens, tags, strict=True):
        pieces = tokenize_with_offsets(w)
        if not pieces:
            continue
        for p in pieces:
            out_t.append(p.text)
            out_y.append("OTH" if p.is_punct else y)
    return out_t, out_y


def load_comi_lingua_lid(max_rows: int | None = None, cache_dir: str | None = None,
                         val_ratio: float = 0.1, seed: int = 13) -> tuple[dict[str, list[dict]], dict]:
    """Load COMI-LINGUA LID → {"train","validation","test"} lists of examples.

    Validation is carved from the official train split, grouping near-duplicate
    sentences so they never straddle train/validation. Sentences that
    duplicate a test sentence are removed from train/validation (leakage).
    """
    from datasets import load_dataset

    ds = load_dataset("LingoIITGN/COMI-LINGUA", "LID", cache_dir=cache_dir)
    stats = {"source": "LingoIITGN/COMI-LINGUA (LID)", "licence": "CC-BY-4.0", "skipped": {}}
    out = {}
    for split in ("train", "test"):
        rows = ds[split]
        if max_rows:
            rows = rows.select(range(min(max_rows, len(rows))))
        exs, skipped = [], 0
        for r in rows:
            ex = comi_row_to_example(r)
            if ex is None:
                skipped += 1
            else:
                exs.append(ex)
        out[split] = exs
        stats["skipped"][split] = skipped

    test_keys = {dedup_key(" ".join(e["tokens"])) for e in out["test"]}
    before = len(out["train"])
    train_pool = [e for e in out["train"] if dedup_key(" ".join(e["tokens"])) not in test_keys]
    stats["removed_train_test_duplicates"] = before - len(train_pool)
    parts = group_split(train_pool, lambda e: dedup_key(" ".join(e["tokens"])),
                        ratios=(1 - val_ratio, val_ratio, 0.0), seed=seed)
    result = {"train": parts["train"], "validation": parts["validation"] + parts["test"], "test": out["test"]}
    stats["sizes"] = {k: len(v) for k, v in result.items()}
    stats["tag_distribution"] = {k: dict(Counter(t for e in v for t in e["tags"])) for k, v in result.items()}
    return result, stats
