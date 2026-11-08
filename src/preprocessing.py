"""Text normalization, offset-preserving tokenization, script detection,
audio loading, and leakage-resistant dataset splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Sequence

TARGET_SR = 16_000

# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------

_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "tamil": (0x0B80, 0x0BFF),
}


def char_script(ch: str) -> str:
    """Coarse script class of a single character."""
    cp = ord(ch)
    for name, (lo, hi) in _SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "other_alpha"
    return "symbol"


def word_script(word: str) -> str:
    """Majority script of the alphabetic characters in ``word``.

    Returns one of ``devanagari``, ``tamil``, ``latin``, ``digit``,
    ``other_alpha``, ``symbol`` or ``mixed`` (no clear majority).
    """
    counts: dict[str, int] = {}
    for ch in word:  # Indic vowel signs/viramas fall inside their script's block
        s = char_script(ch)
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "symbol"
    letters = {k: v for k, v in counts.items() if k not in ("symbol", "digit")}
    if not letters:
        return "digit" if "digit" in counts else "symbol"
    best, n = max(letters.items(), key=lambda kv: kv[1])
    if n / sum(letters.values()) < 0.6:
        return "mixed"
    return best


# ---------------------------------------------------------------------------
# Normalization and tokenization
# ---------------------------------------------------------------------------

_ZW = re.compile("[​‌‍﻿]")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})

# A token is either a run of word characters (letters incl. Indic combining
# marks, digits, apostrophes/hyphens inside words) or a single punctuation mark.
# The danda / double danda (U+0964/5) are punctuation even though they sit in the
# Devanagari block, so they are excluded from the word class.
_WORD_CH = r"[^\W।॥]|[ऀ-ॣ०-ॿ஀-௿]"
_TOKEN_RE = re.compile(rf"(?:{_WORD_CH})+(?:['\-](?:{_WORD_CH})+)*|[^\w\s]|[।॥]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Light, meaning-preserving normalization.

    NFC unicode, unify quotes/dashes, drop zero-width characters, collapse
    whitespace. Does *not* lowercase, strip punctuation, or remove any word —
    Romanized Hindi and English tokens are preserved exactly.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZW.sub("", text)
    text = text.translate(_QUOTES)
    text = text.replace("।", " । ")  # danda as its own token
    return _WS.sub(" ", text).strip()


@dataclass
class Token:
    text: str
    start: int
    end: int
    is_punct: bool


def tokenize_with_offsets(text: str) -> list[Token]:
    """Split ``text`` into word and punctuation tokens with character offsets.

    ``text[t.start:t.end] == t.text`` holds for every token, so edits made on
    tokens can be mapped back onto the original string.
    """
    out = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        is_punct = not any(ch.isalnum() or unicodedata.category(ch).startswith("M") for ch in s)
        out.append(Token(s, m.start(), m.end(), is_punct))
    return out


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, sr: int = TARGET_SR):
    """Load an audio file as mono float32 at ``sr`` Hz. Returns (waveform, sr)."""
    import numpy as np

    try:
        import soundfile as sf

        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:  # mp3/m4a etc. fall back to librosa/audioread
        import librosa

        wav, file_sr = librosa.load(str(path), sr=None, mono=True)
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def audio_duration(wav, sr: int = TARGET_SR) -> float:
    return len(wav) / float(sr)


# ---------------------------------------------------------------------------
# Manifests and splits
# ---------------------------------------------------------------------------

@dataclass
class ManifestRow:
    id: str
    audio_path: str
    transcript: str
    source: str
    licence: str = ""
    speaker_id: str = ""
    duration: float | None = None
    split: str = ""
    extra: dict = field(default_factory=dict)


MANIFEST_FIELDS = ["id", "audio_path", "transcript", "source", "licence", "speaker_id", "duration", "split", "extra"]


def write_manifest(rows: Iterable[ManifestRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d["extra"] = json.dumps(d["extra"], ensure_ascii=False)
            w.writerow(d)


def read_manifest(path: str | Path) -> list[ManifestRow]:
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(ManifestRow(**json.loads(line)))
        return rows
    with path.open(encoding="utf-8", newline="") as f:
        for d in csv.DictReader(f):
            d["duration"] = float(d["duration"]) if d.get("duration") else None
            d["extra"] = json.loads(d["extra"]) if d.get("extra") else {}
            rows.append(ManifestRow(**d))
    return rows


def dedup_key(text: str) -> str:
    """Key under which near-duplicate sentences collide: case-, punctuation-
    and whitespace-insensitive."""
    t = normalize_text(text).lower()
    t = re.sub(r"[^\wऀ-ॿ஀-௿]+", "", t)
    # MD5 is used only as a fast grouping key, never for security.
    return hashlib.md5(t.encode("utf-8"), usedforsecurity=False).hexdigest()


def group_split(
    items: Sequence,
    group_of,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> dict[str, list]:
    """Split ``items`` into train/validation/test so that no group crosses
    splits. ``group_of(item)`` returns a hashable group (speaker id, dedup key,
    or a tuple combining them via union-find in :func:`leakage_safe_groups`).
    """
    groups: dict = {}
    for it in items:
        groups.setdefault(group_of(it), []).append(it)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(items)
    targets = [ratios[0] * n, (ratios[0] + ratios[1]) * n]
    out = {"train": [], "validation": [], "test": []}
    count = 0
    for k in keys:
        name = "train" if count < targets[0] else "validation" if count < targets[1] else "test"
        out[name].extend(groups[k])
        count += len(groups[k])
    return out


def leakage_safe_groups(rows: Sequence[ManifestRow]) -> list[str]:
    """Union speakers and near-duplicate transcripts into connected groups.

    Two rows share a group if they share a speaker id *or* a dedup key, so a
    split on these groups is both speaker-independent and duplicate-safe.
    """
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    first_seen: dict[str, int] = {}
    for i, r in enumerate(rows):
        keys = ["t:" + dedup_key(r.transcript)]
        if r.speaker_id:
            keys.append("s:" + r.speaker_id)
        for k in keys:
            if k in first_seen:
                union(i, first_seen[k])
            else:
                first_seen[k] = i
    return [f"g{find(i)}" for i in range(len(rows))]


def assign_splits(rows: list[ManifestRow], ratios=(0.8, 0.1, 0.1), seed: int = 13) -> list[ManifestRow]:
    """Set ``row.split`` in place using speaker+duplicate-safe groups."""
    groups = leakage_safe_groups(rows)
    idx = list(range(len(rows)))
    parts = group_split(idx, lambda i: groups[i], ratios, seed)
    for name, ids in parts.items():
        for i in ids:
            rows[i].split = name
    return rows
