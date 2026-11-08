"""Day 2: build an audio manifest with leakage-safe splits.

From a Hugging Face dataset (check the dataset card's licence/access first):
    python scripts/build_manifest.py hf --dataset Trelis/CoSHE-500 --split train \
        --text-col transcription --speaker-col speaker_id --name coshe500 --licence "see card"

From a local folder (own consented recordings, HingCoS after access is granted):
    python scripts/build_manifest.py folder --audio-dir data/raw/own --transcripts data/raw/own/transcripts.csv \
        --name own_recordings --licence "consented, private"
  transcripts.csv columns: file,transcript[,speaker_id]
  (or put a same-named .txt next to each audio file and omit --transcripts)

HF audio is decoded and written to data/audio/<name>/ as 16 kHz mono WAV so all
downstream code reads one format. Existing 'split' columns are kept with
--keep-splits; otherwise speaker+duplicate-safe splits are assigned.
"""
import argparse
import csv
from pathlib import Path

from _common import ROOT
from src.preprocessing import ManifestRow, assign_splits, normalize_text, write_manifest

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="mode", required=True)
h = sub.add_parser("hf")
h.add_argument("--dataset", required=True)
h.add_argument("--config", default=None)
h.add_argument("--split", default="train")
h.add_argument("--audio-col", default="audio")
h.add_argument("--text-col", default="text")
h.add_argument("--speaker-col", default=None)
h.add_argument("--max-rows", type=int, default=None)
f = sub.add_parser("folder")
f.add_argument("--audio-dir", required=True)
f.add_argument("--transcripts", default=None)
for p in (h, f):
    p.add_argument("--name", required=True)
    p.add_argument("--licence", default="")
    p.add_argument("--ratios", default="0.8,0.1,0.1")
    p.add_argument("--keep-splits", action="store_true")
    p.add_argument("--all-test", action="store_true", help="mark everything as test (pure evaluation sets)")
a = ap.parse_args()

rows: list[ManifestRow] = []
if a.mode == "hf":
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(a.dataset, a.config, split=a.split)
    if a.max_rows:
        ds = ds.select(range(min(a.max_rows, len(ds))))
    print("columns:", ds.column_names)
    # decode=False → raw bytes; decode ourselves (avoids datasets' torchcodec/FFmpeg dependency)
    import io

    import librosa
    import numpy as np

    ds = ds.cast_column(a.audio_col, Audio(decode=False))
    out_dir = ROOT / "data" / "audio" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(ds):
        raw = ex[a.audio_col]
        src = io.BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]
        try:
            wav, sr = sf.read(src, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        except Exception:
            if hasattr(src, "seek"):
                src.seek(0)
            wav, sr = librosa.load(src, sr=None, mono=True)
        if sr != 16_000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16_000)
        audio = {"array": np.asarray(wav, dtype=np.float32)}
        path = out_dir / f"{a.name}_{i:06d}.wav"
        sf.write(path, audio["array"], 16_000)
        rows.append(ManifestRow(id=f"{a.name}_{i:06d}", audio_path=str(path.relative_to(ROOT)).replace("\\", "/"),
                                transcript=normalize_text(str(ex[a.text_col])), source=a.dataset, licence=a.licence,
                                speaker_id=str(ex[a.speaker_col]) if a.speaker_col else "",
                                duration=round(len(audio["array"]) / 16_000, 3),
                                split=str(ex.get("split", "")) if a.keep_splits else ""))
else:
    adir = Path(a.audio_dir)
    table = {}
    if a.transcripts:
        with open(a.transcripts, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                table[Path(r["file"]).name] = (r["transcript"], r.get("speaker_id", ""))
    for p in sorted(adir.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        if a.transcripts:
            if p.name not in table:
                print("no transcript:", p.name); continue
            text, spk = table[p.name]
        else:
            txt = p.with_suffix(".txt")
            if not txt.exists():
                print("no transcript:", p.name); continue
            text, spk = txt.read_text(encoding="utf-8"), ""
        try:
            rel = str(p.resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            rel = str(p.resolve())
        rows.append(ManifestRow(id=f"{a.name}_{p.stem}", audio_path=rel, transcript=normalize_text(text),
                                source=a.name, licence=a.licence, speaker_id=spk))

if a.all_test:
    for r in rows:
        r.split = "test"
elif not (a.keep_splits and all(r.split for r in rows)):
    assign_splits(rows, tuple(float(x) for x in a.ratios.split(",")))
out = ROOT / "data" / "manifests" / f"{a.name}.csv"
write_manifest(rows, out)
from collections import Counter
print(f"wrote {out}: {len(rows)} rows", dict(Counter(r.split for r in rows)),
      "speakers:", len({r.speaker_id for r in rows if r.speaker_id}) or "n/a")
