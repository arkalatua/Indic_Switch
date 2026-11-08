"""Days 1/6: run one Whisper decoding configuration over a manifest split and
cache raw predictions (E1/E2). Post-processing experiments read these files, so
Whisper only runs once per decoding config.

    python scripts/run_asr.py --manifest data/manifests/coshe500.csv --split test            # E1 auto
    python scripts/run_asr.py --manifest ... --language hindi --beams 3                        # E2
    python scripts/run_asr.py --manifest ... --language english
    python scripts/run_asr.py --manifest ... --prompt "Hinglish: main office ja raha hoon, meeting kal hai."
"""
import argparse
import re
from pathlib import Path

from _common import ROOT, dump_json, write_jsonl
from src.asr import DecodeConfig, WhisperASR
from src.evaluation import corpus_wer, peak_memory, reset_peak_memory
from src.preprocessing import read_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--model", default="openai/whisper-base")
ap.add_argument("--language", default=None)
ap.add_argument("--beams", type=int, default=1)
ap.add_argument("--prompt", default=None)
ap.add_argument("--vad", action="store_true")
ap.add_argument("--suppress-script", action="append", default=[], help="e.g. arabic (blocks Urdu-script output)")
ap.add_argument("--fallback", action="store_true", help="temperature fallback on repetitive/low-confidence output")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "asr"))
a = ap.parse_args()

rows = [r for r in read_manifest(a.manifest) if not a.split or r.split == a.split]
if a.limit:
    rows = rows[:a.limit]
cfg = DecodeConfig(model_name=a.model, language=a.language, num_beams=a.beams, prompt=a.prompt, use_vad=a.vad,
                   suppress_scripts=tuple(a.suppress_script), fallback=a.fallback)
asr = WhisperASR(cfg)
reset_peak_memory()
preds = []
for k, r in enumerate(rows):
    path = r.audio_path if Path(r.audio_path).is_absolute() else str(ROOT / r.audio_path)
    res = asr.transcribe(path)
    preds.append({"id": r.id, "reference": r.transcript, "hypothesis": res.text, "segments": res.segments,
                  "audio_seconds": res.audio_seconds, "elapsed_seconds": res.elapsed_seconds,
                  "speaker_id": r.speaker_id})
    if k < 3 or k % 25 == 0:
        print(f"[{k + 1}/{len(rows)}] REF: {r.transcript}\n           HYP: {res.text}")

tag = re.sub(r"[^\w=,.-]", "_", cfg.tag())
stem = f"{Path(a.manifest).stem}_{a.split}_{tag}"
write_jsonl(preds, f"{a.out_dir}/{stem}.jsonl")
audio_s = sum(p["audio_seconds"] for p in preds)
proc_s = sum(p["elapsed_seconds"] for p in preds)
summary = {"config": cfg.__dict__, "tag": cfg.tag(), "manifest": a.manifest, "split": a.split, "n": len(preds),
           "audio_seconds": audio_s, "processing_seconds": proc_s, "rtf_excl_model_load": proc_s / audio_s if audio_s else None,
           "scores": corpus_wer([p["reference"] for p in preds], [p["hypothesis"] for p in preds]),
           "scores_script_agnostic": corpus_wer([p["reference"] for p in preds], [p["hypothesis"] for p in preds], romanize=True),
           **peak_memory()}
dump_json(summary, f"{a.out_dir}/{stem}.summary.json")
print(f"WER {summary['scores']['wer']:.4f}  CER {summary['scores']['cer']:.4f}  "
      f"script-agnostic WER {summary['scores_script_agnostic']['wer']:.4f}  RTF {summary['rtf_excl_model_load']:.3f}")
