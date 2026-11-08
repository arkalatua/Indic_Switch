"""Day 6: experiment matrix + ablations on cached ASR predictions.

For every ASR prediction file given, evaluates:
  raw                                  (E1/E2: Whisper alone)
  <lid> + full correction              (E4 with XLM-R, E5 with n-gram)
  ablations: −canonicalize, −loop-collapse, script=roman, script=native (E6)
and reports WER/CER (exact and script-agnostic), switch-density breakdown,
#edits, #utterances improved / degraded, and post-processing latency.

    python scripts/run_experiments.py experiments/asr/*.jsonl \
        --lid models/lid_model --lid models/ngram_lid.pkl --lid rule
"""
import argparse
import csv
import time
from pathlib import Path

from _common import ROOT, dump_json, read_jsonl
from src.correction import CorrectionConfig
from src.evaluation import corpus_wer, density_bucket, switch_density, wer_counts
from src.language_identification import load_lid
from src.pipeline import IndicSwitch

ap = argparse.ArgumentParser()
ap.add_argument("predictions", nargs="+")
ap.add_argument("--lid", action="append", default=None, help="LID model spec(s); repeatable")
ap.add_argument("--min-conf", type=float, default=0.80)
ap.add_argument("--out-dir", default=str(ROOT / "experiments"))
ap.add_argument("--n-examples", type=int, default=15)
a = ap.parse_args()
lids = a.lid or ["rule"]

VARIANTS = {
    "full": dict(),
    "-canonicalize": dict(canonicalize=False),
    "-loop_collapse": dict(collapse_loops=False),
    "script=roman": dict(target_script="roman"),
    "script=native": dict(target_script="native"),
    "script=mixed": dict(target_script="mixed"),
}


def score(refs, hyps):
    exact, agn = corpus_wer(refs, hyps), corpus_wer(refs, hyps, romanize=True)
    return {"wer": exact["wer"], "cer": exact["cer"], "wer_script_agnostic": agn["wer"], "cer_script_agnostic": agn["cer"]}


rows_out, details = [], {}
for pred_path in a.predictions:
    preds = read_jsonl(pred_path)
    asr_name = Path(pred_path).stem
    refs = [p["reference"] for p in preds]
    raw = [p["hypothesis"] for p in preds]
    rows_out.append({"asr": asr_name, "system": "raw", "lid": "-", **score(refs, raw), "edits": 0,
                     "improved": 0, "degraded": 0, "post_ms_per_utt": 0.0})
    loaded_lids = {spec: load_lid(spec) for spec in lids}
    for spec, lid in loaded_lids.items():
        # density buckets come from the LID tags on the reference text
        ref_tags = lid.predict_batch([r.split() for r in refs])
        buckets = [density_bucket(switch_density(t)) for t, _ in ref_tags]
        for vname, overrides in VARIANTS.items():
            cfg = CorrectionConfig(min_confidence=a.min_conf, **overrides)
            pipe = IndicSwitch(lid=lid, correction=cfg)
            t = time.perf_counter()
            outs = [pipe.process_text(h) for h in raw]
            ms = 1000 * (time.perf_counter() - t) / max(len(raw), 1)
            hyps = [o["normalized_transcript"] for o in outs]
            improved = degraded = 0
            changed = []
            for r, h0, h1, o in zip(refs, raw, hyps, outs, strict=True):
                e0, e1 = wer_counts(r, h0, romanize=True).rate, wer_counts(r, h1, romanize=True).rate
                improved += e1 < e0; degraded += e1 > e0
                if h0 != h1 and len(changed) < a.n_examples:
                    changed.append({"ref": r, "raw": h0, "normalized": h1, "delta_wer": e1 - e0, "edits": o["edits"]})
            by_bucket = {}
            for b in ("none", "low", "medium", "high"):
                idx = [i for i, x in enumerate(buckets) if x == b]
                if idx:
                    by_bucket[b] = {"n": len(idx), "raw": score([refs[i] for i in idx], [raw[i] for i in idx]),
                                    "system": score([refs[i] for i in idx], [hyps[i] for i in idx])}
            row = {"asr": asr_name, "system": f"{lid.name}+{vname}", "lid": spec, **score(refs, hyps),
                   "edits": sum(len(o["edits"]) for o in outs), "improved": improved, "degraded": degraded,
                   "post_ms_per_utt": round(ms, 3)}
            rows_out.append(row)
            details[f"{asr_name}|{row['system']}"] = {"row": row, "by_switch_density": by_bucket, "examples": changed}
            print(f"{asr_name:40s} {row['system']:28s} WER {row['wer']:.4f}  WER-agn {row['wer_script_agnostic']:.4f}  "
                  f"+{improved}/-{degraded}")

out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader(); w.writerows(rows_out)
print(f"wrote {out / 'results.csv'}")
dump_json(details, out / "ablations.json")
