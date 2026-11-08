"""Evaluate any LID model on a split: metrics, boundary F1, efficiency, and an
error sample (named entities / ambiguous Romanized words / boundaries).

    python scripts/evaluate_lid.py --model models/ngram_lid.pkl --split test
    python scripts/evaluate_lid.py --model models/lid_model --split test
    python scripts/evaluate_lid.py --model rule --split test
"""
import argparse
import time
from collections import Counter

from _common import ROOT, dump_json
from src.evaluation import boundaries, boundary_f1, lid_metrics, peak_memory, reset_peak_memory
from src.language_identification import load_lid, read_lid_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
ap.add_argument("--n-errors", type=int, default=50)
a = ap.parse_args()

rows = read_lid_jsonl(f"{a.data_dir}/lid_{a.split}.jsonl")
lid = load_lid(a.model)
reset_peak_memory()
t = time.perf_counter()
preds = lid.predict_batch([r["tokens"] for r in rows])
secs = time.perf_counter() - t
gold = [r["tags"] for r in rows]
pred = [p[0] for p in preds]

confused = Counter()
examples = []
for r, (ptags, pconf) in zip(rows, preds, strict=True):
    gb = boundaries(r["tags"])
    for i, (w, g, p) in enumerate(zip(r["tokens"], r["tags"], ptags, strict=True)):
        if g != p:
            confused[(w.lower(), g, p)] += 1
            if len(examples) < a.n_errors:
                examples.append({"word": w, "gold": g, "pred": p, "conf": round(pconf[i], 3),
                                 "near_boundary": (i in gb) or (i - 1 in gb),
                                 "capitalized": w[:1].isupper(),
                                 "context": " ".join(r["tokens"][max(0, i - 4):i + 5])})
n_tok = sum(len(g) for g in gold)
report = {"model": lid.name, "model_path": a.model, "split": a.split, "n_sentences": len(rows), "n_tokens": n_tok,
          "lid": lid_metrics(gold, pred), "boundary": boundary_f1(gold, pred),
          "efficiency": {"seconds": secs, "tokens_per_second": n_tok / secs if secs else None, **peak_memory()},
          "top_confusions": [{"word": w, "gold": g, "pred": p, "count": c} for (w, g, p), c in confused.most_common(40)],
          "error_examples": examples}
dump_json(report, ROOT / "experiments" / f"lid_{lid.name}_{a.split}.json")
print(f"{lid.name} [{a.split}] macro-F1 {report['lid']['macro_f1']:.4f}  acc {report['lid']['accuracy']:.4f}  "
      f"boundary-F1 {report['boundary']['f1']:.4f}  {report['efficiency']['tokens_per_second']:.0f} tok/s")
