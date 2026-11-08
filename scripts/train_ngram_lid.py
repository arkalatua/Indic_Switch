"""Day 3a: char n-gram + Logistic Regression LID baseline (CPU, minutes).

    python scripts/train_ngram_lid.py
"""
import argparse
import time

from _common import ROOT, dump_json
from src.evaluation import boundary_f1, lid_metrics
from src.language_identification import NgramLID, read_lid_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
ap.add_argument("--out", default=str(ROOT / "models" / "ngram_lid.pkl"))
ap.add_argument("--C", type=float, default=4.0)
a = ap.parse_args()

train = read_lid_jsonl(f"{a.data_dir}/lid_train.jsonl")
val = read_lid_jsonl(f"{a.data_dir}/lid_validation.jsonl")
t = time.perf_counter()
model = NgramLID(C=a.C).fit(train)
train_s = time.perf_counter() - t
model.save(a.out)

t = time.perf_counter()
preds = model.predict_batch([r["tokens"] for r in val])
infer_s = time.perf_counter() - t
gold = [r["tags"] for r in val]
pred = [p[0] for p in preds]
report = {"model": model.name, "C": a.C, "split": "validation", "train_seconds": train_s,
          "inference_seconds": infer_s, "n_tokens": sum(len(g) for g in gold),
          "lid": lid_metrics(gold, pred), "boundary": boundary_f1(gold, pred)}
dump_json(report, ROOT / "experiments" / "lid_ngram_validation.json")
print(f"val macro-F1 {report['lid']['macro_f1']:.4f}  acc {report['lid']['accuracy']:.4f}  "
      f"boundary-F1 {report['boundary']['f1']:.4f}")
