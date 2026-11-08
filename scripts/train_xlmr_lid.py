"""Day 3b: fine-tune FacebookAI/xlm-roberta-base for token LID (Colab GPU).

    python scripts/train_xlmr_lid.py --batch-size 4 --grad-accum 4 --max-length 128
    # OOM fallback: --batch-size 4 --max-length 96
"""
import argparse

from _common import ROOT, dump_json
from src.language_identification import XLMRTrainConfig, read_lid_jsonl, train_xlmr_lid

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
ap.add_argument("--out-dir", default=str(ROOT / "models" / "lid_model"))
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--grad-accum", type=int, default=4)
ap.add_argument("--max-length", type=int, default=128)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--no-fp16", action="store_true")
ap.add_argument("--max-train", type=int, default=None, help="subsample training rows (quick runs)")
ap.add_argument("--eval-steps", type=int, default=None)
a = ap.parse_args()

train = read_lid_jsonl(f"{a.data_dir}/lid_train.jsonl")
val = read_lid_jsonl(f"{a.data_dir}/lid_validation.jsonl")
if a.max_train:
    import random

    random.Random(13).shuffle(train)
    train = train[:a.max_train]
cfg = XLMRTrainConfig(batch_size=a.batch_size, grad_accum=a.grad_accum, max_length=a.max_length,
                      epochs=a.epochs, learning_rate=a.lr, fp16=not a.no_fp16, eval_steps=a.eval_steps)
metrics = train_xlmr_lid(train, val, a.out_dir, cfg)
metrics["n_train"] = len(train)
dump_json(metrics, ROOT / "experiments" / "lid_xlmr_validation.json")
print(metrics)
