"""Day 2: COMI-LINGUA LID → data/processed/lid_{train,validation,test}.jsonl + stats.

    python scripts/prepare_comi_lingua.py [--max-rows 5000]
"""
import argparse

from _common import ROOT, dump_json, write_jsonl
from src.data_sources import load_comi_lingua_lid

ap = argparse.ArgumentParser()
ap.add_argument("--max-rows", type=int, default=None, help="cap rows per official split (quick runs)")
ap.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
ap.add_argument("--seed", type=int, default=13)
a = ap.parse_args()

splits, stats = load_comi_lingua_lid(max_rows=a.max_rows, seed=a.seed)
for name, rows in splits.items():
    write_jsonl([{"tokens": r["tokens"], "tags": r["tags"]} for r in rows], f"{a.out_dir}/lid_{name}.jsonl")
stats["max_rows"] = a.max_rows
stats["seed"] = a.seed
dump_json(stats, f"{a.out_dir}/lid_stats.json")
print(stats["sizes"], stats["tag_distribution"]["train"])
