#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator for VSI‑Bench (CVPR 2025).
Usage:
    python eval_vsibench.py \
        --pred your_model_vsibench_pred.json \
        --out  report.json
"""

import argparse, json, re
from collections import defaultdict
from datasets import load_dataset

# -------------------------------------------------------------------
# 1.  MRA IMPLEMENTATION —— thresholds 0.5 … 0.95  (10 bins)
# -------------------------------------------------------------------
THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]          # 0.50 … 0.95

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")   # Extract first float number
def _to_float(x):
    m = _num_re.search(str(x))
    return float(m.group()) if m else None

def mra(pred, tgt):
    p, g = _to_float(pred), _to_float(tgt)
    if p is None or g is None:         # Cannot parse → 0
        return 0.0
    if abs(g) < 1e-8:                  # Division by zero protection
        return 1.0 if abs(p - g) < 1e-8 else 0.0
    rel_err = abs(p - g) / abs(g)
    return sum(rel_err < (1 - th) for th in THRESHOLDS) / len(THRESHOLDS)

# Four types of numeric questions —— names from official parquet `question_type`
NUMERIC_QTYPES = {
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
    "object_abs_distance",
}

# -------------------------------------------------------------------
# 2.  MAIN
# -------------------------------------------------------------------
def main(pred, out):
    vsi = load_dataset("VSI-Bench")["test"]    # :contentReference[oaicite:1]{index=1}
    gold = {int(row["id"]): row for row in vsi if row["question_type"] == "object_counting"}
    preds = {int(d["idx"]): d["prediction"] for d in json.load(open(pred))}

    # Completeness check -----------------------------------------------------
    miss = set(gold) - set(preds)
    extra = set(preds) - set(gold)
    # if miss:
    #     raise ValueError(f"Missing {len(miss)} idx, e.g. {next(iter(miss))}")
    # if extra:
    #     print(f"[Warn] Ignore {len(extra)} unknown idx (e.g. {next(iter(extra))}).")

    # Accumulate scores by task -----------------------------------------------
    stat = defaultdict(lambda: {"numer": 0.0, "denom": 0})
    for idx, row in gold.items():
        if idx>10:
            break
        qtype, gt, pred = row["question_type"], row["ground_truth"], preds[idx]

        if qtype in NUMERIC_QTYPES:
            score = mra(pred, gt)
        else:                                              # MCA
            score = float(str(pred).strip().lower() ==
                           str(gt).strip().lower())
        s = stat[qtype]
        s["numer"] += score
        s["denom"] += 1

    # Summary ----------------------------------------------------------
    report = {task: round(rec["numer"] / rec["denom"], 4)
              for task, rec in stat.items()}
    report["overall"] = round(
        sum(r["numer"] for r in stat.values()) /
        sum(r["denom"] for r in stat.values()), 4)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if out:
        json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)

# -------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Prediction results JSON file")
    ap.add_argument("--out",  default=None, help="Output path for evaluation report")
    main(**vars(ap.parse_args()))
