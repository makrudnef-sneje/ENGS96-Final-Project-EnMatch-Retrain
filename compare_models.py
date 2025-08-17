#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare Baseline vs Edited by parsing their training logs to compute average of the last N 'Average Reward' lines.
Usage:
  python compare_models.py --window 50
  python compare_models.py --baseline_log runs/baseline/train.log --edited_log runs/edited/train.log --window 100
"""
from __future__ import print_function
import re, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR  = REPO_ROOT / "runs"

LINE_RE = re.compile(r"Average Reward\s*:\s*(-?\d+(?:\.\d+)?)")

def extract_scores(log_path, window):
    p = Path(log_path)
    if not p.exists():
        return None, []
    scores = []
    try:
        with p.open("r", errors="ignore") as f:
            for line in f:
                m = LINE_RE.search(line)
                if m:
                    try:
                        scores.append(float(m.group(1)))
                    except Exception:
                        pass
    except Exception:
        return None, []
    tail = scores[-window:] if window and len(scores) > window else scores
    avg = sum(tail)/len(tail) if tail else None
    return avg, tail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_log", default=str(RUNS_DIR / "baseline" / "train.log"))
    ap.add_argument("--edited_log",   default=str(RUNS_DIR / "edited" / "train.log"))
    ap.add_argument("--window", type=int, default=50, help="Use last N reward lines (default 50)")
    args = ap.parse_args()

    avg_b, tail_b = extract_scores(args.baseline_log, args.window)
    avg_e, tail_e = extract_scores(args.edited_log, args.window)

    print("Baseline log:", args.baseline_log)
    print("Edited   log:", args.edited_log)

    if avg_b is None or avg_e is None:
        print("\nCould not compute averages (logs missing or no 'Average Reward' lines).")
        print("Make sure you launched training with train_models.py and let it run long enough to log rewards.")
        raise SystemExit(2)

    print("\n============= COMPARISON =============")
    print("Window size:", args.window)
    print("Edited   avg (last N):  %.3f" % avg_e)
    print("Baseline avg (last N):  %.3f" % avg_b)
    print("Delta (Edited - Base):  %+ .3f" % (avg_e - avg_b))

if __name__ == "__main__":
    main()
