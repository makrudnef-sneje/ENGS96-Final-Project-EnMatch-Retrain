import os
import re
import csv
import glob
import argparse
from statistics import mean
from typing import Optional, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_rewards_from_csv(csv_path: str) -> List[float]:
    rewards = []
    with open(csv_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        key = None
        if reader.fieldnames:
            # accept 'reward' or 'Reward'
            if "reward" in reader.fieldnames:
                key = "reward"
            elif "Reward" in reader.fieldnames:
                key = "Reward"
        if key is None:
            return rewards
        for row in reader:
            try:
                rewards.append(float(row[key]))
            except (TypeError, ValueError, KeyError):
                continue
    return rewards

def avg_last_n(values: List[float], n: int) -> Optional[float]:
    if not values:
        return None
    if n <= 0 or n >= len(values):
        return mean(values)
    return mean(values[-n:])

def parse_log_index(path: str) -> Optional[int]:
    """
    Extract the trailing integer from filenames like PPO_SimpleMatch-v0_log_17.csv
    """
    m = re.search(r"_log_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None

def sort_csvs(csv_paths: List[str]) -> List[str]:
    """
    Sort by numeric index if present, otherwise by modification time.
    Returns a new list sorted from oldest -> newest.
    """
    with_idx = []
    without_idx = []
    for p in csv_paths:
        idx = parse_log_index(p)
        if idx is None:
            without_idx.append(p)
        else:
            with_idx.append((idx, p))

    if with_idx:
        with_idx.sort(key=lambda t: t[0])  # by index
        ordered = [p for _, p in with_idx]
        # If there are also files without index, place them by mtime before/after as needed
        if without_idx:
            without_idx.sort(key=lambda p: os.path.getmtime(p))
            # Merge by mtime (append at the end to keep logic simple & stable)
            ordered = without_idx + ordered
    else:
        ordered = sorted(csv_paths, key=lambda p: os.path.getmtime(p))

    # Final guarantee: oldest->newest by mtime if tie
    ordered = sorted(ordered, key=lambda p: os.path.getmtime(p))
    return ordered

def discover_baseline_and_second(csv_dir: str) -> Tuple[Optional[str], Optional[str], int, str]:
    """
    Find baseline (oldest) and second (newest) CSV in csv_dir.
    Returns: (baseline_csv, second_csv, total_count, note)
    """
    pattern = os.path.join(csv_dir, "PPO_SimpleMatch-v0_log_*.csv")
    candidates = glob.glob(pattern)

    if not candidates:
        return None, None, 0, f"No CSVs found at: {pattern}"

    ordered = sort_csvs(candidates)  # oldest -> newest
    total = len(ordered)

    if total == 1:
        return ordered[0], None, total, "Only one CSV found."

    baseline = ordered[0]       # oldest
    second = ordered[-1]        # newest
    return baseline, second, total, f"Picked oldest as baseline and newest as second (total files: {total})."

def summarize_csv(csv_path: str, last_n: int) -> Tuple[Optional[float], str]:
    if not csv_path or not os.path.isfile(csv_path):
        return None, f"Missing CSV: {csv_path}"
    rewards = load_rewards_from_csv(csv_path)
    if not rewards:
        return None, f"No reward rows in CSV: {csv_path}"
    avg_r = avg_last_n(rewards, last_n)
    if avg_r is None:
        return None, f"Unable to compute average from CSV: {csv_path}"
    return avg_r, f"OK (used last {min(last_n, len(rewards))} rows)"

def main():
    parser = argparse.ArgumentParser(description="Compare EnMatch runs using PPO reward CSVs.")
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=os.path.join(ROOT, "code", "script", "strategy", "PPO_logs", "SimpleMatch-v0"),
        help="Directory with PPO_SimpleMatch-v0_log_*.csv (auto-discovery).",
    )
    parser.add_argument("--csv1", type=str, default="", help="Path to first CSV. If omitted, auto-discover baseline (oldest).")
    parser.add_argument("--csv2", type=str, default="", help="Path to second CSV. If omitted, auto-discover newest.")
    parser.add_argument("--last_n", type=int, default=50, help="Average over last N reward rows (default: 50)")
    args = parser.parse_args()

    csv1 = os.path.normpath(args.csv1) if args.csv1 else ""
    csv2 = os.path.normpath(args.csv2) if args.csv2 else ""
    auto_mode = False

    if not csv1 or not csv2:
        auto_mode = True
        d = os.path.normpath(args.csv_dir)
        baseline, second, total, note = discover_baseline_and_second(d)
        print(f"Auto-discovery in: {d}")
        print(note)
        csv1 = csv1 or (baseline or "")
        csv2 = csv2 or (second or "")
        print(f"Baseline (oldest) CSV: {csv1 or '(none)'}")
        print(f"Second (newest)  CSV: {csv2 or '(none)'}\n")

    # Need two CSVs to compare
    if not csv1 or not csv2:
        print("Need two CSVs to compare.")
        print("Provide --csv1 and --csv2 OR ensure the directory contains at least two CSVs.")
        return

    avg1, note1 = summarize_csv(csv1, args.last_n)
    avg2, note2 = summarize_csv(csv2, args.last_n)

    # Build labels
    label1 = "(baseline)"
    label2 = ""
    if auto_mode:
        # Label second model as "(modified)" iff the number of CSV files in the folder is even.
        # Leave unlabeled if odd.
        pattern = os.path.join(os.path.normpath(args.csv_dir), "PPO_SimpleMatch-v0_log_*.csv")
        count = len(glob.glob(pattern))
        if count % 2 == 0:
            label2 = "(modified)"
        else:
            label2 = ""  # explicitly unlabeled

    print(f"[1] {csv1} {label1}")
    print(f"    -> {note1}")
    if avg1 is not None:
        print(f"    Avg(last {args.last_n}): {avg1:.6f}")
    print()
    print(f"[2] {csv2} {label2}")
    print(f"    -> {note2}")
    if avg2 is not None:
        print(f"    Avg(last {args.last_n}): {avg2:.6f}")
    print()

    if avg1 is None or avg2 is None:
        print("Could not compute averages for one or both CSVs.")
        return

    delta = avg2 - avg1
    pct = (delta / abs(avg1)) * 100.0 if avg1 != 0 else float("inf")
    print("========== COMPARISON ==========")
    print(f"Baseline avg (last {args.last_n}): {avg1:.6f}")
    print(f"Second   avg (last {args.last_n}): {avg2:.6f} {label2}")
    print("--------------------------------")
    better = "Second > Baseline" if avg2 > avg1 else ("Baseline > Second" if avg1 > avg2 else "Tie")
    print(f"Δ (second - baseline): {delta:.6f}  ({pct:+.2f}%)   => {better}")
    print("================================")

if __name__ == "__main__":
    main()
