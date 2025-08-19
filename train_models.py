#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train two EnMatch PPO runs (baseline vs edited) by calling the repo's native trainer (enmatch_v2.py).
- Each run is launched in code/script/strategy with PYTHONPATH set so imports resolve.
- Baseline uses repo defaults; Edited bumps model capacity via conf (embed_dim/hidden_dim).
- Logs for each run are saved under: ./runs/{baseline,edited}/train.log

Usage:
  python train_models.py --hours 15 --sample_file code/dataset/simplematch_18_3_100_100000_linear_opt.csv
"""
from __future__ import print_function
import os, sys, json, time, signal, argparse, subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR  = REPO_ROOT / "code"
STRAT_DIR = CODE_DIR / "script" / "strategy"
RUNS_DIR  = REPO_ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

def pick_dataset(default_rel):
    # Ensure the dataset exists; if not, try to pick the first simplematch_*.csv
    p = REPO_ROOT / default_rel
    if p.exists():
        return str(p)
    ds = list((CODE_DIR / "dataset").glob("simplematch_*.csv"))
    if not ds:
        raise SystemExit("Dataset CSV not found. Expected %s or any simplematch_*.csv in code/dataset" % default_rel)
    return str(ds[0])

def launch_train(tag, cfg, hours):
    # Prepare env with PYTHONPATH so enmatch imports work when running from STRAT_DIR
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_DIR) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env and env["PYTHONPATH"] else "")

    # Ensure per-run log folder
    log_dir = RUNS_DIR / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    # Build command
    cmd = [sys.executable, "enmatch_v2.py", "PPO", "train", json.dumps(cfg)]
    print("Launching %s:" % tag, " ".join(cmd))
    print("  cwd:", STRAT_DIR)
    print("  log:", log_path)
    print("  sample_file:", cfg.get("sample_file"))
    sys.stdout.flush()

    # Launch process
    with open(str(log_path), "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(STRAT_DIR), env=env, stdout=lf, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

        # Let it run for the requested number of hours, then terminate gracefully
        try:
            t_end = time.time() + hours * 3600.0
            while True:
                ret = proc.poll()
                if ret is not None:
                    print("[%s] process exited with code %s" % (tag, ret))
                    break
                if time.time() >= t_end:
                    print("[%s] Time budget reached; sending SIGINT..." % tag)
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                    # Wait a bit for cleanup/saving
                    try:
                        proc.wait(timeout=120)
                    except Exception:
                        print("[%s] Forcing terminate..." % tag)
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except Exception:
                            print("[%s] Forcing kill..." % tag)
                            proc.kill()
                    break
                time.sleep(5)
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    print("[%s] training log saved at %s" % (tag, log_path))
    return str(log_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=15.0, help="Hours PER RUN (baseline and edited). Default=15")
    ap.add_argument("--sample_file", default="code/dataset/simplematch_18_3_100_100000_linear_opt.csv",
                    help="Path to dataset CSV (relative to repo root or absolute)")
    args = ap.parse_args()

    dataset = pick_dataset(args.sample_file)

    # Common config keys known by enmatch_v2.py & EnNet
    base_cfg = {
        "gpu": 0,
        "env": "SeqSimpleMatchRecEnv-v0",
        "sample_file": dataset if os.path.isabs(dataset) else str(dataset),
        "is_eval": False,
        "num_matches": 3,
    }

    # Baseline: leave embed_dim/hidden_dim as repo defaults (do not set)
    cfg_baseline = dict(base_cfg)
    cfg_baseline["trial_name"] = "baseline_auto"

    # Edited: increase capacity via conf
    cfg_edited = dict(base_cfg)
    cfg_edited.update({
        "embed_dim": 192,     # bump from default 128
        "hidden_dim": 192,
        "trial_name": "edited_auto",
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
    })


    # Launch baseline then edited
    log_a = launch_train("baseline", cfg_baseline, args.hours)
    log_b = launch_train("edited",   cfg_edited,   args.hours)

    print("\nDone launching both runs.")
    print("Baseline log:", log_a)
    print("Edited log:  ", log_b)

if __name__ == "__main__":
    main()
