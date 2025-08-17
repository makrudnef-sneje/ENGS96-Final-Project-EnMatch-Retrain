
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare two PPO checkpoints (baseline vs edited).
Looks for the latest *_final.pth under code/script/strategy/PPO_runs/checkpoints,
unless paths are passed explicitly.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch

# path fix
REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR  = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# fallback gym
try:
    import gym
except Exception:
    import gymnasium as gym  # noqa: F401

from script.strategy.utils import default_config
from enmatch.env import RecEnvBase, SeqSimpleMatchRecEnv, SeqSimpleMatchState
from enmatch.nets.enmatch.ppo import PPO
import argparse
import torch.nn as nn

CKPT_DIR = REPO_ROOT / "code" / "script" / "strategy" / "PPO_runs" / "checkpoints"

def make_config():
    cfg = dict(default_config)
    cfg.update({
        "env": "SeqSimpleMatchRecEnv-v0",
        "sample_file": str(REPO_ROOT / 'code/dataset/simplematch_18_3_100_100000_linear_opt.csv'),
        "max_steps": 18, "num_players": 18, "action_size": 18,
        "num_per_team": 3, "num_matches": 3,
        "is_eval": True, "gpu": 0,
    })
    cfg["obs_size"] = cfg.get("obs_size", cfg["num_players"] * 3)
    cfg["batch_size"] = int(cfg.get("batch_size", 32))
    return cfg

def make_env(cfg):
    recsim = SeqSimpleMatchRecEnv(config=cfg, state_cls=SeqSimpleMatchState)
    return RecEnvBase(recsim)

def build_ppo(env, cfg):
    return PPO(
        state_dim=cfg["obs_size"],
        action_dim=env.action_space.n,
        lr_actor=cfg.get("lr_actor", 3e-4),
        lr_critic=cfg.get("lr_critic", 1e-3),
        gamma=cfg.get("gamma", 0.99),
        K_epochs=cfg.get("K_epochs", 80),
        eps_clip=cfg.get("eps_clip", 0.2),
        has_continuous_action_space=False,
        action_std=None,
        config=cfg,
    )

def load_ckpt_into(ppo, ckpt_path: Path):
    blob = torch.load(ckpt_path, map_location="cpu")
    ppo.actor.load_state_dict(blob["actor"])
    ppo.critic.load_state_dict(blob["critic"])

def rollout_episode(ppo, env, max_steps):
    state = env.reset()
    total = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        sel = ppo.select_action(state)
        action = sel[0] if isinstance(sel, (tuple, list)) else sel
        state, reward, done, _ = env.step(action)
        r = float(np.mean(reward)) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
        total += r
        steps += 1
    return total

def eval_avg(ppo, env, episodes, max_steps):
    scores = [rollout_episode(ppo, env, max_steps) for _ in range(episodes)]
    return float(np.mean(scores))

def find_latest(tag):
    cands = sorted(CKPT_DIR.glob(f"{tag}_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None, help="Path to baseline .pth (default: latest baseline_*.pth)")
    ap.add_argument("--edited",   default=None, help="Path to edited .pth (default: latest edited_*.pth)")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    cfg = make_config()
    envA = make_env(cfg)
    envB = make_env(cfg)
    ppoA = build_ppo(envA, cfg)
    ppoB = build_ppo(envB, cfg)

    bpath = Path(args.baseline) if args.baseline else find_latest("baseline")
    epath = Path(args.edited)   if args.edited   else find_latest("edited")
    if not bpath or not epath:
        print("Could not find checkpoints. Expected files like:")
        print(str(CKPT_DIR / "baseline_final.pth"))
        print(str(CKPT_DIR / "edited_final.pth"))
        print("Or pass explicit paths via --baseline and --edited.")
        raise SystemExit(2)

    print(f"[compare] baseline={bpath}")
    print(f"[compare] edited  ={epath}")
    load_ckpt_into(ppoA, epath)
    load_ckpt_into(ppoB, bpath)

    avgA = eval_avg(ppoA, envA, args.episodes, cfg["max_steps"])
    avgB = eval_avg(ppoB, envB, args.episodes, cfg["max_steps"])
    print("========== RESULT ==========")
    print(f"Edited   avg_reward: {avgA:.4f}")
    print(f"Baseline avg_reward: {avgB:.4f}")
    print(f"Diff (Edited - Baseline): {avgA-avgB:+.4f}")

if __name__ == "__main__":
    main()
