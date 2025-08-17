
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train two PPO variants (baseline vs edited actor) on EnMatch:
- Baseline uses the repo's native PPO architecture
- Edited replaces the actor with a deeper MLP as specified

Saves checkpoints under: code/script/strategy/PPO_runs/checkpoints/
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# --- Make <repo>/code importable without pip install -e ---
REPO_ROOT = Path(__file__).resolve().parents[3]  # .../Project
CODE_DIR  = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Prefer gym; fall back to gymnasium
try:
    import gym
except Exception:
    import gymnasium as gym  # noqa: F401

from script.strategy.utils import default_config
from enmatch.env import RecEnvBase, SeqSimpleMatchRecEnv, SeqSimpleMatchState
from enmatch.nets.enmatch.ppo import PPO

OUTDIR = REPO_ROOT / "code" / "script" / "strategy" / "PPO_runs"
CKPT_DIR = OUTDIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def make_config(overrides=None):
    cfg = dict(default_config)
    overrides = overrides or {}
    cfg.update(overrides)

    # derive dims from sample_file if provided
    if "sample_file" in cfg and isinstance(cfg["sample_file"], str):
        try:
            name = Path(cfg["sample_file"]).name
            parts = name.split("_")
            # simplematch_18_3_100_100000_linear_opt.csv
            cfg["num_players"]  = int(parts[1])
            cfg["num_per_team"] = int(parts[2])
            cfg["action_size"]  = cfg["num_players"]
            cfg["max_steps"]    = cfg["num_players"]
        except Exception:
            pass
    cfg["obs_size"] = cfg.get("obs_size", cfg["num_players"] * 3)
    cfg["batch_size"] = int(cfg.get("batch_size", 32))
    cfg["max_steps"]  = int(cfg.get("max_steps", cfg["num_players"]))
    return cfg

def make_envs(cfg):
    recsim      = SeqSimpleMatchRecEnv(config=cfg, state_cls=SeqSimpleMatchState)
    env         = RecEnvBase(recsim)
    env.seed(1)
    recsim_copy = SeqSimpleMatchRecEnv(config=cfg, state_cls=SeqSimpleMatchState)
    env_copy    = RecEnvBase(recsim_copy)
    env_copy.seed(1)
    return env, env_copy

def build_baseline(env, cfg):
    state_dim  = cfg["obs_size"]
    action_dim = env.action_space.n
    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=cfg.get("lr_actor", 3e-4),
        lr_critic=cfg.get("lr_critic", 1e-3),
        gamma=cfg.get("gamma", 0.99),
        K_epochs=cfg.get("K_epochs", 80),
        eps_clip=cfg.get("eps_clip", 0.2),
        has_continuous_action_space=False,
        action_std=None,
        config=cfg,
    )
    return ppo

def build_edited(env, cfg):
    """Create PPO then replace its actor with the edited architecture."""
    ppo = build_baseline(env, cfg)
    state_dim  = cfg["obs_size"]
    action_dim = env.action_space.n
    # Edited actor per your spec
    edited_actor = nn.Sequential(
        nn.Linear(state_dim, 64), nn.Tanh(),
        nn.Linear(64, 128),       nn.Tanh(),
        nn.Linear(128, 64),       nn.Tanh(),
        nn.Linear(64, action_dim),nn.Tanh(),
    )
    # Try to keep critic as-is; just swap actor
    ppo.actor = edited_actor
    # reinit weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1/np.sqrt(fan_in) if fan_in>0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
    ppo.actor.apply(init_weights)
    return ppo

def rollout_episode(ppo, env, max_steps):
    state = env.reset()
    total = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        sel = ppo.select_action(state)
        action = sel[0] if isinstance(sel, (tuple, list)) else sel
        state, reward, done, _ = env.step(action)
        # reward maybe batch
        r = float(np.mean(reward)) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
        total += r
        steps += 1
        ppo.buffer.rewards.append(r)
        ppo.buffer.is_terminals.append(done)
    return total

def train_one(tag, ppo, cfg, hours=15, eval_every=6000, save_every=100000):
    env, env_copy = make_envs(cfg)
    max_ep_len = cfg["max_steps"]
    batch_size = cfg["batch_size"]

    t0 = time.time()
    time_step = 0
    ep = 0
    print(f"\n=== TRAIN {tag} for ~{hours}h ===")
    try:
        while (time.time() - t0) < hours * 3600:
            ret = rollout_episode(ppo, env, max_ep_len)
            ep += 1
            time_step += max_ep_len

            # PPO update (match repo cadence: every 12 steps approx., here per episode is fine)
            if (time_step % 12) < max_ep_len:
                ppo.update(max_ep_len, batch_size)

            if eval_every > 0 and (time_step % eval_every) < max_ep_len:
                with torch.no_grad():
                    scores = [rollout_episode(ppo, env_copy, max_ep_len) for _ in range(3)]
                    print(f"[{tag}] t={time_step} eval_avg={float(np.mean(scores)):.3f}")

            if save_every > 0 and (time_step % save_every) < max_ep_len:
                ck = CKPT_DIR / f"{tag}_t{time_step}.pth"
                torch.save({"actor": ppo.actor.state_dict(),
                            "critic": ppo.critic.state_dict()}, ck)
                print(f"[{tag}] checkpoint -> {ck}")
    except KeyboardInterrupt:
        print(f"[{tag}] interrupted, saving emergency checkpoint...")
    # final save
    final = CKPT_DIR / f"{tag}_final.pth"
    torch.save({"actor": ppo.actor.state_dict(),
                "critic": ppo.critic.state_dict()}, final)
    print(f"[{tag}] final -> {final}")

def main():
    # Default config tuned to your 18x3 dataset
    overrides = {
        "env": "SeqSimpleMatchRecEnv-v0",
        "sample_file": str(Path('code/dataset/simplematch_18_3_100_100000_linear_opt.csv')),
        "max_steps": 18, "num_players": 18, "action_size": 18,
        "num_per_team": 3, "num_matches": 3,
        "is_eval": False, "gpu": 0,
    }
    cfg = make_config(overrides)

    # Train baseline then edited
    env_tmp, _ = make_envs(cfg)
    baseline = build_baseline(env_tmp, cfg)
    edited   = build_edited(env_tmp, cfg)

    # train ~15h each (total ~30h)
    train_one("baseline", baseline, cfg, hours=15)
    train_one("edited",   edited,   cfg, hours=15)

if __name__ == "__main__":
    main()
