# EnMatch: PPO Baseline vs Edited Model Comparison

This project extends **EnMatch** to compare the original PPO actor network against a modified (deeper) PPO actor network.  
It provides scripts to **train** both models and **compare** their performance.

---

## 1. Environment Setup

You need **Python 3.10+** (recommended: Anaconda/Miniconda).  
From the project root (the folder containing `code/`):

```bash
# Create and activate a new environment
conda create -n enmatch python=3.10 -y
conda activate enmatch
```

Install requirements automatically:

```bash
python install_requirements.py
```

This script installs:
- `torch`
- `gymnasium` (drop-in replacement for old gym)
- `numpy`
- `ortools`
- plus other required utilities.

---

## 2. Training the Models

Run the training script from the **project root**:

```bash
python train_models.py
```

- **Baseline PPO** (original EnMatch actor):
  ```
  Linear( state_dim → 64 ) → Tanh →
  Linear( 64 → 64 )        → Tanh →
  Linear( 64 → action_dim ) → Tanh
  ```

- **Edited PPO** (deeper actor):
  ```
  Linear( state_dim → 64 ) → Tanh →
  Linear( 64 → 128 )      → Tanh →
  Linear( 128 → 64 )      → Tanh →
  Linear( 64 → action_dim ) → Tanh
  ```

### Training duration
- Configured to run **~15 hours per model** (~30 hours total).  
- Checkpoints are saved periodically, and final models are stored as:

```
code/script/strategy/PPO_runs/checkpoints/
    baseline_final.pth
    edited_final.pth
```

You can stop and resume later without losing progress.

---

## 3. Comparing the Models

After training finishes (or even if you stop early), run:

```bash
python compare_models.py
```

This will:
- Load the most recent `baseline_*.pth` and `edited_*.pth` from the checkpoint folder.
- Evaluate both over a set of episodes.
- Print out **average reward** and side-by-side performance.

Optionally, specify paths manually:

```bash
python compare_models.py   --baseline code/script/strategy/PPO_runs/checkpoints/baseline_final.pth   --edited   code/script/strategy/PPO_runs/checkpoints/edited_final.pth
```

---

## 4. Expected Workflow for User

1. **Setup** environment (Step 1).  
2. **Train** models (Step 2) → ~30 hours total.  
3. **Compare** models (Step 3).  

If trained models are already provided, skip Step 2 and just run the comparison.

---

## 5. Troubleshooting

- Always run scripts from the **project root** so Python can locate the `code/` package.  
- If imports fail (`ModuleNotFoundError: enmatch`), confirm that `code/` is inside the root directory.  
- To speed things up on a machine without GPU, edit `train_models.py` to reduce `num_episodes` or `steps_per_episode`.

---
