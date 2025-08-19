EnMatch Training and Comparison Instructions
===========================================

This guide explains how to set up, train, and compare models for the EnMatch project.

---
## 1. Environment Setup
1. Install Miniconda (if not already installed).
2. Navigate to the EnMatch project folder in PowerShell:
   ```powershell
   cd "C:\Users\<YourUser>\OneDrive\Desktop\ENGS Project\EnMatch"
   ```
3. Install dependencies using the provided script:
   ```powershell
   "C:\Users\<YourUser>\Miniconda3\envs\enmatch39\python.exe" install_requirements.py
   ```

---
## 2. Running Training
To train models, use `train_models.py`.

Example (run for 11 hours on CPU):
```powershell
& "C:\Users\<YourUser>\Miniconda3\envs\enmatch39\python.exe" ".\train_models.py" --hours 11 --sample_file ".\code\dataset\simplematch_18_3_100_100000_linear_opt.csv"
```

This will:
- Launch **baseline** and **edited** models.
- Save logs in:
  - `EnMatch\runs\baseline\train.log`
  - `EnMatch\runs\edited\train.log`
- Save reward CSVs in `EnMatch\code\script\strategy\PPO_logs\SimpleMatch-v0\`.
- Save trained models in `EnMatch\code\script\strategy\PPO_preTrained\SimpleMatch-v0\`.

If you want longer total training (e.g., 22 hours), run the script twice (first 11 hours, then again for another 11).

---
## 3. Comparing Models
To compare baseline vs. edited:
```powershell
& "C:\Users\<YourUser>\Miniconda3\envs\enmatch39\python.exe" ".\compare_models.py"
```

Output:
- Console comparison of rewards/accuracy.
- CSV files in `comparison_results\`.

---
## 4. Example Successful Run
**Training (baseline + edited launched):**
```
Launching baseline: ... baseline_auto
  cwd: ...\code\script\strategy
  log: ...\runs\baseline\train.log
  sample_file: ...\dataset\simplematch_18_3_100_100000_linear_opt.csv
Launching edited: ... edited_auto
  cwd: ...\code\script\strategy
  log: ...\runs\edited\train.log
  sample_file: ...\dataset\simplematch_18_3_100_100000_linear_opt.csv
Done launching both runs.
```

**Comparison:**
```
Baseline avg reward: X.XX
Edited avg reward:   Y.YY
Edited performed better than baseline.
```

---
## 5. Notes
- GPU support works only with NVIDIA CUDA. AMD users must run on CPU.

---
