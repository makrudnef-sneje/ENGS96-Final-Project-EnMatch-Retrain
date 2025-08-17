
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal installer for local CPU usage.
Creates/uses whatever Python is on PATH. If you prefer a venv, create it first.
"""
import subprocess, sys

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

# Upgrade pip
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Core deps (Gymnasium is maintained; repo can import gym, we provide both)
run([sys.executable, "-m", "pip", "install",
     "torch", "numpy", "gymnasium>=0.29", "cloudpickle", "ortools"])

print("\n✅ Done. If imports fail, ensure you run scripts from the repo root so 'code/' is discoverable.")
