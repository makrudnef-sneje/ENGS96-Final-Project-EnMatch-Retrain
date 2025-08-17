#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Install Py3.6-compatible deps required by the EnMatch repo.
Run with:  python install_requirements.py
"""
import subprocess, sys

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

# Keep pip in a Py3.6-friendly range
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip<22"])

# Core packages pinned for Python 3.6 / this repo
for pkg in [
    "torch==1.9.0",
    "numpy==1.19.2",
    "cloudpickle==1.6.0",
    "gym==0.26.2",
]:
    run([sys.executable, "-m", "pip", "install", pkg])

# Try a few ortools versions that supported 3.6
for v in ["8.2.8710", "8.1.8487", "8.0.8283"]:
    try:
        run([sys.executable, "-m", "pip", "install", "ortools==%s" % v])
        break
    except subprocess.CalledProcessError:
        print("... trying next OR-Tools version")
else:
    print("WARNING: Could not install ortools automatically; continue only if not required.")

print("\n✅ Dependencies installed.")
