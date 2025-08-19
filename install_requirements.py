#!/usr/bin/env python
import subprocess, sys
def run(*args): print(">", " ".join(args)); subprocess.check_call(args)
# modern, Py3.9-compatible pins
run(sys.executable, "-m", "pip", "install", "--upgrade", "pip")
for pkg in [
    "torch==1.13.1",
    "numpy==1.23.5",
    "cloudpickle==2.2.1",
    "gym==0.26.2",
    "ortools==9.6.2534",
    "tqdm==4.66.4",
]:
    run(sys.executable, "-m", "pip", "install", pkg)
print("✅ Dependencies installed.")

