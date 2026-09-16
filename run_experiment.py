"""
Bridge runner to allow executing run_experiment.py directly from the project root.
"""
import os
import sys
from pathlib import Path

pkg_dir = Path(__file__).resolve().parent / "VS Code" / "ML-Learning"
sys.path.insert(0, str(pkg_dir))
os.chdir(pkg_dir)

from run_experiment import main

if __name__ == "__main__":
    main()
