"""
config.py
Central configuration and hyperparameter management for Quantum IoT Anomaly Detection.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Data paths to inspect (supports local data directory or root repository N-BaIoT folder)
POTENTIAL_DATA_DIRS = [
    PROJECT_ROOT / "data" / "N-BaIoT",
    PROJECT_ROOT.parent.parent / "N-BaIoT",
    PROJECT_ROOT.parent / "N-BaIoT",
    PROJECT_ROOT / "N-BaIoT",
]

# Results and figures directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure essential output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment Defaults
RANDOM_STATE = 42

# Dimensionality reduction (PCA)
DEFAULT_PCA_COMPONENTS = 4
SUPPORTED_PCA_COMPONENTS = [2, 4, 6, 8, 10]

# Quantum Circuit Settings
DEFAULT_QUANTUM_REPS = 1
DEFAULT_FEATURE_MAP_TYPE = "zz"  # ZZFeatureMap

# Dataset sampling defaults (pairwise kernel requires bounded sample size)
DEFAULT_TRAIN_SIZE = 300
DEFAULT_TEST_SIZE = 200

# One-Class SVM Settings
DEFAULT_NU = 0.10
DEFAULT_RBF_GAMMA = "scale"

# Isolation Forest Settings
DEFAULT_IFOREST_ESTIMATORS = 100
DEFAULT_IFOREST_CONTAMINATION = "auto"

# Safety limits for Quantum Kernel computation:
# N samples -> N x N evaluations. At N=1000, 1M kernel evaluations.
WARN_SAMPLE_THRESHOLD = 1000
HARD_SAMPLE_CAP = 2500

# Device names in N-BaIoT benchmark
DEVICE_NAMES = {
    1: "Danmini Doorbell",
    2: "Ecobee Thermostat",
    3: "Ennio Doorbell",
    4: "Philips B120N/10 Baby Monitor",
    5: "Provision 737E PT Camera",
    6: "Provision 838 PT Camera",
    7: "SimpleHome XCS7-1002-WHT Camera",
    8: "SimpleHome XCS7-1003-WHT Camera",
    9: "Samsung SNH 1011 N Webcam",
}
