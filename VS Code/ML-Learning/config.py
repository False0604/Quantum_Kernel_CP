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

# TON-IoT (telemetry) dataset root candidates
POTENTIAL_TONIOT_DIRS = [
    PROJECT_ROOT / "data" / "TON-IoT",
    PROJECT_ROOT.parent.parent / "TON-IoT",
    PROJECT_ROOT.parent / "TON-IoT",
    PROJECT_ROOT / "TON-IoT",
]

# Results and figures directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = RESULTS_DIR / "runs"

# Ensure essential output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment Defaults
RANDOM_STATE = 42
DEFAULT_SEED_LIST = [0, 1, 2, 3, 4]

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

# Autoencoder Settings (shallow MLP)
DEFAULT_AE_BOTTLENECK = 3
DEFAULT_AE_HIDDEN = 8
DEFAULT_AE_EPOCHS = 60
DEFAULT_AE_BATCH_SIZE = 64
DEFAULT_AE_LR = 1e-3
DEFAULT_AE_PATIENCE = 10

# Quantum noise model defaults (depolarising channel probability p in [0, 1])
DEFAULT_NOISE_PROB = 0.0            # 0 = ideal simulator
NOISE_PRESETS = {
    "Ideal (statevector)": 0.0,
    "Low noise (p = 0.005)": 0.005,
    "Moderate NISQ (p = 0.02)": 0.02,
    "Heavy NISQ (p = 0.05)": 0.05,
}

# Latency benchmark protocol
LATENCY_WARMUP_SAMPLES = 32
LATENCY_TEST_BATCH = 100    # kept modest so quantum runs are tractable
LATENCY_REPEATS = 5

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

# Canonical TON-IoT device families and expected file-name substrings
TONIOT_DEVICE_FAMILIES = {
    "Fridge": ["fridge"],
    "GPS Tracker": ["gps_tracker", "gps"],
    "Motion Light": ["motion_light", "motion"],
    "Garage Door": ["garage_door", "garage"],
    "Modbus": ["modbus"],
    "Thermostat": ["thermostat"],
    "Weather": ["weather"],
}

# Standard column-name candidates for label discovery in TON-IoT
TONIOT_LABEL_CANDIDATES = ["label", "Label", "attack_label", "AttackLabel"]
TONIOT_TYPE_CANDIDATES = ["type", "Type", "attack_type", "AttackType"]
