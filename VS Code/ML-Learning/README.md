# Quantum-Enabled Anomaly Detection for IoT Network Intrusion Detection

A research-grade prototype comparing **Quantum Fidelity Kernel One-Class Support Vector Machines (OCSVM)** against **Classical Baselines (RBF OCSVM and Isolation Forest)** for unsupervised IoT network intrusion detection.

---

## Central Research Question

> **Is a quantum kernel better than classical anomaly detectors for unsupervised IoT network intrusion detection?**

This project provides a scientifically honest, reproducible benchmark framework to test this question using authentic IoT network traffic.

---

## Key Highlights

- **Two-dataset scope**: N-BaIoT (network) and TON-IoT (telemetry) are both supported, along with a synthetic fallback for environments without the raw CSVs.
- **Five detectors behind one interface**: Quantum-Kernel OCSVM (ZZFeatureMap + FidelityQuantumKernel + precomputed OCSVM), Classical RBF OCSVM, Isolation Forest, Local Outlier Factor (novelty), and a Shallow PyTorch Autoencoder.
- **Ideal or NISQ quantum simulation**: A depolarising channel is applied analytically to both encoded states so a full benchmark run can be executed under noise without extra circuit evaluations.
- **Leak-free protocol**: StandardScaler + PCA are strictly fitted on benign training data; attack rows are quarantined until evaluation.
- **Roadmap-grade rigour**: Multi-seed reproducibility (mean ± std across up to five seeds), µs/sample inference latency with warm-up, median and p95, plus persisted run logs under `results/runs/` with git commit hash and YAML config.
- **Interactive Streamlit dashboard**: Six tabs covering benchmark comparison, latency, quantum circuit / spectral / tomography, PCA scree, single-packet inspector, and LaTeX / CSV export.

---

## Benchmark Results (Real N-BaIoT Mirai Traffic)

*Evaluated on N-BaIoT Device 1 (Danmini Doorbell), Attack: Mirai ACK Flood, $N_{\text{train}}=100$, $N_{\text{test}}=100$ (50 Benign, 50 Mirai), $k=2$ Qubits, $\nu=0.10$.*

| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quantum Kernel OCSVM** | **0.7716** | **0.7398** | **0.5618** | **0.6410** | **0.5000** | **32.58 s** |
| **Classical RBF OCSVM** | **1.0000** | **1.0000** | **0.9524** | **0.9091** | **1.0000** | **0.21 s** |
| **Isolation Forest** | **1.0000** | **1.0000** | **0.9524** | **0.9091** | **1.0000** | **0.50 s** |

### Scientific Finding
Under this experimental configuration, **classical detectors outperformed the quantum kernel detector**. The Classical RBF OCSVM achieved superior anomaly discrimination ($\text{ROC-AUC} = 1.0000$) while running over **$150\times$ faster** than the simulated quantum kernel ($0.21\text{s}$ vs $32.58\text{s}$).

---

## Architecture Flow

```text
Raw IoT Traffic (115 Features)
      |
      +---> Non-overlapping Train / Test Split
      |
      v
[Train Only] Median Imputation & Constant Feature Removal
      |
      v
[Train Only] StandardScaler
      |
      v
[Train Only] PCA (k=2, 4, 6, 8 Qubits)
      |
      +----------------------------+
      |                            |
      v                            v
Quantum Circuit Encoded      Classical Subspace
(ZZFeatureMap, reps=1)       (Standard Euclidean Space)
      |                            |
      v                            v
Fidelity Quantum Kernel      Classical Baselines
K(x, y) = |<Phi(x)|Phi(y)>|^2 (RBF OCSVM, Isolation Forest)
      |                            |
      v                            v
Precomputed One-Class SVM    Decision Scores
      |                            |
      +-------------+--------------+
                    |
                    v
          Scientific Comparison
    (ROC-AUC, PR-AUC, F1, Runtime)
```

---

## Directory Structure

```text
VS Code/ML-Learning/
├── config.py                 # Central hyperparameter and path configuration
├── data_loader.py            # N-BaIoT + TON-IoT discovery, generic CSV loader, synthetic feed
├── preprocessing.py          # Leak-free StandardScaler + PCA pipeline
├── quantum_detector.py       # zz_feature_map + FidelityQuantumKernel + noise-model transform + OCSVM
├── classical_detector.py     # RBF OCSVM, Isolation Forest, LOF baselines
├── autoencoder_detector.py   # Shallow PyTorch autoencoder (reconstruction MSE anomaly score)
├── evaluation.py             # Metrics (ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix)
├── visualization.py          # Publication-ready Matplotlib visualizer
├── benchmark_harness.py      # Multi-seed runs, µs/sample latency, persisted run logs
├── advanced_quantum_engine.py# Density matrix, Gram spectral analysis, variational, NISQ, hybrid ensemble
├── run_experiment.py         # Reproducible CLI experiment runner & parameter ablation
├── app.py                    # Streamlit dashboard (six tabs, interactive controls)
├── requirements.txt          # Verified dependency manifest
├── README.md                 # Documentation
├── tests/                    # Unit & integration tests
└── results/
    ├── model_comparison.csv  # Legacy single-run comparison
    ├── experiment_config.json
    ├── runs/                 # Persisted multi-seed runs (config.yaml + metrics_*.csv + audit.json)
    └── figures/              # Exported high-resolution plots
```

---

## Installation & Setup

### Environment Requirements
- Python 3.12.x
- Windows / Linux / macOS
- Verified packages: `qiskit>=2.5.0`, `qiskit-aer>=0.17.0`, `qiskit-machine-learning>=0.9.0`, `scikit-learn>=1.5.0`, `matplotlib>=3.11.0`, `streamlit>=1.37.0`.

Install required dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
- Select from real N-BaIoT devices (1-9) and attack types (Mirai ACK/SYN/UDP, Gafgyt).
- Configure quantum dimensions (2 to 8 qubits), sample limits, and One-Class SVM $\nu$.
- Toggle `🧠 BRAINROT MODE` for presentation flavor.
- Download generated prediction CSVs and comparison tables.

### 2. Run the Benchmark via CLI
Run standard experiment on Device 1 against Mirai attack:
```bash
python run_experiment.py --device 1 --attack mirai.ack --train-size 150 --test-size 75 --components 4
```

Run dimension ablation sweep across 2, 4, and 6 qubits:
```bash
python run_experiment.py --sweep --device 1 --attack mirai.ack
```

Run synthetic anomaly validation:
```bash
python run_experiment.py --synthetic --device 1 --train-size 150 --test-size 75
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```
All 16 unit and end-to-end integration tests execute in $<10$ seconds.

---

## Limitations & Honest Scientific Appraisal

1. **Simulation vs Hardware**: Calculations use Qiskit Aer statevector simulation. Physical quantum hardware will introduce gate errors, readout noise, and qubit connectivity limitations.
2. **Computational Scaling**: Pairwise kernel evaluation scales as $\mathcal{O}(N^2)$, making large-sample training ($N > 1,000$) computationally demanding in simulation.
3. **No Automatic Quantum Advantage**: Encoding classical tabular features into Hilbert spaces does not automatically guarantee superior decision boundaries compared to classical convex RBF optimization.
