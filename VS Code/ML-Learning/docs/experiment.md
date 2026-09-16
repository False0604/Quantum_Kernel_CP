# Experimental Protocol & Methodology

This document outlines the rigorous scientific protocol implemented in this research prototype.

---

## 1. Central Research Question

> **Is a quantum kernel better than classical anomaly detectors for unsupervised IoT network intrusion detection?**

We formulate the experimental test by comparing:
- **Model A**: Quantum Fidelity Kernel One-Class SVM ($ZZFeatureMap$ + Fidelity Kernel).
- **Model B**: Classical RBF One-Class SVM (Gaussian RBF Kernel, $\gamma = \text{scale}$).
- **Model C**: Isolation Forest (Tree-based ensemble benchmark).

---

## 2. Experimental Fairness Principles

To ensure scientific validity and avoid experimental bias:
1. **Identical Partitions**: All models are fitted on the exact same benign training partition and evaluated on the exact same test partition.
2. **Leakage Prevention**:
   - The benign dataset is partitioned *before* computing any descriptive statistics.
   - `StandardScaler` ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) and `PCA` ($\mathbf{W}$) are fitted **strictly on the benign training set**.
   - Test data (both benign and attack) is transformed using the pre-fitted parameters.
3. **Identical Reduced Dimensionality**: Both Quantum Kernel OCSVM and the primary Classical RBF OCSVM operate on the same PCA subspace ($k \in \{2, 4, 6, 8\}$).
4. **Reproducibility**: All random operations (data shuffling, subsampling, PCA initialization, and Isolation Forest tree splits) use fixed seeds (`random_state=42`).

---

## 3. Threat Model & N-BaIoT Dataset

The benchmark utilizes the authentic **N-BaIoT (Network-Based Anomaly Detection of IoT Botnet Attacks)** dataset collected across 9 commercial IoT devices:
- **Devices**: Doorสิ่งbells, Baby Monitors, Security Cameras, and Thermostats.
- **Features (115)**: Extracted from packet arrival streams across 5 time windows ($100\text{ms}, 500\text{ms}, 1.5\text{s}, 10\text{s}, 1\text{min}$) capturing packet size, jitter, bandwidth, and host-to-host covariance.
- **Attack Types**:
  - **Mirai Botnet**: SYN flood (`mirai.syn`), ACK flood (`mirai.ack`), UDP flood (`mirai.udp`), Plain UDP flood (`mirai.udpplain`), and Host scanning (`mirai.scan`).
  - **Gafgyt (BASHLITE) Botnet**: Combination attacks (`gafgyt.combo`), Junk packet flood (`gafgyt.junk`), TCP flood (`gafgyt.tcp`), UDP flood (`gafgyt.udp`), and Scanning (`gafgyt.scan`).

---

## 4. Unsupervised Anomaly Detection Paradigm

Real-world IoT security systems face novel, zero-day botnet vectors. Therefore:
- The detector trains **unsupervised solely on benign network traffic**.
- The model learns the compact manifold representing normal operational state.
- Deviations beyond the learned boundary are flagged as anomalies.
- Supervised metrics (ROC-AUC, PR-AUC, F1) are computed post-hoc for validation using ground-truth labels.
