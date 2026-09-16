# Pipeline Architecture

This document provides a technical walkthrough of the **Quantum-Enabled Anomaly Detection** architecture for IoT network intrusion detection systems.

---

## 1. High-Level Pipeline Flow

```text
               +-------------------------------------------------+
               |              Raw N-BaIoT Dataset                |
               | (115 network traffic statistical features, CSV) |
               +-------------------------------------------------+
                                        |
                                        v
               +-------------------------------------------------+
               |             Strict Partitioning                 |
               |       Benign Train    |    Benign Test          |
               |      (e.g., N=150)    |   (e.g., M=75)          |
               +-------------------------------------------------+
                        |                                |
                        | (fit only)                     | (transform only)
                        v                                v
         +-----------------------------+       +-------------------+
         |      DataPreprocessor       | ----> | Transform Benign  |
         | - Median Imputation         |       |    & Attack Test  |
         | - Constant Feature Removal  |       +-------------------+
         | - StandardScaler            |                 |
         | - PCA (k=4 Qubits)          |                 v
         +-----------------------------+       +-------------------+
                        |                      | Reduced Test Data |
                        v                      |   (2M x 4 dims)   |
         +-----------------------------+       +-------------------+
         |   Reduced Benign Train      |                 |
         |     (N=150 x 4 dims)        |                 |
         +-----------------------------+                 |
                        |                                |
                        v                                |
         +-----------------------------+                 |
         |     Quantum Feature Map     |                 |
         |       (ZZFeatureMap)        |                 |
         +-----------------------------+                 |
                        |                                |
                        v                                |
         +-----------------------------+                 |
         |   Fidelity Quantum Kernel   |                 |
         |  K_train = K(X_tr, X_tr)    |                 |
         |       Shape: (N, N)         |                 |
         +-----------------------------+                 |
                        |                                |
                        v                                v
         +-----------------------------+       +-------------------+
         |    One-Class SVM Model      |       |  Evaluate K_test  |
         |   kernel = 'precomputed'    | <---- | K(X_test, X_train)|
         |      model.fit(K_train)     |       |   Shape: (2M, N)  |
         +-----------------------------+       +-------------------+
                        |                                |
                        +---------------+----------------+
                                        |
                                        v
                        +--------------------------------+
                        |      Inference & Scores        |
                        |  Predictions: +1 (Normal),     |
                        |              -1 (Anomaly)      |
                        |  Score = -decision_function()  |
                        +--------------------------------+
                                        |
                                        v
                        +--------------------------------+
                        |     Scientific Evaluation      |
                        | ROC-AUC, PR-AUC, F1, CM, Time  |
                        +--------------------------------+
```

---

## 2. Mathematical Components

### A. Dimensionality Reduction (PCA)
Physical quantum hardware and statevector simulators operate in a Hilbert space whose dimension scales as $2^k$, where $k$ is the number of qubits. Directly embedding $115$ continuous features would require a $115$-qubit circuit, which is intractable on classical simulators ($2^{115}$ statevector amplitudes) and NISQ hardware.

We project the normalized feature vector $\mathbf{x} \in \mathbb{R}^{115}$ onto the top $k$ principal components $\mathbf{z} \in \mathbb{R}^k$ ($k \in \{2, 4, 6, 8\}$):
$$\mathbf{z} = \mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu})$$
where $\mathbf{W}$ contains the eigenvectors of the covariance matrix computed strictly on benign training data.

### B. Quantum Feature Map ($\mathcal{U}_{\Phi}(\mathbf{z})$)
We use Qiskit's `ZZFeatureMap` to encode classical continuous data into quantum statevectors:
$$|\Phi(\mathbf{z})\rangle = \mathcal{U}_{\Phi}(\mathbf{z}) |0\rangle^{\otimes k}$$

The unitary circuit comprises:
1. Hadamard gates creating uniform superposition: $H^{\otimes k}$.
2. Single-qubit phase rotations: $R_Z(2z_i)$.
3. Two-qubit entangling phase gates parameterized by pairwise feature interactions:
   $$R_{ZZ}(2(\pi - z_i)(\pi - z_j))$$
This creates non-linear quantum entanglement between features.

### C. Quantum Fidelity Kernel ($k_Q$)
The quantum kernel measures the transition fidelity (inner product overlap) between two quantum states in Hilbert space:
$$k_Q(\mathbf{z}_i, \mathbf{z}_j) = |\langle \Phi(\mathbf{z}_i) | \Phi(\mathbf{z}_j) \rangle|^2 = |\langle 0^{\otimes k} | \mathcal{U}_{\Phi}^\dagger(\mathbf{z}_j) \mathcal{U}_{\Phi}(\mathbf{z}_i) | 0^{\otimes k} \rangle|^2$$

Properties:
- **Self-fidelity**: $k_Q(\mathbf{z}_i, \mathbf{z}_i) = 1.0$ (matrix diagonal is unity).
- **Symmetry**: $k_Q(\mathbf{z}_i, \mathbf{z}_j) = k_Q(\mathbf{z}_j, \mathbf{z}_i)$.
- **Range**: $0.0 \le k_Q(\mathbf{z}_i, \mathbf{z}_j) \le 1.0$.

### D. Precomputed One-Class SVM
The One-Class SVM solves the quadratic programming dual problem:
$$\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j K_{ij} \quad \text{s.t.} \quad 0 \le \alpha_i \le \frac{1}{\nu N}, \quad \sum_i \alpha_i = 1$$
where $K_{ij} = k_Q(\mathbf{z}_i, \mathbf{z}_j)$ is the precomputed training Gram matrix.

For test inference, the kernel evaluated is:
$$\mathbf{K}_{\text{test}}[m, n] = k_Q(\mathbf{z}_{\text{test}, m}, \mathbf{z}_{\text{train}, n})$$
with shape $(M, N)$.
The decision function is:
$$f(\mathbf{z}) = \sum_{i \in \text{SVs}} \alpha_i k_Q(\mathbf{z}, \mathbf{z}_i) - \rho$$
- $f(\mathbf{z}) \ge 0 \implies \text{Normal} (+1)$
- $f(\mathbf{z}) < 0 \implies \text{Anomaly} (-1)$
- $\text{Anomaly Score} = -f(\mathbf{z})$
