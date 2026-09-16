# Viva-Voce Oral Defense Guide

Comprehensive, exam-ready answers to the key questions on **Quantum-Enabled IoT Network Intrusion Detection**.

---

### Q1: What is anomaly detection?
**Answer**: Anomaly detection is the task of identifying observations that deviate significantly from the majority of the data (the expected normal profile). In network security, anomalies represent malicious intrusions, unauthorized reconnaissance, or denial-of-service traffic.

---

### Q2: Why unsupervised learning instead of supervised classification?
**Answer**: In real-world IoT environments, cyberattacks evolve rapidly (zero-day exploits). Supervised models require prior labeled examples of every attack type and fail when encountering unseen attack vectors. Unsupervised detection models the normal system baseline and flags any novel deviation without needing prior attack signatures.

---

### Q3: Why train only on benign data?
**Answer**: In one-class classification, the objective is to characterize the support of the benign distribution. Exposing the model to attack traffic during training would contaminate the boundary of normality, causing the detector to accept malicious packets as expected behavior.

---

### Q4: What is One-Class SVM?
**Answer**: One-Class SVM (Schölkopf et al.) is an unsupervised kernel algorithm that maps training data into a high-dimensional feature space and computes a max-margin hyperplane that separates the normal data points from the origin, enclosing the normal data with minimal volume.

---

### Q5: What is a kernel in machine learning?
**Answer**: A kernel function $k(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$ computes the inner product between data points in an implicit high-dimensional feature space without explicitly calculating the high-dimensional coordinates (the "kernel trick").

---

### Q6: What is an RBF kernel?
**Answer**: The Radial Basis Function (RBF) or Gaussian kernel is a stationary classical kernel defined as:
$$k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$
It corresponds to an infinite-dimensional feature space where similarity decays smoothly with Euclidean distance.

---

### Q7: What is a quantum kernel?
**Answer**: A quantum kernel maps classical data vectors into quantum states $|\Phi(\mathbf{x})\rangle$ in a Hilbert space using a parameterized quantum circuit (feature map) and evaluates similarity as the quantum state transition fidelity:
$$k_Q(\mathbf{x}, \mathbf{x}') = |\langle \Phi(\mathbf{x}) | \Phi(\mathbf{x}') \rangle|^2$$

---

### Q8: What is a quantum feature map?
**Answer**: A parameterized unitary circuit $\mathcal{U}_{\Phi}(\mathbf{x})$ that transforms the initial ground state $|0\rangle^{\otimes n}$ into a data-encoded quantum state:
$$|\Phi(\mathbf{x})\rangle = \mathcal{U}_{\Phi}(\mathbf{x}) |0\rangle^{\otimes n}$$
We use `ZZFeatureMap`, which introduces single-qubit rotations ($R_Z$) and two-qubit entangling gates ($R_{ZZ}$) based on pairwise feature products.

---

### Q9: Why use PCA before the quantum circuit?
**Answer**: Physical quantum processors and simulators have strict qubit constraints. PCA reduces the 115 raw continuous features to the top $k$ orthogonal principal components ($k \in \{2, 4, 6, 8\}$) that capture the maximum variance, matching the available qubit register size.

---

### Q10: Why can't we directly encode all 115 features into quantum circuits?
**Answer**: Directly encoding 115 features would require a 115-qubit register. Simulating 115 qubits classically requires storing $2^{115}$ complex amplitudes ($> 10^{34}$ values), which exceeds all global computing memory. Even on real hardware, 115 qubits with deep two-qubit entangling gates suffer from severe gate error rates and decoherence.

---

### Q11: What is a Fidelity Quantum Kernel?
**Answer**: It is a quantum kernel where the similarity measure is defined by quantum state fidelity—the probability of measuring state $|\Phi(\mathbf{x}')\rangle$ in state $|\Phi(\mathbf{x})\rangle$. When states are identical, fidelity is $1.0$; when orthogonal, fidelity is $0.0$.

---

### Q12: Why is the kernel calculation pairwise?
**Answer**: Kernel methods compute pairwise geometric relationships. To construct the training Gram matrix, every training sample $i$ must be evaluated against every training sample $j$, resulting in $N \times N$ circuit evaluations. For testing $M$ samples against $N$ training samples, $M \times N$ circuit evaluations are required.

---

### Q13: Why is quantum kernel computation computationally expensive?
**Answer**: Because pairwise evaluation requires $\mathcal{O}(N^2)$ circuit simulations. In statevector simulation, each evaluation executes matrix-vector multiplications across $2^k$-dimensional complex vectors.

---

### Q14: Why use a quantum simulator instead of physical hardware?
**Answer**: High-performance quantum simulators (such as Qiskit Aer) provide noise-free, reproducible baselines to evaluate theoretical algorithmic capability without confounding factors like thermal noise, readout errors, and queue delays on cloud quantum providers.

---

### Q15: What is ROC-AUC?
**Answer**: Receiver Operating Characteristic Area Under the Curve (ROC-AUC) measures the ability of the classifier to rank anomalous samples higher than benign samples across all classification thresholds. A score of $1.0$ is perfect; $0.5$ represents random guessing.

---

### Q16: What is PR-AUC?
**Answer**: Precision-Recall Area Under the Curve (PR-AUC / Average Precision) evaluates detection performance across varying thresholds focusing specifically on the positive (minority anomaly) class. It is especially critical in cyber intrusion detection where anomalies are rare.

---

### Q17: What is F1-score?
**Answer**: The harmonic mean of Precision and Recall:
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
It balances the trade-off between false alarms (false positives) and missed attacks (false negatives).

---

### Q18: What is data leakage, and how did we prevent it?
**Answer**: Data leakage occurs when information from outside the training partition influences model training. We prevented leakage by:
1. Partitioning benign data into train and test sets *before* fitting any transformer.
2. Fitting `StandardScaler` and `PCA` **strictly on the benign training set**.
3. Applying the pre-fitted transformations to test and attack sets without re-estimating statistics.

---

### Q19: What is the fundamental difference between classical and quantum kernels?
**Answer**:
- **Classical RBF**: Projects data into an implicit infinite-dimensional space based purely on Euclidean distance $\|\mathbf{x} - \mathbf{x}'\|$.
- **Quantum Kernel**: Projects data into a non-linear $2^k$-dimensional Hilbert space parameterized by quantum entanglement and non-local phase correlations that cannot be efficiently factored into simple Euclidean distance metrics.

---

### Q20: Does this experiment prove Quantum Advantage?
**Answer**: **No.** Quantum advantage requires demonstrating that a quantum algorithm solves a practical problem faster or better than the best-known classical algorithms. In our benchmark, classical RBF OCSVM and Isolation Forest achieved higher accuracy ($\text{ROC-AUC} = 1.0000$ vs $0.7716$) and ran over $150\times$ faster.

---

### Q21: What are the main limitations of this research prototype?
**Answer**:
1. **Computational Scalability**: Quadratic pairwise complexity limits sample sizes ($N \le 500$) in simulation.
2. **Feature Compression**: PCA down-projection from 115 to $2-8$ dimensions discards fine-grained packet variance.
3. **Simulation vs Hardware**: Evaluated on an ideal simulator rather than noisy physical NISQ processors.

---

### Q22: What are the logical next steps for future research?
**Answer**:
1. **Trainable Quantum Kernels**: Train parameterized variational feature maps (quantum metric learning) to optimize separation.
2. **Noise-Aware Modeling**: Test on physical IBM Quantum backends with error mitigation (ZNE/M3).
3. **Hybrid Feature Selection**: Use mutual-information or autoencoder bottleneck layers instead of linear PCA.
