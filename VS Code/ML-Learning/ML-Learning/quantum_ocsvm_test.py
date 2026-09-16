import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import OneClassSVM

# Normal training data
train_data = np.array([
    [0.5, 1.0],
    [0.6, 0.9],
    [0.55, 1.05],
    [0.45, 0.95]
])

# Test data: last point is an anomaly
test_data = np.array([
    [0.52, 0.98],
    [0.58, 0.92],
    [5.0, 5.0]
])

# Quantum feature map
feature_map = zz_feature_map(
    feature_dimension=2,
    reps=1
)

# Quantum kernel
kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

# Kernel between training samples
K_train = kernel.evaluate(train_data)

# Kernel between test samples and training samples
K_test = kernel.evaluate(test_data, train_data)

# One-Class SVM
model = OneClassSVM(
    kernel="precomputed",
    nu=0.25
)

# Train ONLY on normal data
model.fit(K_train)

# Predict test data
predictions = model.predict(K_test)

print("Predictions:")
print(predictions)

for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Test sample {i + 1}: Normal")
    else:
        print(f"Test sample {i + 1}: ANOMALY")