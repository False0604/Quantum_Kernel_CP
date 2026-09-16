import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from qiskit.circuit.library import ZZFeatureMap

feature_map = ZZFeatureMap(feature_dimension=2, reps=1)

kernel = FidelityQuantumKernel(feature_map=feature_map)

x = np.array([[0.5, 1.0]])
y = np.array([[0.6, 0.9]])

result = kernel.evaluate(x, y)

print(result)