import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

feature_map = zz_feature_map(feature_dimension=2, reps=1)

kernel = FidelityQuantumKernel(feature_map=feature_map)

data = np.array([
    [0.5, 1.0],
    [0.6, 0.9],
    [2.0, 2.5]
])

matrix = kernel.evaluate(data)

print(matrix)