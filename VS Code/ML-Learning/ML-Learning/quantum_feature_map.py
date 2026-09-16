from qiskit import QuantumCircuit

def feature_map(features):
    qc = QuantumCircuit(len(features))

    for i, value in enumerate(features):
        qc.ry(value, i)

    return qc

features = [0.5, 1.0]

qc = feature_map(features)

print(qc)