import pandas as pd

from quantum_detector import QuantumAnomalyDetector


# ---------------------------------------
# SETTINGS
# ---------------------------------------

BENIGN_FILE = "data/N-BaIoT/1.benign.csv"

ATTACK_FILE = "data/N-BaIoT/1.mirai.csv"

TRAIN_SIZE = 300
TEST_SIZE = 100


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

print("\nLoading N-BaIoT data...")

benign = pd.read_csv(BENIGN_FILE)

attack = pd.read_csv(ATTACK_FILE)

print("Benign shape:", benign.shape)
print("Attack shape:", attack.shape)


# ---------------------------------------
# SAMPLE DATA
# ---------------------------------------

benign_train = benign.sample(
    n=min(TRAIN_SIZE, len(benign)),
    random_state=42
)

benign_test = benign.sample(
    n=min(TEST_SIZE, len(benign)),
    random_state=123
)

attack_test = attack.sample(
    n=min(TEST_SIZE, len(attack)),
    random_state=42
)


# ---------------------------------------
# TRAIN QUANTUM MODEL
# ---------------------------------------

detector = QuantumAnomalyDetector(
    n_components=4,
    sample_size=300,
    nu=0.10,
    reps=1
)

detector.fit(benign_train)


# ---------------------------------------
# TEST
# ---------------------------------------

test_data = pd.concat(
    [
        benign_test,
        attack_test
    ],
    ignore_index=True
)

predictions, scores = detector.predict(
    test_data
)


# ---------------------------------------
# RESULTS
# ---------------------------------------

results = test_data.copy()

results["prediction"] = predictions

results["anomaly_score"] = scores

results["actual"] = (
    ["Benign"] * len(benign_test)
    + ["Mirai"] * len(attack_test)
)


results["detected"] = results["prediction"].map({
    1: "Normal",
    -1: "Anomaly"
})


print("\n==============================")
print("QUANTUM ANOMALY DETECTION")
print("==============================")

print(
    "\nNormal detected:",
    sum(predictions == 1)
)

print(
    "Anomalies detected:",
    sum(predictions == -1)
)

print(
    "Total samples:",
    len(predictions)
)

print(
    "Anomaly rate:",
    f"{sum(predictions == -1) / len(predictions) * 100:.2f}%"
)


print("\nDetection breakdown:")

print(
    pd.crosstab(
        results["actual"],
        results["detected"]
    )
)


# ---------------------------------------
# SAVE RESULTS
# ---------------------------------------

results.to_csv(
    "quantum_results.csv",
    index=False
)

print(
    "\nResults saved to quantum_results.csv"
)