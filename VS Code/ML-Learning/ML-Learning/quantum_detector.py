import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class QuantumAnomalyDetector:

    def __init__(
        self,
        n_components=4,
        sample_size=300,
        nu=0.10,
        reps=1
    ):
        self.n_components = n_components
        self.sample_size = sample_size
        self.nu = nu
        self.reps = reps

        self.scaler = StandardScaler()
        self.pca = None
        self.feature_map = None
        self.kernel = None
        self.model = None

        self.feature_columns = None
        self.X_train_pca = None

    def preprocess(self, data):
        """Prepare numerical data."""

        numeric_data = data.select_dtypes(
            include=np.number
        ).copy()

        if numeric_data.empty:
            raise ValueError(
                "No numerical columns found."
            )

        numeric_data = numeric_data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        numeric_data = numeric_data.fillna(
            numeric_data.median()
        )

        # Remove constant columns
        numeric_data = numeric_data.loc[
            :, numeric_data.nunique() > 1
        ]

        if numeric_data.empty:
            raise ValueError(
                "No useful features remain."
            )

        self.feature_columns = numeric_data.columns.tolist()

        return numeric_data

    def fit(self, benign_data):

        X = self.preprocess(benign_data)

        # Scale using benign data only
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        n_components = min(
            self.n_components,
            X_scaled.shape[1],
            X_scaled.shape[0]
        )

        self.pca = PCA(
            n_components=n_components,
            random_state=42
        )

        X_pca = self.pca.fit_transform(X_scaled)

        # Limit number of quantum training samples
        if len(X_pca) > self.sample_size:

            rng = np.random.default_rng(42)

            indices = rng.choice(
                len(X_pca),
                self.sample_size,
                replace=False
            )

            X_pca = X_pca[indices]

        self.X_train_pca = X_pca

        print(
            f"Quantum training samples: {len(X_pca)}"
        )

        print(
            f"Quantum features / qubits: "
            f"{X_pca.shape[1]}"
        )

        # Quantum feature map
        self.feature_map = zz_feature_map(
            feature_dimension=X_pca.shape[1],
            reps=self.reps
        )

        # Quantum kernel
        self.kernel = FidelityQuantumKernel(
            feature_map=self.feature_map
        )

        print("Calculating quantum training kernel...")

        K_train = self.kernel.evaluate(
            X_pca
        )

        print("Training One-Class SVM...")

        self.model = OneClassSVM(
            kernel="precomputed",
            nu=self.nu
        )

        self.model.fit(K_train)

        print("Quantum model trained successfully!")

        return self

    def predict(self, data):

        if self.model is None:
            raise ValueError(
                "Model has not been trained."
            )

        X = self.preprocess(data)

        X_scaled = self.scaler.transform(X)

        X_pca = self.pca.transform(X_scaled)

        print(
            f"Calculating quantum kernel for "
            f"{len(X_pca)} test samples..."
        )

        K_test = self.kernel.evaluate(
            X_pca,
            self.X_train_pca
        )

        predictions = self.model.predict(K_test)

        scores = self.model.decision_function(
            K_test
        )

        return predictions, scores