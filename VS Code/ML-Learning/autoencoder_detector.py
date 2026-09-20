"""
autoencoder_detector.py
Shallow feed-forward autoencoder for unsupervised anomaly detection.
Trained strictly on benign data. Anomaly score = per-sample mean squared reconstruction error.
"""

from typing import Dict, Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_AE_BOTTLENECK,
    DEFAULT_AE_HIDDEN,
    DEFAULT_AE_EPOCHS,
    DEFAULT_AE_BATCH_SIZE,
    DEFAULT_AE_LR,
    DEFAULT_AE_PATIENCE,
    RANDOM_STATE,
)
from preprocessing import DataPreprocessor


class _ShallowAutoencoder(nn.Module):
    """Encoder-bottleneck-decoder MLP with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck: int):
        super().__init__()
        # Guarantee monotonically shrinking capacity into the bottleneck
        h = max(hidden_dim, bottleneck + 1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h),
            nn.ReLU(),
            nn.Linear(h, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderAnomalyDetector:
    """
    Shallow autoencoder trained only on benign IoT feature vectors after PCA.
    Per-sample reconstruction MSE is used as the anomaly score
    (higher = more anomalous), matching the roadmap protocol.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_PCA_COMPONENTS,
        hidden_dim: int = DEFAULT_AE_HIDDEN,
        bottleneck: int = DEFAULT_AE_BOTTLENECK,
        epochs: int = DEFAULT_AE_EPOCHS,
        batch_size: int = DEFAULT_AE_BATCH_SIZE,
        learning_rate: float = DEFAULT_AE_LR,
        patience: int = DEFAULT_AE_PATIENCE,
        random_state: int = RANDOM_STATE,
    ):
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state

        self.name = "Shallow Autoencoder"
        self.preprocessor: Optional[DataPreprocessor] = None
        self.model: Optional[_ShallowAutoencoder] = None
        self.threshold: Optional[float] = None
        self.loss_history: list = []

        self.timings: Dict[str, float] = {
            "preprocessing_time": 0.0,
            "fit_time": 0.0,
            "predict_time": 0.0,
            "total_runtime": 0.0,
        }

    def _prepare_input_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, benign_train_df: pd.DataFrame) -> "AutoencoderAnomalyDetector":
        start_total = time.perf_counter()

        # Deterministic training
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        t0 = time.perf_counter()
        self.preprocessor = DataPreprocessor(
            n_components=self.n_components, random_state=self.random_state
        )
        X_pca = self.preprocessor.fit_transform(benign_train_df)
        self.timings["preprocessing_time"] = time.perf_counter() - t0

        input_dim = X_pca.shape[1]
        bottleneck = min(self.bottleneck, max(1, input_dim - 1))

        self.model = _ShallowAutoencoder(
            input_dim=input_dim, hidden_dim=self.hidden_dim, bottleneck=bottleneck
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Hold out 20% of benign train as a validation split for early stopping
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(X_pca))
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(idx)))
        train_idx, val_idx = idx[:cut], idx[cut:]

        X_tr = self._prepare_input_tensor(X_pca[train_idx])
        X_val = self._prepare_input_tensor(X_pca[val_idx]) if len(val_idx) > 0 else X_tr

        dataset = TensorDataset(X_tr)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        t_fit = time.perf_counter()
        best_val = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                x_hat = self.model(batch_x)
                loss = loss_fn(x_hat, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            avg_loss = epoch_loss / max(1, len(loader))

            # Validation
            self.model.eval()
            with torch.no_grad():
                x_hat_val = self.model(X_val)
                val_loss = float(loss_fn(x_hat_val, X_val).item())

            self.loss_history.append({"epoch": epoch, "train": avg_loss, "val": val_loss})

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Set anomaly threshold at 95th percentile of benign-train reconstruction error
        self.model.eval()
        with torch.no_grad():
            recon = self.model(self._prepare_input_tensor(X_pca))
            errors = ((recon - self._prepare_input_tensor(X_pca)) ** 2).mean(dim=1).cpu().numpy()
        self.threshold = float(np.quantile(errors, 0.95))
        self.timings["fit_time"] = time.perf_counter() - t_fit

        self.timings["total_runtime"] = time.perf_counter() - start_total
        return self

    def predict(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None or self.preprocessor is None or self.threshold is None:
            raise RuntimeError("AutoencoderAnomalyDetector must be fitted before predict.")

        t0 = time.perf_counter()
        X_test_pca = self.preprocessor.transform(test_df)
        X_tensor = self._prepare_input_tensor(X_test_pca)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = ((recon - X_tensor) ** 2).mean(dim=1).cpu().numpy()

        # Convention: anomaly_score is higher = more anomalous (reconstruction MSE fits directly)
        anomaly_scores = errors.astype(float)
        # sklearn-style predictions: +1 normal, -1 anomaly
        predictions = np.where(anomaly_scores <= self.threshold, 1, -1).astype(int)
        decision_raw = self.threshold - anomaly_scores  # positive = inside benign band

        self.timings["predict_time"] = time.perf_counter() - t0
        return predictions, anomaly_scores, decision_raw
