"""
data_loader.py
Dataset discovery, loading, and sampling for N-BaIoT and arbitrary numerical CSVs.
Guarantees zero overlap between train and test partitions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob
import re

import numpy as np
import pandas as pd

from config import (
    POTENTIAL_DATA_DIRS,
    RANDOM_STATE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_SIZE,
)


class DatasetDiscovery:
    """Discovers and catalogs available N-BaIoT CSV datasets in the repository."""

    @staticmethod
    def get_data_directories() -> List[Path]:
        """Returns all existing data directories from the potential list."""
        return [d for d in POTENTIAL_DATA_DIRS if d.is_dir()]

    @classmethod
    def discover_files(cls) -> Dict[str, Dict[str, Path]]:
        """
        Discovers all available CSV files and groups them by device ID.
        Returns:
            Dict mapping device_id -> { 'benign': Path, 'attacks': { attack_name: Path } }
        """
        devices: Dict[str, Dict] = {}
        scanned_dirs = cls.get_data_directories()

        for data_dir in scanned_dirs:
            for csv_path in data_dir.glob("*.csv"):
                filename = csv_path.name

                # Skip non-traffic summary/info files
                if filename in ("features.csv", "device_info.csv", "data_summary.csv", "iot_data.csv"):
                    continue

                # Match N-BaIoT naming convention: <device_id>.<traffic_type>.<attack_subtype>.csv
                # e.g., 1.benign.csv, 1.mirai.ack.csv, 1.gafgyt.tcp.csv
                parts = filename.split(".")
                if len(parts) < 2:
                    continue

                device_str = parts[0]
                if not device_str.isdigit():
                    continue

                device_id = int(device_str)
                if device_id not in devices:
                    devices[device_id] = {"benign": None, "attacks": {}}

                if "benign" in filename.lower():
                    if devices[device_id]["benign"] is None:
                        devices[device_id]["benign"] = csv_path
                else:
                    # Attack file (e.g. gafgyt.combo, mirai.ack)
                    attack_name = ".".join(parts[1:-1])
                    if attack_name not in devices[device_id]["attacks"]:
                        devices[device_id]["attacks"][attack_name] = csv_path

        return devices

    @classmethod
    def summary_report(cls) -> str:
        """Generates a text summary of discovered datasets."""
        devices = cls.discover_files()
        if not devices:
            return "No N-BaIoT CSV files discovered in known data paths."

        lines = ["=== N-BaIoT Dataset Discovery ==="]
        for dev_id in sorted(devices.keys()):
            info = devices[dev_id]
            benign_status = "Available" if info["benign"] else "Missing"
            attack_count = len(info["attacks"])
            attack_names = ", ".join(sorted(info["attacks"].keys())) if attack_count > 0 else "None"
            lines.append(f"Device {dev_id}: Benign [{benign_status}], Attacks ({attack_count}): [{attack_names}]")

        return "\n".join(lines)


class TabularDataLoader:
    """Loads, validates, and cleans numerical tabular data."""

    @staticmethod
    def load_csv(
        file_path: Union[str, Path],
        nrows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
        """
        Loads a CSV file, selects numeric columns, and returns data and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            df = pd.read_csv(path, nrows=nrows)
        except Exception as e:
            raise ValueError(f"Failed to read CSV '{path.name}': {str(e)}")

        total_rows, total_cols = df.shape

        # Filter to numerical features
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise ValueError(
                f"The file '{path.name}' contains no numerical features. Tabular anomaly detection requires numeric data."
            )

        # Handle infinities
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        missing_count = int(numeric_df.isna().sum().sum())

        meta = {
            "file_name": path.name,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "numeric_columns": numeric_df.shape[1],
            "missing_values": missing_count,
        }

        return numeric_df, meta

    @staticmethod
    def partition_benign(
        benign_df: pd.DataFrame,
        train_size: int = DEFAULT_TRAIN_SIZE,
        test_size: int = DEFAULT_TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partitions benign data into non-overlapping training and testing subsets.
        CRITICAL: Prevents data leakage between train and test splits.
        """
        total_needed = train_size + test_size
        n_available = len(benign_df)

        if n_available < total_needed:
            # Adjust sizes proportionally if dataset has fewer rows
            train_ratio = train_size / total_needed
            actual_train = int(n_available * train_ratio)
            actual_test = n_available - actual_train
            train_size = max(1, actual_train)
            test_size = max(1, actual_test)

        # Shuffle deterministically
        shuffled = benign_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        train_part = shuffled.iloc[:train_size].copy()
        test_part = shuffled.iloc[train_size:train_size + test_size].copy()

        return train_part, test_part

    @staticmethod
    def sample_attack(
        attack_df: pd.DataFrame,
        sample_size: int = DEFAULT_TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> pd.DataFrame:
        """Samples attack test records deterministically."""
        n_samples = min(sample_size, len(attack_df))
        return attack_df.sample(n=n_samples, random_state=random_state).reset_index(drop=True).copy()


class SyntheticAnomalyGenerator:
    """
    Generates controlled synthetic anomalies for software validation
    when real attack files are absent or explicitly requested for verification.
    """

    @staticmethod
    def generate(
        benign_reference: pd.DataFrame,
        n_samples: int = 100,
        perturbation_factor: float = 3.5,
        random_state: int = RANDOM_STATE,
    ) -> pd.DataFrame:
        """
        Creates synthetic anomalies by applying heavy-tail perturbation and feature injection
        to reference benign distributions.
        """
        rng = np.random.default_rng(random_state)
        sample_pool = benign_reference.sample(
            n=min(n_samples, len(benign_reference)),
            replace=len(benign_reference) < n_samples,
            random_state=random_state
        ).copy().reset_index(drop=True)

        # Calculate statistics
        means = sample_pool.mean()
        stds = sample_pool.std().replace(0, 1.0).fillna(1.0)

        perturbed = sample_pool.copy()
        for col in perturbed.columns:
            # Inject extreme z-score shifts into random subsets of features
            mask = rng.random(len(perturbed)) > 0.4
            shift_direction = rng.choice([-1, 1], size=len(perturbed))
            shifts = shift_direction * (perturbation_factor * stds[col] + rng.exponential(stds[col], len(perturbed)))
            perturbed.loc[mask, col] = perturbed.loc[mask, col] + shifts[mask]

        return perturbed
