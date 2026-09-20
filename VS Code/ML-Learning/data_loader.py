"""
data_loader.py
Dataset discovery, loading, and sampling for N-BaIoT, TON-IoT, and arbitrary numerical CSVs.
Guarantees zero overlap between train and test partitions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    POTENTIAL_DATA_DIRS,
    POTENTIAL_TONIOT_DIRS,
    RANDOM_STATE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_SIZE,
    TONIOT_DEVICE_FAMILIES,
    TONIOT_LABEL_CANDIDATES,
    TONIOT_TYPE_CANDIDATES,
)


class DatasetDiscovery:
    """Discovers and catalogs available N-BaIoT CSV datasets in the repository."""

    @staticmethod
    def get_data_directories() -> List[Path]:
        """Returns all existing data directories from the potential list."""
        return [d for d in POTENTIAL_DATA_DIRS if d.is_dir()]

    @classmethod
    def discover_files(cls) -> Dict[int, Dict[str, Union[Path, Dict[str, Path], None]]]:
        """
        Discovers all available CSV files and groups them by device ID.
        Returns:
            Dict mapping device_id -> { 'benign': Path, 'attacks': { attack_name: Path } }
        """
        devices: Dict[int, Dict] = {}
        scanned_dirs = cls.get_data_directories()

        for data_dir in scanned_dirs:
            for csv_path in data_dir.glob("*.csv"):
                filename = csv_path.name

                # Skip non-traffic summary/info files
                if filename in ("features.csv", "device_info.csv", "data_summary.csv", "iot_data.csv"):
                    continue

                # Match N-BaIoT naming convention: <device_id>.<traffic_type>.<attack_subtype>.csv
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
                    attack_name = ".".join(parts[1:-1])
                    if attack_name not in devices[device_id]["attacks"]:
                        devices[device_id]["attacks"][attack_name] = csv_path

        return devices

    @classmethod
    def summary_report(cls) -> str:
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


class TONIoTDiscovery:
    """
    Discovers TON-IoT telemetry CSVs. TON-IoT files typically carry mixed data types plus
    a 'label' column (0 benign, 1 attack) and a 'type' column naming the attack sub-family.
    """

    @staticmethod
    def get_data_directories() -> List[Path]:
        return [d for d in POTENTIAL_TONIOT_DIRS if d.is_dir()]

    @classmethod
    def discover_files(cls) -> Dict[str, Path]:
        """Maps device-family name -> path to a single labelled telemetry CSV."""
        catalogue: Dict[str, Path] = {}
        for data_dir in cls.get_data_directories():
            for csv_path in data_dir.rglob("*.csv"):
                fname = csv_path.name.lower()
                for family, patterns in TONIOT_DEVICE_FAMILIES.items():
                    if any(p in fname for p in patterns):
                        catalogue.setdefault(family, csv_path)
                        break
        return catalogue

    @classmethod
    def summary_report(cls) -> str:
        found = cls.discover_files()
        if not found:
            return "No TON-IoT CSV files discovered."
        lines = ["=== TON-IoT Dataset Discovery ==="]
        for family in sorted(found.keys()):
            lines.append(f"{family}: {found[family].name}")
        return "\n".join(lines)


class TabularDataLoader:
    """Loads, validates, and cleans numerical tabular data."""

    @staticmethod
    def load_csv(
        file_path: Union[str, Path],
        nrows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str]]]:
        """Loads a CSV file, selects numeric columns, returns data and metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            df = pd.read_csv(path, nrows=nrows)
        except Exception as e:
            raise ValueError(f"Failed to read CSV '{path.name}': {str(e)}")

        total_rows, total_cols = df.shape

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise ValueError(
                f"The file '{path.name}' contains no numerical features."
            )

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
    def load_toniot_csv(
        file_path: Union[str, Path],
        nrows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Union[int, float, str]]]:
        """
        Loads a TON-IoT telemetry CSV and cleanly separates benign rows from
        attack rows using the 'label' column.
        Returns (benign_numeric_df, attack_numeric_df, meta).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path, nrows=nrows, low_memory=False)

        label_col = None
        for cand in TONIOT_LABEL_CANDIDATES:
            if cand in df.columns:
                label_col = cand
                break

        type_col = None
        for cand in TONIOT_TYPE_CANDIDATES:
            if cand in df.columns:
                type_col = cand
                break

        if label_col is None:
            raise ValueError(
                f"TON-IoT file '{path.name}' does not contain a label column "
                f"(expected one of {TONIOT_LABEL_CANDIDATES})."
            )

        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        drop_cols = [c for c in [label_col, type_col] if c is not None]
        feature_df = df.drop(columns=drop_cols, errors="ignore")

        # Encode object columns; drop timestamp-like columns
        for col in feature_df.columns:
            if feature_df[col].dtype == "object":
                # Simple deterministic categorical encoding by hashed factorization
                feature_df[col] = pd.factorize(feature_df[col].astype(str), sort=True)[0].astype(float)

        feature_df = feature_df.select_dtypes(include=[np.number]).copy()
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        benign_mask = labels == 0
        attack_mask = labels != 0

        meta = {
            "file_name": path.name,
            "total_rows": int(len(df)),
            "benign_rows": int(benign_mask.sum()),
            "attack_rows": int(attack_mask.sum()),
            "numeric_columns": int(feature_df.shape[1]),
            "attack_types": sorted(df[type_col].dropna().astype(str).unique().tolist())
            if type_col is not None else [],
        }

        return (
            feature_df.loc[benign_mask].reset_index(drop=True),
            feature_df.loc[attack_mask].reset_index(drop=True),
            meta,
        )

    @staticmethod
    def partition_benign(
        benign_df: pd.DataFrame,
        train_size: int = DEFAULT_TRAIN_SIZE,
        test_size: int = DEFAULT_TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Partitions benign data into non-overlapping training and testing subsets."""
        total_needed = train_size + test_size
        n_available = len(benign_df)

        if n_available < total_needed:
            train_ratio = train_size / total_needed
            actual_train = int(n_available * train_ratio)
            actual_test = n_available - actual_train
            train_size = max(1, actual_train)
            test_size = max(1, actual_test)

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
        rng = np.random.default_rng(random_state)
        sample_pool = benign_reference.sample(
            n=min(n_samples, len(benign_reference)),
            replace=len(benign_reference) < n_samples,
            random_state=random_state
        ).copy().reset_index(drop=True)

        stds = sample_pool.std().replace(0, 1.0).fillna(1.0)

        perturbed = sample_pool.copy()
        for col in perturbed.columns:
            mask = rng.random(len(perturbed)) > 0.4
            shift_direction = rng.choice([-1, 1], size=len(perturbed))
            shifts = shift_direction * (perturbation_factor * stds[col] + rng.exponential(stds[col], len(perturbed)))
            perturbed.loc[mask, col] = perturbed.loc[mask, col] + shifts[mask]

        return perturbed


class SyntheticBenignGenerator:
    """
    Produces reproducible, purely synthetic benign IoT-like feature vectors when no
    real dataset files are available in the environment. This is used only for
    software validation and dashboard demos; results generated from it are labelled
    'synthetic dataset' in the UI so they are never confused with real benchmarks.
    """

    @staticmethod
    def generate(
        n_samples: int = 800,
        n_features: int = 20,
        random_state: int = RANDOM_STATE,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        base = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
        # Introduce a mild covariance structure
        mix = rng.normal(0.0, 0.6, size=(n_features, n_features))
        mixed = base @ mix
        cols = [f"feat_{i:02d}" for i in range(n_features)]
        return pd.DataFrame(mixed, columns=cols)
