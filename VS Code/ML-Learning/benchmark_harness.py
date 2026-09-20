"""
benchmark_harness.py
Multi-seed benchmark orchestrator with roadmap-compliant latency measurement
and persistent, audit-trail run logging.

For every (dataset, detector) pair the harness executes the whole train-on-normal
pipeline under each seed. Metrics are reduced to mean and standard deviation
across seeds; latency is reported in microseconds per sample (median + p95)
against a warm-up-then-time protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from config import (
    LATENCY_WARMUP_SAMPLES,
    LATENCY_TEST_BATCH,
    LATENCY_REPEATS,
    RUNS_DIR,
    RANDOM_STATE,
)
from data_loader import (
    TabularDataLoader,
    SyntheticAnomalyGenerator,
)
from evaluation import AnomalyEvaluation


DetectorFactory = Callable[[int], Any]  # takes seed -> detector instance


def _git_commit_hash() -> str:
    """Best-effort git commit sha of the working tree, empty string on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


@dataclass
class LatencyReport:
    """Latency benchmark for a fitted detector on a fixed test batch."""
    detector: str
    batch_size: int
    warmup_samples: int
    repeats: int
    us_per_sample_median: float
    us_per_sample_p95: float
    us_per_sample_min: float
    us_per_sample_max: float


@dataclass
class SeedRun:
    """One seed's evaluation output for one detector."""
    detector: str
    seed: int
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    runtime_seconds: float
    train_samples: int
    test_samples: int


@dataclass
class AggregatedResult:
    """Mean +/- std over seeds, plus curve arrays from the first seed for visualisation."""
    detector: str
    seeds: List[int]
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    f1_mean: float
    f1_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    runtime_seconds_mean: float
    runtime_seconds_std: float
    per_seed: List[SeedRun] = field(default_factory=list)
    latency: Optional[LatencyReport] = None
    representative_result: Optional[Dict[str, Any]] = None  # first-seed full dict for plotting


def measure_latency(
    detector: Any,
    warmup_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = LATENCY_TEST_BATCH,
    warmup_samples: int = LATENCY_WARMUP_SAMPLES,
    repeats: int = LATENCY_REPEATS,
) -> LatencyReport:
    """
    Roadmap latency protocol:
      - single-threaded, warm-up discarded,
      - median + p95 over `repeats` runs on a fixed `batch_size` sample,
      - reported in microseconds per sample.
    """
    # Warmup
    if warmup_samples > 0:
        warmup = warmup_df.iloc[:min(warmup_samples, len(warmup_df))]
        detector.predict(warmup)

    batch = test_df.iloc[:min(batch_size, len(test_df))]
    per_call_us: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        detector.predict(batch)
        elapsed = time.perf_counter() - t0
        per_call_us.append((elapsed * 1e6) / len(batch))

    per_call_us_arr = np.asarray(per_call_us, dtype=float)
    return LatencyReport(
        detector=getattr(detector, "name", type(detector).__name__),
        batch_size=len(batch),
        warmup_samples=warmup_samples,
        repeats=len(per_call_us_arr),
        us_per_sample_median=float(np.median(per_call_us_arr)),
        us_per_sample_p95=float(np.percentile(per_call_us_arr, 95)),
        us_per_sample_min=float(np.min(per_call_us_arr)),
        us_per_sample_max=float(np.max(per_call_us_arr)),
    )


def _partition_for_seed(
    benign_df: pd.DataFrame,
    attack_df: Optional[pd.DataFrame],
    train_size: int,
    test_size: int,
    seed: int,
    use_synthetic: bool,
    perturbation_factor: float = 3.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Deterministic split for a given seed, keeping identical semantics across detectors."""
    benign_train, benign_test = TabularDataLoader.partition_benign(
        benign_df, train_size=train_size, test_size=test_size, random_state=seed,
    )
    if use_synthetic or attack_df is None or len(attack_df) == 0:
        atk_test = SyntheticAnomalyGenerator.generate(
            benign_test,
            n_samples=test_size,
            perturbation_factor=perturbation_factor,
            random_state=seed,
        )
    else:
        atk_test = TabularDataLoader.sample_attack(
            attack_df, sample_size=test_size, random_state=seed
        )
    test_df = pd.concat([benign_test, atk_test], ignore_index=True)
    y_true = np.array([0] * len(benign_test) + [1] * len(atk_test), dtype=int)
    return benign_train, benign_test, test_df, y_true


def run_multi_seed_benchmark(
    detector_factories: Dict[str, DetectorFactory],
    benign_df: pd.DataFrame,
    attack_df: Optional[pd.DataFrame],
    dataset_label: str,
    evaluation_type: str,
    train_size: int,
    test_size: int,
    seeds: List[int],
    use_synthetic: bool,
    latency_batch: int = LATENCY_TEST_BATCH,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, AggregatedResult], Dict[str, Any]]:
    """
    Runs the full pipeline for every (detector, seed) pair and returns
    aggregated results plus a metadata dictionary suitable for persistence.
    """
    per_detector_seeds: Dict[str, List[Tuple[Dict[str, Any], SeedRun]]] = {
        name: [] for name in detector_factories
    }
    per_detector_last: Dict[str, Any] = {}
    per_detector_first_test: Dict[str, pd.DataFrame] = {}
    per_detector_first_benign_train: Dict[str, pd.DataFrame] = {}

    features_before = int(benign_df.shape[1])
    total_iters = len(seeds) * len(detector_factories)
    step = 0

    for seed in seeds:
        benign_train, _benign_test, test_df, y_true = _partition_for_seed(
            benign_df,
            attack_df,
            train_size=train_size,
            test_size=test_size,
            seed=seed,
            use_synthetic=use_synthetic,
        )

        for name, factory in detector_factories.items():
            step += 1
            if progress is not None:
                progress(f"[{step}/{total_iters}] seed={seed}, detector={name}")

            det = factory(seed)
            t0 = time.perf_counter()
            det.fit(benign_train)
            preds, scores, _raw = det.predict(test_df)
            runtime = time.perf_counter() - t0

            full = AnomalyEvaluation.evaluate(
                model_name=name,
                predictions=preds,
                anomaly_scores=scores,
                runtime_seconds=runtime,
                y_true=y_true,
                evaluation_type=evaluation_type,
                dataset_name=dataset_label,
                train_samples=len(benign_train),
                test_samples=len(test_df),
                features_before_pca=features_before,
                quantum_features=getattr(det, "n_components", 0),
            )
            seed_run = SeedRun(
                detector=name,
                seed=seed,
                roc_auc=full.get("roc_auc"),
                pr_auc=full.get("pr_auc"),
                f1=full.get("f1"),
                precision=full.get("precision"),
                recall=full.get("recall"),
                runtime_seconds=float(runtime),
                train_samples=len(benign_train),
                test_samples=len(test_df),
            )
            per_detector_seeds[name].append((full, seed_run))
            per_detector_last[name] = det
            if name not in per_detector_first_test:
                per_detector_first_test[name] = test_df
                per_detector_first_benign_train[name] = benign_train

    # Latency measurement uses the last-fitted detector for each name
    aggregated: Dict[str, AggregatedResult] = {}
    for name, runs in per_detector_seeds.items():
        seed_runs = [r for _, r in runs]
        auc_vals = np.array([r.roc_auc for r in seed_runs if r.roc_auc is not None], dtype=float)
        pr_vals = np.array([r.pr_auc for r in seed_runs if r.pr_auc is not None], dtype=float)
        f1_vals = np.array([r.f1 for r in seed_runs if r.f1 is not None], dtype=float)
        prec_vals = np.array([r.precision for r in seed_runs if r.precision is not None], dtype=float)
        rec_vals = np.array([r.recall for r in seed_runs if r.recall is not None], dtype=float)
        rt_vals = np.array([r.runtime_seconds for r in seed_runs], dtype=float)

        det = per_detector_last[name]
        lat_report: Optional[LatencyReport] = None
        try:
            lat_report = measure_latency(
                det,
                warmup_df=per_detector_first_test[name],
                test_df=per_detector_first_test[name],
                batch_size=latency_batch,
            )
        except Exception as e:
            lat_report = None
            if progress is not None:
                progress(f"Latency measurement failed for {name}: {e}")

        aggregated[name] = AggregatedResult(
            detector=name,
            seeds=list(seeds),
            roc_auc_mean=float(np.mean(auc_vals)) if len(auc_vals) else float("nan"),
            roc_auc_std=float(np.std(auc_vals)) if len(auc_vals) else float("nan"),
            pr_auc_mean=float(np.mean(pr_vals)) if len(pr_vals) else float("nan"),
            pr_auc_std=float(np.std(pr_vals)) if len(pr_vals) else float("nan"),
            f1_mean=float(np.mean(f1_vals)) if len(f1_vals) else float("nan"),
            f1_std=float(np.std(f1_vals)) if len(f1_vals) else float("nan"),
            precision_mean=float(np.mean(prec_vals)) if len(prec_vals) else float("nan"),
            precision_std=float(np.std(prec_vals)) if len(prec_vals) else float("nan"),
            recall_mean=float(np.mean(rec_vals)) if len(rec_vals) else float("nan"),
            recall_std=float(np.std(rec_vals)) if len(rec_vals) else float("nan"),
            runtime_seconds_mean=float(np.mean(rt_vals)) if len(rt_vals) else float("nan"),
            runtime_seconds_std=float(np.std(rt_vals)) if len(rt_vals) else float("nan"),
            per_seed=seed_runs,
            latency=lat_report,
            representative_result=runs[0][0] if runs else None,
        )

    metadata = {
        "dataset_label": dataset_label,
        "evaluation_type": evaluation_type,
        "train_size": train_size,
        "test_size": test_size,
        "seeds": list(seeds),
        "features_before_pca": features_before,
        "git_commit": _git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return aggregated, metadata


def aggregated_to_dataframe(agg: Dict[str, AggregatedResult]) -> pd.DataFrame:
    """Flattens the aggregated result map into a display-ready DataFrame."""
    rows = []
    for name, r in agg.items():
        row = {
            "detector": name,
            "seeds_n": len(r.seeds),
            "roc_auc": f"{r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}",
            "pr_auc": f"{r.pr_auc_mean:.4f} ± {r.pr_auc_std:.4f}",
            "f1": f"{r.f1_mean:.4f} ± {r.f1_std:.4f}",
            "precision": f"{r.precision_mean:.4f} ± {r.precision_std:.4f}",
            "recall": f"{r.recall_mean:.4f} ± {r.recall_std:.4f}",
            "runtime_s": f"{r.runtime_seconds_mean:.3f} ± {r.runtime_seconds_std:.3f}",
            "roc_auc_mean": r.roc_auc_mean,
            "roc_auc_std": r.roc_auc_std,
            "pr_auc_mean": r.pr_auc_mean,
            "pr_auc_std": r.pr_auc_std,
            "f1_mean": r.f1_mean,
            "runtime_seconds_mean": r.runtime_seconds_mean,
        }
        if r.latency is not None:
            row["latency_us_median"] = round(r.latency.us_per_sample_median, 3)
            row["latency_us_p95"] = round(r.latency.us_per_sample_p95, 3)
        else:
            row["latency_us_median"] = float("nan")
            row["latency_us_p95"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def latency_to_dataframe(agg: Dict[str, AggregatedResult]) -> pd.DataFrame:
    rows = []
    for name, r in agg.items():
        if r.latency is None:
            continue
        lat = r.latency
        rows.append({
            "detector": name,
            "batch_size": lat.batch_size,
            "warmup_samples": lat.warmup_samples,
            "repeats": lat.repeats,
            "us_per_sample_median": round(lat.us_per_sample_median, 3),
            "us_per_sample_p95": round(lat.us_per_sample_p95, 3),
            "us_per_sample_min": round(lat.us_per_sample_min, 3),
            "us_per_sample_max": round(lat.us_per_sample_max, 3),
        })
    return pd.DataFrame(rows)


def persist_run(
    agg: Dict[str, AggregatedResult],
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path = RUNS_DIR,
) -> Path:
    """Writes YAML config, CSV metrics, and JSON audit blob into a timestamped folder."""
    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    dst = Path(output_dir) / run_id
    dst.mkdir(parents=True, exist_ok=True)

    (dst / "config.yaml").write_text(
        yaml.safe_dump({"metadata": metadata, "config": config}, sort_keys=False, allow_unicode=True)
    )

    df = aggregated_to_dataframe(agg)
    df.to_csv(dst / "metrics_aggregated.csv", index=False)
    latency_to_dataframe(agg).to_csv(dst / "metrics_latency.csv", index=False)

    per_seed_rows = []
    for name, r in agg.items():
        for sr in r.per_seed:
            per_seed_rows.append(asdict(sr))
    pd.DataFrame(per_seed_rows).to_csv(dst / "metrics_per_seed.csv", index=False)

    audit = {
        "metadata": metadata,
        "config": config,
        "detectors": list(agg.keys()),
    }
    (dst / "audit.json").write_text(json.dumps(audit, indent=2, default=str))

    return dst
