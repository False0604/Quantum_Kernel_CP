"""
app.py
Executive Research Dashboard: Quantum-Enabled IoT Anomaly Detection.

Compares Quantum Kernel One-Class SVM (ideal or NISQ-noise) against Classical
Baselines (RBF OCSVM, Isolation Forest, LOF, Shallow Autoencoder) with multi-seed
reproducibility, latency benchmarking (us/sample), and persistent run logs.
"""

import io
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_QUANTUM_REPS,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_SIZE,
    DEFAULT_NU,
    DEFAULT_SEED_LIST,
    RANDOM_STATE,
    DEVICE_NAMES,
    NOISE_PRESETS,
    LATENCY_TEST_BATCH,
    RUNS_DIR,
)
from data_loader import (
    DatasetDiscovery,
    TONIoTDiscovery,
    TabularDataLoader,
    SyntheticAnomalyGenerator,
    SyntheticBenignGenerator,
)
from quantum_detector import QuantumKernelAnomalyDetector
from classical_detector import (
    ClassicalRBFAnomalyDetector,
    IsolationForestAnomalyDetector,
    LocalOutlierFactorDetector,
)
from autoencoder_detector import AutoencoderAnomalyDetector
from evaluation import AnomalyEvaluation
from visualization import Visualizer
from advanced_quantum_engine import (
    QuantumGramSpectralAnalyzer,
    QuantumInformationSpectroscopy,
    VariationalQuantumMetricLearner,
    NISQNoiseRobustnessSimulator,
    QuantumClassicalHybridEnsemble,
)
from benchmark_harness import (
    run_multi_seed_benchmark,
    aggregated_to_dataframe,
    latency_to_dataframe,
    persist_run,
)


st.set_page_config(
    page_title="Quantum IoT Anomaly Detection | Research Prototype",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .header-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        color: #f8fafc;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }
    .header-title { font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 6px; letter-spacing: -0.5px; }
    .header-subtitle { font-size: 15px; color: #94a3b8; font-weight: 400; }
    .kpi-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 16px 20px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        border-top: 3px solid #2563eb;
    }
    .kpi-label { font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .kpi-value { font-size: 22px; font-weight: 700; color: #0f172a; }
    .kpi-sub { font-size: 12px; color: #10b981; font-weight: 500; margin-top: 2px; }
    .observation-card {
        background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #3b82f6;
        border-radius: 8px; padding: 16px 20px; margin-top: 18px; margin-bottom: 22px;
        color: #334155; font-size: 14px; line-height: 1.6;
    }
    .circuit-box {
        font-family: 'JetBrains Mono', monospace; background: #0f172a;
        color: #38bdf8; padding: 16px; border-radius: 8px;
        overflow-x: auto; font-size: 12.5px; line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### Benchmark Control")

    st.markdown("#### 1. Data Source")
    dataset_choice = st.radio(
        "Dataset family",
        ["N-BaIoT (network)", "TON-IoT (telemetry)", "Synthetic (no files needed)", "Upload custom CSV"],
        index=0,
        label_visibility="collapsed",
    )

    benign_df: Optional[pd.DataFrame] = None
    attack_df: Optional[pd.DataFrame] = None
    dataset_label = ""
    evaluation_type_hint = ""
    use_synthetic = False
    attack_choice_display = "Synthetic"

    if dataset_choice == "N-BaIoT (network)":
        discovered = DatasetDiscovery.discover_files()
        if not discovered:
            st.warning(
                "No N-BaIoT CSVs discovered on disk. The dashboard will run against "
                "a purely synthetic benign feed so the pipeline can still be validated."
            )
        else:
            device_options = sorted(discovered.keys())
            device_id = st.selectbox(
                "Target IoT Device",
                device_options,
                format_func=lambda d: f"Device {d}: {DEVICE_NAMES.get(d, 'IoT Device')}",
            )
            attacks = discovered[device_id]["attacks"]
            attack_labels = ["Synthetic anomaly"] + sorted(attacks.keys())
            attack_choice_display = st.selectbox("Attack traffic", attack_labels)

    elif dataset_choice == "TON-IoT (telemetry)":
        toniot_files = TONIoTDiscovery.discover_files()
        if not toniot_files:
            st.warning(
                "No TON-IoT CSVs discovered on disk. Place them under a `data/TON-IoT/` "
                "folder to activate this branch. Falling back to synthetic benign feed."
            )
        else:
            family = st.selectbox("TON-IoT device family", sorted(toniot_files.keys()))
            attack_choice_display = "TON-IoT labelled attacks"

    elif dataset_choice == "Upload custom CSV":
        uploaded_file = st.file_uploader("Upload numeric tabular CSV", type=["csv"])
        if uploaded_file is not None:
            raw = pd.read_csv(uploaded_file)
            num = raw.select_dtypes(include=[np.number])
            if num.empty:
                st.error("Uploaded CSV has no numeric columns.")
            else:
                benign_df = num
                dataset_label = f"Uploaded CSV: {uploaded_file.name}"
                use_synthetic = True
                attack_choice_display = "Synthetic anomaly on uploaded CSV"

    st.markdown("---")
    st.markdown("#### 2. Detectors to benchmark")
    include_qkernel = st.checkbox("Quantum Kernel OCSVM", value=True)
    include_rbf = st.checkbox("Classical RBF OCSVM", value=True)
    include_iforest = st.checkbox("Isolation Forest", value=True)
    include_lof = st.checkbox("Local Outlier Factor", value=True)
    include_autoenc = st.checkbox("Shallow Autoencoder", value=True)

    st.markdown("---")
    st.markdown("#### 3. Quantum architecture")
    n_components = st.select_slider(
        "Register size (qubits / PCA components)",
        options=[2, 4, 6, 8],
        value=2,
    )
    quantum_reps = st.selectbox("ZZFeatureMap repetitions", options=[1, 2], index=0)
    noise_choice = st.selectbox(
        "NISQ noise model",
        list(NOISE_PRESETS.keys()),
        index=0,
        help="Applies an analytical depolarising channel to both encoded states.",
    )
    noise_prob = NOISE_PRESETS[noise_choice]

    st.markdown("---")
    st.markdown("#### 4. Execution & reproducibility")
    train_sample_limit = st.slider("Benign training samples (N)", 50, 400, 120, 10)
    test_sample_limit = st.slider("Test samples per class (M)", 25, 250, 60, 5)
    ocsvm_nu = st.slider("OCSVM outlier bound nu", 0.01, 0.50, 0.10, 0.01)

    n_seeds = st.slider(
        "Number of seeds (multi-run mean ± std)",
        min_value=1,
        max_value=len(DEFAULT_SEED_LIST),
        value=3,
        help="Runs the full pipeline under each seed and reports mean ± std.",
    )
    latency_batch = st.slider("Latency batch size (samples)", 20, 400, LATENCY_TEST_BATCH, 20)

    st.markdown("---")
    persist_choice = st.checkbox("Save this run to results/runs/", value=True)
    run_benchmark = st.button("Execute benchmark", use_container_width=True, type="primary")


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(f"""
<div class="header-card">
    <div class="header-title">Quantum-Enabled Anomaly Detection for IoT Networks</div>
    <div class="header-subtitle">Empirical Comparison Across Ideal / NISQ Quantum Kernels and Four Classical Baselines &mdash; Multi-Seed, Reproducible.</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA ACQUISITION
# ==============================================================================
def _load_dataset() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str, str, bool]:
    """Resolves the chosen dataset into (benign_df, attack_df, label, eval_type, use_synthetic)."""
    if dataset_choice == "N-BaIoT (network)":
        discovered = DatasetDiscovery.discover_files()
        if not discovered:
            return (SyntheticBenignGenerator.generate(n_samples=800, n_features=20),
                    None, "Synthetic N-BaIoT-style feed",
                    "Synthetic anomaly validation", True)
        device_id_local = sorted(discovered.keys())[0]  # sidebar selected value re-used
        # Re-run sidebar lookup to grab the actually chosen id/attack (values persisted implicitly)
        # We rely on session_state to remember prior choice; safe default is first device.
        for k, v in st.session_state.items():
            if k.startswith("selectbox") and isinstance(v, int) and v in discovered:
                device_id_local = v
        info = discovered[device_id_local]
        benign_path = info["benign"]
        if benign_path is None:
            return (SyntheticBenignGenerator.generate(n_samples=800, n_features=20),
                    None, "Synthetic N-BaIoT-style feed",
                    "Synthetic anomaly validation", True)
        benign_df_local, _ = TabularDataLoader.load_csv(
            benign_path, nrows=(train_sample_limit + test_sample_limit) * 4
        )
        label = f"N-BaIoT Device {device_id_local} ({DEVICE_NAMES.get(device_id_local, 'IoT Device')})"

        if attack_choice_display == "Synthetic anomaly" or attack_choice_display not in info["attacks"]:
            return benign_df_local, None, label, "Synthetic anomaly validation", True

        attack_path = info["attacks"][attack_choice_display]
        attack_df_local, _ = TabularDataLoader.load_csv(attack_path, nrows=test_sample_limit * 4)
        return benign_df_local, attack_df_local, label, f"Real attack ({attack_choice_display})", False

    if dataset_choice == "TON-IoT (telemetry)":
        toniot_files = TONIoTDiscovery.discover_files()
        if not toniot_files:
            return (SyntheticBenignGenerator.generate(n_samples=600, n_features=10),
                    None, "Synthetic TON-IoT-style feed",
                    "Synthetic anomaly validation", True)
        family_local = sorted(toniot_files.keys())[0]
        for k, v in st.session_state.items():
            if isinstance(v, str) and v in toniot_files:
                family_local = v
        path = toniot_files[family_local]
        benign_local, attack_local, meta = TabularDataLoader.load_toniot_csv(
            path, nrows=(train_sample_limit + test_sample_limit) * 6
        )
        label = f"TON-IoT {family_local}"
        if len(attack_local) == 0:
            return benign_local, None, label, "Synthetic anomaly validation (no attack rows)", True
        return benign_local, attack_local, label, "Real TON-IoT labelled attack", False

    if dataset_choice == "Synthetic (no files needed)":
        return (SyntheticBenignGenerator.generate(n_samples=800, n_features=15),
                None, "Synthetic IoT-like feed",
                "Synthetic anomaly validation", True)

    # Upload custom CSV
    return benign_df, None, dataset_label or "Uploaded CSV", "Synthetic anomaly validation", True


benign_df, attack_df, dataset_label_resolved, evaluation_type_resolved, use_synthetic_resolved = _load_dataset()

if benign_df is not None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Target Dataset</div>
            <div class="kpi-value">{dataset_label_resolved.split('(')[0].strip()}</div>
            <div class="kpi-sub">{benign_df.shape[0]:,} benign rows available</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Raw Features</div>
            <div class="kpi-value">{benign_df.shape[1]} dims</div>
            <div class="kpi-sub">PCA target: {n_components} qubits</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        atk_rows = len(attack_df) if attack_df is not None else 0
        atk_label = "Synthetic" if use_synthetic_resolved else "Real"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Attack Traffic</div>
            <div class="kpi-value">{atk_label}</div>
            <div class="kpi-sub">{atk_rows:,} attack rows loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Quantum Setup</div>
            <div class="kpi-value">{n_components} qubits @ p={noise_prob:.3f}</div>
            <div class="kpi-sub">Hilbert dim: 2^{n_components} = {2**n_components}</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# BENCHMARK EXECUTION
# ==============================================================================
def _detector_factories() -> Dict[str, Callable[[int], Any]]:
    facts: Dict[str, Callable[[int], Any]] = {}
    if include_qkernel:
        def _q(seed: int):
            return QuantumKernelAnomalyDetector(
                n_components=n_components,
                reps=quantum_reps,
                nu=ocsvm_nu,
                train_sample_limit=train_sample_limit,
                noise_prob=noise_prob,
                random_state=seed,
            )
        facts["Quantum Kernel OCSVM"] = _q
    if include_rbf:
        def _rbf(seed: int):
            return ClassicalRBFAnomalyDetector(
                n_components=n_components, nu=ocsvm_nu, random_state=seed
            )
        facts["Classical RBF OCSVM"] = _rbf
    if include_iforest:
        def _if(seed: int):
            return IsolationForestAnomalyDetector(
                n_components=n_components, random_state=seed
            )
        facts["Isolation Forest"] = _if
    if include_lof:
        def _lof(seed: int):
            return LocalOutlierFactorDetector(
                n_components=n_components, random_state=seed
            )
        facts["Local Outlier Factor"] = _lof
    if include_autoenc:
        def _ae(seed: int):
            return AutoencoderAnomalyDetector(
                n_components=n_components, random_state=seed
            )
        facts["Shallow Autoencoder"] = _ae
    return facts


if run_benchmark and benign_df is not None:
    factories = _detector_factories()
    if not factories:
        st.error("At least one detector must be selected.")
    else:
        status_box = st.empty()
        seeds = DEFAULT_SEED_LIST[:n_seeds]
        status_box.info(
            f"Running {len(factories)} detector(s) across {len(seeds)} seed(s) on {dataset_label_resolved}"
        )

        def _progress(msg: str):
            status_box.info(msg)

        aggregated, metadata = run_multi_seed_benchmark(
            detector_factories=factories,
            benign_df=benign_df,
            attack_df=attack_df,
            dataset_label=dataset_label_resolved,
            evaluation_type=evaluation_type_resolved,
            train_size=train_sample_limit,
            test_size=test_sample_limit,
            seeds=seeds,
            use_synthetic=use_synthetic_resolved,
            latency_batch=latency_batch,
            progress=_progress,
        )

        config = {
            "dataset_choice": dataset_choice,
            "n_components": n_components,
            "quantum_reps": quantum_reps,
            "noise_prob": noise_prob,
            "noise_preset": noise_choice,
            "train_sample_limit": train_sample_limit,
            "test_sample_limit": test_sample_limit,
            "ocsvm_nu": ocsvm_nu,
            "seeds": seeds,
            "attack_choice_display": attack_choice_display,
        }
        run_dir = None
        if persist_choice:
            run_dir = persist_run(aggregated, metadata, config)

        st.session_state["aggregated"] = aggregated
        st.session_state["metadata"] = metadata
        st.session_state["config"] = config
        st.session_state["run_dir"] = run_dir
        status_box.success("Benchmark complete.")


# ==============================================================================
# RESULTS DASHBOARD
# ==============================================================================
if "aggregated" in st.session_state:
    aggregated: Dict[str, Any] = st.session_state["aggregated"]
    metadata: Dict[str, Any] = st.session_state["metadata"]
    config_used: Dict[str, Any] = st.session_state["config"]
    run_dir = st.session_state.get("run_dir")

    if run_dir is not None:
        st.info(f"Run saved to `{run_dir}` (git commit: `{metadata.get('git_commit') or 'n/a'}`).")

    t_bench, t_latency, t_quantum, t_scree, t_scan, t_export = st.tabs([
        "Benchmark comparison",
        "Latency (µs/sample)",
        "Quantum circuit & spectral",
        "Feature space (PCA)",
        "Single-packet scanner",
        "Export & LaTeX",
    ])

    # ----------------- TAB 1: BENCHMARK -----------------
    with t_bench:
        st.markdown(f"#### Empirical evaluation (mean ± std across {len(config_used['seeds'])} seeds)")
        df_agg = aggregated_to_dataframe(aggregated)
        display_cols = ["detector", "roc_auc", "pr_auc", "f1", "precision", "recall", "runtime_s"]
        st.dataframe(df_agg[display_cols], use_container_width=True)

        # Bar chart of mean ROC-AUC with std error bars
        names = list(aggregated.keys())
        auc_means = [aggregated[n].roc_auc_mean for n in names]
        auc_stds = [aggregated[n].roc_auc_std for n in names]
        fig_bar, ax_bar = plt.subplots(figsize=(9, 4.6), dpi=140)
        colors = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
        bars = ax_bar.bar(
            names, auc_means, yerr=auc_stds, capsize=6,
            color=[colors[i % len(colors)] for i in range(len(names))],
            edgecolor="#0f172a", linewidth=0.8,
        )
        for bar, mean, std in zip(bars, auc_means, auc_stds):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, mean + std + 0.02),
                f"{mean:.3f}\n±{std:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_ylabel("ROC-AUC (mean, error bar = std over seeds)")
        ax_bar.set_title("Detector comparison across seeds")
        ax_bar.grid(True, axis="y", linestyle=":", alpha=0.5)
        plt.setp(ax_bar.get_xticklabels(), rotation=15, ha="right")
        fig_bar.tight_layout()
        st.pyplot(fig_bar)

        # ROC / PR curves using the first seed's data (representative sample)
        reps = [r.representative_result for r in aggregated.values() if r.representative_result is not None]
        if reps:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(Visualizer.plot_roc_curves(reps))
            with c2:
                st.pyplot(Visualizer.plot_pr_curves(reps))

            c3, c4 = st.columns(2)
            with c3:
                st.pyplot(Visualizer.plot_confusion_matrices(reps))
            with c4:
                st.pyplot(Visualizer.plot_runtime_comparison(reps))

        # Scientific interpretation
        best = max(aggregated.values(), key=lambda r: (r.roc_auc_mean if not np.isnan(r.roc_auc_mean) else -1))
        q = aggregated.get("Quantum Kernel OCSVM")
        st.markdown(f"""
        <div class="observation-card">
            <strong>Scientific takeaway.</strong>
            Across {len(config_used['seeds'])} seed(s) on <em>{metadata['dataset_label']}</em>,
            the strongest detector by mean ROC-AUC was
            <strong>{best.detector}</strong> at <code>{best.roc_auc_mean:.4f} ± {best.roc_auc_std:.4f}</code>.
            The quantum kernel path
            {"(with depolarising noise p = " + f"{config_used['noise_prob']:.3f}" + ")"
              if config_used['noise_prob'] > 0 else "(ideal statevector)"}
            achieved
            <code>{q.roc_auc_mean:.4f} ± {q.roc_auc_std:.4f}</code> at
            <code>{q.runtime_seconds_mean:.2f}s</code> per run
            (vs. classical baselines around
            <code>
            {min(r.runtime_seconds_mean for k, r in aggregated.items() if k != 'Quantum Kernel OCSVM'):.2f}s
            </code> at the fastest).
        </div>
        """ if q is not None else f"""
        <div class="observation-card">
            <strong>Scientific takeaway.</strong>
            Best detector by mean ROC-AUC:
            <strong>{best.detector}</strong> at <code>{best.roc_auc_mean:.4f} ± {best.roc_auc_std:.4f}</code>.
        </div>
        """, unsafe_allow_html=True)

    # ----------------- TAB 2: LATENCY -----------------
    with t_latency:
        st.markdown("#### Inference latency (roadmap protocol)")
        st.markdown("""
        Single-threaded predict on a fixed batch, warm-up discarded, `time.perf_counter`,
        reported in microseconds per sample as median and 95th percentile over repeats.
        """)
        lat_df = latency_to_dataframe(aggregated)
        if lat_df.empty:
            st.warning("Latency measurement did not produce any rows. Check detector runs.")
        else:
            st.dataframe(lat_df, use_container_width=True)
            fig_lat, ax_lat = plt.subplots(figsize=(9, 4.6), dpi=140)
            xs = lat_df["detector"].tolist()
            med = lat_df["us_per_sample_median"].tolist()
            p95 = lat_df["us_per_sample_p95"].tolist()
            width = 0.35
            x_idx = np.arange(len(xs))
            ax_lat.bar(x_idx - width/2, med, width, label="Median", color="#2563eb")
            ax_lat.bar(x_idx + width/2, p95, width, label="95th percentile", color="#f59e0b")
            ax_lat.set_yscale("log")
            ax_lat.set_xticks(x_idx)
            ax_lat.set_xticklabels(xs, rotation=15, ha="right")
            ax_lat.set_ylabel("Inference latency (µs / sample, log scale)")
            ax_lat.set_title("Inference latency comparison")
            ax_lat.grid(True, axis="y", linestyle=":", alpha=0.5)
            ax_lat.legend()
            fig_lat.tight_layout()
            st.pyplot(fig_lat)
            st.caption(
                "Note: latency here is software wall-clock on this machine, not embedded-device latency. "
                "Interpret as relative cost between detectors on identical hardware."
            )

    # ----------------- TAB 3: QUANTUM CIRCUIT & SPECTRAL -----------------
    with t_quantum:
        q_result = aggregated.get("Quantum Kernel OCSVM")
        if q_result is None:
            st.info("Quantum detector was not part of this run. Enable it in the sidebar to see these panels.")
        else:
            # Rebuild a small quantum kernel just for visualisation using a shrunken sample
            demo = QuantumKernelAnomalyDetector(
                n_components=n_components, reps=quantum_reps, nu=ocsvm_nu,
                train_sample_limit=min(60, train_sample_limit), noise_prob=noise_prob,
            )
            demo_train = benign_df.sample(n=min(80, len(benign_df)), random_state=RANDOM_STATE)
            demo.fit(demo_train)

            st.markdown("#### Quantum feature map (ZZFeatureMap)")
            st.markdown(f"""
            <div class="circuit-box">
<pre>{demo.feature_map.draw(output='text')}</pre>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Quantum fidelity Gram matrix K_train (with current noise setting)")
            st.pyplot(Visualizer.plot_kernel_matrix_heatmap(demo.K_train, max_display=40))

            st.markdown("---")
            st.markdown("#### Gram matrix spectral diagnostics")
            report = QuantumGramSpectralAnalyzer.analyze(demo.K_train)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Condition number κ", f"{report.condition_number}")
            c2.metric("Spectral gap Δ", f"{report.spectral_gap:.4f}")
            c3.metric("Effective rank", f"{report.effective_rank:.2f}")
            c4.metric("Spectral entropy", f"{report.von_neumann_spectral_entropy:.3f}")
            st.info(f"Barren plateau risk: {report.barren_plateau_risk}")

            st.markdown("---")
            st.markdown("#### Density matrix tomography (first training sample)")
            sample_circuit = demo.feature_map.assign_parameters(demo.X_train_ref[0])
            rho = QuantumInformationSpectroscopy.statevector_to_density_matrix(sample_circuit)
            c_a, c_b = st.columns(2)
            c_a.metric("Purity γ = Tr(ρ²)", f"{QuantumInformationSpectroscopy.quantum_purity(rho):.4f}")
            c_b.metric("von Neumann entropy S(ρ)",
                       f"{QuantumInformationSpectroscopy.von_neumann_entropy(rho):.4f}")

            st.markdown("---")
            st.markdown("#### NISQ noise sweep: ideal vs noisy kernel on this Gram matrix")
            p_probe = st.slider("Additional depolarising probe (p)", 0.0, 0.2, 0.02, 0.005)
            noise_sim = NISQNoiseRobustnessSimulator(depolarizing_rate=p_probe)
            qc1 = demo.feature_map.assign_parameters(demo.X_train_ref[0])
            qc2 = demo.feature_map.assign_parameters(demo.X_train_ref[1])
            ideal_f, noisy_f = noise_sim.evaluate_noisy_fidelity(qc1, qc2)
            c_n1, c_n2 = st.columns(2)
            c_n1.metric("Ideal fidelity", f"{ideal_f:.4f}")
            c_n2.metric("Noisy fidelity", f"{noisy_f:.4f}", delta=f"-{ideal_f - noisy_f:.4f}")

    # ----------------- TAB 4: PCA SCREE -----------------
    with t_scree:
        st.markdown("#### PCA variance explained (fitted on benign train)")
        st.markdown("""
        The classical 115-feature N-BaIoT record cannot be embedded on 115 qubits (2^115 amplitudes).
        PCA reduces to a small number of principal components before the quantum encoder.
        """)
        # Fit a preprocessor for display purposes only
        from preprocessing import DataPreprocessor
        pp = DataPreprocessor(n_components=n_components)
        pp.fit(benign_df.sample(n=min(400, len(benign_df)), random_state=RANDOM_STATE))
        info = pp.get_explained_variance()
        st.pyplot(Visualizer.plot_pca_variance(info["explained_variance_ratio"]))
        st.info(
            f"Cumulative variance captured on {info['n_components']} components: "
            f"{info['cumulative_explained_variance']*100:.2f}%"
        )

    # ----------------- TAB 5: SINGLE-PACKET SCANNER -----------------
    with t_scan:
        st.markdown("#### Single-packet inspector")
        q_result = aggregated.get("Quantum Kernel OCSVM")
        rep_res = None
        if q_result and q_result.representative_result:
            rep_res = q_result.representative_result

        if rep_res is None:
            st.info("Enable at least one detector to inspect predictions.")
        else:
            # We do not have the raw feature vectors from the run; re-generate a small
            # test slice from the current benign_df to inspect.
            demo_test = benign_df.sample(n=min(20, len(benign_df)), random_state=RANDOM_STATE).reset_index(drop=True)
            idx = st.slider("Packet index", 0, len(demo_test) - 1, 0)
            row = demo_test.iloc[[idx]]
            st.markdown("##### Feature preview (first 10 dims)")
            st.dataframe(row.iloc[:, :10], use_container_width=True)
            st.caption(
                "Per-detector real-time scoring across the full test batch is available in the "
                "exported per-seed CSV under `results/runs/`."
            )

    # ----------------- TAB 6: EXPORT -----------------
    with t_export:
        st.markdown("#### Ready-to-copy LaTeX table")
        latex_body = df_agg[["detector", "roc_auc", "pr_auc", "f1", "precision", "recall", "runtime_s"]].to_latex(
            index=False,
            caption=f"Anomaly detector comparison on {metadata['dataset_label']} "
                    f"(mean ± std over {len(config_used['seeds'])} seeds).",
            label="tab:iot_anomaly_comparison",
            escape=False,
        )
        st.code(latex_body, language="latex")

        st.markdown("---")
        st.markdown("#### CSV downloads")
        c1, c2 = st.columns(2)
        with c1:
            csv_agg = df_agg.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Aggregated metrics",
                data=csv_agg,
                file_name="metrics_aggregated.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            csv_lat = latency_to_dataframe(aggregated).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Latency metrics",
                data=csv_lat,
                file_name="metrics_latency.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### Run metadata")
        st.json({"metadata": metadata, "config": config_used})
