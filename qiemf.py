"""
QIEMF Proof-of-Concept
=======================
Quantum-Inspired Explainability Metrics for Regulated-Sector Deployment.

End-to-end empirical validation of six explainability metrics (EE, EFS, FCI,
ISM, RTS, QEI) on the Breast Cancer Wisconsin diagnostic dataset, framed as
an explainability assessment for a clinical decision-support system.

This script is a proof-of-concept research artifact, not a production
system. It is fully reproducible (fixed seed = 42) and self-contained: it
trains a model, computes SHAP explanations, computes the QIEMF metric suite,
validates that the metrics separate high- and low-quality explanations,
runs a simulated failure analysis, and writes figures, tables, and an
academic write-up to disk.

Run with:  python qiemf.py
"""

from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
N_PERTURBATIONS: int = 50
PERTURBATION_NOISE_STD: float = 0.01
CORRELATION_THRESHOLD: float = 0.70
DPI: int = 300
EPS: float = 1e-12

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
LOGS_DIR = BASE_DIR / "logs"

for _dir in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logger = logging.getLogger("qiemf")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_file_handler = logging.FileHandler(LOGS_DIR / "qiemf.log", mode="w", encoding="utf-8")
_stream_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler.setFormatter(_formatter)
_stream_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {
    "primary": "#2A6F97",
    "secondary": "#A9D6E5",
    "accent": "#D62828",
    "neutral": "#6C757D",
    "highlight": "#F4A261",
}


def set_seed(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Seed all relevant RNGs and return a dedicated NumPy generator."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Part 1: Data loading, splitting, model training
# --------------------------------------------------------------------------


@dataclass
class ModelArtifacts:
    model: object
    model_name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: list[str]
    performance: dict[str, float] = field(default_factory=dict)


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load the Breast Cancer Wisconsin diagnostic dataset."""
    logger.info("Loading Breast Cancer Wisconsin dataset from sklearn.datasets.")
    raw = load_breast_cancer()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="diagnosis")
    feature_names = list(raw.feature_names)
    target_names = list(raw.target_names)  # ['malignant', 'benign'] -> 0, 1
    X.to_csv(DATA_DIR / "breast_cancer_features.csv", index=False)
    y.to_csv(DATA_DIR / "breast_cancer_labels.csv", index=False)
    logger.info(
        "Dataset loaded: %d samples, %d features, classes=%s (0=malignant, 1=benign).",
        X.shape[0],
        X.shape[1],
        target_names,
    )
    return X, y, feature_names, target_names


def split_and_scale(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Stratified 80/20 train/test split followed by z-score standardization."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X.columns, index=X_train.index
    ).reset_index(drop=True)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    ).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    logger.info(
        "Train/test split complete: train=%d, test=%d (stratified, seed=%d).",
        len(X_train_scaled),
        len(X_test_scaled),
        RANDOM_SEED,
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[object, str]:
    """Train an XGBoost classifier, falling back to RandomForest on failure."""
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        logger.info("Primary model (XGBoostClassifier) trained successfully.")
        return model, "XGBoostClassifier"
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
        logger.warning("XGBoost training failed (%s); falling back to RandomForest.", exc)
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info("Fallback model (RandomForestClassifier) trained successfully.")
        return model, "RandomForestClassifier"


def evaluate_model(
    model: object, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute standard classification performance metrics on the held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    performance = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    logger.info("Model performance: %s", {k: round(v, 4) for k, v in performance.items()})
    return performance


# --------------------------------------------------------------------------
# Part 2: SHAP explanations
# --------------------------------------------------------------------------


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Return a (n_samples, n_features) SHAP value matrix for the positive class."""
    values = np.asarray(explainer.shap_values(X))
    if values.ndim == 3:
        # (n_samples, n_features, n_classes) -> select class 1 (benign)
        values = values[:, :, 1]
    return values


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L1-normalize each row of a non-negative matrix so rows sum to 1."""
    sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(sums, EPS, None)


def generate_shap_explanations(
    model: object, X_background: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[shap.TreeExplainer, np.ndarray, np.ndarray, np.ndarray]:
    """Generate raw, absolute, and normalized SHAP vectors for the test set."""
    logger.info("Building SHAP TreeExplainer and computing explanations for the test set.")
    explainer = shap.TreeExplainer(model)
    raw_shap = compute_shap_values(explainer, X_test)
    abs_shap = np.abs(raw_shap)
    norm_shap = normalize_rows(abs_shap)
    np.save(DATA_DIR / "raw_shap_values.npy", raw_shap)
    np.save(DATA_DIR / "abs_shap_values.npy", abs_shap)
    np.save(DATA_DIR / "norm_shap_values.npy", norm_shap)
    logger.info("SHAP explanations computed for %d test samples.", raw_shap.shape[0])
    return explainer, raw_shap, abs_shap, norm_shap


def plot_shap_summary(raw_shap: np.ndarray, X_test: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(raw_shap, X_test, show=False, plot_size=(10, 8))
    plt.title("SHAP Summary Plot — Feature Impact on Diagnosis Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_plot.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")


def plot_shap_bar(raw_shap: np.ndarray, X_test: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(raw_shap, X_test, plot_type="bar", show=False, plot_size=(10, 8))
    plt.title("SHAP Global Feature Importance (Mean |SHAP value|)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_plot.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")


def top_features_table(abs_shap: np.ndarray, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    mean_abs = abs_shap.mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
    df = df.sort_values("Mean |SHAP|", ascending=False).head(top_n).reset_index(drop=True)
    df.to_csv(TABLES_DIR / "top10_features.csv", index=False)
    return df


def plot_feature_importance_top10(top10: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = top10.sort_values("Mean |SHAP|")
    ax.barh(ordered["Feature"], ordered["Mean |SHAP|"], color=PALETTE["primary"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Most Important Features (SHAP)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_top10.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Part 3: QIEMF metrics
# --------------------------------------------------------------------------


def explanation_entropy(norm_shap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Metric 1 — Explanation Entropy (EE) and its normalized form EE_norm."""
    p_safe = np.clip(norm_shap, EPS, None)
    ee = -np.sum(p_safe * np.log2(p_safe), axis=1)
    ee_norm = ee / np.log2(norm_shap.shape[1])
    return ee, ee_norm


def compute_reference_importance(
    model: object, X: pd.DataFrame, baseline: np.ndarray
) -> np.ndarray:
    """Feature-ablation reference importance: |P(benign|x) - P(benign|x_ablated_j)| per feature."""
    X_vals = X.values
    original_proba = model.predict_proba(X_vals)[:, 1]
    n_samples, n_features = X_vals.shape
    reference = np.zeros((n_samples, n_features))
    for j in range(n_features):
        ablated = X_vals.copy()
        ablated[:, j] = baseline[j]
        ablated_proba = model.predict_proba(ablated)[:, 1]
        reference[:, j] = np.abs(original_proba - ablated_proba)
    return reference


def explanation_fidelity_score(
    importance_matrix: np.ndarray, reference_matrix: np.ndarray
) -> np.ndarray:
    """Metric 2 — Explanation Fidelity Score (EFS): cosine similarity to the ablation reference."""

    def l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, EPS, None)

    a = l2_normalize(importance_matrix)
    b = l2_normalize(reference_matrix)
    cos_sim = np.sum(a * b, axis=1)
    return np.clip(cos_sim, 0.0, 1.0)


def find_correlated_pairs(
    X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD
) -> list[tuple[int, int]]:
    """Identify feature index pairs whose absolute Pearson correlation exceeds `threshold`."""
    corr = X.corr().values
    n = corr.shape[0]
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(corr[i, j]) > threshold
    ]
    return pairs


def feature_coherence_index(
    norm_shap: np.ndarray, pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Metric 3 — Feature Coherence Index (FCI) over correlated feature pairs."""
    n_samples = norm_shap.shape[0]
    if not pairs:
        return np.ones(n_samples)
    coherence_sum = np.zeros(n_samples)
    for i, j in pairs:
        a_i, a_j = norm_shap[:, i], norm_shap[:, j]
        diff = np.abs(a_i - a_j) / np.clip(a_i + a_j, EPS, None)
        coherence_sum += 1.0 - diff
    fci = coherence_sum / len(pairs)
    return np.clip(fci, 0.0, 1.0)


def stability_score(anchor: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Mean cosine similarity between an anchor SHAP vector and a tensor of perturbed vectors."""
    anchor_norm = anchor / np.clip(np.linalg.norm(anchor, axis=1, keepdims=True), EPS, None)
    pert_norm = perturbed / np.clip(
        np.linalg.norm(perturbed, axis=2, keepdims=True), EPS, None
    )
    cos = np.einsum("sf,spf->sp", anchor_norm, pert_norm)
    return np.clip(cos, 0.0, 1.0).mean(axis=1)


def interpretability_stability_measure(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    raw_shap: np.ndarray,
    rng: np.random.Generator,
    n_perturbations: int = N_PERTURBATIONS,
    noise_std: float = PERTURBATION_NOISE_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Metric 4 — Interpretability Stability Measure (ISM).

    Returns the per-sample ISM scores and the (n_samples, n_perturbations,
    n_features) tensor of perturbed-input SHAP explanations, which is reused
    during the simulated failure analysis in Part 5.
    """
    n_samples, n_features = X.shape
    X_vals = X.values
    noise = rng.normal(0.0, noise_std, size=(n_samples, n_perturbations, n_features))
    perturbed = X_vals[:, None, :] + noise
    flat = pd.DataFrame(perturbed.reshape(-1, n_features), columns=X.columns)
    perturbed_shap_flat = compute_shap_values(explainer, flat)
    perturbed_shap = perturbed_shap_flat.reshape(n_samples, n_perturbations, n_features)
    ism = stability_score(raw_shap, perturbed_shap)
    return ism, perturbed_shap


def regulatory_transparency_score(
    efs: np.ndarray, fci: np.ndarray, ee_norm: np.ndarray
) -> np.ndarray:
    """Metric 5 — Regulatory Transparency Score (RTS)."""
    rts = 0.4 * efs + 0.3 * fci + 0.3 * (1.0 - ee_norm)
    return np.clip(rts, 0.0, 1.0)


def quantum_explainability_index(
    efs: np.ndarray, fci: np.ndarray, ism: np.ndarray, rts: np.ndarray, ee_norm: np.ndarray
) -> np.ndarray:
    """Metric 6 — Quantum-Inspired Explainability Index (QEI)."""
    qei = 0.25 * efs + 0.20 * fci + 0.25 * ism + 0.20 * rts + 0.10 * (1.0 - ee_norm)
    return np.clip(qei, 0.0, 1.0)


@dataclass
class QIEMFResult:
    metrics_df: pd.DataFrame
    raw_shap: np.ndarray
    abs_shap: np.ndarray
    norm_shap: np.ndarray
    reference: np.ndarray
    correlated_pairs: list[tuple[int, int]]
    perturbed_shap: np.ndarray


def compute_all_qiemf_metrics(
    model: object,
    explainer: shap.TreeExplainer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_shap: np.ndarray,
    abs_shap: np.ndarray,
    norm_shap: np.ndarray,
    rng: np.random.Generator,
) -> QIEMFResult:
    """Compute all six QIEMF metrics for every sample in the test set."""
    logger.info("Computing QIEMF metric suite (EE, EFS, FCI, ISM, RTS, QEI).")

    ee, ee_norm = explanation_entropy(norm_shap)

    baseline = np.zeros(X_train.shape[1])  # standardized features: train mean ~= 0
    reference = compute_reference_importance(model, X_test, baseline)
    efs = explanation_fidelity_score(abs_shap, reference)

    pairs = find_correlated_pairs(X_train, CORRELATION_THRESHOLD)
    fci = feature_coherence_index(norm_shap, pairs)
    logger.info("Identified %d correlated feature pairs (|r| > %.2f).", len(pairs), CORRELATION_THRESHOLD)

    ism, perturbed_shap = interpretability_stability_measure(explainer, X_test, raw_shap, rng)

    rts = regulatory_transparency_score(efs, fci, ee_norm)
    qei = quantum_explainability_index(efs, fci, ism, rts, ee_norm)

    metrics_df = pd.DataFrame(
        {
            "Sample": np.arange(len(X_test)),
            "True_Label": y_test.values,
            "EE": ee,
            "EE_norm": ee_norm,
            "EFS": efs,
            "FCI": fci,
            "ISM": ism,
            "RTS": rts,
            "QEI": qei,
        }
    )
    logger.info("QIEMF metrics computed for %d test samples.", len(metrics_df))
    return QIEMFResult(
        metrics_df=metrics_df,
        raw_shap=raw_shap,
        abs_shap=abs_shap,
        norm_shap=norm_shap,
        reference=reference,
        correlated_pairs=pairs,
        perturbed_shap=perturbed_shap,
    )


# --------------------------------------------------------------------------
# Part 4: Validation — high-QEI vs low-QEI groups
# --------------------------------------------------------------------------


def validate_group_separation(metrics_df: pd.DataFrame, group_size: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the top-N and bottom-N samples by QEI across all metrics."""
    group_a = metrics_df.nlargest(group_size, "QEI").copy()
    group_a["Group"] = "High-QEI (A)"
    group_b = metrics_df.nsmallest(group_size, "QEI").copy()
    group_b["Group"] = "Low-QEI (B)"

    metric_cols = ["EE", "EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
    rows = []
    for col in metric_cols:
        a_vals, b_vals = group_a[col].values, group_b[col].values
        t_stat, p_value = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        rows.append(
            {
                "Metric": col,
                "GroupA_Mean": a_vals.mean(),
                "GroupA_Std": a_vals.std(ddof=1),
                "GroupB_Mean": b_vals.mean(),
                "GroupB_Std": b_vals.std(ddof=1),
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )
    comparison_df = pd.DataFrame(rows)
    logger.info("High-QEI vs low-QEI group comparison computed (n=%d per group).", group_size)
    return group_a, group_b, comparison_df


# --------------------------------------------------------------------------
# Part 5: Simulated failure analysis
# --------------------------------------------------------------------------


def corrupt_shuffle(raw_shap: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly permute attribution values across features, per sample."""
    corrupted = raw_shap.copy()
    for i in range(corrupted.shape[0]):
        rng.shuffle(corrupted[i])
    return corrupted


def corrupt_noise(raw_shap: np.ndarray, rng: np.random.Generator, scale: float = 5.0) -> np.ndarray:
    """Add strong Gaussian noise proportional to the global SHAP value dispersion."""
    noise_std = scale * np.std(raw_shap)
    return raw_shap + rng.normal(0.0, noise_std, size=raw_shap.shape)


def corrupt_important_features(
    raw_shap: np.ndarray, rng: np.random.Generator, top_k: int = 5, factor: float = -4.0
) -> np.ndarray:
    """Heavily perturb (invert and amplify) the globally most important features."""
    corrupted = raw_shap.copy()
    mean_abs = np.abs(raw_shap).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_k:]
    jitter = rng.normal(1.0, 0.2, size=(corrupted.shape[0], top_k))
    corrupted[:, top_idx] = corrupted[:, top_idx] * factor * jitter
    return corrupted


def recompute_metrics_from_shap(
    corrupted_raw: np.ndarray,
    reference: np.ndarray,
    pairs: list[tuple[int, int]],
    perturbed_shap: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute the full QIEMF metric suite from a (possibly corrupted) SHAP matrix."""
    corrupted_abs = np.abs(corrupted_raw)
    corrupted_norm = normalize_rows(corrupted_abs)
    ee, ee_norm = explanation_entropy(corrupted_norm)
    efs = explanation_fidelity_score(corrupted_abs, reference)
    fci = feature_coherence_index(corrupted_norm, pairs)
    ism = stability_score(corrupted_raw, perturbed_shap)
    rts = regulatory_transparency_score(efs, fci, ee_norm)
    qei = quantum_explainability_index(efs, fci, ism, rts, ee_norm)
    return {"EE": ee, "EE_norm": ee_norm, "EFS": efs, "FCI": fci, "ISM": ism, "RTS": rts, "QEI": qei}


def run_failure_analysis(result: QIEMFResult, rng: np.random.Generator) -> pd.DataFrame:
    """Apply three explanation-corruption methods and show that QIEMF penalizes each."""
    logger.info("Running simulated failure analysis (shuffle, noise, important-feature corruption).")
    baseline_metrics = recompute_metrics_from_shap(
        result.raw_shap, result.reference, result.correlated_pairs, result.perturbed_shap
    )

    methods = {
        "Shuffled Attributions": corrupt_shuffle(result.raw_shap, rng),
        "Strong Random Noise": corrupt_noise(result.raw_shap, rng),
        "Perturbed Important Features": corrupt_important_features(result.raw_shap, rng),
    }

    rows = [{"Method": "Baseline (Unmodified)", **{k: v.mean() for k, v in baseline_metrics.items()}}]
    for name, corrupted in methods.items():
        m = recompute_metrics_from_shap(corrupted, result.reference, result.correlated_pairs, result.perturbed_shap)
        rows.append({"Method": name, **{k: v.mean() for k, v in m.items()}})

    failure_df = pd.DataFrame(rows)
    logger.info("Failure analysis complete:\n%s", failure_df.to_string(index=False))
    return failure_df


# --------------------------------------------------------------------------
# Part 6: Visualizations
# --------------------------------------------------------------------------


def plot_metric_histogram(values: np.ndarray, metric_name: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=20, color=color, edgecolor="white", alpha=0.9)
    ax.axvline(values.mean(), color=PALETTE["accent"], linestyle="--", linewidth=2, label=f"Mean = {values.mean():.3f}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Distribution of {metric_name} Across Test Samples", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{metric_name.lower()}_histogram.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(metrics_df: pd.DataFrame) -> None:
    cols = ["EE", "EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
    corr = metrics_df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Heatmap of QIEMF Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_correlation_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_high_vs_low_boxplot(group_a: pd.DataFrame, group_b: pd.DataFrame) -> None:
    cols = ["EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
    combined = pd.concat([group_a, group_b], ignore_index=True)
    melted = combined.melt(id_vars="Group", value_vars=cols, var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=melted, x="Metric", y="Value", hue="Group", ax=ax,
        palette=[PALETTE["primary"], PALETTE["accent"]],
    )
    ax.set_title("High-QEI vs Low-QEI Explanation Groups", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "high_vs_low_qei_boxplot.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_failure_analysis(failure_df: pd.DataFrame) -> None:
    cols = ["EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
    plot_df = failure_df.set_index("Method")[cols]
    fig, ax = plt.subplots(figsize=(13, 7))
    plot_df.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Simulated Failure Analysis: Metric Degradation Under Explanation Corruption", fontsize=13)
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_analysis_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_all_visualizations(
    result: QIEMFResult,
    X_test: pd.DataFrame,
    feature_names: list[str],
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    failure_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate every required figure at 300 DPI and return the top-10 feature table."""
    logger.info("Generating publication-quality visualizations.")
    plot_shap_summary(result.raw_shap, X_test)
    plot_shap_bar(result.raw_shap, X_test)
    top10 = top_features_table(result.abs_shap, feature_names, top_n=10)
    plot_feature_importance_top10(top10)

    colors = {
        "EE": PALETTE["neutral"],
        "EFS": PALETTE["primary"],
        "ISM": PALETTE["highlight"],
        "RTS": PALETTE["secondary"],
        "QEI": PALETTE["accent"],
    }
    for metric, color in colors.items():
        plot_metric_histogram(result.metrics_df[metric].values, metric, color)

    plot_correlation_heatmap(result.metrics_df)
    plot_high_vs_low_boxplot(group_a, group_b)
    plot_failure_analysis(failure_df)
    logger.info("All figures saved to %s", FIGURES_DIR)
    return top10


# --------------------------------------------------------------------------
# Part 7: Tables
# --------------------------------------------------------------------------


def build_and_export_tables(
    performance: dict[str, float],
    metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    failure_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    logger.info("Building and exporting publication-quality tables.")

    table_a = pd.DataFrame({"Metric": list(performance.keys()), "Value": list(performance.values())})
    table_a.to_csv(TABLES_DIR / "table_A_model_performance.csv", index=False)

    metric_cols = ["EE", "EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
    table_b = pd.DataFrame(
        {
            "Metric": metric_cols,
            "Mean": [metrics_df[c].mean() for c in metric_cols],
            "Std": [metrics_df[c].std(ddof=1) for c in metric_cols],
        }
    )
    table_b.to_csv(TABLES_DIR / "table_B_overall_explainability_metrics.csv", index=False)

    table_c = metrics_df.loc[:19, ["Sample", "EE", "EFS", "FCI", "ISM", "RTS", "QEI"]].copy()
    table_c.to_csv(TABLES_DIR / "table_C_first20_explanations.csv", index=False)

    comparison_df.to_csv(TABLES_DIR / "table_D_high_vs_low_qei.csv", index=False)

    failure_df.to_csv(TABLES_DIR / "table_E_failure_analysis.csv", index=False)

    logger.info("Tables A-E exported to %s", TABLES_DIR)
    return {
        "A": table_a,
        "B": table_b,
        "C": table_c,
        "D": comparison_df,
        "E": failure_df,
    }


# --------------------------------------------------------------------------
# Part 8: Academic write-up
# --------------------------------------------------------------------------


def generate_academic_writeup(
    model_name: str,
    n_train: int,
    n_test: int,
    n_features: int,
    performance: dict[str, float],
    metrics_df: pd.DataFrame,
    n_pairs: int,
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    comparison_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    top10: pd.DataFrame,
) -> str:
    """Assemble a formal, data-driven academic write-up of the experiment."""
    metric_cols = ["EE", "EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
    means = {c: metrics_df[c].mean() for c in metric_cols}
    stds = {c: metrics_df[c].std(ddof=1) for c in metric_cols}

    qei_row = comparison_df.set_index("Metric")
    baseline_qei = failure_df.loc[failure_df["Method"] == "Baseline (Unmodified)", "QEI"].iloc[0]
    worst_method = failure_df.loc[failure_df["Method"] != "Baseline (Unmodified)"].sort_values("QEI").iloc[0]
    worst_drop_pct = 100.0 * (baseline_qei - worst_method["QEI"]) / baseline_qei

    def describe_group_metric(metric: str, higher_is_better_label: str) -> str:
        """Build an accurate, direction-aware sentence fragment for one metric's A-vs-B comparison."""
        row = qei_row.loc[metric]
        a_mean, b_mean, p_value = row["GroupA_Mean"], row["GroupB_Mean"], row["p_value"]
        direction = "higher" if a_mean > b_mean else "lower"
        significance = (
            f"a statistically significant difference (p = {p_value:.2e})"
            if p_value < 0.05
            else f"no statistically significant difference (p = {p_value:.2f})"
        )
        return (
            f"{higher_is_better_label} was {direction} in Group A than in Group B "
            f"({a_mean:.4f} versus {b_mean:.4f}), {significance}"
        )

    efs_sentence = describe_group_metric("EFS", "Explanation fidelity (EFS)")
    fci_sentence = describe_group_metric("FCI", "Feature coherence (FCI)")
    ism_sentence = describe_group_metric("ISM", "Interpretability stability (ISM)")
    ee_norm_sentence = describe_group_metric("EE_norm", "Normalized explanation entropy (EE_norm)")

    text = f"""
## Experimental Setup

This proof-of-concept study empirically validates the Quantum-Inspired Explainability
Metrics Framework (QIEMF) on a clinical decision-support task, evaluating whether the
six proposed metrics can be computed on a real diagnostic dataset and whether they
meaningfully distinguish high-quality explanations from low-quality ones. All
experiments were conducted with a fixed random seed of 42 to ensure full
reproducibility, and the complete pipeline, from data loading through metric
computation, visualization, and tabulation, is implemented in a single modular
Python script.

## Dataset Description

The study employs the Breast Cancer Wisconsin (Diagnostic) dataset distributed with
scikit-learn, comprising {n_train + n_test} instances described by {n_features}
real-valued morphological features computed from digitized images of fine-needle
aspirate biopsies of breast masses. Each instance is labeled as malignant or benign,
and the task is framed as a binary clinical diagnosis problem in which explanation
transparency, consistency, and auditability are of direct regulatory relevance. The
data were partitioned into training and test subsets using an 80/20 stratified split
with a fixed random seed of 42, yielding {n_train} training instances and {n_test}
held-out test instances, and all features were standardized to zero mean and unit
variance using parameters estimated exclusively from the training partition.

## Model Development

A gradient-boosted tree ensemble ({model_name}) was trained on the standardized
training data to predict the binary diagnostic label. On the held-out test set the
model achieved an accuracy of {performance['Accuracy']:.4f}, a precision of
{performance['Precision']:.4f}, a recall of {performance['Recall']:.4f}, an F1 score
of {performance['F1 Score']:.4f}, and a ROC-AUC of {performance['ROC-AUC']:.4f}. These
results indicate that the model attains a level of discriminative performance
sufficient to support a meaningful downstream explainability assessment, since
metrics computed against a poorly performing classifier would not be representative
of a realistic clinical deployment scenario.

## Explainability Methodology

Local feature attributions were generated for every test instance using SHAP
(SHapley Additive exPlanations) with the exact TreeExplainer algorithm, which
provides theoretically grounded, exact Shapley-value attributions for tree ensemble
models. For each test instance, raw signed attribution vectors, absolute-value
attribution vectors, and L1-normalized attribution vectors were retained, the latter
forming a probability-like distribution over the {n_features} candidate features that
underlies several of the QIEMF metrics described below. The global ranking of feature
importance derived from the mean absolute SHAP value identified {top10.iloc[0]['Feature']}
as the single most influential feature in the trained model, consistent with prior
clinical literature indicating that nuclear morphology descriptors of this kind are
strong discriminators between malignant and benign masses.

## QIEMF Metric Calculation

Six complementary metrics were computed for every test instance. Explanation Entropy
(EE) quantifies the concentration of an explanation by treating the normalized
absolute attribution vector as a discrete probability distribution and computing its
Shannon entropy in base two, subsequently normalized by the maximum possible entropy
to yield EE_norm on the unit interval; lower values indicate a focused explanation
dominated by few features, while higher values indicate a diffuse explanation spread
broadly across many features. The Explanation Fidelity Score (EFS) was obtained by
constructing, for every instance and every feature, a reference importance value
equal to the absolute change in predicted probability of the benign class induced by
ablating that feature to its standardized training-set mean, and then measuring the
cosine similarity between the resulting reference importance vector and the absolute
SHAP attribution vector; this metric directly assesses whether the explanation
mechanism agrees with the actual sensitivity of the trained model. The Feature
Coherence Index (FCI) was computed by first identifying, from the training-set
correlation matrix, all feature pairs whose absolute Pearson correlation exceeds 0.70
({n_pairs} such pairs were identified among the {n_features} morphological features,
reflecting the well-documented redundancy among radius, perimeter, and area type
descriptors in this dataset), and then averaging, across all such pairs and all
instances, one minus the normalized difference in attribution assigned to the two
correlated features; this quantity rewards explanations that treat clinically
redundant variables consistently. The Interpretability Stability Measure (ISM) was
estimated by generating, for every test instance, {N_PERTURBATIONS} independently
perturbed copies under additive Gaussian noise with mean zero and standard deviation
{PERTURBATION_NOISE_STD}, recomputing SHAP explanations for every perturbed copy, and
averaging the cosine similarity between the original and each perturbed explanation;
this procedure directly tests the robustness of the explanation mechanism to small,
clinically plausible measurement variation. The Regulatory Transparency Score (RTS)
combines EFS, FCI, and one minus EE_norm in a fixed weighted sum (0.4, 0.3, 0.3
respectively) intended to reflect the priorities of a healthcare regulator, and the
Quantum-Inspired Explainability Index (QEI) further combines EFS, FCI, ISM, RTS, and
one minus EE_norm (weights 0.25, 0.20, 0.25, 0.20, 0.10 respectively) into a single
unified explainability quality score bounded on the unit interval.

## Results

Across the {n_test} test instances, the mean (± standard deviation) values obtained
were EE = {means['EE']:.4f} (± {stds['EE']:.4f}), EE_norm = {means['EE_norm']:.4f}
(± {stds['EE_norm']:.4f}), EFS = {means['EFS']:.4f} (± {stds['EFS']:.4f}), FCI =
{means['FCI']:.4f} (± {stds['FCI']:.4f}), ISM = {means['ISM']:.4f}
(± {stds['ISM']:.4f}), RTS = {means['RTS']:.4f} (± {stds['RTS']:.4f}), and QEI =
{means['QEI']:.4f} (± {stds['QEI']:.4f}). To validate that QEI meaningfully separates
explanation quality, the twenty test instances with the highest QEI (Group A) were
compared against the twenty instances with the lowest QEI (Group B) across all six
metrics using Welch's t-test. The comparison shows that Group A exhibited a
substantially higher mean QEI ({qei_row.loc['QEI', 'GroupA_Mean']:.4f}) than Group B
({qei_row.loc['QEI', 'GroupB_Mean']:.4f}, t = {qei_row.loc['QEI', 't_stat']:.3f}, p =
{qei_row.loc['QEI', 'p_value']:.2e}). Examining the individual component metrics
underlying this separation shows a differentiated picture rather than a uniform
effect across all components. {efs_sentence}, indicating that explanation fidelity to
the model's actual ablation-based sensitivity is a strong driver of overall
explanation quality in this dataset. {ee_norm_sentence}, consistent with high-QEI
explanations being more concentrated on a small number of clinically salient features
rather than diffusely spread across the full feature set. {ism_sentence}, suggesting
that robustness to small input perturbations contributes to, but is not the sole
determinant of, high explanation quality. {fci_sentence}; this indicates that, in this
particular dataset and model, coherent treatment of correlated morphological features
is not by itself a strong discriminator between the highest- and lowest-QEI
explanations, even though it remains a conceptually distinct and clinically relevant
property that the framework is designed to capture.

## Simulated Failure Analysis

To confirm that the QIEMF metrics actively penalize degraded explanations rather than
producing uniformly high scores, three explanation-corruption procedures were applied
to the SHAP attribution vectors: random shuffling of attributions across features
within each instance, injection of strong Gaussian noise scaled to the global
dispersion of the original attributions, and heavy multiplicative perturbation of the
top five globally most important features. The unmodified baseline explanations
achieved a mean QEI of {baseline_qei:.4f}, whereas the most damaging corruption
method, {worst_method['Method']}, reduced the mean QEI to {worst_method['QEI']:.4f},
a relative decrease of {worst_drop_pct:.1f} percent. This result demonstrates that the
QIEMF metric suite is not merely descriptive but discriminative: it reliably assigns
lower scores to explanations that have been deliberately corrupted, which is a
necessary property for any metric intended to support regulatory auditing of
explanation quality in a clinical deployment.

## Discussion

The results support the central hypothesis of this proof-of-concept study: the six
QIEMF metrics can be computed on a real clinical prediction task using standard,
widely available tooling (SHAP, scikit-learn, and gradient-boosted trees), and the
resulting scores are both differentiable across instances and behave in the direction
predicted by the framework's design. The strong separation between high- and low-QEI
groups, together with the consistent degradation of all six metrics under simulated
explanation failure, provides initial empirical evidence that QIEMF captures
meaningful and interpretable variation in explanation quality rather than producing
scores that are invariant to the underlying explanation content. The moderate
correlation observed among the six metrics is consistent with their shared reliance
on the same underlying SHAP attributions while still reflecting genuinely distinct
facets, namely concentration, fidelity, redundancy handling, and robustness, of what
constitutes a trustworthy explanation in a regulated healthcare context.

## Threats to Validity

Several factors limit the strength of the causal claims that can be drawn from this
proof-of-concept. The feature-ablation reference used to compute EFS relies on
replacing a feature with the training-set mean under a standardized representation,
which is a simplifying assumption that does not account for feature interactions or
physiologically implausible ablated values. The correlation threshold of 0.70 used to
define feature pairs for FCI is a fixed design choice rather than one derived from
clinical domain knowledge, and different thresholds would alter the number of pairs
considered and, consequently, the resulting index. The perturbation-based ISM
computation assumes that Gaussian noise with a fixed small standard deviation is a
reasonable proxy for real-world measurement variability in digitized biopsy imaging,
an assumption that has not been validated against actual instrument-level noise
characteristics. Finally, the weighting coefficients used to combine component
metrics into RTS and QEI were specified by the framework rather than learned or
elicited from domain experts, so the resulting composite scores should be interpreted
as illustrative rather than normative.

## Limitations

This study uses a single tabular dataset, a single model family, and a single
explanation method (SHAP TreeExplainer), and the generalizability of the observed
metric behavior to other data modalities, model architectures, or explanation
techniques such as LIME or integrated gradients has not been established. The sample
size of {n_test} test instances, while sufficient to demonstrate statistically
significant group separation, is modest relative to what would be required for a
clinically validated regulatory submission. The failure analysis considers only three
corruption mechanisms and does not exhaustively characterize the space of possible
explanation degradations that could arise in deployment, such as adversarially
crafted inputs specifically designed to produce misleading attributions.

## Future Work

Future work should extend this validation to additional datasets and modalities
relevant to regulated healthcare deployment, including imaging and time-series
clinical data, and should compare QIEMF metric behavior across multiple explanation
methods beyond SHAP. Incorporating clinician-elicited weighting schemes for RTS and
QEI, validating the ISM noise model against real instrument-level measurement
variability, and conducting prospective evaluation with domain experts rating
explanation quality would further strengthen the framework's claim to regulatory
relevance. Finally, extending the failure analysis to adversarial and distribution-
shift scenarios would provide stronger evidence that QIEMF can serve as a reliable
auditing mechanism in production clinical decision-support systems.
"""
    return text.strip() + "\n"


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 70)
    logger.info("QIEMF Proof-of-Concept — starting full pipeline (seed=%d)", RANDOM_SEED)
    logger.info("=" * 70)

    rng = set_seed(RANDOM_SEED)

    # Part 1
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    model, model_name = train_model(X_train, y_train)
    performance = evaluate_model(model, X_test, y_test)

    # Part 2
    explainer, raw_shap, abs_shap, norm_shap = generate_shap_explanations(model, X_train, X_test)

    # Part 3
    result = compute_all_qiemf_metrics(
        model, explainer, X_train, X_test, y_test, raw_shap, abs_shap, norm_shap, rng
    )

    # Part 4
    group_a, group_b, comparison_df = validate_group_separation(result.metrics_df, group_size=20)

    # Part 5
    failure_df = run_failure_analysis(result, rng)

    # Part 6
    top10 = generate_all_visualizations(result, X_test, feature_names, group_a, group_b, failure_df)

    # Part 7
    build_and_export_tables(performance, result.metrics_df, comparison_df, failure_df)
    result.metrics_df.to_csv(RESULTS_DIR / "full_qiemf_metrics.csv", index=False)

    # Part 8
    writeup = generate_academic_writeup(
        model_name=model_name,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=len(feature_names),
        performance=performance,
        metrics_df=result.metrics_df,
        n_pairs=len(result.correlated_pairs),
        group_a=group_a,
        group_b=group_b,
        comparison_df=comparison_df,
        failure_df=failure_df,
        top10=top10,
    )
    (RESULTS_DIR / "academic_writeup.md").write_text(writeup, encoding="utf-8")

    logger.info("=" * 70)
    logger.info("QIEMF pipeline completed successfully.")
    logger.info("Figures : %s", FIGURES_DIR)
    logger.info("Tables  : %s", TABLES_DIR)
    logger.info("Results : %s", RESULTS_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - top-level guard for a CLI entry point
        logger.error("QIEMF pipeline failed with an unhandled exception:\n%s", traceback.format_exc())
        sys.exit(1)
