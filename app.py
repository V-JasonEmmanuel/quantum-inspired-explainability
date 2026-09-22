"""
QIEMF Live Demo — Interactive Streamlit Frontend
==================================================
A live, browser-based demonstration of the Quantum-Inspired Explainability
Metrics Framework (QIEMF) on the Breast Cancer Wisconsin diagnostic dataset.

This app does not replay pre-rendered screenshots: it imports the exact
functions defined in qiemf.py (model training, SHAP explanation generation,
and all six QIEMF metric computations) and runs them live, in-process, so
every number and chart on screen is computed in front of the viewer. This
keeps a single source of truth between the batch pipeline (qiemf.py) and
this interactive frontend.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

import qiemf as core

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="QIEMF Live Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2A6F97"
ACCENT = "#D62828"
GOOD = "#2A9D8F"
NEUTRAL = "#6C757D"

st.markdown(
    """
    <style>
    .qiemf-banner {
        padding: 1.1rem 1.4rem;
        border-radius: 0.6rem;
        background: linear-gradient(90deg, #0E3B52 0%, #2A6F97 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .qiemf-banner h1 { margin: 0; font-size: 1.6rem; }
    .qiemf-banner p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    .qiemf-disclaimer {
        font-size: 0.8rem; color: #6C757D; border-top: 1px solid #DDD;
        padding-top: 0.6rem; margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

METRIC_DEFINITIONS = {
    "EE_norm": "Explanation Entropy (normalized) — lower means a more focused explanation.",
    "EFS": "Explanation Fidelity Score — agreement between SHAP and ablation-based sensitivity.",
    "FCI": "Feature Coherence Index — consistency of attribution across correlated features.",
    "ISM": "Interpretability Stability Measure — robustness of the explanation under small input noise.",
    "RTS": "Regulatory Transparency Score — weighted combination tuned for healthcare auditability.",
    "QEI": "Quantum-Inspired Explainability Index — the unified explanation-quality score.",
}


# --------------------------------------------------------------------------
# Cached pipeline: trains the model and computes everything exactly once
# per running process, reusing qiemf.py functions directly.
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Training model, computing SHAP explanations, and scoring QIEMF metrics…")
def run_pipeline() -> dict:
    rng = core.set_seed(core.RANDOM_SEED)

    X, y, feature_names, target_names = core.load_data()
    X_train, X_test, y_train, y_test, scaler = core.split_and_scale(X, y)
    model, model_name = core.train_model(X_train, y_train)
    performance = core.evaluate_model(model, X_test, y_test)

    explainer, raw_shap, abs_shap, norm_shap = core.generate_shap_explanations(model, X_train, X_test)
    result = core.compute_all_qiemf_metrics(
        model, explainer, X_train, X_test, y_test, raw_shap, abs_shap, norm_shap, rng
    )
    group_a, group_b, comparison_df = core.validate_group_separation(result.metrics_df, group_size=20)
    failure_df = core.run_failure_analysis(result, rng)
    top10 = core.generate_all_visualizations(result, X_test, feature_names, group_a, group_b, failure_df)
    core.build_and_export_tables(performance, result.metrics_df, comparison_df, failure_df)
    writeup = core.generate_academic_writeup(
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
    (core.RESULTS_DIR / "academic_writeup.md").write_text(writeup, encoding="utf-8")

    X_test_raw = pd.DataFrame(scaler.inverse_transform(X_test.values), columns=feature_names)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    base_value = np.asarray(explainer.expected_value).reshape(-1)[-1]

    return {
        "rng": rng,
        "X_raw": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_names,
        "X_train": X_train,
        "X_test": X_test,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "model": model,
        "model_name": model_name,
        "performance": performance,
        "explainer": explainer,
        "base_value": float(base_value),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "result": result,
        "group_a": group_a,
        "group_b": group_b,
        "comparison_df": comparison_df,
        "failure_df": failure_df,
        "top10": top10,
    }


DATA = run_pipeline()

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.title("🧬 QIEMF Live Demo")
st.sidebar.caption("Quantum-Inspired Explainability Metrics Framework")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🗂️ Dataset Explorer",
        "📊 Model Performance",
        "🔍 Explanation Viewer",
        "🧪 Failure Lab (Live)",
        "📈 Validation & Figures",
        "📄 Academic Write-up",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Run configuration**")
st.sidebar.write(f"Model: `{DATA['model_name']}`")
st.sidebar.write(f"Random seed: `{core.RANDOM_SEED}`")
st.sidebar.write(f"Train / Test: `{len(DATA['X_train'])}` / `{len(DATA['X_test'])}`")
st.sidebar.write(f"Correlated feature pairs (|r| > 0.70): `{len(DATA['result'].correlated_pairs)}`")
st.sidebar.divider()
st.sidebar.caption(
    "Proof-of-concept research artifact. Not a certified medical device and "
    "not intended for real clinical decision-making."
)


def metric_row(values: dict[str, float]) -> None:
    cols = st.columns(len(values))
    for col, (name, value) in zip(cols, values.items()):
        col.metric(name, f"{value:.4f}")


# --------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------

if page == "🏠 Overview":
    st.markdown(
        """
        <div class="qiemf-banner">
            <h1>Quantum-Inspired Explainability Metrics for Regulated-Sector Deployment</h1>
            <p>Live proof-of-concept validation on clinical breast cancer diagnosis (Breast Cancer Wisconsin dataset)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "This app is a **live, interactive demonstration** of the QIEMF framework, not a set of static "
        "screenshots. Every model, explanation, and metric shown here was computed by this running process "
        "using the exact same functions as the batch pipeline (`qiemf.py`)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", DATA["model_name"])
    c2.metric("Test ROC-AUC", f"{DATA['performance']['ROC-AUC']:.4f}")
    c3.metric("Mean QEI", f"{DATA['result'].metrics_df['QEI'].mean():.4f}")
    c4.metric("Test samples", len(DATA["X_test"]))

    st.subheader("The Six QIEMF Metrics")
    for name, desc in METRIC_DEFINITIONS.items():
        st.markdown(f"- **{name}** — {desc}")

    st.subheader("How to explore this demo")
    st.markdown(
        """
        - **Dataset Explorer** — browse, filter, and download the raw Breast Cancer Wisconsin dataset.
        - **Model Performance** — accuracy, ROC curve, and confusion matrix for the trained classifier.
        - **Explanation Viewer** — pick any test patient and see their SHAP explanation and QIEMF scores live.
        - **Failure Lab** — deliberately corrupt an explanation and watch QIEMF scores fall in real time.
        - **Validation & Figures** — high-QEI vs low-QEI group statistics and all generated research figures.
        - **Academic Write-up** — the full, data-driven manuscript text generated from this exact run.
        """
    )

# --------------------------------------------------------------------------
# Page: Dataset Explorer
# --------------------------------------------------------------------------

elif page == "🗂️ Dataset Explorer":
    st.header("Dataset Explorer — Breast Cancer Wisconsin (Diagnostic)")
    st.caption(
        "569 instances, 30 real-valued morphological features per instance, digitized from fine-needle "
        "aspirate biopsy images. Label: 0 = malignant, 1 = benign."
    )

    display_df = DATA["X_raw"].copy()
    display_df.insert(0, "Diagnosis", DATA["y"].map({0: "Malignant", 1: "Benign"}))

    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            diagnosis_filter = st.multiselect(
                "Diagnosis", options=["Malignant", "Benign"], default=["Malignant", "Benign"]
            )
        with fc2:
            search_feature = st.selectbox("Filter by feature", options=DATA["feature_names"])
        with fc3:
            f_min, f_max = float(display_df[search_feature].min()), float(display_df[search_feature].max())
            value_range = st.slider(
                f"{search_feature} range", min_value=f_min, max_value=f_max, value=(f_min, f_max)
            )

    filtered = display_df[
        display_df["Diagnosis"].isin(diagnosis_filter)
        & display_df[search_feature].between(*value_range)
    ]

    st.write(f"Showing **{len(filtered)}** of **{len(display_df)}** instances.")
    st.dataframe(filtered, use_container_width=True, height=380)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV", data=csv_bytes, file_name="breast_cancer_filtered.csv", mime="text/csv"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Class balance")
        counts = display_df["Diagnosis"].value_counts()
        fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.45,
                                      marker_colors=[ACCENT, PRIMARY])])
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Summary statistics")
        st.dataframe(display_df[DATA["feature_names"]].describe().T, use_container_width=True, height=340)

    with st.expander("Feature correlation heatmap (full dataset)"):
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(display_df[DATA["feature_names"]].corr(), cmap="vlag", vmin=-1, vmax=1, ax=ax,
                    cbar_kws={"label": "Pearson r"})
        st.pyplot(fig)
        plt.close(fig)

# --------------------------------------------------------------------------
# Page: Model Performance
# --------------------------------------------------------------------------

elif page == "📊 Model Performance":
    st.header("Model Performance")
    st.caption(f"Model: {DATA['model_name']} — trained on {len(DATA['X_train'])} instances, "
               f"evaluated on {len(DATA['X_test'])} held-out instances.")

    perf = DATA["performance"]
    metric_row(perf)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(DATA["y_test"], DATA["y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(DATA["y_test"], DATA["y_proba"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                  line=dict(color=NEUTRAL, width=1, dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           height=380, margin=dict(t=10, b=10, l=10, r=10),
                           annotations=[dict(x=0.6, y=0.1, text=f"AUC = {perf['ROC-AUC']:.4f}", showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Most Important Features (Global SHAP Importance)")
    st.dataframe(DATA["top10"], use_container_width=True)

# --------------------------------------------------------------------------
# Page: Explanation Viewer
# --------------------------------------------------------------------------

elif page == "🔍 Explanation Viewer":
    st.header("Per-Patient Explanation Viewer")
    st.caption("Select any held-out test instance to view its live SHAP explanation and QIEMF metric scores.")

    n_test = len(DATA["X_test"])
    idx = st.slider("Test sample index", min_value=0, max_value=n_test - 1, value=0)

    metrics_row = DATA["result"].metrics_df.iloc[idx]
    true_label = DATA["target_names"][int(DATA["y_test"].iloc[idx])]
    pred_label = DATA["target_names"][int(DATA["y_pred"][idx])]
    proba = DATA["y_proba"][idx]
    correct = true_label == pred_label

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True diagnosis", true_label.capitalize())
    c2.metric("Predicted diagnosis", pred_label.capitalize())
    c3.metric("P(benign)", f"{proba:.3f}")
    c4.metric("Prediction correct?", "✅ Yes" if correct else "❌ No")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("SHAP Waterfall Explanation")
        expl = shap.Explanation(
            values=DATA["result"].raw_shap[idx],
            base_values=DATA["base_value"],
            data=DATA["X_test_raw"].iloc[idx].values,
            feature_names=DATA["feature_names"],
        )
        fig = plt.figure(figsize=(9, 7))
        shap.plots.waterfall(expl, show=False, max_display=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("QIEMF Metric Scores")
        for name in ["EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]:
            dataset_mean = DATA["result"].metrics_df[name].mean()
            st.metric(
                name,
                f"{metrics_row[name]:.4f}",
                delta=f"{metrics_row[name] - dataset_mean:+.4f} vs. test-set mean",
            )
            st.caption(METRIC_DEFINITIONS[name])

    st.subheader("This Sample vs. Test-Set Average")
    categories = ["EFS", "FCI", "ISM", "RTS", "QEI"]
    sample_vals = [float(metrics_row[c]) for c in categories]
    mean_vals = [float(DATA["result"].metrics_df[c].mean()) for c in categories]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=sample_vals + [sample_vals[0]], theta=categories + [categories[0]],
                                   fill="toself", name=f"Sample #{idx}", line=dict(color=PRIMARY)))
    fig.add_trace(go.Scatterpolar(r=mean_vals + [mean_vals[0]], theta=categories + [categories[0]],
                                   fill="toself", name="Test-set mean", opacity=0.5, line=dict(color=NEUTRAL)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=420)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Page: Failure Lab (interactive corruption demo)
# --------------------------------------------------------------------------

elif page == "🧪 Failure Lab (Live)":
    st.header("Failure Lab — Watch QIEMF Penalize a Corrupted Explanation")
    st.caption(
        "Pick a test patient, choose a corruption method, and click Run. This recomputes all six QIEMF "
        "metrics live from the corrupted attribution vector using the same functions as the batch pipeline."
    )

    n_test = len(DATA["X_test"])
    idx = st.selectbox("Test sample index", options=list(range(n_test)), index=0, key="failure_idx")

    method = st.radio(
        "Corruption method",
        ["Shuffled Attributions", "Strong Random Noise", "Perturbed Important Features"],
        horizontal=True,
    )

    params: dict = {}
    if method == "Strong Random Noise":
        params["scale"] = st.slider("Noise scale (× local SHAP std. dev.)", 1.0, 10.0, 5.0, 0.5)
    elif method == "Perturbed Important Features":
        params["top_k"] = st.slider("Number of top features to corrupt", 1, 10, 5)
        params["factor"] = st.slider("Inversion / amplification factor", -8.0, -1.0, -4.0, 0.5)

    run_clicked = st.button("▶ Run Corruption", type="primary")

    if run_clicked:
        result = DATA["result"]
        raw_row = result.raw_shap[idx : idx + 1].copy()
        rng = DATA["rng"]

        if method == "Shuffled Attributions":
            corrupted_row = core.corrupt_shuffle(raw_row, rng)
        elif method == "Strong Random Noise":
            corrupted_row = core.corrupt_noise(raw_row, rng, scale=params["scale"])
        else:
            corrupted_row = core.corrupt_important_features(
                raw_row, rng, top_k=params["top_k"], factor=params["factor"]
            )

        corrupted_metrics = core.recompute_metrics_from_shap(
            corrupted_row,
            result.reference[idx : idx + 1],
            result.correlated_pairs,
            result.perturbed_shap[idx : idx + 1],
        )

        original_row = result.metrics_df.iloc[idx]
        metric_names = ["EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
        original_vals = [float(original_row[m]) for m in metric_names]
        corrupted_vals = [float(corrupted_metrics[m][0]) for m in metric_names]

        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig = go.Figure(
                data=[
                    go.Bar(name="Original", x=metric_names, y=original_vals, marker_color=PRIMARY),
                    go.Bar(name="Corrupted", x=metric_names, y=corrupted_vals, marker_color=ACCENT),
                ]
            )
            fig.update_layout(
                barmode="group", yaxis_range=[0, 1.05], template="plotly_white", height=440,
                title=f"Sample #{idx} — {method}",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            qei_before = original_row["QEI"]
            qei_after = corrupted_vals[metric_names.index("QEI")]
            drop_pct = 100.0 * (qei_before - qei_after) / qei_before if qei_before else 0.0
            st.metric("QEI before corruption", f"{qei_before:.4f}")
            st.metric("QEI after corruption", f"{qei_after:.4f}", delta=f"{qei_after - qei_before:+.4f}")
            st.metric("Relative QEI drop", f"{drop_pct:.1f}%")
            if qei_after < qei_before:
                st.success("QIEMF correctly penalized the corrupted explanation.")
            else:
                st.warning("QEI did not decrease for this configuration — try a stronger corruption setting.")

        with st.expander("Full before/after metric table"):
            comp_table = pd.DataFrame(
                {"Metric": metric_names, "Original": original_vals, "Corrupted": corrupted_vals}
            )
            comp_table["Δ"] = comp_table["Corrupted"] - comp_table["Original"]
            st.dataframe(comp_table, use_container_width=True)
    else:
        st.info("Choose a sample and corruption method, then click **Run Corruption** to see live results.")

    st.divider()
    st.subheader("Reference: Full Test-Set Failure Analysis")
    st.caption("Mean metric values across all test instances, from the batch pipeline run.")
    st.dataframe(DATA["failure_df"], use_container_width=True)

# --------------------------------------------------------------------------
# Page: Validation & Figures
# --------------------------------------------------------------------------

elif page == "📈 Validation & Figures":
    st.header("Validation: High-QEI vs Low-QEI Explanation Groups")
    st.caption(
        "The 20 test instances with the highest QEI (Group A) compared against the 20 with the lowest "
        "QEI (Group B), across every component metric, using Welch's t-test."
    )
    st.dataframe(DATA["comparison_df"], use_container_width=True)

    st.subheader("Overall QIEMF Metric Statistics")
    metric_cols = ["EE", "EE_norm", "EFS", "FCI", "ISM", "RTS", "QEI"]
    overall = pd.DataFrame(
        {
            "Metric": metric_cols,
            "Mean": [DATA["result"].metrics_df[c].mean() for c in metric_cols],
            "Std": [DATA["result"].metrics_df[c].std(ddof=1) for c in metric_cols],
        }
    )
    st.dataframe(overall, use_container_width=True)

    st.subheader("Generated Research Figures")
    st.caption("All figures below were generated at 300 DPI by this exact run and saved to figures/.")

    figure_files = [
        ("shap_summary_plot.png", "SHAP Summary Plot"),
        ("shap_bar_plot.png", "SHAP Global Feature Importance"),
        ("feature_importance_top10.png", "Top 10 Feature Importance"),
        ("metrics_correlation_heatmap.png", "QIEMF Metrics Correlation Heatmap"),
        ("high_vs_low_qei_boxplot.png", "High-QEI vs Low-QEI Boxplot"),
        ("failure_analysis_comparison.png", "Failure Analysis Comparison"),
        ("ee_histogram.png", "EE Distribution"),
        ("efs_histogram.png", "EFS Distribution"),
        ("ism_histogram.png", "ISM Distribution"),
        ("rts_histogram.png", "RTS Distribution"),
        ("qei_histogram.png", "QEI Distribution"),
    ]
    cols = st.columns(2)
    for i, (filename, caption) in enumerate(figure_files):
        path = core.FIGURES_DIR / filename
        if path.exists():
            cols[i % 2].image(str(path), caption=caption, use_container_width=True)

# --------------------------------------------------------------------------
# Page: Academic Write-up
# --------------------------------------------------------------------------

elif page == "📄 Academic Write-up":
    st.header("Academic Write-up")
    st.caption("Generated live from this run's actual numbers — suitable as a manuscript draft section.")
    writeup_path = core.RESULTS_DIR / "academic_writeup.md"
    if writeup_path.exists():
        st.markdown(writeup_path.read_text(encoding="utf-8"))
    else:
        st.warning("Write-up not yet generated.")

st.markdown(
    """
    <div class="qiemf-disclaimer">
    QIEMF Proof-of-Concept — a research validation artifact for explainability metric design.
    Not a certified medical device; not intended for real clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)
