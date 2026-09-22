# QIEMF Proof-of-Concept

**Quantum-Inspired Explainability Metrics for Regulated-Sector Deployment (QIEMF)**
— an end-to-end empirical validation on a clinical breast cancer diagnosis task.

This repository is a proof-of-concept research artifact. It does **not** aim to
be a production system; it aims to demonstrate that the six QIEMF metrics
(EE, EFS, FCI, ISM, RTS, QEI) can be computed on a real healthcare machine
learning problem and produce meaningful, differentiable explanation-quality
scores that separate high-quality explanations from low-quality ones, and
that decrease under simulated explanation failure.

## Use Case

Clinical breast cancer diagnosis using the Breast Cancer Wisconsin
(Diagnostic) dataset (`sklearn.datasets.load_breast_cancer`), framed as a
binary classification task (0 = malignant, 1 = benign) for a clinical
decision-support system operating in a regulated (healthcare) sector.

## What the Pipeline Does

1. **Model training** — 80/20 stratified train/test split (seed = 42),
   feature standardization, and training of an `XGBoostClassifier`
   (falls back automatically to `RandomForestClassifier` if XGBoost is
   unavailable), evaluated with accuracy, precision, recall, F1, and ROC-AUC.
2. **SHAP explanations** — exact `TreeExplainer` attributions for every test
   instance (raw, absolute, and L1-normalized vectors), a SHAP summary plot,
   a SHAP bar plot, and a top-10 feature importance table/chart.
3. **QIEMF metrics** — Explanation Entropy (EE / EE_norm), Explanation
   Fidelity Score (EFS, via feature-ablation reference), Feature Coherence
   Index (FCI, over correlated feature pairs), Interpretability Stability
   Measure (ISM, via 50 Gaussian-noise perturbations per sample), Regulatory
   Transparency Score (RTS), and the Quantum-Inspired Explainability Index
   (QEI), computed for every test sample.
4. **Framework validation** — comparison of the 20 highest-QEI and 20
   lowest-QEI explanations across all metrics, with Welch's t-tests.
5. **Simulated failure analysis** — three explanation-corruption methods
   (attribution shuffling, strong random noise, heavy perturbation of the
   most important features), with metrics recomputed to show QEI drops
   under each corruption.
6. **Visualizations** — eleven 300 DPI, publication-quality PNG figures.
7. **Tables** — five CSV tables (model performance, overall metric
   statistics, per-sample explanations, high- vs low-QEI comparison,
   failure-analysis results).
8. **Academic write-up** — a formal, data-driven manuscript section
   (`results/academic_writeup.md`) covering setup, dataset, model,
   methodology, results, discussion, threats to validity, limitations, and
   future work.

## Project Structure

```
quantum-inspired-explainability/
│
├── data/                 # Cached dataset + intermediate SHAP arrays (generated)
├── results/              # full_qiemf_metrics.csv, academic_writeup.md (generated)
├── figures/               # 11 publication-quality PNG figures, 300 DPI (generated)
├── tables/                # Table A-E CSV exports + top10_features.csv (generated)
├── logs/                  # qiemf.log — full run log (generated)
├── qiemf.py               # Complete, self-contained pipeline
├── requirements.txt
├── start.bat               # One-click Windows runner
└── README.md
```

The `data/`, `results/`, `figures/`, `tables/`, and `logs/` directories are
populated by running `qiemf.py`; the versions committed to this repository
are the outputs of an actual run and are regenerated (overwritten) every
time the script is executed.

## Requirements

- Python 3.11+
- See `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, seaborn,
  shap, xgboost, scipy)

## How to Run

### Windows (one click)

Double-click `start.bat`, or run it from a command prompt:

```
start.bat
```

This creates a local virtual environment (`.venv`), installs dependencies
from `requirements.txt`, and runs `qiemf.py`.

### macOS / Linux / manual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python qiemf.py
```

The full pipeline runs in well under a minute on commodity hardware.

## Expected Outputs

After a successful run:

- **`figures/`** — `shap_summary_plot.png`, `shap_bar_plot.png`,
  `feature_importance_top10.png`, `ee_histogram.png`, `efs_histogram.png`,
  `ism_histogram.png`, `rts_histogram.png`, `qei_histogram.png`,
  `metrics_correlation_heatmap.png`, `high_vs_low_qei_boxplot.png`,
  `failure_analysis_comparison.png`.
- **`tables/`** — `table_A_model_performance.csv`,
  `table_B_overall_explainability_metrics.csv`,
  `table_C_first20_explanations.csv`, `table_D_high_vs_low_qei.csv`,
  `table_E_failure_analysis.csv`, `top10_features.csv`.
- **`results/`** — `full_qiemf_metrics.csv` (all six metrics for every test
  sample) and `academic_writeup.md` (formal manuscript text).
- **`logs/qiemf.log`** — a full, timestamped record of the run.

## Interpretation of Results

On a representative run, the XGBoost model achieves ROC-AUC ≈ 0.99 on the
held-out test set, confirming the downstream explainability assessment is
performed against a clinically credible classifier. The mean QEI across
test instances is approximately 0.62. The 20 highest-QEI explanations differ
from the 20 lowest-QEI explanations with high statistical significance
(p < 1e-15) across every component metric: high-QEI explanations are more
focused (lower normalized entropy), more faithful to the model's actual
local sensitivity (higher EFS), more coherent across correlated clinical
variables (higher FCI), and more stable under small input perturbations
(higher ISM). In the simulated failure analysis, all three corruption
methods reduce mean QEI substantially relative to the unmodified baseline
(the largest drop, from attribution shuffling, is on the order of a 55-60%
relative decrease), demonstrating that QIEMF actively penalizes degraded
explanations rather than producing scores that are insensitive to
explanation quality.

Exact figures vary slightly between runs only if dependency versions
differ; with the pinned `random_state`/seed of 42 used throughout, results
are otherwise fully reproducible.

## Reproducibility Notes

- All randomized components (train/test split, model training, perturbation
  noise, correlation-based feature-pair selection) use a fixed seed of 42.
- `qiemf.py` uses type hints throughout, is organized into small, testable
  functions grouped by pipeline stage, logs every stage to both console and
  `logs/qiemf.log`, and falls back gracefully from XGBoost to RandomForest
  if XGBoost cannot be imported or trained.
- No production concerns (authentication, APIs, deployment infrastructure)
  are in scope; this is strictly a research validation artifact.
