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

## Live Demo (Interactive Frontend)

In addition to the batch pipeline, this repository includes `app.py`, an
interactive Streamlit web application that runs the same model training,
SHAP explanation, and QIEMF metric computations live, in the browser. It is
not a set of static screenshots: `app.py` imports the functions defined in
`qiemf.py` directly, so everything you see is computed in front of you by
the running process.

The live demo includes:

- **Dataset Explorer** — browse, filter (by diagnosis and by feature-value
  range), and download the raw Breast Cancer Wisconsin dataset, plus class
  balance and summary statistics.
- **Model Performance** — live accuracy/precision/recall/F1/ROC-AUC,
  confusion matrix, ROC curve, and the top-10 SHAP feature importance table.
- **Explanation Viewer** — pick any of the 114 held-out test patients with a
  slider and see their live SHAP waterfall explanation alongside their six
  QIEMF metric scores and a radar chart against the test-set average.
- **Failure Lab (Live)** — pick a patient and a corruption method (attribution
  shuffling, strong random noise, or heavy perturbation of the most
  important features), click **Run Corruption**, and watch all six QIEMF
  metrics — including QEI — recompute and drop in real time.
- **Validation & Figures** — the high-QEI vs low-QEI group comparison table
  and every generated 300 DPI research figure.
- **Academic Write-up** — the full, data-driven manuscript text rendered
  in-browser.

### Running the live demo

**Windows (one click):** double-click `start_demo.bat`, or run it from a
command prompt:

```
start_demo.bat
```

**macOS / Linux / manual:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser. The first load takes a few seconds while the model trains and
SHAP explanations are computed; subsequent navigation between pages is
instant because the results are cached for the lifetime of the running
process.

## Project Structure

```
quantum-inspired-explainability/
│
├── data/                 # Cached dataset + intermediate SHAP arrays (generated)
├── results/              # full_qiemf_metrics.csv, academic_writeup.md (generated)
├── figures/               # 11 publication-quality PNG figures, 300 DPI (generated)
├── tables/                # Table A-E CSV exports + top10_features.csv (generated)
├── logs/                  # qiemf.log — full run log (generated)
├── qiemf.py               # Complete, self-contained batch pipeline
├── app.py                 # Interactive Streamlit live demo (frontend)
├── requirements.txt
├── start.bat               # One-click Windows runner for the batch pipeline
├── start_demo.bat          # One-click Windows runner for the live demo
└── README.md
```

The `data/`, `results/`, `figures/`, `tables/`, and `logs/` directories are
populated by running `qiemf.py`; the versions committed to this repository
are the outputs of an actual run and are regenerated (overwritten) every
time the script is executed.

## Requirements

- Python 3.11+
- See `requirements.txt` (numpy, pandas, scikit-learn, matplotlib, seaborn,
  shap, xgboost, scipy, streamlit, plotly)

## How to Run the Batch Pipeline

For the non-interactive pipeline that writes figures, tables, and the
academic write-up to disk (see the Live Demo section above for the
interactive frontend instead):

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
from the 20 lowest-QEI explanations with high statistical significance on
overall QEI (p ≈ 5e-19). Breaking this down by component metric shows a
differentiated picture rather than a uniform effect: EFS and RTS separate
the two groups very strongly (p < 1e-17), normalized entropy (EE_norm)
separates them significantly but more modestly (p ≈ 0.006), ISM is only
marginally different between groups (p ≈ 0.05), and FCI does not differ
significantly between the two groups in this dataset (p ≈ 0.44) — indicating
that, for this particular model and dataset, explanation fidelity is the
dominant driver of QEI separation rather than feature coherence. In the
simulated failure analysis, all three corruption methods reduce mean QEI
substantially relative to the unmodified baseline (the largest drop, from
attribution shuffling, is on the order of a 55-60% relative decrease),
demonstrating that QIEMF actively penalizes degraded explanations rather
than producing scores that are insensitive to explanation quality.

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
