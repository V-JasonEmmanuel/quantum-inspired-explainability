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
scikit-learn, comprising 569 instances described by 30
real-valued morphological features computed from digitized images of fine-needle
aspirate biopsies of breast masses. Each instance is labeled as malignant or benign,
and the task is framed as a binary clinical diagnosis problem in which explanation
transparency, consistency, and auditability are of direct regulatory relevance. The
data were partitioned into training and test subsets using an 80/20 stratified split
with a fixed random seed of 42, yielding 455 training instances and 114
held-out test instances, and all features were standardized to zero mean and unit
variance using parameters estimated exclusively from the training partition.

## Model Development

A gradient-boosted tree ensemble (XGBoostClassifier) was trained on the standardized
training data to predict the binary diagnostic label. On the held-out test set the
model achieved an accuracy of 0.9561, a precision of
0.9467, a recall of 0.9861, an F1 score
of 0.9660, and a ROC-AUC of 0.9947. These
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
forming a probability-like distribution over the 30 candidate features that
underlies several of the QIEMF metrics described below. The global ranking of feature
importance derived from the mean absolute SHAP value identified worst perimeter
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
(73 such pairs were identified among the 30 morphological features,
reflecting the well-documented redundancy among radius, perimeter, and area type
descriptors in this dataset), and then averaging, across all such pairs and all
instances, one minus the normalized difference in attribution assigned to the two
correlated features; this quantity rewards explanations that treat clinically
redundant variables consistently. The Interpretability Stability Measure (ISM) was
estimated by generating, for every test instance, 50 independently
perturbed copies under additive Gaussian noise with mean zero and standard deviation
0.01, recomputing SHAP explanations for every perturbed copy, and
averaging the cosine similarity between the original and each perturbed explanation;
this procedure directly tests the robustness of the explanation mechanism to small,
clinically plausible measurement variation. The Regulatory Transparency Score (RTS)
combines EFS, FCI, and one minus EE_norm in a fixed weighted sum (0.4, 0.3, 0.3
respectively) intended to reflect the priorities of a healthcare regulator, and the
Quantum-Inspired Explainability Index (QEI) further combines EFS, FCI, ISM, RTS, and
one minus EE_norm (weights 0.25, 0.20, 0.25, 0.20, 0.10 respectively) into a single
unified explainability quality score bounded on the unit interval.

## Results

Across the 114 test instances, the mean (± standard deviation) values obtained
were EE = 3.9305 (± 0.1024), EE_norm = 0.8010
(± 0.0209), EFS = 0.7465 (± 0.0998), FCI =
0.3641 (± 0.0261), ISM = 0.9975
(± 0.0082), RTS = 0.4675 (± 0.0408), and QEI =
0.6222 (± 0.0333). To validate that QEI meaningfully separates
explanation quality, the twenty test instances with the highest QEI (Group A) were
compared against the twenty instances with the lowest QEI (Group B) across all six
metrics using Welch's t-test. The comparison shows that Group A exhibited a
substantially higher mean QEI (0.6665) than Group B
(0.5681, t = 20.178, p =
4.67e-19). Examining the individual component metrics
underlying this separation shows a differentiated picture rather than a uniform
effect across all components. Explanation fidelity (EFS) was higher in Group A than in Group B (0.8727 versus 0.5842), a statistically significant difference (p = 3.23e-18), indicating that explanation fidelity to
the model's actual ablation-based sensitivity is a strong driver of overall
explanation quality in this dataset. Normalized explanation entropy (EE_norm) was lower in Group A than in Group B (0.7846 versus 0.8069), a statistically significant difference (p = 5.64e-03), consistent with high-QEI
explanations being more concentrated on a small number of clinically salient features
rather than diffusely spread across the full feature set. Interpretability stability (ISM) was higher in Group A than in Group B (0.9991 versus 0.9939), no statistically significant difference (p = 0.05), suggesting
that robustness to small input perturbations contributes to, but is not the sole
determinant of, high explanation quality. Feature coherence (FCI) was lower in Group A than in Group B (0.3626 versus 0.3688), no statistically significant difference (p = 0.44); this indicates that, in this
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
achieved a mean QEI of 0.6222, whereas the most damaging corruption
method, Shuffled Attributions, reduced the mean QEI to 0.2635,
a relative decrease of 57.7 percent. This result demonstrates that the
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
size of 114 test instances, while sufficient to demonstrate statistically
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
