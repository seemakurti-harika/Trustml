# TrustML ML Methodology

1. Dataset preparation: select a benchmark or upload a CSV. Identify target; require at least two classes and numeric features in this version.
2. Hold-out split: stratified split where class counts allow, with configurable seed and test fraction.
3. Preprocessing: median imputation and optional standard scaling are inside the training pipeline, avoiding train/test preprocessing leakage.
4. Models: Logistic Regression, Random Forest, Extra Trees, SVM and Gradient Boosting. Fit only training data.
5. Optional calibration: sigmoid or isotonic through `CalibratedClassifierCV` on training folds. Final metrics use the held-out set.
6. Metrics: predictive scores and probabilistic metrics including ECE, log loss, Brier score, plus confusion matrix.
7. Uncertainty: confidence is max probability; entropy is normalized by log(K); margin is top-two probability difference.
8. Selective prediction: confidence threshold defines acceptance; coverage is accepted fraction, risk is accepted error rate. Threshold sweeps expose trade-off.
9. Robustness: perturb numeric test features with controlled Gaussian noise and measure accuracy delta.
10. Explainability: permutation importance measures reliance by shuffled feature degradation, not causality.
11. Drift: numeric KS statistics compare reference and current distributions. Thresholds are heuristic and require context; drift does not prove performance loss.
