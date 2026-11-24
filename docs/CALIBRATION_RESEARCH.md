# Phase 3: Calibration Research

## Objective
Compare raw classifier probabilities with sigmoid (Platt-style) and isotonic calibration on a common held-out test set.

## Protocol
1. Split the selected dataset into train and test once using the experiment seed.
2. Fit preprocessing and the base estimator using training data only.
3. Fit calibrated estimators with cross-validation confined to the training partition.
4. Generate probabilities for the held-out test partition for all methods.
5. Report Accuracy, ECE, multiclass Brier score, and log loss. Accuracy is included because calibration may affect class rankings, but it is not the calibration objective.
6. Draw reliability curves class-wise using test predictions. Treat the test results as descriptive; model selection should use a separate validation set or nested evaluation if repeatedly tuning.

## Endpoint
`POST /api/calibration/compare`

Body: `{ "model_id": "...", "methods": ["none", "sigmoid", "isotonic"] }`

## Interpretation
- **Overconfidence:** average confidence is higher than observed frequency in bins.
- **Underconfidence:** average confidence is lower than observed frequency in bins.
- **Calibration:** among predictions assigned a similar probability, the observed frequency approaches that probability.

Calibration improves alignment between predicted probabilities and observed frequencies. It does not guarantee individual predictions are correct.


## Limitations
Calibration can vary across datasets, subgroups, and time. Isotonic regression can overfit small calibration samples. Reliability bins are noisy when they contain few samples. Calibration is not a substitute for uncertainty estimation, drift monitoring, domain validation, or decision analysis.
