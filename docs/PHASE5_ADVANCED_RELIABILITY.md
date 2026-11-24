# Phase 5: Advanced Drift and Split Conformal Prediction

## Split conformal classification
TrustML exposes `POST /api/conformal/analyze` with `model_id`, `alpha`, `calibration_fraction`, and `seed`. It reserves a calibration subset from the model training partition, refits the base pipeline on proper-training data, computes nonconformity `1 - probability(true class)`, and derives a finite-sample quantile. Prediction sets include classes whose nonconformity is below the quantile.

The test partition is untouched by fitting and is used only for empirical coverage and argmax accuracy. Returned metrics include nominal coverage, empirical test coverage, mean set size, singleton rate, quantile, prediction sets, and predictions/truth.

**Limitation:** conformal coverage is marginal, not a per-example guarantee, and depends on exchangeability between calibration and future observations. Distribution shift can invalidate coverage. Empty prediction sets can occur.

## Advanced drift
`POST /api/drift/advanced` compares two uploaded datasets on shared columns:
- Numeric: Kolmogorov-Smirnov statistic and p-value, plus Population Stability Index (PSI).
- Categorical: chi-square test and p-value.
- Both: missing-value rates and change.

Thresholds are configurable (default KS statistic 0.1 and PSI 0.2); categorical drift uses p<0.05. These are heuristics. Drift is not synonymous with performance degradation; labeled performance monitoring is required to establish that.

## API examples
- `POST /api/conformal/analyze` JSON: `{"model_id":"<trained-id>","alpha":0.1,"calibration_fraction":0.25,"seed":42}`
- `POST /api/drift/advanced?reference_key=<upload-key>&current_key=<upload-key>&ks_threshold=0.1&psi_threshold=0.2`
