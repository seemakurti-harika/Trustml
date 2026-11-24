# Experiment Plan & Results Log

Do not fill this log with fabricated values. Save experiment JSON from the app and summarize observed results.


| ID | Research experiment | Variables | Outcome measures | Status |
|---|---|---|---|
| E1 | Baseline model comparison | Models, fixed dataset/split/seed | Accuracy, weighted F1, log loss, Brier, ECE | Pending |
| E2 | Calibration | Raw, sigmoid, isotonic | ECE, Brier, log loss | Pending |
| E3 | Uncertainty | Confidence, entropy, margin | Signal distributions, correctness associations | Pending |
| E4 | Selective prediction | Confidence threshold sweep | Coverage, accepted accuracy, risk | Pending |
| E5 | Robustness | Gaussian noise scale | Accuracy degradation, uncertainty changes | Pending |
| E6 | Data drift | Reference/current dataset | KS statistic, drift feature count | Pending |
| E7 | Generalization | Dataset/domain transfer | Performance and calibration gap | Pending |

## Result table template
| Dataset | Model | Seed | Accuracy | F1 | ROC-AUC | Log loss | Brier | ECE | Coverage | Selective accuracy |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — |

## Reproducibility checklist
- Dataset and version/source
- Target and feature columns
- Split size and random seed
- Model parameters
- Preprocessing settings
- Calibration setting
- Saved experiment ID and exported JSON
- Limitations / anomalies noted
