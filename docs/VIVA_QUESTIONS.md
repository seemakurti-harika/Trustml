# TrustML Viva Questions & Short Answers

1. **What is TrustML?** A local research tool for evaluating model prediction reliability beyond accuracy.
2. **What is uncertainty?** A measure of ambiguity or lack of confidence associated with model predictions.
3. **Confidence vs calibration?** Confidence is the model's probability output; calibration checks whether probabilities align with empirical frequencies.
4. **What is entropy?** `-Σ p log(p)`; it is higher when probability mass is spread across classes.
5. **Why normalize entropy?** Divide by `log(K)` to compare scores for different class counts.
6. **What is prediction margin?** Difference between top and second class probability.
7. **What is ECE?** Weighted absolute gap between average confidence and accuracy within bins.
8. **What is Brier score?** Mean squared error of predicted probabilities against one-hot labels.
9. **Why accuracy is insufficient?** It does not show calibration, per-sample confidence, coverage, or performance under shift.
10. **What is selective prediction?** The model accepts only predictions meeting criteria and abstains on others.
11. **What is coverage?** Fraction of all cases accepted.
12. **What is selective risk?** Error rate among accepted cases.
13. **Does more abstention mean safer?** No. It can increase accepted accuracy while reducing coverage; safety depends on context and error costs.
14. **What is probability calibration?** Adjusting probability outputs so predicted confidence better matches observed frequency.
15. **Platt/sigmoid vs isotonic?** Sigmoid fits a parametric calibration mapping; isotonic learns a monotonic nonparametric mapping and can overfit smaller samples.
16. **What is data drift?** A change in feature distributions between reference and current data.
17. **Does drift prove performance decreased?** No; performance impact requires labels or downstream evaluation.
18. **What is KS test?** A statistic comparing cumulative distributions for numeric samples.
19. **What is permutation importance?** Shuffling a feature and measuring performance degradation to estimate model reliance.
20. **Does importance imply causality?** No.
21. **How prevent leakage?** Fit preprocessing only within training folds/pipeline; keep final test data separate for evaluation.
22. **Why use random seed?** Reproducibility of splits and stochastic estimators.
23. **Why multiple models?** To compare approaches and assess sensitivity to algorithm choice.
24. **What is log loss?** Penalizes inaccurate probability distributions, especially confident wrong predictions.
25. **What is ROC-AUC?** Ranking metric based on true-positive vs false-positive rates across thresholds.
26. **Why use a confusion matrix?** Shows which classes are confused and how often.
27. **Why median imputation?** A simple robust baseline for numeric missing values.
28. **What is StandardScaler?** Transforms numeric features using training mean and standard deviation.
29. **Why SVM probability?** TrustML needs probability estimates; SVC probability mode estimates them with internal calibration.
30. **Limitations of TrustML?** Numeric-only training, process-memory models/uploads, heuristic drift thresholds, and benchmark datasets.
