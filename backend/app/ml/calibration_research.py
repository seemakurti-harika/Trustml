"""Leakage-aware calibration comparison evaluated on a held-out test set."""
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from .evaluation import ece

def _classwise_brier(y, probabilities, classes):
    return float(np.mean([brier_score_loss((y == cls).astype(int), probabilities[:, i]) for i, cls in enumerate(classes)]))

def _calibration_bins(y, probabilities, classes, n_bins=10):
    result=[]
    for idx, cls in enumerate(classes):
        observed, predicted = calibration_curve((y == cls).astype(int), probabilities[:, idx], n_bins=n_bins, strategy='uniform')
        result.append({'class':str(cls),'observed':observed.tolist(),'predicted':predicted.tolist()})
    return result

def compare_calibration(base_pipeline, X_train, y_train, X_test, y_test, classes, methods=('none','sigmoid','isotonic'), seed=42):
    rows=[]
    curves={}
    for method in methods:
        if method not in ('none','sigmoid','isotonic'):
            raise ValueError(f'Unsupported calibration method: {method}')
        estimator=clone(base_pipeline)
        if method=='none':
            estimator.fit(X_train,y_train)
        else:
            # Calibration folds are created strictly inside the training partition.
            estimator=CalibratedClassifierCV(estimator, method=method, cv=3)
            estimator.fit(X_train,y_train)
        p=np.asarray(estimator.predict_proba(X_test),dtype=float)
        p=np.clip(p,1e-15,1.0)
        p=p/p.sum(axis=1,keepdims=True)
        pred=estimator.classes_[np.argmax(p,axis=1)]
        ece_score=ece(np.asarray(y_test),p)
        row={'method':method,'accuracy':float(np.mean(pred==np.asarray(y_test))),
             'ece':float(ece_score),'brier_score':_classwise_brier(np.asarray(y_test),p,classes),
             'log_loss':float(log_loss(y_test,p,labels=classes))}
        rows.append(row)
        curves[method]=_calibration_bins(np.asarray(y_test),p,classes)
    return {'results':rows,'curves':curves,'classes':[str(c) for c in classes],
            'note':'Calibrators are fitted using training data only. Metrics and reliability curves are calculated on the held-out test partition. Calibration aligns predicted probabilities with observed frequencies; it does not guarantee individual predictions are correct.'}
