import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

def ece(y_true, probs, bins=10):
    conf=probs.max(axis=1); pred=probs.argmax(axis=1); edges=np.linspace(0,1,bins+1); score=0.0
    for i in range(bins):
        mask=(conf>edges[i]) & (conf<=edges[i+1])
        if mask.any(): score += mask.mean()*abs((pred[mask]==y_true[mask]).mean()-conf[mask].mean())
    return float(score)

def evaluate(y_true, probs, labels):
    preds=probs.argmax(axis=1)
    try: auc=float(roc_auc_score(y_true,probs,multi_class='ovr',average='macro')) if len(labels)>2 else float(roc_auc_score(y_true,probs[:,1]))
    except Exception: auc=None
    try: brier=float(np.mean([brier_score_loss((y_true==c).astype(int),probs[:,i]) for i,c in enumerate(labels)]))
    except Exception: brier=None
    return {'accuracy':float(accuracy_score(y_true,preds)), 'precision':float(precision_score(y_true,preds,average='weighted',zero_division=0)),
      'recall':float(recall_score(y_true,preds,average='weighted',zero_division=0)), 'f1':float(f1_score(y_true,preds,average='weighted',zero_division=0)),
      'roc_auc':auc,'log_loss':float(log_loss(y_true,probs,labels=labels)),'brier_score':brier,'ece':ece(y_true,probs),
      'confusion_matrix':confusion_matrix(y_true,preds,labels=labels).tolist(), 'labels':[str(x) for x in labels]}
