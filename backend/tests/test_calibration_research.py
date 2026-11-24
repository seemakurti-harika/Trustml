import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from app.ml.calibration_research import compare_calibration

def test_compare_calibration_methods_returns_metrics_and_curves():
    data=load_wine(as_frame=True)
    X=data.data; y=data.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=7,stratify=y)
    pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000))])
    result=compare_calibration(pipeline,X_train,y_train,X_test,y_test,np.unique(y),methods=('none','sigmoid','isotonic'))
    assert [r['method'] for r in result['results']]==['none','sigmoid','isotonic']
    for r in result['results']:
        assert 0 <= r['accuracy'] <= 1
        assert 0 <= r['ece'] <= 1
        assert 0 <= r['brier_score'] <= 1
        assert r['log_loss'] >= 0
    assert set(result['curves'])=={'none','sigmoid','isotonic'}
    assert len(result['curves']['none'])==3
