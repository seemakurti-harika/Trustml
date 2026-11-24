import numpy as np
from app.ml.uncertainty import scores, categorize
from app.ml.datasets import get_dataset
from app.ml.training import train_models
from app.ml.evaluation import ece
from app.database.db import init_db

def test_uncertainty_scores():
    p=np.array([[.9,.1],[.5,.5]])
    u=scores(p)
    assert u['confidence'][0]>.8
    assert u['normalized_entropy'][1]>.99
    assert u['margin'][0]>.7
    assert categorize(.1)=='Low uncertainty'

def test_dataset_shapes():
    assert len(get_dataset('wine'))==178
    assert len(get_dataset('digits'))==1797

def test_ece_range():
    assert 0<=ece(np.array([0,1]),np.array([[.8,.2],[.2,.8]]))<=1

def test_train_and_store():
    init_db()
    df=get_dataset('wine')
    df.attrs['dataset_name']='Wine test'
    results=train_models(df,'target',['Logistic Regression'],.25,42,True,'none')
    assert results and results[0]['metrics']['accuracy']>0
    assert results[0]['model_id']
