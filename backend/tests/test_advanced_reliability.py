import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.ml.advanced_drift import analyze_drift, population_stability_index


def test_psi_detects_numeric_shift():
    ref=np.linspace(0,1,300)
    shifted=np.linspace(5,6,300)
    assert population_stability_index(ref,shifted) > 0.2
    report=analyze_drift(pd.DataFrame({'x':ref}),pd.DataFrame({'x':shifted}))
    assert report['features'][0]['drift'] is True


def test_chi_square_detects_categorical_change():
    ref=pd.DataFrame({'kind':['a']*90+['b']*10})
    cur=pd.DataFrame({'kind':['a']*10+['b']*90})
    result=analyze_drift(ref,cur)
    assert result['features'][0]['type']=='categorical'
    assert result['features'][0]['drift'] is True


def test_conformal_api_returns_prediction_sets():
    with TestClient(app) as client:
        train=client.post('/api/train',json={'dataset_id':'wine','target':'target','models':['Random Forest'],'seed':53,'test_size':0.25})
        assert train.status_code==200,train.text
        model_id=train.json()['experiments'][0]['model_id']
        response=client.post('/api/conformal/analyze',json={'model_id':model_id,'alpha':0.1})
        assert response.status_code==200,response.text
        data=response.json()
        assert data['prediction_sets'] and 0<=data['empirical_test_coverage']<=1
        assert data['mean_set_size']>=0  # empty prediction sets are valid conformal outcomes


def test_advanced_drift_endpoint():
    with TestClient(app) as client:
        a=client.post('/api/datasets/upload',files={'file':('a.csv',b'value,category\n0,a\n1,a\n2,b\n3,b\n','text/csv')})
        b=client.post('/api/datasets/upload',files={'file':('b.csv',b'value,category\n20,c\n21,c\n22,c\n23,c\n','text/csv')})
        result=client.post('/api/drift/advanced',params={'reference_key':a.json()['dataset_key'],'current_key':b.json()['dataset_key']})
        assert result.status_code==200,result.text
        assert result.json()['drifted_features']>=1
