from fastapi.testclient import TestClient
from app.main import app

def test_api_training_prediction_flow():
    with TestClient(app) as client:
        assert client.get('/api/health').status_code == 200
        trained=client.post('/api/train',json={'dataset_id':'wine','target':'target','models':['Logistic Regression'],'test_size':0.25,'seed':9,'scaling':True,'calibration':'none'})
        assert trained.status_code==200,trained.text
        result=trained.json()['experiments'][0]
        model_id=result['model_id']
        schema=client.get(f'/api/models/{model_id}/schema')
        assert schema.status_code==200 and schema.json()['columns']
        predict=client.post('/api/predict',json={'model_id':model_id,'values':schema.json()['sample'],'confidence_threshold':.5,'entropy_threshold':.65})
        assert predict.status_code==200
        assert 'probabilities' in predict.json()
        uncertainty=client.post('/api/uncertainty/analyze',params={'model_id':model_id})
        assert uncertainty.status_code==200

def test_reliability_and_drift_endpoints():
    with TestClient(app) as client:
        trained=client.post('/api/train',json={'dataset_id':'synthetic','target':'target','models':['Extra Trees'],'test_size':0.25,'seed':123,'scaling':False})
        assert trained.status_code==200,trained.text
        model_id=trained.json()['experiments'][0]['model_id']
        cal=client.post('/api/calibration/analyze',params={'model_id':model_id})
        assert cal.status_code==200 and cal.json()['ece'] is not None
        sel=client.post('/api/selective-prediction/analyze',params={'model_id':model_id,'threshold':0.8})
        assert sel.status_code==200 and 0<=sel.json()['coverage']<=1
        rob=client.post('/api/robustness/test',params={'model_id':model_id,'noise':0.2})
        assert rob.status_code==200 and 'perturbed_accuracy' in rob.json()
        csv_a=b'feature,target\n0.1,0\n0.2,0\n0.3,1\n0.4,1\n'
        csv_b=b'feature,target\n10.1,0\n10.2,0\n10.3,1\n10.4,1\n'
        a=client.post('/api/datasets/upload',files={'file':('reference.csv',csv_a,'text/csv')})
        b=client.post('/api/datasets/upload',files={'file':('current.csv',csv_b,'text/csv')})
        assert a.status_code==200 and b.status_code==200
        drift=client.post('/api/drift/analyze',params={'reference_key':a.json()['dataset_key'],'current_key':b.json()['dataset_key'],'threshold':0.1})
        assert drift.status_code==200 and drift.json()['features']

def test_ensemble_disagreement_endpoint():
    with TestClient(app) as client:
        trained = client.post('/api/train', json={
            'dataset_id': 'wine', 'target': 'target',
            'models': ['Random Forest', 'Extra Trees'],
            'test_size': 0.25, 'seed': 31, 'scaling': True,
            'calibration': 'none'
        })
        assert trained.status_code == 200, trained.text
        ids = [e['model_id'] for e in trained.json()['experiments']]
        result = client.post('/api/ensemble/analyze', json=ids)
        assert result.status_code == 200, result.text
        data = result.json()
        assert data['model_count'] == 2
        assert len(data['vote_disagreement']) > 0
        assert 0 <= data['mean_vote_disagreement'] <= 1
        assert data['ensemble_predictions']

def test_calibration_comparison_api():
    with TestClient(app) as client:
        trained=client.post('/api/train',json={'dataset_id':'wine','target':'target','models':['Logistic Regression'],'test_size':0.25,'seed':44,'scaling':True,'calibration':'none'})
        assert trained.status_code==200,trained.text
        model_id=trained.json()['experiments'][0]['model_id']
        response=client.post('/api/calibration/compare',json={'model_id':model_id,'methods':['none','sigmoid','isotonic']})
        assert response.status_code==200,response.text
        payload=response.json()
        assert len(payload['results'])==3
        assert 'none' in payload['curves'] and 'sigmoid' in payload['curves'] and 'isotonic' in payload['curves']
