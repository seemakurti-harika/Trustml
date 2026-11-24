from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd, io, json, numpy as np, uuid
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from .database.db import init_db, connect
from .ml.datasets import get_dataset, DATASET_INFO
from .ml.training import train_models, MODEL_STORE
from .ml.uncertainty import scores, categorize
from .ml.evaluation import evaluate
from .ml.calibration_research import compare_calibration
from .ml.advanced_drift import analyze_drift

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize persistent storage before serving requests."""
    init_db()
    yield

app = FastAPI(
    title='TrustML API',
    version='0.1.0',
    description='Uncertainty-aware ML reliability research platform',
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
DATA_STORE = {}
class TrainRequest(BaseModel):
 dataset_id: str | None=None
 dataset_name: str | None=None
 target: str='target'
 models: list[str]=Field(default_factory=lambda:['Logistic Regression','Random Forest'])
 test_size: float=Field(default=0.25,ge=0.1,le=0.5)
 seed: int=42
 scaling: bool=True
 calibration: str='none'
 dataset_key: str | None=None
 noise: float=0.12
class PredictRequest(BaseModel):
 model_id: str
 values: dict[str,float]
 confidence_threshold: float=0.55
 entropy_threshold: float=0.65
class DriftRequest(BaseModel):
 reference_key: str
 current_key: str
 threshold: float=0.1
class CalibrationCompareRequest(BaseModel):
 model_id: str
 methods: list[str]=Field(default_factory=lambda:['none','sigmoid','isotonic'])
@app.get('/api/health')
def health(): return {'status':'ok','service':'TrustML API'}
@app.get('/api/datasets')
def datasets(): return DATASET_INFO
@app.get('/api/datasets/uploads')
def uploaded_datasets():
 return [{'dataset_key':k,'name':k.split(':',1)[-1],'columns':list(df.columns),'rows':len(df)} for k,df in DATA_STORE.items() if k.startswith('upload:')]
@app.post('/api/datasets/upload')
async def upload_dataset(file: UploadFile=File(...)):
 try:
  raw=await file.read()
  if len(raw)>15*1024*1024: raise ValueError('CSV exceeds 15 MB upload limit')
  if not raw: raise ValueError('Uploaded file is empty')
  df=pd.read_csv(io.BytesIO(raw))
  if df.empty or len(df.columns)<2: raise ValueError('CSV needs at least one feature and one target column')
  key=file.filename or 'upload.csv'; storekey=f'upload:{uuid.uuid4().hex[:8]}:{key}'
  DATA_STORE[storekey]=df
  return {'dataset_key':storekey,'name':key,'rows':int(len(df)),'columns':list(df.columns),'preview':df.head(8).to_dict(orient='records')}
 except Exception as e: raise HTTPException(status_code=400,detail=f'CSV validation failed: {e}')
@app.get('/api/datasets/{dataset_id}/summary')
def dataset_summary(dataset_id: str, noise: float=0.12, dataset_key: str | None=None):
 try:
  df=DATA_STORE.get(dataset_key) if dataset_key else get_dataset(dataset_id,noise)
  if df is None: raise ValueError('Uploaded dataset not found. Upload it again before using this endpoint.')
  return {'name':dataset_key.split(':')[-1] if dataset_key else dataset_id,'rows':int(len(df)),'features':int(len(df.columns)-1),'columns':list(df.columns),'missing':df.isna().sum().to_dict(),'duplicates':int(df.duplicated().sum()),'dtypes':df.dtypes.astype(str).to_dict(),'stats':df.describe(include='all').replace({np.nan:None}).to_dict(),'class_distribution':df.iloc[:,-1].value_counts(dropna=False).to_dict(),'preview':df.head(10).replace({np.nan:None}).to_dict(orient='records')}
 except Exception as e: raise HTTPException(status_code=400,detail=str(e))
@app.post('/api/train')
def train(req:TrainRequest):
 try:
  df=DATA_STORE.get(req.dataset_key) if req.dataset_key else get_dataset(req.dataset_id or 'wine',req.noise)
  if df is None: raise ValueError('Dataset upload not found in this session')
  df.attrs['dataset_name']=req.dataset_name or req.dataset_id or 'Uploaded Dataset'
  if req.target!='target' and req.dataset_key: pass
  if req.dataset_key and req.target not in df.columns: raise ValueError('Target column missing')
  return {'experiments':train_models(df,req.target,req.models,req.test_size,req.seed,req.scaling,req.calibration)}
 except Exception as e: raise HTTPException(status_code=400,detail=str(e))
@app.get('/api/experiments')
def experiments():
 with connect() as c: rows=c.execute('SELECT * FROM experiments ORDER BY created_at DESC').fetchall()
 return [dict(r,parameters=json.loads(r['parameters']),metrics=json.loads(r['metrics'])) for r in rows]
@app.get('/api/experiments/{experiment_id}')
def experiment(experiment_id: str):
 with connect() as c: row=c.execute('SELECT * FROM experiments WHERE id=?',(experiment_id,)).fetchone()
 if not row: raise HTTPException(status_code=404,detail='Experiment not found')
 return dict(row,parameters=json.loads(row['parameters']),metrics=json.loads(row['metrics']))
@app.post('/api/predict')
def predict(req:PredictRequest):
 if req.model_id not in MODEL_STORE: raise HTTPException(status_code=404,detail='Model not found. Train a model first.')
 item=MODEL_STORE[req.model_id];
 try:
  x=pd.DataFrame([{c:req.values.get(c) for c in item['columns']}]); probs=item['model'].predict_proba(x)[0]; pred=item['model'].classes_[int(np.argmax(probs))]
  u=scores(probs.reshape(1,-1)); ent=float(u['normalized_entropy'][0]); conf=float(u['confidence'][0]); margin=float(u['margin'][0]);
  status=categorize(ent); accepted=(conf>=req.confidence_threshold and ent<=req.entropy_threshold)
  return {'predicted_class':str(pred),'probabilities':{str(c):float(p) for c,p in zip(item['model'].classes_,probs)},'confidence':conf,'entropy':ent,'margin':margin,'uncertainty':status,'accepted':accepted,'decision':'Accepted' if accepted else 'Abstained / Review required','message':'Prediction accepted — uncertainty is below the configured threshold.' if accepted else 'Prediction abstained — review recommended due to uncertainty.'}
 except Exception as e: raise HTTPException(status_code=400,detail=str(e))
@app.post('/api/calibration/analyze')
def calibration_analysis(model_id:str):
 if model_id not in MODEL_STORE: raise HTTPException(404,'Model not found')
 item=MODEL_STORE[model_id]; y=item['y_test'].to_numpy(); p=item['probs']; labels=item['model'].classes_
 bins=[]
 for i,c in enumerate(labels):
  truth=(y==c).astype(int); conf=p[:,i]
  frac,mean=calibration_curve(truth,conf,n_bins=10,strategy='uniform')
  bins.append({'class':str(c),'observed':frac.tolist(),'predicted':mean.tolist()})
 metrics=evaluate(y,p,labels)
 return {'model':item['name'],'calibration_curve':bins,'ece':metrics['ece'],'brier_score':metrics['brier_score'],'log_loss':metrics['log_loss'],'note':'Calibration curves are calculated on held-out test predictions. Compare separate training runs for raw, sigmoid, and isotonic methods.'}
@app.post('/api/calibration/compare')
def calibration_compare(req:CalibrationCompareRequest):
 if req.model_id not in MODEL_STORE: raise HTTPException(404,'Model not found. Train a model first.')
 if not req.methods or len(req.methods)>3: raise HTTPException(400,'Choose between one and three calibration methods')
 item=MODEL_STORE[req.model_id]
 try:
  return compare_calibration(item['base_pipeline'],item['X_train'],item['y_train'],item['X_test'],item['y_test'],item['model'].classes_,req.methods,item.get('seed',42))
 except Exception as e: raise HTTPException(400,detail=f'Calibration comparison failed: {e}')
@app.post('/api/uncertainty/analyze')
def uncertainty_analysis(model_id:str):
 if model_id not in MODEL_STORE: raise HTTPException(404,'Model not found')
 item=MODEL_STORE[model_id]; u=scores(item['probs']);
 return {'confidence':u['confidence'].tolist(),'entropy':u['normalized_entropy'].tolist(),'margin':u['margin'].tolist(),'means':{k:float(np.mean(v)) for k,v in u.items()},'labels':item['labels']}
@app.post('/api/selective-prediction/analyze')
def selective(model_id:str,threshold:float=0.5):
 if model_id not in MODEL_STORE: raise HTTPException(404,'Model not found')
 item=MODEL_STORE[model_id]; conf=item['probs'].max(axis=1); preds=item['probs'].argmax(axis=1); y=item['y_test'].to_numpy(); accepted=conf>=threshold; coverage=float(accepted.mean()); selective_accuracy=float((preds[accepted]==y[accepted]).mean()) if accepted.any() else None
 curve=[]
 for t in np.linspace(0.25,0.95,15):
  a=conf>=t; curve.append({'threshold':float(t),'coverage':float(a.mean()),'risk':float(1-(preds[a]==y[a]).mean()) if a.any() else None})
 return {'coverage':coverage,'selective_accuracy':selective_accuracy,'risk':1-selective_accuracy if selective_accuracy is not None else None,'abstention_rate':1-coverage,'curve':curve}
@app.post('/api/robustness/test')
def robustness(model_id:str,noise:float=0.1):
 if model_id not in MODEL_STORE: raise HTTPException(404,'Model not found')
 item=MODEL_STORE[model_id]; X=item['X_test'].copy(); y=item['y_test']; baseline=float(accuracy_score(y,item['probs'].argmax(axis=1)))
 Xnoise=X.copy(); num=Xnoise.select_dtypes(include='number').columns; Xnoise[num]=Xnoise[num]+np.random.default_rng(42).normal(0,float(noise),Xnoise[num].shape)
 pert=item['model'].predict(Xnoise); acc=float(accuracy_score(y,pert));
 return {'baseline_accuracy':baseline,'perturbed_accuracy':acc,'degradation':baseline-acc,'noise_scale':noise,'description':'Gaussian perturbations applied to numeric features. Noise scale is in raw feature units.'}
@app.post('/api/drift/analyze')
def drift(reference_key:str,current_key:str,threshold:float=0.1):
 ref=DATA_STORE.get(reference_key); cur=DATA_STORE.get(current_key)
 if ref is None or cur is None: raise HTTPException(400,'Upload both datasets first')
 from scipy.stats import ks_2samp
 features=[c for c in ref.columns if c in cur.columns and pd.api.types.is_numeric_dtype(ref[c]) and pd.api.types.is_numeric_dtype(cur[c])]
 results=[]
 for c in features:
  a=ref[c].dropna(); b=cur[c].dropna(); ks=float(ks_2samp(a,b).statistic) if len(a) and len(b) else None
  results.append({'feature':c,'ks_statistic':ks,'drift':bool(ks is not None and ks>threshold),'reference_missing':float(ref[c].isna().mean()),'current_missing':float(cur[c].isna().mean())})
 drifted=sum(x['drift'] for x in results)
 return {'status':'Significant' if drifted/ max(1,len(results))>0.5 else 'Warning' if drifted else 'Stable','drifted_features':drifted,'total_features':len(results),'features':results,'threshold':threshold,'note':'KS-statistic threshold is a configurable heuristic; data drift alone does not imply performance degradation.'}
@app.get('/api/models/{model_id}/schema')
def model_schema(model_id:str):
 if model_id not in MODEL_STORE: raise HTTPException(404,'Model not found')
 item=MODEL_STORE[model_id]
 return {'columns':item['columns'],'sample':item['X_test'].iloc[0].to_dict(),'dataset':item['dataset'],'name':item['name']}
@app.get('/api/models/{model_id}/explain')
def explain(model_id:str):
 if model_id not in MODEL_STORE: raise HTTPException(404,'Model not found')
 item=MODEL_STORE[model_id]; model=item['model']; X=item['X_test']; y=item['y_test']
 try:
  imp=permutation_importance(model,X,y,n_repeats=4,random_state=42,scoring='accuracy')
  ranked=sorted([{'feature':f,'importance':float(v),'std':float(s)} for f,v,s in zip(item['columns'],imp.importances_mean,imp.importances_std)],key=lambda x:x['importance'],reverse=True)[:20]
  return {'method':'Permutation importance on held-out samples','features':ranked,'warning':'Importance reflects model reliance, not causality.'}
 except Exception as e: raise HTTPException(400,detail=f'Explanation failed: {e}')

@app.post('/api/ensemble/analyze')
def ensemble_analysis(model_ids: list[str]):
 """Aggregate compatible trained models and quantify predictive disagreement."""
 if len(model_ids) < 2:
  raise HTTPException(400, 'Select at least two trained models for ensemble analysis')
 missing=[mid for mid in model_ids if mid not in MODEL_STORE]
 if missing:
  raise HTTPException(404, f'Models not found: {missing}')
 items=[MODEL_STORE[mid] for mid in model_ids]
 dataset_names={it['dataset'] for it in items}
 if len(dataset_names)!=1:
  raise HTTPException(400, 'Selected models must be trained on the same dataset')
 # Require identical held-out truth and row order so predictions can be compared meaningfully.
 y0=items[0]['y_test'].to_numpy()
 if any(not np.array_equal(y0,it['y_test'].to_numpy()) for it in items[1:]):
  raise HTTPException(400, 'Models have different test partitions; retrain using the same split configuration')
 class_labels=list(items[0]['model'].classes_)
 if any(list(it['model'].classes_)!=class_labels for it in items[1:]):
  raise HTTPException(400, 'Models use incompatible class labels')
 probs=np.stack([it['probs'] for it in items],axis=0)
 mean_probs=probs.mean(axis=0)
 individual_preds=probs.argmax(axis=2)
 ensemble_pred=mean_probs.argmax(axis=1)
 # Vote disagreement is 1 - fraction of models voting for the majority class.
 vote_counts=np.stack([np.sum(individual_preds==class_idx,axis=0) for class_idx in range(len(class_labels))],axis=0)
 vote_disagreement=1-(vote_counts.max(axis=0)/len(items))
 # Probability disagreement is the mean class-wise variance across model probabilities.
 probability_variance=probs.var(axis=0).mean(axis=1)
 from .ml.uncertainty import scores as uncertainty_scores
 u=uncertainty_scores(mean_probs)
 return {
  'models':[it['name'] for it in items], 'model_count':len(items),
  'ensemble_probabilities':mean_probs.tolist(),
  'ensemble_predictions':[str(class_labels[i]) for i in ensemble_pred],
  'labels':[str(c) for c in class_labels],
  'vote_disagreement':vote_disagreement.tolist(),
  'mean_vote_disagreement':float(vote_disagreement.mean()),
  'probability_variance':probability_variance.tolist(),
  'mean_probability_variance':float(probability_variance.mean()),
  'mean_confidence':float(u['confidence'].mean()),
  'mean_normalized_entropy':float(u['normalized_entropy'].mean()),
  'note':'Disagreement detects some model instability, but models may share biases and agree while being wrong.'
 }

class ConformalRequest(BaseModel):
 model_id: str
 alpha: float=Field(default=0.1,gt=0.01,lt=0.5)
 calibration_fraction: float=Field(default=0.25,ge=0.1,le=0.5)
 seed: int=42

@app.post('/api/conformal/analyze')
def conformal_analyze(req:ConformalRequest):
 """Split-conformal classification with a calibration split carved from training data."""
 if req.model_id not in MODEL_STORE: raise HTTPException(404,'Model not found')
 item=MODEL_STORE[req.model_id]
 from sklearn.model_selection import train_test_split
 from sklearn.base import clone
 from sklearn.metrics import accuracy_score
 try:
  strat=item['y_train'] if item['y_train'].value_counts().min()>=2 else None
  X_fit,X_cal,y_fit,y_cal=train_test_split(item['X_train'],item['y_train'],test_size=req.calibration_fraction,random_state=req.seed,stratify=strat)
  fit_model=clone(item['base_pipeline']).fit(X_fit,y_fit)
  cal_probs=fit_model.predict_proba(X_cal); test_probs=fit_model.predict_proba(item['X_test'])
  classes=fit_model.classes_; cal_true_idx=np.array([np.where(classes==label)[0][0] for label in y_cal])
  nonconformity=1-cal_probs[np.arange(len(cal_probs)),cal_true_idx]
  q_level=min(1.0,float(np.ceil((len(nonconformity)+1)*(1-req.alpha))/len(nonconformity)))
  q=float(np.quantile(nonconformity,q_level,method='higher'))
  sets=[[str(classes[j]) for j,pv in enumerate(row) if 1-float(pv)<=q] for row in test_probs]
  y_test=item['y_test'].to_numpy(); covered=[str(y) in ps for y,ps in zip(y_test,sets)]
  preds=[str(classes[int(np.argmax(row))]) for row in test_probs]
  return {'alpha':req.alpha,'nominal_coverage':1-req.alpha,'calibration_samples':len(y_cal),'quantile':q,'empirical_test_coverage':float(np.mean(covered)),'mean_set_size':float(np.mean([len(ps) for ps in sets])),'singleton_rate':float(np.mean([len(ps)==1 for ps in sets])),'test_accuracy_of_argmax':float(accuracy_score(y_test,np.argmax(test_probs,axis=1))),'prediction_sets':sets,'predictions':preds,'true_labels':[str(v) for v in y_test],'note':'Split conformal classification. Coverage guarantee is marginal and relies on exchangeability between calibration and future/test samples; it is not a per-case guarantee.'}
 except Exception as e: raise HTTPException(400,detail=f'Conformal analysis failed: {e}')

@app.post('/api/drift/advanced')
def advanced_drift(reference_key:str,current_key:str,ks_threshold:float=0.1,psi_threshold:float=0.2):
 ref=DATA_STORE.get(reference_key); cur=DATA_STORE.get(current_key)
 if ref is None or cur is None: raise HTTPException(400,'Upload both datasets first')
 return analyze_drift(ref,cur,ks_threshold,psi_threshold)
