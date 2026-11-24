import uuid, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from .evaluation import evaluate
from .uncertainty import scores
from ..database.db import connect

MODEL_MAP={'Logistic Regression':LogisticRegression(max_iter=2000),'Random Forest':RandomForestClassifier(n_estimators=160,random_state=42),'Extra Trees':ExtraTreesClassifier(n_estimators=160,random_state=42),'SVM':SVC(probability=True),'Gradient Boosting':GradientBoostingClassifier(random_state=42)}
MODEL_STORE={}
def train_models(df, target, model_names, test_size=0.25, seed=42, scaling=True, calibration='none'):
    if target not in df.columns: raise ValueError('Target column not found')
    if df[target].isna().any(): raise ValueError('Target contains missing values')
    X=df.drop(columns=[target]); y=df[target]
    if X.empty or X.select_dtypes(include='number').shape[1] != X.shape[1]: raise ValueError('Features must all be numeric. Encode categorical columns before training.')
    if y.nunique()<2: raise ValueError('Target must contain at least two classes')
    if y.nunique()>20: raise ValueError('Target has too many classes for this classification demo')
    if X.isna().mean().max()>0.8: raise ValueError('A feature has more than 80% missing values. Remove or repair it before training.')
    if X.isna().all().any(): raise ValueError('A feature is completely missing. Remove or repair it before training.')
    strat=y if y.value_counts().min()>=2 else None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=float(test_size),random_state=int(seed),stratify=strat)
    labels=np.unique(y_train)
    results=[]
    for name in model_names:
        if name not in MODEL_MAP: continue
        base=clone(MODEL_MAP[name])
        pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler() if scaling else 'passthrough'),('model',base)])
        model=pipe
        if calibration in ('sigmoid','isotonic'):
            model=CalibratedClassifierCV(pipe,method=calibration,cv=3)
        model.fit(X_train,y_train)
        probs=model.predict_proba(X_test)
        met=evaluate(y_test.to_numpy(),probs,model.classes_)
        u=scores(probs)
        model_id=str(uuid.uuid4())
        MODEL_STORE[model_id]={'model':model,'base_pipeline':pipe,'columns':list(X.columns),'labels':[str(x) for x in model.classes_], 'dataset':str(df.attrs.get('dataset_name','Uploaded dataset')),'name':name,'X_train':X_train.copy(),'y_train':y_train.copy(),'X_test':X_test.copy(),'y_test':y_test.copy(),'probs':probs,'seed':int(seed)}
        met.update({'mean_confidence':float(u['confidence'].mean()),'mean_entropy':float(u['normalized_entropy'].mean()),'mean_margin':float(u['margin'].mean()),'model_id':model_id})
        expid=str(uuid.uuid4()); timestamp=datetime.now(timezone.utc).isoformat()
        params={'test_size':test_size,'seed':seed,'scaling':scaling,'calibration':calibration}
        with connect() as c: c.execute('INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)',(expid,timestamp,MODEL_STORE[model_id]['dataset'],len(df),X.shape[1],target,name,json.dumps(params),json.dumps(met),'completed'))
        results.append({'id':expid,'model_id':model_id,'model':name,'metrics':met,'created_at':timestamp,'dataset':MODEL_STORE[model_id]['dataset']})
    return results
