from sklearn.datasets import load_breast_cancer, load_wine, load_digits, make_classification
import pandas as pd

DATASET_INFO = [
    {"id":"breast_cancer","name":"Breast Cancer Wisconsin","type":"classification","note":"Educational benchmark only — not a medical diagnostic tool."},
    {"id":"wine","name":"Wine Recognition","type":"classification","note":"Educational ML benchmark dataset."},
    {"id":"digits","name":"Optical Digits","type":"classification","note":"Educational ML benchmark dataset."},
    {"id":"synthetic","name":"Synthetic Classification","type":"classification","note":"Generated locally; configurable noise."},
]
def get_dataset(dataset_id, noise=0.12):
    if dataset_id == 'breast_cancer':
        d=load_breast_cancer(); df=pd.DataFrame(d.data, columns=d.feature_names); df['target']=d.target
    elif dataset_id == 'wine':
        d=load_wine(); df=pd.DataFrame(d.data, columns=d.feature_names); df['target']=d.target
    elif dataset_id == 'digits':
        d=load_digits(); df=pd.DataFrame(d.data, columns=[f'pixel_{i}' for i in range(d.data.shape[1])]); df['target']=d.target
    elif dataset_id == 'synthetic':
        x,y=make_classification(n_samples=800,n_features=12,n_informative=7,n_redundant=2,n_classes=3,flip_y=float(noise),random_state=42)
        df=pd.DataFrame(x,columns=[f'feature_{i}' for i in range(x.shape[1])]); df['target']=y
    else: raise ValueError('Unknown dataset id')
    return df
