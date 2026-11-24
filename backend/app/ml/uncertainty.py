import numpy as np

def scores(probabilities):
    p=np.asarray(probabilities,dtype=float)
    p=np.clip(p,1e-12,1)
    entropy=-(p*np.log(p)).sum(axis=1)
    normalized=entropy/np.log(p.shape[1]) if p.shape[1]>1 else np.zeros(len(p))
    sorted_p=np.sort(p,axis=1)
    margin=sorted_p[:,-1]-(sorted_p[:,-2] if p.shape[1]>1 else 0)
    return {'confidence':p.max(axis=1),'entropy':entropy,'normalized_entropy':normalized,'margin':margin}

def categorize(normalized_entropy, low=0.25, high=0.65):
    if normalized_entropy < low: return 'Low uncertainty'
    if normalized_entropy < high: return 'Moderate uncertainty'
    return 'High uncertainty'
