"""Statistical drift diagnostics for numerical and categorical columns."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def population_stability_index(reference, current, bins=10):
    ref=np.asarray(reference, dtype=float); cur=np.asarray(current, dtype=float)
    if not len(ref) or not len(cur): return None
    edges=np.unique(np.quantile(ref, np.linspace(0,1,bins+1)))
    if len(edges)<3: edges=np.linspace(float(np.min(ref)),float(np.max(ref))+1e-9,bins+1)
    edges[0]=-np.inf; edges[-1]=np.inf
    rp=np.histogram(ref,bins=edges)[0]/len(ref); cp=np.histogram(cur,bins=edges)[0]/len(cur)
    rp=np.clip(rp,1e-6,None); cp=np.clip(cp,1e-6,None)
    return float(np.sum((cp-rp)*np.log(cp/rp)))


def analyze_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold=0.1, psi_threshold=0.2):
    common=[c for c in reference.columns if c in current.columns]
    results=[]
    for col in common:
        a=reference[col].dropna(); b=current[col].dropna()
        missing_delta=float(current[col].isna().mean()-reference[col].isna().mean())
        if pd.api.types.is_numeric_dtype(reference[col]) and pd.api.types.is_numeric_dtype(current[col]):
            ks=ks_2samp(a,b) if len(a)>0 and len(b)>0 else None
            psi=population_stability_index(a,b) if len(a)>0 and len(b)>0 else None
            score=float(ks.statistic) if ks else None
            flagged=bool((score is not None and score>=threshold) or (psi is not None and psi>=psi_threshold))
            results.append({'feature':col,'type':'numeric','ks_statistic':score,'ks_p_value':float(ks.pvalue) if ks else None,'psi':psi,'reference_missing_rate':float(reference[col].isna().mean()),'current_missing_rate':float(current[col].isna().mean()),'missing_rate_delta':missing_delta,'drift':flagged,'method':'KS statistic and PSI (heuristic thresholds)'})
        else:
            if len(a) and len(b):
                cats=list(set(a.astype(str))|set(b.astype(str))); ac=a.astype(str).value_counts().reindex(cats,fill_value=0); bc=b.astype(str).value_counts().reindex(cats,fill_value=0)
                table=np.vstack([ac.to_numpy(),bc.to_numpy()]); chi=chi2_contingency(table, correction=False) if table.shape[1]>1 else None
                statistic=float(chi.statistic) if chi else 0.0; pval=float(chi.pvalue) if chi else 1.0
            else: statistic=None; pval=None
            flagged=bool(pval is not None and pval<0.05)
            results.append({'feature':col,'type':'categorical','chi_square':statistic,'p_value':pval,'reference_missing_rate':float(reference[col].isna().mean()),'current_missing_rate':float(current[col].isna().mean()),'missing_rate_delta':missing_delta,'drift':flagged,'method':'Chi-square test (p<0.05)'})
    drifted=sum(r['drift'] for r in results); n=max(1,len(results))
    status='Significant' if drifted/n>=0.5 and drifted else 'Warning' if drifted else 'Stable'
    return {'status':status,'drifted_features':int(drifted),'total_features':len(results),'features':results,'thresholds':{'ks':threshold,'psi':psi_threshold,'categorical_p_value':0.05},'note':'Drift thresholds are heuristics. Distribution shifts do not by themselves establish model performance degradation.'}
