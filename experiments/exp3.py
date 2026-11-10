"""Diagnostics: what actually limits the official score, per language."""
import os, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from eot_features import FEATURE_NAMES

CACHE=os.path.join(os.environ["SCRATCH"],"eot_cache")
TIMEOUT_S=1.6; THR=np.round(np.arange(0.05,1.0,0.05),3); DLY=np.round(np.arange(0.10,1.65,0.05),3)

def load(l):
    d=np.load(os.path.join(CACHE,f"{l}.npz"),allow_pickle=True); return d["X"],d["y"],d["groups"],d["durs"]
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s)
    if len(np.unique(y))<2: return float("nan")
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1,n0=y.sum(),len(y)-y.sum(); return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def score_detail(tid,y,dur,p,b=0.05):
    tid=np.asarray(tid); y=np.asarray(y); dur=np.asarray(dur); p=np.asarray(p)
    nt=len(set(tid.tolist())); eot=y==1; best=None
    for t in THR:
        fires=p>=t
        for d in DLY:
            cut=set(tid[(~eot)&fires&(d<dur)].tolist())
            if len(cut)/max(1,nt)>b: continue
            lat=np.where(fires[eot],d,TIMEOUT_S); ml=float(np.mean(lat)) if lat.size else TIMEOUT_S
            if best is None or ml<best[0]: best=(ml,len(cut)/nt,float(t),float(d),int(fires[eot].sum()))
    return best or (TIMEOUT_S,0,1.0,TIMEOUT_S,0)

def lr():
    return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
        LogisticRegression(max_iter=3000,class_weight="balanced",C=0.5))

for lang in ["english","hindi"]:
    X,y,g,d=load(lang)
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        m=lr(); m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    ml,cut,th,dl,nfire=score_detail(g,y,d,p)
    print(f"\n### {lang}: score={ml*1000:.0f}ms cutoff={cut*100:.1f}% th={th} delay={dl*1000:.0f} eots_firing={nfire}/{int((y==1).sum())}",flush=True)
    print(f"  AUC(all holds)      = {auc(y,p):.3f}",flush=True)
    for thr_dur in [0.5,0.6,0.7,0.8,1.0]:
        mask=(y==1)|((y==0)&(d>thr_dur))
        yy=y[mask]; pp=p[mask]
        nlong=((y==0)&(d>thr_dur)).sum()
        print(f"  AUC(eot vs holds dur>{thr_dur}) = {auc(yy,pp):.3f}   [{nlong} long holds]",flush=True)
    # score curve: mean delay achievable vs the delay grid (all eots must fire)
    eot=y==1
    print("  --- how low can delay go? (need <=5 turns cut) ---",flush=True)
    for dl_try in [0.4,0.5,0.6,0.7,0.8]:
        # best threshold at this delay
        bestfire=-1; bestcut=99
        for t in THR:
            fires=p>=t
            cut=len(set(g[(~eot)&fires&(dl_try<d)].tolist()))
            if cut<=5:
                nf=int(fires[eot].sum())
                if nf>bestfire: bestfire=nf; bestcut=cut; bestt=t
        if bestfire>=0:
            print(f"    delay={dl_try*1000:.0f}: max eots firing @<=5cut = {bestfire}/{int(eot.sum())} (th={bestt}, cut={bestcut})",flush=True)
        else:
            print(f"    delay={dl_try*1000:.0f}: infeasible",flush=True)
