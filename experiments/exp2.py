"""Forward feature selection + model blend, off cached features."""
import os, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from eot_features import FEATURE_NAMES

CACHE = os.path.join(os.environ["SCRATCH"], "eot_cache")
TIMEOUT_S = 1.6
THR = np.round(np.arange(0.05, 1.0, 0.05), 3); DLY = np.round(np.arange(0.10, 1.65, 0.05), 3)


def load(l):
    d = np.load(os.path.join(CACHE, f"{l}.npz"), allow_pickle=True)
    return d["X"], d["y"], d["groups"], d["durs"]


def score_g(tid, y, dur, p, b=0.05):
    tid=np.asarray(tid); y=np.asarray(y); dur=np.asarray(dur); p=np.asarray(p)
    nt=len(set(tid.tolist())); eot=y==1; best=None
    for t in THR:
        fires=p>=t
        for d in DLY:
            cut=set(tid[(~eot)&fires&(d<dur)].tolist())
            if len(cut)/max(1,nt)>b: continue
            lat=np.where(fires[eot],d,TIMEOUT_S); ml=float(np.mean(lat)) if lat.size else TIMEOUT_S
            if best is None or ml<best[0]: best=(ml,)
    return (best or (TIMEOUT_S,))[0]*1000


def auc(y,s):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1,n0=y.sum(),len(y)-y.sum()
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else .5


Xe,ye,ge,de=load("english"); Xh,yh,gh,dh=load("hindi")
X=np.vstack([Xe,Xh]); y=np.concatenate([ye,yh]); g=np.concatenate([ge,gh])
lang=np.array(["e"]*len(ye)+["h"]*len(yh))


def lr(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
    LogisticRegression(max_iter=3000,class_weight="balanced",C=0.5))
def gb(): return HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,
    l2_regularization=1.0,min_samples_leaf=25,class_weight="balanced",random_state=0)


def oof_cols(cols, model=lr):
    Xs=X[:,cols]; p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(Xs,y,g):
        m=model(); m.fit(Xs[tr],y[tr]); p[te]=m.predict_proba(Xs[te])[:,1]
    return p


def report(p,tag):
    a=auc(y,p); me=score_g(ge,ye,de,p[lang=="e"]); mh=score_g(gh,yh,dh,p[lang=="h"])
    print(f"  {tag:28s} AUC={a:.3f}  eng={me:.0f}  hin={mh:.0f}  avg={(me+mh)/2:.0f}",flush=True)
    return a,me,mh


# ---- forward selection maximising combined OOF AUC ----
print("== forward selection (LR, maximise combined OOF AUC) ==",flush=True)
remaining=list(range(len(FEATURE_NAMES))); chosen=[]; best_auc=0.5; trace=[]
while remaining and len(chosen)<16:
    scored=[]
    for c in remaining:
        p=oof_cols(chosen+[c]); scored.append((auc(y,p),c))
    scored.sort(reverse=True)
    a,c=scored[0]
    if a<=best_auc+0.002 and len(chosen)>=6:
        break
    best_auc=a; chosen.append(c); remaining.remove(c); trace.append((len(chosen),a,FEATURE_NAMES[c]))
    print(f"  +{FEATURE_NAMES[c]:22s} -> AUC={a:.3f}",flush=True)

print("\nselected:",[FEATURE_NAMES[c] for c in chosen],flush=True)
print("\n== evaluate on selected feature set ==",flush=True)
p_lr=oof_cols(chosen,lr); report(p_lr,"LR(selected)")
p_gb_all=oof_cols(list(range(len(FEATURE_NAMES))),gb); report(p_gb_all,"GB(all)")
p_gb_sel=oof_cols(chosen,gb); report(p_gb_sel,"GB(selected)")
# blends
report((p_lr+p_gb_all)/2,"blend LR(sel)+GB(all)")
report((p_lr+p_gb_sel)/2,"blend LR(sel)+GB(sel)")
report(0.6*p_lr+0.4*p_gb_all,"blend .6LR(sel)+.4GB(all)")

print("\nCHOSEN_IDX="+repr(chosen),flush=True)
