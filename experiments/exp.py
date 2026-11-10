"""Fast experiments off the cached feature matrices.

Tests (a) LR hyperparameters and (b) DURATION-AWARE TRAINING WEIGHTS: weight
each hold by how 'dangerous' it is (long holds can be cut off, short holds are
free), so the loss aligns with the official metric. dur is used ONLY as a
training weight, never as an inference feature (causality preserved).
"""
import os, sys, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

CACHE = os.path.join(os.environ["SCRATCH"], "eot_cache")
TIMEOUT_S = 1.6
THR = np.round(np.arange(0.05, 1.0, 0.05), 3)
DLY = np.round(np.arange(0.10, 1.65, 0.05), 3)


def load(lang):
    d = np.load(os.path.join(CACHE, f"{lang}.npz"), allow_pickle=True)
    return d["X"], d["y"], d["groups"], d["durs"]


def score_g(tid, y, dur, p, budget=0.05):
    tid = np.asarray(tid); y = np.asarray(y); dur = np.asarray(dur); p = np.asarray(p)
    nt = len(set(tid.tolist())); eot = y == 1; best = None
    for t in THR:
        fires = p >= t
        for d in DLY:
            cut = set(tid[(~eot) & fires & (d < dur)].tolist())
            if len(cut) / max(1, nt) > budget:
                continue
            lat = np.where(fires[eot], d, TIMEOUT_S)
            ml = float(np.mean(lat)) if lat.size else TIMEOUT_S
            if best is None or ml < best[0]:
                best = (ml, len(cut) / max(1, nt), float(t), float(d))
    return best or (TIMEOUT_S, 0.0, 1.0, TIMEOUT_S)


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


Xe, ye, ge, de = load("english"); Xh, yh, gh, dh = load("hindi")
X = np.vstack([Xe, Xh]); y = np.concatenate([ye, yh])
g = np.concatenate([ge, gh]); d = np.concatenate([de, dh])
lang = np.array(["e"] * len(ye) + ["h"] * len(yh))
print(f"X={X.shape} eot={y.sum()}", flush=True)


def make_w(y, dur, gamma, eot_w):
    """Weight holds by dur**gamma (danger), eots by eot_w; balance the totals."""
    w = np.ones(len(y))
    hold = y == 0
    w[hold] = (np.minimum(dur[hold], 1.6)) ** gamma
    w[hold] *= (hold.sum() / (w[hold].sum() + 1e-9))   # normalise hold mass to N_hold
    w[y == 1] = eot_w * ((y == 1).sum() / max(1, (y == 1).sum()))
    return w


def oof(Cval, gamma=None, eot_w=1.0):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=3000,
                                 class_weight=("balanced" if gamma is None else None), C=Cval))
        if gamma is None:
            pipe.fit(X[tr], y[tr])
        else:
            w = make_w(y[tr], d[tr], gamma, eot_w)
            pipe.fit(X[tr], y[tr], logisticregression__sample_weight=w)
        p[te] = pipe.predict_proba(X[te])[:, 1]
    a = auc(y, p)
    me = score_g(ge, ye, de, p[lang == "e"])[0] * 1000
    mh = score_g(gh, yh, dh, p[lang == "h"])[0] * 1000
    return a, me, mh, (me + mh) / 2


print("\n== LR C-grid (class_weight=balanced) ==", flush=True)
for C in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    a, me, mh, av = oof(C)
    print(f"  C={C:<4} AUC={a:.3f}  eng={me:.0f}  hin={mh:.0f}  avg={av:.0f}", flush=True)

print("\n== duration-aware sample weights (C=0.3) ==", flush=True)
for gamma in [0.5, 1.0, 1.5, 2.0]:
    for eot_w in [1.0, 1.5, 2.0]:
        a, me, mh, av = oof(0.3, gamma=gamma, eot_w=eot_w)
        print(f"  gamma={gamma} eot_w={eot_w}: AUC={a:.3f}  eng={me:.0f}  hin={mh:.0f}  avg={av:.0f}", flush=True)
