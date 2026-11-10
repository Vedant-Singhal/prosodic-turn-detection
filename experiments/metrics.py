"""Compute all numbers/series the SUMMARY.html charts need -> metrics.json."""
import os, json, joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from eot_features import FEATURE_NAMES
from train_model import get_lang, new_lr, dur_weights, official_score_g, auc, oof, CANDIDATES

OUT = "metrics.json"

Xe, ye, ge, de = get_lang("english")
Xh, yh, gh, dh = get_lang("hindi")
X = np.vstack([Xe, Xh]); y = np.concatenate([ye, yh])
g = np.concatenate([ge, gh]); d = np.concatenate([de, dh])
lang = np.array(["e"] * len(ye) + ["h"] * len(yh))


def uni_auc(col, yy):
    c = col.copy(); c[np.isnan(c)] = np.nanmedian(c); return auc(yy, c)


# univariate AUCs (combined)
uni = sorted(((FEATURE_NAMES[i], float(uni_auc(X[:, i], y))) for i in range(len(FEATURE_NAMES))),
             key=lambda kv: -abs(kv[1] - 0.5))

# final model OOF (winner = dur-weighted LR C=0.5) for distributions + coefs
p = oof(lambda: new_lr(0.5, balanced=False), True, X, y, g, d)
me = official_score_g(ge, ye, de, p[lang == "e"])
mh = official_score_g(gh, yh, dh, p[lang == "h"])

# duration-restricted AUC (the key Hindi finding)
def dur_restricted(yy, dd, pp):
    out = []
    for thr in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        m = (yy == 1) | ((yy == 0) & (dd > thr))
        a = auc(yy[m], pp[m]) if len(np.unique(yy[m])) > 1 else float("nan")
        out.append([thr, float(a), int(((yy == 0) & (dd > thr)).sum())])
    return out

# p_eot histograms (eot vs hold) per lang
def hist(pp, yy, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    he = np.histogram(pp[yy == 1], bins=edges)[0].tolist()
    hh = np.histogram(pp[yy == 0], bins=edges)[0].tolist()
    return {"edges": edges.tolist(), "eot": he, "hold": hh}

# LR coefficients (standardized) from a full-data fit
final = new_lr(0.5, balanced=False)
final.fit(X, y, logisticregression__sample_weight=dur_weights(y, d))
coefs = final.named_steps["logisticregression"].coef_[0]
coef_rank = sorted(zip(FEATURE_NAMES, coefs.tolist()), key=lambda kv: -abs(kv[1]))

metrics = {
    "n": {"turns_per_lang": 100, "pauses_eng": int(len(ye)), "pauses_hin": int(len(yh)),
          "eot_per_lang": 100, "n_features": len(FEATURE_NAMES)},
    "baseline": {"english": 1600, "hindi": 850},
    "cv_final": {"english_ms": round(me[0] * 1000), "hindi_ms": round(mh[0] * 1000),
                 "auc": round(float(auc(y, p)), 3)},
    "op": {"english": {"threshold": me[2], "delay_ms": round(me[3] * 1000)},
           "hindi": {"threshold": mh[2], "delay_ms": round(mh[3] * 1000)}},
    "univariate_auc": uni,
    "coef_rank": coef_rank,
    "dur_restricted": {"english": dur_restricted(ye, de, p[lang == "e"]),
                       "hindi": dur_restricted(yh, dh, p[lang == "h"])},
    "hist": {"english": hist(p[lang == "e"], ye), "hindi": hist(p[lang == "h"], yh)},
}
json.dump(metrics, open(OUT, "w"), indent=1)
print("wrote", OUT)
print("CV final eng=%.0f hin=%.0f auc=%.3f" % (me[0]*1000, mh[0]*1000, auc(y, p)))
