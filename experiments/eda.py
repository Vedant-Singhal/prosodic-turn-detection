"""EDA + cross-validation harness.

Reports: univariate feature AUCs, out-of-fold model AUC, the OFFICIAL score
metric (mean response delay @<=5% cutoffs) computed on grouped held-out
folds, and cross-language generalisation (train Eng -> test Hindi, etc).
"""
import os, sys, argparse
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from eot_data import build_dataset
from eot_features import FEATURE_NAMES

HANDOUT = "eot_handout/eot_data"
CACHE = os.path.join(os.environ.get("SCRATCH", "."), "eot_cache")

# ---- official scorer, reimplemented for in-memory CV evaluation ----------
TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def official_score(labels, durs, p, budget=0.05):
    """labels: 1=eot,0=hold; durs: pause duration; p: p_eot; grouped per turn.
    Returns best mean-delay(ms) subject to cutoff<=budget, plus AUC."""
    labels = np.asarray(labels); durs = np.asarray(durs); p = np.asarray(p)
    # per-turn grouping via the provided turn ids (passed separately)
    return None  # replaced below by official_score_g


def official_score_g(turn_ids, labels, durs, p, budget=0.05):
    turn_ids = np.asarray(turn_ids); labels = np.asarray(labels)
    durs = np.asarray(durs); p = np.asarray(p)
    uniq_turns = set(turn_ids.tolist())
    n_turns = len(uniq_turns)
    is_eot = labels == 1
    best = None
    for t in THRESHOLDS:
        fires = p >= t
        for d in DELAYS:
            cut = set(turn_ids[(~is_eot) & fires & (d < durs)].tolist())
            cutoff = len(cut) / max(1, n_turns)
            if cutoff > budget:
                continue
            lat = np.where(fires[is_eot], d, TIMEOUT_S)
            ml = float(np.mean(lat)) if lat.size else TIMEOUT_S
            if best is None or ml < best[0]:
                best = (ml, cutoff, float(t), float(d))
    if best is None:
        best = (TIMEOUT_S, 0.0, 1.0, TIMEOUT_S)
    return best


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def get_data(lang):
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{lang}.npz")
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True)
        return d["X"], d["y"], d["groups"], d["durs"]
    X, y, groups, keys, meta = build_dataset(os.path.join(HANDOUT, lang))
    import pandas as pd
    lab = pd.read_csv(os.path.join(HANDOUT, lang, "labels.csv")).sort_values(["turn_id", "pause_index"])
    # align durs to keys order
    durmap = {(r.turn_id, int(r.pause_index)): float(r.pause_end) - float(r.pause_start)
              for r in lab.itertuples()}
    durs = np.array([durmap[k] for k in keys])
    np.savez(fp, X=X, y=y, groups=groups, durs=durs)
    return X, y, groups, durs


def model_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))


def model_gb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                          max_iter=300, l2_regularization=1.0,
                                          min_samples_leaf=20, class_weight="balanced",
                                          random_state=0)


def cv_oof(X, y, groups, durs, model_fn, n_splits=5):
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = model_fn(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    a = auc(y, oof)
    ms, cut, th, dl = official_score_g(groups, y, durs, oof)
    return oof, a, ms, cut, th, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    args = ap.parse_args()

    Xe, ye, ge, de = get_data("english")
    Xh, yh, gh, dh = get_data("hindi")
    print(f"english X={Xe.shape} eot={ye.sum()}  hindi X={Xh.shape} eot={yh.sum()}\n")

    # ---- univariate AUC per feature (combined) ----
    Xc = np.vstack([Xe, Xh]); yc = np.concatenate([ye, yh])
    print("== univariate AUC per feature (|AUC-0.5| ranked, combined Eng+Hin) ==")
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        col = Xc[:, i].copy()
        med = np.nanmedian(col); col[np.isnan(col)] = med
        a = auc(yc, col)
        rows.append((abs(a - 0.5), a, name))
    for strength, a, name in sorted(rows, reverse=True):
        print(f"  {name:22s} AUC={a:.3f}  (|.5|={strength:.3f})")

    # ---- model CV within each language + combined ----
    print("\n== grouped 5-fold CV (official metric = mean delay ms @<=5% cutoff) ==")
    for lang, (X, y, g, d) in [("english", (Xe, ye, ge, de)),
                               ("hindi", (Xh, yh, gh, dh))]:
        for mname, mfn in [("LR", model_lr), ("GB", model_gb)]:
            _, a, ms, cut, th, dl = cv_oof(X, y, g, d, mfn)
            print(f"  {lang:8s} {mname}: AUC={a:.3f}  score={ms*1000:.0f}ms  cutoff={cut*100:.1f}%  (th={th},delay={dl*1000:.0f})")

    # combined (turn ids unique across langs already: en__/hi__ prefixes)
    Xall = np.vstack([Xe, Xh]); yall = np.concatenate([ye, yh])
    gall = np.concatenate([ge, gh]); dall = np.concatenate([de, dh])
    for mname, mfn in [("LR", model_lr), ("GB", model_gb)]:
        _, a, ms, cut, th, dl = cv_oof(Xall, yall, gall, dall, mfn)
        print(f"  COMBINED {mname}: AUC={a:.3f}  score={ms*1000:.0f}ms  cutoff={cut*100:.1f}%  (th={th},delay={dl*1000:.0f})")

    # ---- cross-language generalisation ----
    print("\n== cross-language (train A -> test B), official metric ==")
    for (na, Xa, ya, ga, da), (nb, Xb, yb, gb, db) in [
        (("eng", Xe, ye, ge, de), ("hin", Xh, yh, gh, dh)),
        (("hin", Xh, yh, gh, dh), ("eng", Xe, ye, ge, de))]:
        for mname, mfn in [("LR", model_lr), ("GB", model_gb)]:
            m = mfn(); m.fit(Xa, ya); p = m.predict_proba(Xb)[:, 1]
            a = auc(yb, p); ms, cut, th, dl = official_score_g(gb, yb, db, p)
            print(f"  {na}->{nb} {mname}: AUC={a:.3f}  score={ms*1000:.0f}ms  cutoff={cut*100:.1f}%")


if __name__ == "__main__":
    main()
