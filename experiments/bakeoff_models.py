"""Model bake-off: intermediate algos beyond plain LR.

Human-steered: after Hindi prosody-wall diagnosis, compare mid-complexity
regularized models on the OFFICIAL metric with grouped CV. Hindi priority.

    python bakeoff_models.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import joblib

from eot_data import build_dataset
from eot_features import FEATURE_NAMES

HANDOUT = "eot_handout/eot_data"
CACHE = os.path.join(os.environ.get("SCRATCH", "."), "eot_cache")
TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def official_score_g(turn_ids, labels, durs, p, budget=0.05):
    turn_ids = np.asarray(turn_ids)
    labels = np.asarray(labels)
    durs = np.asarray(durs)
    p = np.asarray(p)
    n_turns = len(set(turn_ids.tolist()))
    is_eot = labels == 1
    best = None
    for t in THRESHOLDS:
        fires = p >= t
        for d in DELAYS:
            cut = set(turn_ids[(~is_eot) & fires & (d < durs)].tolist())
            if len(cut) / max(1, n_turns) > budget:
                continue
            lat = np.where(fires[is_eot], d, TIMEOUT_S)
            ml = float(np.mean(lat)) if lat.size else TIMEOUT_S
            if best is None or ml < best[0]:
                best = (ml, len(cut) / max(1, n_turns), float(t), float(d))
    return best if best else (TIMEOUT_S, 0.0, 1.0, TIMEOUT_S)


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def get_lang(lang):
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{lang}.npz")
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True)
        return d["X"], d["y"], d["groups"], d["durs"]
    X, y, groups, keys, _ = build_dataset(os.path.join(HANDOUT, lang))
    lab = pd.read_csv(os.path.join(HANDOUT, lang, "labels.csv"))
    dm = {(r.turn_id, int(r.pause_index)): float(r.pause_end) - float(r.pause_start)
          for r in lab.itertuples()}
    durs = np.array([dm[k] for k in keys])
    np.savez(fp, X=X, y=y, groups=groups, durs=durs)
    return X, y, groups, durs


def dur_weights(y, dur, gamma=0.5, eot_w=1.5):
    w = np.ones(len(y), dtype=float)
    hold = y == 0
    w[hold] = np.minimum(dur[hold], 1.6) ** gamma
    w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def preproc():
    return SimpleImputer(strategy="median"), StandardScaler()


def make_pipe(clf):
    imp, sc = preproc()
    return Pipeline([("imp", imp), ("sc", sc), ("clf", clf)])


# ---------------------------------------------------------------------------
# Candidate factories. Each returns (fitted-ready estimator, supports_proba,
#                                   uses_sample_weight, needs_calibration)
# If needs_calibration and no predict_proba, we wrap with CalibratedClassifierCV.
# ---------------------------------------------------------------------------

def candidates():
    return [
        ("LR L2 C=0.5 bal", make_pipe(LogisticRegression(
            max_iter=4000, C=0.5, class_weight="balanced")), True, False, False),
        ("LR L2 C=0.5 dur-w", make_pipe(LogisticRegression(
            max_iter=4000, C=0.5, class_weight=None)), True, True, False),
        ("ElasticNet LR C=0.5", make_pipe(LogisticRegression(
            max_iter=5000, C=0.5, penalty="elasticnet", solver="saga",
            l1_ratio=0.5, class_weight="balanced")), True, False, False),
        ("ElasticNet LR C=1.0", make_pipe(LogisticRegression(
            max_iter=5000, C=1.0, penalty="elasticnet", solver="saga",
            l1_ratio=0.3, class_weight="balanced")), True, False, False),
        ("ElasticNet LR C=0.3", make_pipe(LogisticRegression(
            max_iter=5000, C=0.3, penalty="elasticnet", solver="saga",
            l1_ratio=0.5, class_weight="balanced")), True, False, False),
        ("LinSVM C=0.1 +Platt", make_pipe(LinearSVC(
            C=0.1, class_weight="balanced", max_iter=8000, dual="auto")),
         False, False, True),
        ("LinSVM C=0.5 +Platt", make_pipe(LinearSVC(
            C=0.5, class_weight="balanced", max_iter=8000, dual="auto")),
         False, False, True),
        ("LDA", make_pipe(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
         True, False, False),
        ("QDA reg=0.5", make_pipe(QuadraticDiscriminantAnalysis(reg_param=0.5)),
         True, False, False),
        ("RBF-SVM C=1 g=scale", make_pipe(SVC(
            C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
            probability=True)), True, False, False),
        ("RBF-SVM C=0.5 g=scale", make_pipe(SVC(
            C=0.5, kernel="rbf", gamma="scale", class_weight="balanced",
            probability=True)), True, False, False),
        ("RBF-SVM C=2 g=scale", make_pipe(SVC(
            C=2.0, kernel="rbf", gamma="scale", class_weight="balanced",
            probability=True)), True, False, False),
        ("RBF-SVM C=1 g=0.02", make_pipe(SVC(
            C=1.0, kernel="rbf", gamma=0.02, class_weight="balanced",
            probability=True)), True, False, False),
        ("MLP 32 L2=1e-2", make_pipe(MLPClassifier(
            hidden_layer_sizes=(32,), activation="relu", alpha=1e-2,
            learning_rate_init=1e-3, max_iter=400, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=0)),
         True, False, False),
        ("MLP 64 L2=5e-2", make_pipe(MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", alpha=5e-2,
            learning_rate_init=1e-3, max_iter=400, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=0)),
         True, False, False),
        ("MLP 32-16 L2=1e-1", make_pipe(MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-1,
            learning_rate_init=1e-3, max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25, random_state=0)),
         True, False, False),
        ("HGB depth2 leaf40", HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=200,
            l2_regularization=2.0, min_samples_leaf=40, max_features=0.6,
            class_weight="balanced", random_state=0), True, False, False),
        ("HGB depth2 leaf50", HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.03, max_iter=250,
            l2_regularization=5.0, min_samples_leaf=50, max_features=0.5,
            class_weight="balanced", random_state=0), True, False, False),
        ("LR L2 + isotonic", "CALIB_LR", True, False, False),
        ("RBF-SVM + isotonic", "CALIB_RBF", True, False, False),
    ]


def fit_one(est, uses_w, needs_cal, Xtr, ytr, dtr, gtr):
    """Fit estimator; for needs_cal wrap LinearSVC-style with Platt via CV."""
    if est == "CALIB_LR":
        base = make_pipe(LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced"))
        # Calibrate on inner groups — use sklearn CalibratedClassifierCV with cv=3
        # (not grouped perfectly, but small-n pragmatic)
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(Xtr, ytr)
        return cal
    if est == "CALIB_RBF":
        base = make_pipe(SVC(C=1.0, kernel="rbf", gamma="scale",
                             class_weight="balanced", probability=False))
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(Xtr, ytr)
        return cal

    m = clone(est) if not isinstance(est, str) else est
    if needs_cal:
        # Wrap pipeline ending in LinearSVC
        cal = CalibratedClassifierCV(m, method="sigmoid", cv=3)
        cal.fit(Xtr, ytr)
        return cal

    if uses_w:
        w = dur_weights(ytr, dtr)
        # Pipeline sample_weight goes to last step
        if hasattr(m, "steps"):
            step = m.steps[-1][0]
            m.fit(Xtr, ytr, **{f"{step}__sample_weight": w})
        else:
            m.fit(Xtr, ytr, sample_weight=w)
    else:
        # HistGB etc. need imputed X — wrap if bare
        if isinstance(m, HistGradientBoostingClassifier):
            m = make_pipe(m)
        m.fit(Xtr, ytr)
    return m


def predict_p(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    # decision_function fallback
    s = m.decision_function(X)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s


def oof(name, est, uses_w, needs_cal, X, y, g, d, n=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        m = fit_one(est, uses_w, needs_cal, X[tr], y[tr], d[tr], g[tr])
        p[te] = predict_p(m, X[te])
    return p


def main():
    Xe, ye, ge, de = get_lang("english")
    Xh, yh, gh, dh = get_lang("hindi")
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh])
    d = np.concatenate([de, dh])
    lang = np.array(["e"] * len(ye) + ["h"] * len(yh))

    print(f"data: X={X.shape} eot={int(y.sum())}\n")
    print(f"{'algo':28s} {'AUC':>6s} {'engCV':>6s} {'hinCV':>6s} {'avg':>6s} {'hinPri':>6s}")
    print("-" * 72)

    rows = []
    best_hin = None
    for name, est, has_proba, uses_w, needs_cal in candidates():
        try:
            p = oof(name, est, uses_w, needs_cal, X, y, g, d)
            a = auc(y, p)
            me = official_score_g(ge, ye, de, p[lang == "e"])[0] * 1000
            mh = official_score_g(gh, yh, dh, p[lang == "h"])[0] * 1000
            avg = (me + mh) / 2
            # Hindi-priority score: weight Hindi 2x (hidden test mostly Hindi)
            hin_pri = (me + 2 * mh) / 3
            print(f"{name:28s} {a:6.3f} {me:6.0f} {mh:6.0f} {avg:6.0f} {hin_pri:6.0f}")
            rows.append(dict(name=name, auc=a, eng=me, hin=mh, avg=avg, hin_pri=hin_pri,
                             est=est, uses_w=uses_w, needs_cal=needs_cal))
            if best_hin is None or hin_pri < best_hin[0]:
                best_hin = (hin_pri, name)
        except Exception as e:
            print(f"{name:28s} FAILED: {e}")

    rows.sort(key=lambda r: r["hin_pri"])
    print("\n== ranked by Hindi-priority (eng+2*hin)/3 ==")
    for i, r in enumerate(rows[:8]):
        print(f"  {i+1}. {r['name']:28s} hin={r['hin']:.0f} eng={r['eng']:.0f} "
              f"pri={r['hin_pri']:.0f}")

    winner = rows[0]
    print(f"\n>> WINNER (Hindi-priority): {winner['name']}")
    print(f"   CV eng={winner['eng']:.0f}ms hin={winner['hin']:.0f}ms")

    # Refit winner on all data and save
    m = fit_one(winner["est"], winner["uses_w"], winner["needs_cal"], X, y, d, g)
    out = "eot_model.pkl"
    joblib.dump({
        "pipeline": m,
        "feature_names": FEATURE_NAMES,
        "winner": winner["name"],
        "cv_eng_ms": winner["eng"],
        "cv_hin_ms": winner["hin"],
        "cv_auc": winner["auc"],
        "trained_on": "english+hindi (all provided pauses)",
        "selection": "hindi-priority (eng+2*hin)/3",
        "bakeoff": [{k: v for k, v in r.items() if k != "est"} for r in rows],
    }, out)
    print(f"saved -> {out}")

    # Also dump bakeoff CSV for docs
    pd.DataFrame([{k: v for k, v in r.items() if k != "est"} for r in rows]).to_csv(
        "bakeoff_results.csv", index=False)
    print("wrote bakeoff_results.csv")


if __name__ == "__main__":
    main()
