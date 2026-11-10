"""Train the EOT model and save it for predict.py.

Human-steered mid-complexity choice after Hindi diagnosis:
Elastic-Net logistic with hard-negative sample weights on holds whose
duration > 0.8 s (the only holds that can cause false cutoffs), plus
4 acoustic-completion features. Official score.py: Eng 1015 / Hin 831.

    python train_model.py
"""
import os, argparse, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from eot_data import build_dataset
from eot_features import FEATURE_NAMES

warnings.filterwarnings("ignore", category=FutureWarning)

HANDOUT = "eot_handout/eot_data"
CACHE = os.path.join(os.environ.get("SCRATCH", "."), "eot_cache")
TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def official_score_g(turn_ids, labels, durs, p, budget=0.05):
    turn_ids = np.asarray(turn_ids); labels = np.asarray(labels)
    durs = np.asarray(durs); p = np.asarray(p)
    n_turns = len(set(turn_ids.tolist())); is_eot = labels == 1; best = None
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
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def get_lang(lang):
    os.makedirs(CACHE, exist_ok=True)
    # bust cache when feature dim changes
    fp = os.path.join(CACHE, f"{lang}_f{len(FEATURE_NAMES)}.npz")
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


def new_enet(C=0.5, l1_ratio=0.5):
    # sklearn>=1.8: elastic-net via l1_ratio in (0,1) with saga
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=6000, C=C, solver="saga", l1_ratio=l1_ratio,
                           penalty="elasticnet"),
    )


def dur_weights(y, dur, gamma=0.5, eot_w=1.5):
    w = np.ones(len(y), dtype=float)
    hold = y == 0
    w[hold] = np.minimum(dur[hold], 1.6) ** gamma
    w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def hardneg_weights(y, dur, eot_w=1.5):
    """Heavy weight on holds with dur>0.8s (the only ones that can false-cut)."""
    w = np.ones(len(y), dtype=float)
    hold = y == 0
    d = np.asarray(dur, float)
    w[hold] = np.where(d[hold] > 0.8, 12.0, np.minimum(d[hold], 1.6) ** 0.5)
    w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def oof(X, y, g, d, n=5, weight_fn=None):
    if weight_fn is None:
        weight_fn = hardneg_weights
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        m = new_enet()
        w = weight_fn(y[tr], d[tr])
        m.fit(X[tr], y[tr], logisticregression__sample_weight=w)
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eot_model.pkl")
    args = ap.parse_args()

    # Train on the original 43 acoustic+completion features (hard-neg weights).
    # Extra MFCC/history dims may exist in FEATURE_NAMES; we slice to 43 for the
    # shipped model that beat Hindi silence on official score.py.
    N_USE = 43
    Xe, ye, ge, de = get_lang("english")
    Xh, yh, gh, dh = get_lang("hindi")
    Xe, Xh = Xe[:, :N_USE], Xh[:, :N_USE]
    X = np.vstack([Xe, Xh]); y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh]); d = np.concatenate([de, dh])
    lang = np.array(["e"] * len(ye) + ["h"] * len(yh))

    print(f"data: X={X.shape} eot={int(y.sum())}  features={N_USE} (hard-neg holds>0.8s)")
    p = oof(X, y, g, d, weight_fn=hardneg_weights)
    a = auc(y, p)
    me = official_score_g(ge, ye, de, p[lang == "e"])[0] * 1000
    mh = official_score_g(gh, yh, dh, p[lang == "h"])[0] * 1000
    print(f"CV ElasticNet+AC+hardneg: eng={me:.0f}ms hin={mh:.0f}ms AUC={a:.3f}")

    final = new_enet()
    final.fit(X, y, logisticregression__sample_weight=hardneg_weights(y, d))
    joblib.dump({
        "pipeline": final,
        "feature_names": FEATURE_NAMES[:N_USE],
        "winner": "ElasticNet C=0.5 l1=0.5 + AC + hard-neg (holds dur>0.8s ×12)",
        "cv_eng_ms": me,
        "cv_hin_ms": mh,
        "cv_auc": a,
        "n_features_used": N_USE,
        "trained_on": "english+hindi (all provided pauses)",
        "selection": "hard-negative duration weights; Hindi silence beat on score.py",
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
