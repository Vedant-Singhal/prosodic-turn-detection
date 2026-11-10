"""Hindi-priority follow-ups: acoustic-completion features + mid-complexity models.

Original implementations from first principles (AR residuals as completion proxy,
Hindi-upweighted training, Elastic-Net with duration weights).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import joblib

from eot_features import (
    compute_frame_features, pause_features, FEATURE_NAMES, frames_before,
)
from eot_data import load_wav

HANDOUT = "eot_handout/eot_data"
TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)
AC_NAMES = ["ac_e_rmse", "ac_f0_rmse", "ac_e_surprise", "ac_contour_curl"]


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


def ar_rmse(y, n_eval=40):
    y = np.asarray(y, dtype=float)
    if len(y) < n_eval + 20:
        return np.nan
    Y = y[2:]
    X = np.column_stack([y[1:-1], y[:-2], np.ones(len(y) - 2)])
    te0 = len(Y) - n_eval
    coef, _, _, _ = np.linalg.lstsq(X[:te0], Y[:te0], rcond=None)
    resid = Y[te0:] - X[te0:] @ coef
    return float(np.sqrt(np.mean(resid ** 2)))


def acoustic_completion(fr, pause_start):
    out = np.full(4, np.nan, dtype=np.float32)
    k = frames_before(fr, pause_start)
    if k < 50:
        return out
    edb = fr["edb"][:k]
    f0 = fr["f0"][:k]
    voiced = fr["voiced"][:k]
    t = fr["t_end"][:k]
    vi = np.where(voiced)[0]
    last = int(vi[-1]) if len(vi) else k - 1
    e = edb[: last + 1]
    out[0] = ar_rmse(e, n_eval=min(40, max(10, (last + 1) // 5)))
    f0s = f0.copy()
    if voiced.sum() >= 10:
        idx = np.arange(k)
        f0s[~voiced] = np.interp(idx[~voiced], idx[voiced], f0[voiced])
        out[1] = ar_rmse(f0s[: last + 1], n_eval=min(40, max(10, (last + 1) // 5)))
    if last > 30:
        y = edb[: last + 1]
        Y = y[2:-5]
        X = np.column_stack([y[1:-6], y[:-7], np.ones(len(Y))])
        if len(Y) >= 10:
            coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            last5 = y[-5:]
            pred = np.array([np.dot(coef, [y[-6 + i], y[-7 + i], 1.0]) for i in range(5)])
            out[2] = float(np.mean(np.abs(last5 - pred)))
    m = t >= (t[last] - 0.40)
    ee = edb[m]
    if len(ee) >= 6:
        out[3] = float(np.mean(np.diff(ee, n=2)))
    return out


def build_extended(lang):
    cache = os.path.join("eot_cache", f"{lang}_ext.npz")
    os.makedirs("eot_cache", exist_ok=True)
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["groups"], d["durs"]
    data_dir = os.path.join(HANDOUT, lang)
    lab = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    Xs, ys, gs, ds = [], [], [], []
    for tid, g in lab.groupby("turn_id", sort=False):
        g = g.sort_values("pause_index")
        x, sr = load_wav(os.path.join(data_dir, g.iloc[0]["audio_file"]))
        fr = compute_frame_features(x, sr)
        prior, prev_end = [], None
        for _, r in g.iterrows():
            ps = float(r["pause_start"])
            base = pause_features(fr, ps, int(r["pause_index"]), tuple(prior), prev_end)
            ac = acoustic_completion(fr, ps)
            Xs.append(np.concatenate([base, ac]))
            ys.append(1 if r["label"] == "eot" else 0)
            gs.append(tid)
            ds.append(float(r["pause_end"]) - ps)
            prior.append(float(r["pause_end"]) - ps)
            prev_end = float(r["pause_end"])
    X = np.asarray(Xs, np.float32)
    y = np.asarray(ys, np.int64)
    groups = np.array(gs)
    durs = np.asarray(ds)
    np.savez(cache, X=X, y=y, groups=groups, durs=durs)
    return X, y, groups, durs


def make_pipe(clf):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", clf),
    ])


def dur_weights(y, dur, gamma=0.5, eot_w=1.5):
    w = np.ones(len(y), dtype=float)
    hold = y == 0
    w[hold] = np.minimum(dur[hold], 1.6) ** gamma
    w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def oof(est, X, y, g, d, sw_extra=None, uses_dur_w=False, n=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        m = clone(est)
        w = np.ones(len(tr))
        if uses_dur_w:
            w = dur_weights(y[tr], d[tr])
        if sw_extra is not None:
            w = w * sw_extra[tr]
        step = m.steps[-1][0]
        m.fit(X[tr], y[tr], **{f"{step}__sample_weight": w})
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main():
    Xe, ye, ge, de = build_extended("english")
    Xh, yh, gh, dh = build_extended("hindi")
    print(f"ext eng={Xe.shape} hin={Xh.shape}")

    rows = []
    for use_ac, tag in [(False, "39f"), (True, "43f")]:
        X = np.vstack([Xe if use_ac else Xe[:, :39], Xh if use_ac else Xh[:, :39]])
        y = np.concatenate([ye, yh])
        g = np.concatenate([ge, gh])
        d = np.concatenate([de, dh])
        lang = np.array(["e"] * len(ye) + ["h"] * len(yh))
        w_hin = np.where(lang == "h", 2.5, 1.0).astype(float)

        cands = [
            ("ENET C=0.5 bal", make_pipe(LogisticRegression(
                max_iter=5000, C=0.5, penalty="elasticnet", solver="saga",
                l1_ratio=0.5, class_weight="balanced")), False, None),
            ("ENET C=0.5 dur-w", make_pipe(LogisticRegression(
                max_iter=5000, C=0.5, penalty="elasticnet", solver="saga",
                l1_ratio=0.5)), True, None),
            ("ENET C=0.5 hin×2.5", make_pipe(LogisticRegression(
                max_iter=5000, C=0.5, penalty="elasticnet", solver="saga",
                l1_ratio=0.5)), False, w_hin),
            ("ENET C=0.5 dur+hin", make_pipe(LogisticRegression(
                max_iter=5000, C=0.5, penalty="elasticnet", solver="saga",
                l1_ratio=0.5)), True, w_hin),
            ("LR C=0.5 dur-w", make_pipe(LogisticRegression(
                max_iter=4000, C=0.5)), True, None),
            ("LR C=0.5 dur+hin", make_pipe(LogisticRegression(
                max_iter=4000, C=0.5)), True, w_hin),
        ]
        for name, est, udw, sw in cands:
            p = oof(est, X, y, g, d, sw_extra=sw, uses_dur_w=udw)
            me = official_score_g(ge, ye, de, p[lang == "e"])[0] * 1000
            mh = official_score_g(gh, yh, dh, p[lang == "h"])[0] * 1000
            pri = (me + 2 * mh) / 3
            print(f"{tag:4s} {name:22s} eng={me:6.0f} hin={mh:6.0f} pri={pri:6.0f}")
            rows.append(dict(tag=tag, name=name, eng=me, hin=mh, pri=pri,
                             use_ac=use_ac, udw=udw, sw=sw is not None))

    rows.sort(key=lambda r: r["pri"])
    print("\nTOP:")
    for r in rows[:8]:
        print(f"  {r['tag']} {r['name']:22s} hin={r['hin']:.0f} eng={r['eng']:.0f} pri={r['pri']:.0f}")

    best = rows[0]
    use_ac = best["use_ac"]
    X = np.vstack([Xe if use_ac else Xe[:, :39], Xh if use_ac else Xh[:, :39]])
    y = np.concatenate([ye, yh])
    d = np.concatenate([de, dh])
    lang = np.array(["e"] * len(ye) + ["h"] * len(yh))
    w_hin = np.where(lang == "h", 2.5, 1.0).astype(float) if best["sw"] else None

    # Rebuild matching estimator
    if "ENET" in best["name"]:
        clf = LogisticRegression(max_iter=5000, C=0.5, penalty="elasticnet",
                                 solver="saga", l1_ratio=0.5,
                                 class_weight="balanced" if "bal" in best["name"] else None)
    else:
        clf = LogisticRegression(max_iter=4000, C=0.5)
    est = make_pipe(clf)
    w = np.ones(len(y))
    if best["udw"]:
        w = dur_weights(y, d)
    if w_hin is not None:
        w = w * w_hin
    step = est.steps[-1][0]
    est.fit(X, y, **{f"{step}__sample_weight": w})

    feat_names = list(FEATURE_NAMES) + (AC_NAMES if use_ac else [])
    bundle = {
        "pipeline": est,
        "feature_names": feat_names,
        "winner": f"{best['tag']}/{best['name']}",
        "cv_eng_ms": best["eng"],
        "cv_hin_ms": best["hin"],
        "use_acoustic_completion": use_ac,
        "trained_on": "english+hindi",
        "selection": "hindi-priority (eng+2*hin)/3",
    }
    joblib.dump(bundle, "eot_model.pkl")
    print(f"\nsaved eot_model.pkl :: {bundle['winner']} CV eng={best['eng']:.0f} hin={best['hin']:.0f}")
    pd.DataFrame(rows).to_csv("followup_results.csv", index=False)

    # If AC features used, predict.py must build them — write a flag file for train_model
    with open("model_config.txt", "w") as f:
        f.write(f"use_ac={int(use_ac)}\nwinner={bundle['winner']}\n")


if __name__ == "__main__":
    main()
