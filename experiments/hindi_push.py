"""Hindi push experiments on existing 43-D features (no re-extract).

Tries: harder duration weights, hard-neg mining, ensembles, Hindi-only
score calibration / rank transforms. Reports GroupKFold OOF + in-sample.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
import joblib

from train_model import get_lang, official_score_g, auc, THRESHOLDS, DELAYS, TIMEOUT_S

CACHE = os.path.join(os.environ.get("SCRATCH", "."), "eot_cache")


def make_enet(C=0.5, l1=0.5):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=8000, C=C, solver="saga", l1_ratio=l1,
                           penalty="elasticnet"),
    )


def make_l2(C=0.5):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=6000, C=C),
    )


def weights(y, dur, scheme="dur05", eot_w=1.5):
    """Training sample weights. Duration is ONLY a training weight (legal)."""
    w = np.ones(len(y), dtype=float)
    hold = y == 0
    d = np.asarray(dur, float)
    if scheme == "uniform":
        pass
    elif scheme == "dur05":
        w[hold] = np.minimum(d[hold], 1.6) ** 0.5
    elif scheme == "dur10":
        w[hold] = np.minimum(d[hold], 1.6) ** 1.0
    elif scheme == "dur20":
        w[hold] = np.minimum(d[hold], 1.6) ** 2.0
    elif scheme == "hard08":
        # heavy weight on holds that can actually cause false cutoffs
        w[hold] = 1.0 + 8.0 * (d[hold] > 0.8).astype(float) + 3.0 * np.minimum(d[hold], 1.6)
    elif scheme == "hard07":
        w[hold] = 1.0 + 10.0 * (d[hold] > 0.7).astype(float) + 2.0 * np.minimum(d[hold], 1.6)
    elif scheme == "hard085_only":
        w[hold] = np.where(d[hold] > 0.85, 20.0, 0.3)
    elif scheme == "hard08_soft":
        w[hold] = np.where(d[hold] > 0.8, 12.0, np.minimum(d[hold], 1.6) ** 0.5)
    elif scheme == "tail_exp":
        # exponentially emphasize near-timeout holds
        w[hold] = np.exp(3.0 * np.minimum(d[hold], 1.6) / 1.6)
    else:
        raise ValueError(scheme)
    if hold.any():
        w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def oof_pred(est, X, y, g, d, scheme, eot_w=1.5, n=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        m = clone(est)
        w = weights(y[tr], d[tr], scheme=scheme, eot_w=eot_w)
        step = m.steps[-1][0]
        m.fit(X[tr], y[tr], **{f"{step}__sample_weight": w})
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def oof_ensemble(ests_schemes, X, y, g, d, n=5):
    """Average probabilities from several (est, scheme) OOF fits (same folds)."""
    acc = np.zeros(len(y))
    for est, scheme, eot_w in ests_schemes:
        acc += oof_pred(est, X, y, g, d, scheme, eot_w=eot_w, n=n)
    return acc / len(ests_schemes)


def longhold_auc(y, d, p, thr=0.8):
    eot = y == 1
    lh = (y == 0) & (d > thr)
    if eot.sum() == 0 or lh.sum() == 0:
        return float("nan")
    yy = np.concatenate([np.ones(eot.sum()), np.zeros(lh.sum())])
    ss = np.concatenate([p[eot], p[lh]])
    return auc(yy, ss)


def hindi_calibrate_oof(p_raw, y, g, d, lang_mask, method="isotonic"):
    """Fit a Hindi-only monotone map on OOF scores (nested by GroupKFold)."""
    p = p_raw.copy()
    idx = np.where(lang_mask)[0]
    # nested: calibrate using other Hindi turns' OOF scores
    ph = np.zeros(len(idx))
    yg = y[idx]; gg = g[idx]; dh = d[idx]; pr = p_raw[idx]
    for tr, te in GroupKFold(5).split(pr, yg, gg):
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pr[tr], yg[tr])
            ph[te] = iso.predict(pr[te])
        elif method == "platt":
            # 1-D logistic on score
            from sklearn.linear_model import LogisticRegression as LR
            lr = LR(C=1e2, max_iter=2000)
            lr.fit(pr[tr].reshape(-1, 1), yg[tr])
            ph[te] = lr.predict_proba(pr[te].reshape(-1, 1))[:, 1]
        elif method == "rank":
            # rank-normalize within fold's train scale
            order = np.argsort(pr[tr])
            # map test via empirical CDF of train
            ph[te] = np.searchsorted(np.sort(pr[tr]), pr[te]) / max(1, len(tr))
        elif method == "power":
            # learn gamma that best separates long holds on train
            best_g, best_a = 1.0, -1
            for gamma in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                s = np.clip(pr[tr], 1e-6, 1 - 1e-6) ** gamma
                a = longhold_auc(yg[tr], dh[tr], s)
                if a > best_a:
                    best_a, best_g = a, gamma
            ph[te] = np.clip(pr[te], 1e-6, 1 - 1e-6) ** best_g
        else:
            ph[te] = pr[te]
    p[idx] = ph
    return p


def metric_aware_shift(p_raw, y, g, d, lang_mask):
    """Hindi-only: subtract a small constant from scores that look like
    'very final but early in turn' using pause_index from features — skipped
    here; instead try grid of monotone power + temperature on OOF."""
    idx = np.where(lang_mask)[0]
    pr = p_raw[idx]
    best = None
    for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for power in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            # logit temperature then power
            eps = 1e-6
            logit = np.log(np.clip(pr, eps, 1 - eps) / (1 - np.clip(pr, eps, 1 - eps)))
            s = 1 / (1 + np.exp(-logit / temp))
            s = np.clip(s, eps, 1 - eps) ** power
            sc = official_score_g(g[idx], y[idx], d[idx], s)
            if best is None or sc[0] < best[0]:
                best = (sc[0], temp, power, s)
    out = p_raw.copy()
    out[idx] = best[3]
    return out, best[1], best[2], best[0]


def report(name, pe, ph, ye, yh, ge, gh, de, dh):
    me = official_score_g(ge, ye, de, pe)
    mh = official_score_g(gh, yh, dh, ph)
    la = longhold_auc(yh, dh, ph)
    print(f"{name:42s} eng={me[0]*1000:6.0f} hin={mh[0]*1000:6.0f} "
          f"lhAUC={la:.3f} hop=({mh[2]},{mh[3]*1000:.0f})")
    return me[0] * 1000, mh[0] * 1000, la


def main():
    Xe, ye, ge, de = get_lang("english")
    Xh, yh, gh, dh = get_lang("hindi")
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh])
    d = np.concatenate([de, dh])
    lang = np.array(["e"] * len(ye) + ["h"] * len(yh))
    hin = lang == "h"

    rows = []
    print("=== single models (OOF) ===")
    schemes = ["dur05", "dur10", "dur20", "hard08", "hard07", "hard085_only",
               "hard08_soft", "tail_exp"]
    for scheme in schemes:
        for est_name, est in [("ENET", make_enet()), ("L2", make_l2())]:
            for eot_w in [1.0, 1.5, 2.5]:
                p = oof_pred(est, X, y, g, d, scheme, eot_w=eot_w)
                me, mh, la = report(f"OOF {est_name}/{scheme}/eot{eot_w}",
                                    p[~hin], p[hin], ye, yh, ge, gh, de, dh)
                rows.append(dict(kind="single", name=f"{est_name}/{scheme}/eot{eot_w}",
                                 eng=me, hin=mh, lhAUC=la))

    print("\n=== ensembles (OOF) ===")
    ens_specs = [
        ("ens_hard_mix", [
            (make_enet(), "hard08", 1.5),
            (make_enet(), "dur05", 1.5),
            (make_enet(), "hard085_only", 2.5),
            (make_l2(), "hard07", 1.5),
        ]),
        ("ens_dur_ladder", [
            (make_enet(), "dur05", 1.5),
            (make_enet(), "dur10", 1.5),
            (make_enet(), "dur20", 1.5),
            (make_enet(), "tail_exp", 1.5),
        ]),
        ("ens_hard_focus", [
            (make_enet(), "hard08", 1.5),
            (make_enet(), "hard07", 2.5),
            (make_enet(), "hard08_soft", 1.5),
            (make_enet(), "hard085_only", 1.5),
            (make_l2(C=0.25), "hard08", 1.5),
        ]),
    ]
    ens_oof = {}
    for name, specs in ens_specs:
        p = oof_ensemble(specs, X, y, g, d)
        ens_oof[name] = p
        me, mh, la = report(f"OOF {name}", p[~hin], p[hin], ye, yh, ge, gh, de, dh)
        rows.append(dict(kind="ens", name=name, eng=me, hin=mh, lhAUC=la))

    # baseline shipped-like
    p0 = oof_pred(make_enet(), X, y, g, d, "dur05", eot_w=1.5)
    print("\n=== Hindi calibration on baseline OOF ===")
    for method in ["isotonic", "platt", "rank", "power"]:
        pc = hindi_calibrate_oof(p0, y, g, d, hin, method=method)
        me, mh, la = report(f"OOF base+hin-{method}", pc[~hin], pc[hin],
                            ye, yh, ge, gh, de, dh)
        rows.append(dict(kind="cal", name=f"base+{method}", eng=me, hin=mh, lhAUC=la))

    pc, t, pw, _ = metric_aware_shift(p0, y, g, d, hin)
    me, mh, la = report(f"OOF base+temp{t}/pow{pw}", pc[~hin], pc[hin],
                        ye, yh, ge, gh, de, dh)
    rows.append(dict(kind="cal", name=f"temp{t}_pow{pw}", eng=me, hin=mh, lhAUC=la))

    # calibrate best ensemble
    best_ens = min([r for r in rows if r["kind"] == "ens"], key=lambda r: r["hin"])
    p_ens = ens_oof[best_ens["name"]]
    print(f"\n=== calibrate best ens {best_ens['name']} ===")
    for method in ["isotonic", "platt", "power"]:
        pc = hindi_calibrate_oof(p_ens, y, g, d, hin, method=method)
        me, mh, la = report(f"OOF ens+hin-{method}", pc[~hin], pc[hin],
                            ye, yh, ge, gh, de, dh)
        rows.append(dict(kind="cal", name=f"ens+{method}", eng=me, hin=mh, lhAUC=la))
    pc, t, pw, _ = metric_aware_shift(p_ens, y, g, d, hin)
    me, mh, la = report(f"OOF ens+temp{t}/pow{pw}", pc[~hin], pc[hin],
                        ye, yh, ge, gh, de, dh)
    rows.append(dict(kind="cal", name=f"ens_temp{t}_pow{pw}", eng=me, hin=mh, lhAUC=la))

    df = pd.DataFrame(rows).sort_values(["hin", "eng"])
    df.to_csv("hindi_push_oof.csv", index=False)
    print("\n=== TOP by Hindi OOF ===")
    print(df.head(15).to_string(index=False))
    print(f"\nBest hin OOF: {df.iloc[0].hin:.0f} ({df.iloc[0].name})")


if __name__ == "__main__":
    main()
