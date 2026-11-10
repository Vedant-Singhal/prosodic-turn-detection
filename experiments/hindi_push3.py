"""Metric-aware Hindi score adjuster + surgical feature subsets.

Uses nested GroupKFold: outer OOF base scores, inner grid over causal
adjustments that demote mid-turn 'fake finals' without using pause duration.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from eot_features import FEATURE_NAMES
from train_model import official_score_g, auc
from hindi_push2 import get_lang, make_enet, weights, longhold_auc, fit_pipe

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def base_oof(X, y, g, d, cols, scheme="dur05", eot_w=1.5, n=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        est = make_enet()
        w = weights(y[tr], d[tr], scheme=scheme, eot_w=eot_w)
        m = fit_pipe(est, X[tr][:, cols], y[tr], w)
        p[te] = m.predict_proba(X[te][:, cols])[:, 1]
    return p


def adjust(p, X, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0):
    """p' = sigmoid(logit(p) + a*z_fade + b*(-z_ix) + c*z_etrend + d*z_rel).

    Positive e_fade (less negative) and energy_trend favor EOT; high pause_index
    favors demotion (thinking holds). All inputs are causal features.
    """
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    fade = X[:, IDX["e_fade_200"]]
    ix = X[:, IDX["pause_index"]]
    et = X[:, IDX["energy_trend_turn"]]
    # prefer milder fades for EOT (fade is negative slope; higher=better for EOT)
    fade_z = (fade - np.nanmean(fade)) / (np.nanstd(fade) + 1e-9)
    ix_z = (ix - np.nanmean(ix)) / (np.nanstd(ix) + 1e-9)
    et_z = (et - np.nanmean(et)) / (np.nanstd(et) + 1e-9)
    if "rel_e_drop" in IDX:
        rel = X[:, IDX["rel_e_drop"]]
        rel_z = (rel - np.nanmean(rel)) / (np.nanstd(rel) + 1e-9)
    else:
        rel_z = 0.0
    adj = logit + alpha * fade_z - beta * ix_z + gamma * et_z - delta * np.abs(rel_z)
    return 1 / (1 + np.exp(-adj))


def grid_adjust(p, X, y, g, d, budget_grid=None):
    """Pick (a,b,c,d) minimizing official Hindi score on this fold's data."""
    best = None
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    betas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    gammas = [0.0, 0.25, 0.5, 1.0]
    deltas = [0.0, 0.25, 0.5]
    for a in alphas:
        for b in betas:
            for c in gammas:
                for dlt in deltas:
                    if a == b == c == dlt == 0:
                        s = p
                    else:
                        s = adjust(p, X, a, b, c, dlt)
                    sc = official_score_g(g, y, d, s)
                    key = (sc[0], -longhold_auc(y, d, s))
                    if best is None or key < best[0]:
                        best = (key, a, b, c, dlt, s, sc)
    return best


def nested_hindi_adjust(p_all, Xh, yh, gh, dh, n=5):
    """Nested CV: choose adjuster params on train Hindi turns, apply to test."""
    p_out = np.zeros(len(yh))
    params = []
    for tr, te in GroupKFold(n).split(Xh, yh, gh):
        # optimize on train portion of OOF scores
        best = grid_adjust(p_all[tr], Xh[tr], yh[tr], gh[tr], dh[tr])
        _, a, b, c, dlt, _, sc = best
        params.append((a, b, c, dlt, sc[0] * 1000))
        p_out[te] = adjust(p_all[te], Xh[te], a, b, c, dlt)
    return p_out, params


def report(name, pe, ph, ye, yh, ge, gh, de, dh):
    me = official_score_g(ge, ye, de, pe)
    mh = official_score_g(gh, yh, dh, ph)
    la = longhold_auc(yh, dh, ph)
    print(f"{name:48s} eng={me[0]*1000:6.0f} hin={mh[0]*1000:6.0f} "
          f"lhAUC={la:.3f} hop=({mh[2]},{mh[3]*1000:.0f})")
    return dict(name=name, eng=me[0]*1000, hin=mh[0]*1000, lhAUC=la)


def main():
    Xe, ye, ge, de = get_lang("english")
    Xh, yh, gh, dh = get_lang("hindi")
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh])
    d = np.concatenate([de, dh])
    hin = np.array([False] * len(ye) + [True] * len(yh))

    cols43 = list(range(43))
    cols_mfcc = list(range(51))  # 43 + 8 mfcc
    # hand-picked separators for long-hold problem
    names_sep = ["e_fade_200", "e_slope_300", "energy_trend_turn", "tilt_slope_final",
                 "pause_index", "pause_start", "voiced_time_sofar", "n_voiced_runs",
                 "ac_e_surprise", "f0_min_final_st", "e_std_final", "speech_rate_final",
                 "prior_pause_mean_dur", "creak_frac", "f0_end_vs_p10", "e_drop_final"]
    cols_sep = [IDX[n] for n in names_sep if n in IDX]

    rows = []
    configs = [
        ("43f dur05", cols43, "dur05"),
        ("43+mfcc dur05", cols_mfcc, "dur05"),
        ("sep16 dur05", cols_sep, "dur05"),
        ("sep16 hard08", cols_sep, "hard08_soft"),
        ("43f hard08", cols43, "hard08_soft"),
    ]

    print("=== base OOF ===")
    store = {}
    for name, cols, scheme in configs:
        p = base_oof(X, y, g, d, cols, scheme=scheme)
        store[name] = (p, cols)
        rows.append(report(name, p[~hin], p[hin], ye, yh, ge, gh, de, dh))

    print("\n=== nested Hindi adjust (honest) ===")
    for name, (p, cols) in store.items():
        ph_adj, params = nested_hindi_adjust(p[hin], Xh, yh, gh, dh)
        print(f"  params for {name}: {params}")
        rows.append(report(f"{name}+nestedAdj", p[~hin], ph_adj, ye, yh, ge, gh, de, dh))

    print("\n=== oracle adjust on full Hindi (UPPER BOUND, not honest) ===")
    for name, (p, cols) in store.items():
        best = grid_adjust(p[hin], Xh, yh, gh, dh)
        _, a, b, c, dlt, s, sc = best
        print(f"  {name}: a={a} b={b} c={c} d={dlt} -> {sc[0]*1000:.0f}ms")
        rows.append(report(f"{name}+oracleAdj", p[~hin], s, ye, yh, ge, gh, de, dh))

    # multiplicative demotion by pause_index (simple rule)
    print("\n=== simple pause_index demotion grid (nested) ===")
    p0 = store["43+mfcc dur05"][0]
    ph = np.zeros(len(yh))
    for tr, te in GroupKFold(5).split(Xh, yh, gh):
        best = None
        for lam in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            for thr in [0, 1, 2, 3]:
                s = p0[hin][tr].copy()
                ix = Xh[tr, IDX["pause_index"]]
                s = s * (1.0 / (1.0 + lam * np.maximum(ix - thr, 0)))
                sc = official_score_g(gh[tr], yh[tr], dh[tr], s)
                if best is None or sc[0] < best[0]:
                    best = (sc[0], lam, thr)
        lam, thr = best[1], best[2]
        s = p0[hin][te].copy()
        ix = Xh[te, IDX["pause_index"]]
        ph[te] = s * (1.0 / (1.0 + lam * np.maximum(ix - thr, 0)))
        print(f"  fold lam={lam} thr={thr} train={best[0]*1000:.0f}")
    rows.append(report("43+mfcc+ixDemote nested", p0[~hin], ph, ye, yh, ge, gh, de, dh))

    # Combine: promote high e_fade, demote high pause_index
    print("\n=== fade-promote + ix-demote nested ===")
    ph = np.zeros(len(yh))
    for tr, te in GroupKFold(5).split(Xh, yh, gh):
        best = None
        pr = p0[hin][tr]
        for lam_ix in [0.0, 0.1, 0.2, 0.3, 0.5]:
            for thr in [0, 1, 2]:
                for lam_f in [0.0, 0.1, 0.2, 0.4]:
                    fade = Xh[tr, IDX["e_fade_200"]]
                    fz = (fade - fade.mean()) / (fade.std() + 1e-9)
                    ix = Xh[tr, IDX["pause_index"]]
                    s = pr * (1.0 + lam_f * np.maximum(fz, 0))
                    s = s / (1.0 + lam_ix * np.maximum(ix - thr, 0))
                    s = np.clip(s, 0, 1)
                    sc = official_score_g(gh[tr], yh[tr], dh[tr], s)
                    if best is None or sc[0] < best[0]:
                        best = (sc[0], lam_ix, thr, lam_f)
        lam_ix, thr, lam_f = best[1], best[2], best[3]
        fade = Xh[te, IDX["e_fade_200"]]
        # use train stats for z — approximate with te local (slight leak ok? NO — use train)
        fade_tr = Xh[tr, IDX["e_fade_200"]]
        fz = (fade - fade_tr.mean()) / (fade_tr.std() + 1e-9)
        ix = Xh[te, IDX["pause_index"]]
        s = p0[hin][te] * (1.0 + lam_f * np.maximum(fz, 0))
        s = s / (1.0 + lam_ix * np.maximum(ix - thr, 0))
        ph[te] = np.clip(s, 0, 1)
        print(f"  fold ix={lam_ix}/{thr} fade={lam_f} train={best[0]*1000:.0f}")
    rows.append(report("43+mfcc+fadeIx nested", p0[~hin], ph, ye, yh, ge, gh, de, dh))

    df = pd.DataFrame(rows).sort_values(["hin", "eng"])
    df.to_csv("hindi_push3_oof.csv", index=False)
    print("\n=== TOP ===")
    print(df.to_string(index=False))
    print("beat 850:", (df.hin < 849.9).sum(), "best hin", df.hin.min())


if __name__ == "__main__":
    main()
