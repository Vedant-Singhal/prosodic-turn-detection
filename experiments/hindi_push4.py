"""Stack second-stage LR + bagged adjusters; inspect sep16 operating points."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from eot_features import FEATURE_NAMES
from train_model import official_score_g, THRESHOLDS, DELAYS, TIMEOUT_S
from hindi_push2 import get_lang, make_enet, weights, longhold_auc, fit_pipe
from hindi_push3 import adjust, IDX

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def base_oof(X, y, g, d, cols, scheme="hard08_soft", eot_w=1.5, n=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        w = weights(y[tr], d[tr], scheme=scheme, eot_w=eot_w)
        m = fit_pipe(make_enet(), X[tr][:, cols], y[tr], w)
        p[te] = m.predict_proba(X[te][:, cols])[:, 1]
    return p


def stage2_features(p, X):
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    fade = X[:, IDX["e_fade_200"]]
    ix = X[:, IDX["pause_index"]]
    et = X[:, IDX["energy_trend_turn"]]
    es = X[:, IDX["e_slope_300"]]
    ac = X[:, IDX["ac_e_surprise"]]
    return np.column_stack([logit, fade, ix, et, es, ac])


def oof_stage2(p, X, y, g, d, n=5, C=0.5, hard_w=True):
    """Second-stage LR on [logit(p), causal cues], GroupKFold OOF."""
    Z = stage2_features(p, X)
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(Z, y, g):
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Ztr = sc.fit_transform(imp.fit_transform(Z[tr]))
        w = np.ones(len(tr))
        if hard_w:
            hold = y[tr] == 0
            w[hold] = 1.0 + 10.0 * (d[tr][hold] > 0.8)
            w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
            w[y[tr] == 1] = 1.5
        clf = LogisticRegression(max_iter=5000, C=C)
        clf.fit(Ztr, y[tr], sample_weight=w)
        Zte = sc.transform(imp.transform(Z[te]))
        out[te] = clf.predict_proba(Zte)[:, 1]
    return out


def bagged_adjust(p, X, y, g, d, n=5):
    """Average adjust() over a fixed menu of param tuples (no metric pick)."""
    menu = [
        (0.5, 0.5, 0.0, 0.25),
        (0.5, 0.75, 0.25, 0.0),
        (0.75, 1.0, 0.5, 0.0),
        (1.0, 0.75, 0.5, 0.0),
        (0.25, 1.0, 0.0, 0.0),
        (1.0, 0.5, 1.0, 0.0),
    ]
    # Still need honesty: choose menu subset on train by averaging top-k
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        scored = []
        for params in menu:
            s = adjust(p[tr], X[tr], *params)
            sc = official_score_g(g[tr], y[tr], d[tr], s)[0]
            scored.append((sc, params))
        scored.sort()
        top = [params for _, params in scored[:3]]
        preds = [adjust(p[te], X[te], *params) for params in top]
        out[te] = np.mean(preds, axis=0)
    return out


def dump_ops(name, g, y, d, p, k=8):
    """Show best operating points even above 850."""
    n_turns = len(set(g.tolist()))
    is_eot = y == 1
    rows = []
    for t in THRESHOLDS:
        fires = p >= t
        for delay in DELAYS:
            cut = set(g[(~is_eot) & fires & (delay < d)].tolist())
            cr = len(cut) / max(1, n_turns)
            if cr > 0.05:
                continue
            lat = np.where(fires[is_eot], delay, TIMEOUT_S)
            rows.append((float(np.mean(lat)) * 1000, cr, t, delay * 1000))
    rows.sort()
    print(f"\n{name} top feasible ops:")
    for r in rows[:k]:
        print(f"  {r[0]:6.0f} ms  cut={r[1]*100:.1f}%  th={r[2]} d={r[3]:.0f}")
    return rows[0] if rows else None


def main():
    Xe, ye, ge, de = get_lang("english")
    Xh, yh, gh, dh = get_lang("hindi")
    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh])
    d = np.concatenate([de, dh])
    hin = np.array([False] * len(ye) + [True] * len(yh))

    names_sep = ["e_fade_200", "e_slope_300", "energy_trend_turn", "tilt_slope_final",
                 "pause_index", "pause_start", "voiced_time_sofar", "n_voiced_runs",
                 "ac_e_surprise", "f0_min_final_st", "e_std_final", "speech_rate_final",
                 "prior_pause_mean_dur", "creak_frac", "f0_end_vs_p10", "e_drop_final"]
    cols_sep = [IDX[n] for n in names_sep]
    cols43 = list(range(43))
    cols_mfcc = list(range(51))

    rows = []
    for tag, cols, scheme in [
        ("sep16/hard", cols_sep, "hard08_soft"),
        ("43/dur05", cols43, "dur05"),
        ("51/dur05", cols_mfcc, "dur05"),
        ("51/hard", cols_mfcc, "hard08_soft"),
    ]:
        p = base_oof(X, y, g, d, cols, scheme=scheme)
        pe, ph = p[~hin], p[hin]
        me = official_score_g(ge, ye, de, pe)[0] * 1000
        mh = official_score_g(gh, yh, dh, ph)[0] * 1000
        print(f"BASE {tag}: eng={me:.0f} hin={mh:.0f} lhAUC={longhold_auc(yh,dh,ph):.3f}")
        dump_ops(f"BASE {tag} Hindi", gh, yh, dh, ph)

        # stage2 on Hindi only (Eng unchanged)
        for C in [0.25, 0.5, 1.0, 2.0]:
            ph2 = oof_stage2(ph, Xh, yh, gh, dh, C=C, hard_w=True)
            mh2 = official_score_g(gh, yh, dh, ph2)[0] * 1000
            la = longhold_auc(yh, dh, ph2)
            print(f"  stage2 C={C}: hin={mh2:.0f} lhAUC={la:.3f}")
            rows.append(dict(base=tag, method=f"stage2_C{C}", eng=me, hin=mh2, lhAUC=la))
            if mh2 < 849.9:
                dump_ops(f"stage2 {tag} C={C}", gh, yh, dh, ph2)

        phb = bagged_adjust(ph, Xh, yh, gh, dh)
        mhb = official_score_g(gh, yh, dh, phb)[0] * 1000
        print(f"  baggedAdj: hin={mhb:.0f} lhAUC={longhold_auc(yh,dh,phb):.3f}")
        rows.append(dict(base=tag, method="baggedAdj", eng=me, hin=mhb,
                         lhAUC=longhold_auc(yh, dh, phb)))
        if mhb < 849.9:
            dump_ops(f"bagged {tag}", gh, yh, dh, phb)

        # stage2 on joint then take Hindi
        p2 = oof_stage2(p, X, y, g, d, C=0.5, hard_w=True)
        mh2 = official_score_g(gh, yh, dh, p2[hin])[0] * 1000
        me2 = official_score_g(ge, ye, de, p2[~hin])[0] * 1000
        print(f"  jointStage2: eng={me2:.0f} hin={mh2:.0f}")
        rows.append(dict(base=tag, method="jointStage2", eng=me2, hin=mh2,
                         lhAUC=longhold_auc(yh, dh, p2[hin])))

    df = pd.DataFrame(rows).sort_values(["hin", "eng"])
    df.to_csv("hindi_push4_oof.csv", index=False)
    print("\nTOP:")
    print(df.head(20).to_string(index=False))
    print("beat", (df.hin < 849.9).sum(), "best", df.hin.min())


if __name__ == "__main__":
    main()
