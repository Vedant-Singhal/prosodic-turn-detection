"""Round-2 Hindi push: new causal feats + pairwise ranking + lang-conditional.

Rebuilds feature cache (MFCC + within-turn history). Reports GroupKFold OOF.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import joblib

from eot_features import FEATURE_NAMES
from train_model import official_score_g, auc

HANDOUT = "eot_handout/eot_data"
CACHE = os.path.join(os.environ.get("SCRATCH", "."), "eot_cache")


def get_lang(lang):
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{lang}_f{len(FEATURE_NAMES)}.npz")
    if os.path.exists(fp):
        d = np.load(fp, allow_pickle=True)
        return d["X"], d["y"], d["groups"], d["durs"]
    from eot_data import build_dataset
    print(f"building features for {lang} ({len(FEATURE_NAMES)}-D)...")
    X, y, groups, keys, _ = build_dataset(os.path.join(HANDOUT, lang), verbose=False)
    lab = pd.read_csv(os.path.join(HANDOUT, lang, "labels.csv"))
    dm = {(r.turn_id, int(r.pause_index)): float(r.pause_end) - float(r.pause_start)
          for r in lab.itertuples()}
    durs = np.array([dm[k] for k in keys])
    np.savez(fp, X=X, y=y, groups=groups, durs=durs)
    return X, y, groups, durs


def make_enet(C=0.5, l1=0.5):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=10000, C=C, solver="saga", l1_ratio=l1,
                           penalty="elasticnet"),
    )


def make_l2(C=0.5):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=8000, C=C),
    )


def weights(y, dur, scheme="dur05", eot_w=1.5):
    w = np.ones(len(y), dtype=float)
    hold = y == 0
    d = np.asarray(dur, float)
    if scheme == "dur05":
        w[hold] = np.minimum(d[hold], 1.6) ** 0.5
    elif scheme == "hard08":
        w[hold] = 1.0 + 8.0 * (d[hold] > 0.8) + 3.0 * np.minimum(d[hold], 1.6)
    elif scheme == "hard08_soft":
        w[hold] = np.where(d[hold] > 0.8, 12.0, np.minimum(d[hold], 1.6) ** 0.5)
    elif scheme == "hard07":
        w[hold] = 1.0 + 10.0 * (d[hold] > 0.7) + 2.0 * np.minimum(d[hold], 1.6)
    elif scheme == "pairlike":
        # very heavy on cutoff-capable holds
        w[hold] = np.where(d[hold] >= 0.75, 25.0, 0.25 * np.minimum(d[hold], 1.6))
    else:
        raise ValueError(scheme)
    if hold.any():
        w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def longhold_auc(y, d, p, thr=0.8):
    eot = y == 1
    lh = (y == 0) & (d > thr)
    if not eot.any() or not lh.any():
        return float("nan")
    return auc(np.concatenate([np.ones(eot.sum()), np.zeros(lh.sum())]),
               np.concatenate([p[eot], p[lh]]))


def fit_pipe(est, X, y, w):
    m = clone(est)
    step = m.steps[-1][0]
    m.fit(X, y, **{f"{step}__sample_weight": w})
    return m


def oof_weighted(est, X, y, g, d, scheme, eot_w=1.5, n=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        w = weights(y[tr], d[tr], scheme=scheme, eot_w=eot_w)
        m = fit_pipe(est, X[tr], y[tr], w)
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def oof_pairwise(X, y, g, d, hold_thr=0.75, C=0.5, n=5, max_pairs=4000):
    """Pairwise logistic on (x_eot - x_hold) for holds with dur>=hold_thr.

    Score at inference = sigmoid(w·x + b) from the difference model fitted
    as LR on feature differences (classic RankNet-linear).
    """
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        Xt, yt, dt = X[tr], y[tr], d[tr]
        # impute+scale on train
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xt_s = sc.fit_transform(imp.fit_transform(Xt))
        eot_i = np.where(yt == 1)[0]
        hold_i = np.where((yt == 0) & (dt >= hold_thr))[0]
        if len(hold_i) == 0:
            hold_i = np.where(yt == 0)[0]
        # also include some short holds at low weight via fewer samples
        short_i = np.where((yt == 0) & (dt < hold_thr))[0]
        rng = np.random.RandomState(0)
        diffs, labels = [], []
        # all eot vs long-hold pairs (cap)
        for i in eot_i:
            for j in hold_i:
                diffs.append(Xt_s[i] - Xt_s[j])
                labels.append(1)
                diffs.append(Xt_s[j] - Xt_s[i])
                labels.append(0)
        # subsample short-hold pairs
        n_short = min(len(eot_i) * 3, len(short_i), 800)
        if n_short > 0 and len(short_i):
            sj = rng.choice(short_i, size=n_short, replace=False)
            ei = rng.choice(eot_i, size=n_short, replace=True)
            for i, j in zip(ei, sj):
                diffs.append(Xt_s[i] - Xt_s[j]); labels.append(1)
                diffs.append(Xt_s[j] - Xt_s[i]); labels.append(0)
        D = np.asarray(diffs, float)
        lab = np.asarray(labels, int)
        if len(D) > max_pairs:
            sel = rng.choice(len(D), size=max_pairs, replace=False)
            D, lab = D[sel], lab[sel]
        clf = LogisticRegression(max_iter=8000, C=C, solver="lbfgs")
        clf.fit(D, lab)
        # score: w·x  (bias absorbed; use decision_function on single vectors
        # via difference from zero vector)
        Xe_s = sc.transform(imp.transform(X[te]))
        # decision on x vs 0: w·x + b/2-ish — use predict_proba on (x - 0)
        # Better: score = X @ coef
        raw = Xe_s @ clf.coef_.ravel()
        # convert to (0,1) via logistic of centered raw
        raw = raw - np.median(raw)
        p[te] = 1 / (1 + np.exp(-raw))
    return p


def oof_blend(p_list, weights=None):
    W = np.ones(len(p_list)) if weights is None else np.asarray(weights, float)
    W = W / W.sum()
    return sum(w * p for w, p in zip(W, p_list))


def report(name, pe, ph, ye, yh, ge, gh, de, dh):
    me = official_score_g(ge, ye, de, pe)
    mh = official_score_g(gh, yh, dh, ph)
    la = longhold_auc(yh, dh, ph)
    print(f"{name:44s} eng={me[0]*1000:6.0f} hin={mh[0]*1000:6.0f} "
          f"lhAUC={la:.3f} hop=({mh[2]},{mh[3]*1000:.0f})")
    return dict(name=name, eng=me[0]*1000, hin=mh[0]*1000, lhAUC=la,
                th=mh[2], delay=mh[3])


def main():
    print(f"FEATURE_NAMES={len(FEATURE_NAMES)}")
    Xe, ye, ge, de = get_lang("english")
    Xh, yh, gh, dh = get_lang("hindi")
    print(f"eng={Xe.shape} hin={Xh.shape}")

    # sanity: new feat separation
    eot, lh = yh == 1, (yh == 0) & (dh > 0.8)
    for name in ["e_fade_200", "pause_index", "rel_e_drop", "ix_x_edrop",
                 "mfcc1_mean_500", "prior_pause_max_dur", "n_prior_pauses"]:
        i = FEATURE_NAMES.index(name)
        a = auc(np.concatenate([np.ones(eot.sum()), np.zeros(lh.sum())]),
                np.concatenate([Xh[eot, i], Xh[lh, i]]))
        print(f"  uni {name:22s} eot-vs-lh AUC={a:.3f}")

    X = np.vstack([Xe, Xh])
    y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh])
    d = np.concatenate([de, dh])
    lang = np.array(["e"] * len(ye) + ["h"] * len(yh))
    hin = lang == "h"
    rows = []

    print("\n=== joint models ===")
    for scheme in ["dur05", "hard08", "hard08_soft", "hard07", "pairlike"]:
        for est_name, est in [("ENET", make_enet()), ("L2", make_l2())]:
            p = oof_weighted(est, X, y, g, d, scheme, eot_w=1.5)
            rows.append(report(f"joint {est_name}/{scheme}",
                               p[~hin], p[hin], ye, yh, ge, gh, de, dh))

    print("\n=== Hindi-only models (Eng kept as joint dur05) ===")
    p_eng_base = oof_weighted(make_enet(), X, y, g, d, "dur05", eot_w=1.5)
    for scheme in ["dur05", "hard08", "hard08_soft", "pairlike"]:
        ph = oof_weighted(make_enet(), Xh, yh, gh, dh, scheme, eot_w=1.5)
        # english from joint
        pe = p_eng_base[~hin]
        rows.append(report(f"hinOnly ENET/{scheme}", pe, ph, ye, yh, ge, gh, de, dh))

    print("\n=== pairwise ranking ===")
    for thr, C in [(0.75, 0.5), (0.8, 0.5), (0.7, 0.25), (0.75, 1.0)]:
        p = oof_pairwise(X, y, g, d, hold_thr=thr, C=C)
        rows.append(report(f"pair thr{thr}/C{C}", p[~hin], p[hin], ye, yh, ge, gh, de, dh))
    # Hindi-only pairwise
    ph = oof_pairwise(Xh, yh, gh, dh, hold_thr=0.75, C=0.5)
    rows.append(report("pair hinOnly thr0.75", p_eng_base[~hin], ph, ye, yh, ge, gh, de, dh))

    print("\n=== blends ===")
    p_base = oof_weighted(make_enet(), X, y, g, d, "dur05", eot_w=1.5)
    p_hard = oof_weighted(make_enet(), X, y, g, d, "hard08_soft", eot_w=1.5)
    p_pair = oof_pairwise(X, y, g, d, hold_thr=0.75, C=0.5)
    p_hin_hard = np.zeros(len(y))
    p_hin_hard[~hin] = p_base[~hin]
    p_hin_hard[hin] = oof_weighted(make_enet(), Xh, yh, gh, dh, "hard08_soft", eot_w=2.5)

    for name, plist, ww in [
        ("blend base+hard", [p_base, p_hard], None),
        ("blend base+pair", [p_base, p_pair], None),
        ("blend hard+pair", [p_hard, p_pair], None),
        ("blend 3way", [p_base, p_hard, p_pair], None),
        ("blend base+hinHard", [p_base, p_hin_hard], [0.4, 0.6]),
        ("blend pair+hinHard", [p_pair, p_hin_hard], [0.5, 0.5]),
    ]:
        p = oof_blend(plist, ww)
        rows.append(report(name, p[~hin], p[hin], ye, yh, ge, gh, de, dh))

    # feature ablations: 43 vs 61, and history-only add-on
    print("\n=== feature ablations ===")
    for tag, cols in [
        ("43f", list(range(43))),
        ("43+mfcc", list(range(51))),
        ("43+hist", list(range(43)) + list(range(51, 61))),
        ("61f", list(range(61))),
    ]:
        p = oof_weighted(make_enet(), X[:, cols], y, g, d, "hard08_soft", eot_w=1.5)
        rows.append(report(f"ablate {tag} hard08_soft",
                           p[~hin], p[hin], ye, yh, ge, gh, de, dh))
        p = oof_weighted(make_enet(), X[:, cols], y, g, d, "dur05", eot_w=1.5)
        rows.append(report(f"ablate {tag} dur05",
                           p[~hin], p[hin], ye, yh, ge, gh, de, dh))

    df = pd.DataFrame(rows).sort_values(["hin", "eng"])
    df.to_csv("hindi_push2_oof.csv", index=False)
    print("\n=== TOP by Hindi OOF ===")
    print(df.head(20).to_string(index=False))
    beat = df[df.hin < 849.9]
    print(f"\nRuns beating 850: {len(beat)}")
    if len(beat):
        print(beat.to_string(index=False))


if __name__ == "__main__":
    main()
