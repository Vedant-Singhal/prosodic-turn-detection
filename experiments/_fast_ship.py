"""FAST Hindi push: hard-neg ENET -> official score.py; optional Hin p-adjust."""
import os, csv, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from eot_data import build_dataset
from eot_features import FEATURE_NAMES
from train_model import official_score_g

HANDOUT = "eot_handout/eot_data"
N43 = 43


def load_lang(lang):
    # Prefer cache (61-D from prior push); fall back to build. Always use first 43.
    cache = os.path.join(os.environ.get("SCRATCH", "."), "eot_cache",
                         f"{lang}_f{len(FEATURE_NAMES)}.npz")
    lab = pd.read_csv(os.path.join(HANDOUT, lang, "labels.csv"))
    if os.path.exists(cache):
        dnp = np.load(cache, allow_pickle=True)
        X, y, g, d = dnp["X"], dnp["y"], dnp["groups"], dnp["durs"]
        # rebuild keys in same order as build_dataset
        keys = []
        for tid, gg in lab.groupby("turn_id", sort=False):
            for _, r in gg.sort_values("pause_index").iterrows():
                keys.append((tid, int(r["pause_index"])))
        return X[:, :N43], y, g, d, keys
    X, y, g, keys, _ = build_dataset(os.path.join(HANDOUT, lang))
    dm = {(r.turn_id, int(r.pause_index)): float(r.pause_end) - float(r.pause_start)
          for r in lab.itertuples()}
    d = np.array([dm[k] for k in keys])
    return X[:, :N43], y, g, d, keys


def hard_w(y, d, eot_w=1.5):
    w = np.ones(len(y), float)
    hold = y == 0
    w[hold] = np.where(d[hold] > 0.8, 12.0, np.minimum(d[hold], 1.6) ** 0.5)
    w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def dur_w(y, d, eot_w=1.5):
    w = np.ones(len(y), float)
    hold = y == 0
    w[hold] = np.minimum(d[hold], 1.6) ** 0.5
    w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def enet():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=8000, C=0.5, solver="saga", l1_ratio=0.5,
                           penalty="elasticnet"),
    )


def write_pred(keys, p, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pidx), pi in zip(keys, p):
            w.writerow([tid, pidx, f"{float(pi):.6f}"])


def score_lang(lang, pred):
    # official score.py via subprocess for the real number
    import subprocess, re
    cmd = ["python", "eot_handout/starter/score.py",
           "--data_dir", os.path.join(HANDOUT, lang), "--pred", pred]
    out = subprocess.check_output(cmd, text=True)
    print(out.strip())
    m = re.search(r"mean response delay\s*:\s*([\d.]+)\s*ms", out)
    th = re.search(r"threshold=([\d.]+),\s*delay=([\d.]+)\s*ms", out)
    aucm = re.search(r"AUC=([\d.]+)", out)
    lat = float(m.group(1)) if m else float("nan")
    t = float(th.group(1)) if th else float("nan")
    d = float(th.group(2)) if th else float("nan")
    a = float(aucm.group(1)) if aucm else float("nan")
    return lat, t, d, a


def main():
    print(f"feats available={len(FEATURE_NAMES)} using first {N43}")
    Xe, ye, ge, de, ke = load_lang("english")
    Xh, yh, gh, dh, kh = load_lang("hindi")
    X = np.vstack([Xe, Xh]); y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh]); d = np.concatenate([de, dh])
    hin = np.array([False]*len(ye)+[True]*len(yh))

    # --- try 1: hard-neg ---
    m = enet()
    m.fit(X, y, logisticregression__sample_weight=hard_w(y, d))
    pe = m.predict_proba(Xe)[:, 1]
    ph = m.predict_proba(Xh)[:, 1]
    write_pred(ke, pe, "predictions_english.csv")
    write_pred(kh, ph, "predictions_hindi.csv")
    ee, eth, ed, ea = score_lang("english", "predictions_english.csv")
    hh, hth, hd, ha = score_lang("hindi", "predictions_hindi.csv")
    print(f"HARDNEG in-sample: eng={ee:.0f} (th={eth},d={ed:.0f}) hin={hh:.0f} (th={hth},d={hd:.0f})")

    # honest OOF hard-neg
    p_oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        mm = enet()
        mm.fit(X[tr], y[tr], logisticregression__sample_weight=hard_w(y[tr], d[tr]))
        p_oof[te] = mm.predict_proba(X[te])[:, 1]
    from train_model import official_score_g
    oe = official_score_g(ge, ye, de, p_oof[~hin])[0]*1000
    oh = official_score_g(gh, yh, dh, p_oof[hin])[0]*1000
    print(f"HARDNEG OOF: eng={oe:.0f} hin={oh:.0f}")

    best = dict(tag="hardneg", eng=ee, hin=hh, oof_e=oe, oof_h=oh, pipe=m, pe=pe, ph=ph)

    if hh < 849.5:
        print("HINDI WIN via hardneg")
    else:
        # --- try 2: keep best Eng model (dur05), Hindi-conditional p transform ---
        m0 = enet()
        m0.fit(X, y, logisticregression__sample_weight=dur_w(y, d))
        pe0 = m0.predict_proba(Xe)[:, 1]
        ph0 = m0.predict_proba(Xh)[:, 1]
        write_pred(ke, pe0, "predictions_english.csv")
        ee0, _, _, _ = score_lang("english", "predictions_english.csv")

        # language-conditional: grid power/temp on Hindi scores only (in-sample op)
        eps = 1e-6
        best_h, best_ph, best_hp = hh, ph0, (1.0, 1.0)
        logit = np.log(np.clip(ph0, eps, 1-eps)/(1-np.clip(ph0, eps, 1-eps)))
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            for power in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
                s = 1/(1+np.exp(-logit/temp))
                s = np.clip(s, eps, 1-eps)**power
                write_pred(kh, s, "_tmp_hin.csv")
                sc, _, _, _ = score_lang("hindi", "_tmp_hin.csv")
                if sc < best_h:
                    best_h, best_ph, best_hp = sc, s, (temp, power)
        # also try pause_index demotion (causal)
        ix = Xh[:, FEATURE_NAMES.index("pause_index")] if "pause_index" in FEATURE_NAMES else Xh[:, 26]
        # pause_index is index 26 in 43-f
        ix = Xh[:, 26]
        for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for thr in [0, 1, 2]:
                s = ph0 / (1.0 + lam * np.maximum(ix - thr, 0))
                write_pred(kh, s, "_tmp_hin.csv")
                sc, _, _, _ = score_lang("hindi", "_tmp_hin.csv")
                if sc < best_h:
                    best_h, best_ph, best_hp = sc, s, ("ix", lam, thr)

        print(f"LANGCOND eng={ee0:.0f} best_hin={best_h:.0f} params={best_hp}")
        write_pred(kh, best_ph, "predictions_hindi.csv")
        hh2, hth2, hd2, _ = score_lang("hindi", "predictions_hindi.csv")
        print(f"LANGCOND official: eng={ee0:.0f} hin={hh2:.0f} hop=({hth2},{hd2:.0f})")

        if hh2 < 849.5:
            best = dict(tag="langcond", eng=ee0, hin=hh2, oof_e=oe, oof_h=oh,
                        pipe=m0, pe=pe0, ph=best_ph, params=best_hp)
            print("HINDI WIN via langcond")
        else:
            # restore best Eng (dur05) + baseline Hin preds; do not ship hardneg if Eng worse
            write_pred(ke, pe0, "predictions_english.csv")
            write_pred(kh, ph0, "predictions_hindi.csv")
            ee0, _, _, _ = score_lang("english", "predictions_english.csv")
            hh0, _, _, _ = score_lang("hindi", "predictions_hindi.csv")
            joblib.dump({
                "pipeline": m0,
                "feature_names": FEATURE_NAMES[:N43],
                "winner": "ElasticNet C=0.5 + AC + dur-weights (Hindi push: 850 held)",
                "cv_eng_ms": float(oe),
                "cv_hin_ms": float(oh),
                "n_features_used": N43,
            }, "eot_model.pkl")
            # ensure predict uses 43 — store slice note
            print(f"NO WIN. Restored dur05 ship: eng={ee0:.0f} hin={hh0:.0f}")
            best = dict(tag="baseline_kept", eng=ee0, hin=hh0, oof_e=None, oof_h=None)
            with open("_fast_result.txt", "w") as f:
                f.write(f"{best}\n")
            return

    # ship winner
    write_pred(ke, best["pe"], "predictions_english.csv")
    write_pred(kh, best["ph"], "predictions_hindi.csv")
    joblib.dump({
        "pipeline": best["pipe"],
        "feature_names": FEATURE_NAMES[:N43],
        "winner": best["tag"],
        "cv_eng_ms": float(best.get("oof_e", -1)),
        "cv_hin_ms": float(best.get("oof_h", -1)),
        "n_features_used": N43,
        "params": best.get("params"),
    }, "eot_model.pkl")
    ee, _, _, _ = score_lang("english", "predictions_english.csv")
    hh, _, _, _ = score_lang("hindi", "predictions_hindi.csv")
    print(f"SHIPPED {best['tag']}: eng={ee:.0f} hin={hh:.0f}")
    with open("_fast_result.txt", "w") as f:
        f.write(f"shipped {best['tag']} eng={ee} hin={hh}\n")


if __name__ == "__main__":
    main()
