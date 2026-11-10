"""Human-steered Hindi diagnosis: longest holds vs true EOTs.

Prints acoustic summaries so we can listen + compare qualitatively.
Does NOT use future pause duration as a feature — only for analysis.
"""
import os
import numpy as np
import pandas as pd
import soundfile as sf

from eot_features import compute_frame_features, pause_features, FEATURE_NAMES, frames_before

HANDOUT = "eot_handout/eot_data/hindi"


def main():
    lab = pd.read_csv(os.path.join(HANDOUT, "labels.csv"))
    lab["dur"] = lab["pause_end"] - lab["pause_start"]

    holds = lab[lab["label"] == "hold"].sort_values("dur", ascending=False)
    eots = lab[lab["label"] == "eot"]

    print("=== Hindi pause duration stats ===")
    print(f"holds: n={len(holds)} median={holds.dur.median():.3f} p90={holds.dur.quantile(0.9):.3f} max={holds.dur.max():.3f}")
    print(f"eots:  n={len(eots)} median={eots.dur.median():.3f} p90={eots.dur.quantile(0.9):.3f}")

    long_holds = holds.head(12)
    print("\n=== Top 12 longest Hindi HOLDS (listen to these) ===")
    for r in long_holds.itertuples():
        print(f"  {r.turn_id} pause#{r.pause_index} dur={r.dur:.3f}s  "
              f"[{r.pause_start:.2f}->{r.pause_end:.2f}]  file={r.audio_file}")

    # Sample EOTs for comparison (spread of pause_start)
    eot_sample = eots.sample(12, random_state=0) if len(eots) >= 12 else eots
    print("\n=== Sample Hindi EOTs (for A/B listening) ===")
    for r in eot_sample.itertuples():
        print(f"  {r.turn_id} pause#{r.pause_index} dur={r.dur:.3f}s  "
              f"start={r.pause_start:.2f}  file={r.audio_file}")

    # Acoustic feature contrast: long holds (dur>0.7) vs all eots
    danger = holds[holds.dur > 0.7]
    print(f"\n=== Feature means: eot (n={len(eots)}) vs hold dur>0.7s (n={len(danger)}) ===")

    rows = []
    for subset, name in [(eots, "eot"), (danger, "long_hold"), (long_holds, "top12_hold")]:
        for r in subset.itertuples():
            wav = os.path.join(HANDOUT, r.audio_file)
            x, sr = sf.read(wav, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            fr = compute_frame_features(x, sr)
            # prior pauses in same turn
            tg = lab[lab.turn_id == r.turn_id].sort_values("pause_index")
            prior = []
            prev_end = None
            for _, pr in tg.iterrows():
                if int(pr.pause_index) >= int(r.pause_index):
                    break
                prior.append(float(pr.pause_end) - float(pr.pause_start))
                prev_end = float(pr.pause_end)
            feat = pause_features(fr, float(r.pause_start), int(r.pause_index),
                                  tuple(prior), prev_end)
            rows.append((name, feat))

    # compare eot vs long_hold means
    eot_X = np.vstack([f for n, f in rows if n == "eot"])
    lh_X = np.vstack([f for n, f in rows if n == "long_hold"])
    print(f"{'feature':28s} {'eot_mean':>9s} {'lh_mean':>9s} {'delta':>8s} {'sep':>6s}")
    diffs = []
    for i, name in enumerate(FEATURE_NAMES):
        em = np.nanmean(eot_X[:, i]); lm = np.nanmean(lh_X[:, i])
        # rough separation: mean diff / pooled std
        pooled = np.nanstd(np.concatenate([eot_X[:, i], lh_X[:, i]])) + 1e-9
        sep = (em - lm) / pooled
        diffs.append((abs(sep), name, em, lm, em - lm, sep))
    diffs.sort(reverse=True)
    for _, name, em, lm, d, sep in diffs[:20]:
        print(f"{name:28s} {em:9.3f} {lm:9.3f} {d:8.3f} {sep:6.3f}")

    # Extra: residual energy predictability near pause (acoustic completion proxy preview)
    print("\n=== Acoustic-completion proxy (next-frame energy residual RMSE, last 0.4s) ===")
    def completion_rmse(fr, pause_start):
        k = frames_before(fr, pause_start)
        if k < 30:
            return np.nan
        edb = fr["edb"][:k]
        # predict edb[t] from edb[t-1], edb[t-2] via linear AR(2) fit on earlier frames
        y = edb[2:k]
        X = np.column_stack([edb[1:k-1], edb[0:k-2], np.ones(k-2)])
        # fit on all-but-last 0.4s, evaluate on last 0.4s
        n_eval = min(40, (k - 2) // 3)
        if n_eval < 5 or (k - 2 - n_eval) < 10:
            return np.nan
        train = slice(0, k - 2 - n_eval)
        test = slice(k - 2 - n_eval, k - 2)
        try:
            coef, _, _, _ = np.linalg.lstsq(X[train], y[train], rcond=None)
            pred = X[test] @ coef
            return float(np.sqrt(np.mean((y[test] - pred) ** 2)))
        except Exception:
            return np.nan

    for label, subset in [("eot", eots), ("long_hold>0.7", danger)]:
        vals = []
        for r in subset.itertuples():
            wav = os.path.join(HANDOUT, r.audio_file)
            x, sr = sf.read(wav, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            fr = compute_frame_features(x, sr)
            vals.append(completion_rmse(fr, float(r.pause_start)))
        vals = np.array(vals, dtype=float)
        print(f"  {label:16s} mean_rmse={np.nanmean(vals):.4f}  median={np.nanmedian(vals):.4f}")


if __name__ == "__main__":
    main()
