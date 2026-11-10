"""Shared dataset builder: labels.csv + wavs -> causal feature matrix.

Used by eda.py, train_model.py and predict.py so feature extraction is
identical everywhere.  Loads each wav once, computes frame features once,
then extracts a causal feature vector per pause (with strictly-prior
pause context).
"""
import os
import numpy as np
import pandas as pd
import soundfile as sf

from eot_features import (
    compute_frame_features, pause_features, FEATURE_NAMES, boundary_stats_update,
)


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def build_dataset(data_dir, verbose=False):
    """Return X, y, groups, keys, meta for one data folder.

    y is None-safe: if a 'label' column is absent (unseen test folder), y is
    returned as all -1 and should be ignored.
    """
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    has_label = "label" in labels.columns
    X, y, groups, keys = [], [], [], []
    for tid, g in labels.groupby("turn_id", sort=False):
        g = g.sort_values("pause_index")
        wav_path = os.path.join(data_dir, g.iloc[0]["audio_file"])
        x, sr = load_wav(wav_path)
        fr = compute_frame_features(x, sr)
        prior_durs = []
        prev_end = None
        prior_stats = {}
        for _, r in g.iterrows():
            ps = float(r["pause_start"])
            pidx = int(r["pause_index"])
            feat = pause_features(fr, ps, pause_index=pidx,
                                  prior_pause_durs=tuple(prior_durs),
                                  prev_pause_end=prev_end,
                                  prior_stats=prior_stats)
            X.append(feat)
            groups.append(tid)
            keys.append((tid, pidx))
            if has_label:
                y.append(1 if r["label"] == "eot" else 0)
            else:
                y.append(-1)
            # update causal history AFTER using it for this pause
            prior_stats = boundary_stats_update(prior_stats, feat)
            if "pause_end" in r and not pd.isna(r["pause_end"]):
                prior_durs.append(float(r["pause_end"]) - ps)
                prev_end = float(r["pause_end"])
        if verbose:
            print(f"  {tid}: {len(g)} pauses")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y, np.array(groups), keys, {"feature_names": FEATURE_NAMES}
