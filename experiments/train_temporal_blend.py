"""Tiny causal frame-level temporal model + Elastic-Net blend.

Original from first principles: last N frames of (F0-st, energy, voiced,
centroid, tilt) -> small BiLSTM -> p_eot; blend with tabular Elastic-Net.
Strictly causal (frames ending <= pause_start only). No pretrained weights.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import joblib

from eot_features import (
    compute_frame_features, pause_features, FEATURE_NAMES, frames_before, HOP_S,
)
from eot_data import load_wav

HANDOUT = "eot_handout/eot_data"
TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)
SEQ_LEN = 80          # 80 * 10ms = 0.8 s of pre-pause frames
N_CH = 5              # f0_st, edb_norm, voiced, cent_norm, tilt
DEVICE = "cpu"
SEED = 0


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


def seq_window(fr, pause_start, seq_len=SEQ_LEN):
    """Causal (seq_len, N_CH) window ending at last frame before pause."""
    k = frames_before(fr, pause_start)
    out = np.zeros((seq_len, N_CH), dtype=np.float32)
    if k < 3:
        return out
    t = fr["t_end"][:k]
    edb = fr["edb"][:k]
    f0 = fr["f0"][:k]
    voiced = fr["voiced"][:k].astype(np.float32)
    cent = fr["cent"][:k]
    tilt = fr["tilt"][:k]
    # speaker norms from prefix
    f0v = f0[voiced > 0.5]
    ref = float(np.median(f0v)) if len(f0v) >= 3 else 150.0
    f0_st = 12.0 * np.log2(np.maximum(f0, 1e-3) / max(ref, 1e-3))
    f0_st = np.where(voiced > 0.5, f0_st, 0.0)
    e_mu, e_sd = float(edb.mean()), float(edb.std() + 1e-6)
    c_mu, c_sd = float(cent.mean()), float(cent.std() + 1e-6)
    feats = np.stack([
        f0_st,
        (edb - e_mu) / e_sd,
        voiced,
        (cent - c_mu) / c_sd,
        tilt,
    ], axis=1).astype(np.float32)
    take = min(seq_len, k)
    out[-take:] = feats[k - take:k]
    return out


def build_all(lang):
    cache = os.path.join("eot_cache", f"{lang}_tab_seq.npz")
    os.makedirs("eot_cache", exist_ok=True)
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["S"], d["y"], d["groups"], d["durs"]
    data_dir = os.path.join(HANDOUT, lang)
    lab = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    Xs, Ss, ys, gs, ds = [], [], [], [], []
    for tid, g in lab.groupby("turn_id", sort=False):
        g = g.sort_values("pause_index")
        x, sr = load_wav(os.path.join(data_dir, g.iloc[0]["audio_file"]))
        fr = compute_frame_features(x, sr)
        prior, prev_end = [], None
        for _, r in g.iterrows():
            ps = float(r["pause_start"])
            tab = pause_features(fr, ps, int(r["pause_index"]), tuple(prior), prev_end)
            seq = seq_window(fr, ps)
            Xs.append(tab)
            Ss.append(seq)
            ys.append(1 if r["label"] == "eot" else 0)
            gs.append(tid)
            ds.append(float(r["pause_end"]) - ps)
            prior.append(float(r["pause_end"]) - ps)
            prev_end = float(r["pause_end"])
    X = np.asarray(Xs, np.float32)
    S = np.asarray(Ss, np.float32)
    y = np.asarray(ys, np.int64)
    groups = np.array(gs)
    durs = np.asarray(ds)
    np.savez(cache, X=X, S=S, y=y, groups=groups, durs=durs)
    return X, S, y, groups, durs


class TinyGRU(nn.Module):
    def __init__(self, n_ch=N_CH, hidden=32, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: (B, T, C)
        o, h = self.gru(x)
        # attention-ish: mean of last 10 frames of outputs
        pooled = o[:, -10:, :].mean(dim=1)
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(-1)


def train_gru(S, y, epochs=25, lr=1e-3, batch=64, seed=SEED):
    torch.manual_seed(seed)
    model = TinyGRU().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    # class balance
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ds = TensorDataset(torch.from_numpy(S), torch.from_numpy(y.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_gru(model, S):
    model.eval()
    out = []
    for i in range(0, len(S), 256):
        xb = torch.from_numpy(S[i:i + 256]).to(DEVICE)
        logits = model(xb)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def make_enet():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, C=0.5, penalty="elasticnet", solver="saga",
            l1_ratio=0.5, class_weight="balanced")),
    ])


def dur_weights(y, dur, gamma=0.5, eot_w=1.5):
    w = np.ones(len(y), dtype=float)
    hold = y == 0
    w[hold] = np.minimum(dur[hold], 1.6) ** gamma
    w[hold] *= hold.sum() / (w[hold].sum() + 1e-9)
    w[y == 1] = eot_w
    return w


def main():
    Xe, Se, ye, ge, de = build_all("english")
    Xh, Sh, yh, gh, dh = build_all("hindi")
    X = np.vstack([Xe, Xh])
    S = np.vstack([Se, Sh])
    y = np.concatenate([ye, yh])
    g = np.concatenate([ge, gh])
    d = np.concatenate([de, dh])
    lang = np.array(["e"] * len(ye) + ["h"] * len(yh))
    print(f"tab={X.shape} seq={S.shape}")

    # OOF for tabular ENET, GRU, and blends
    p_tab = np.zeros(len(y))
    p_seq = np.zeros(len(y))
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        print(f"fold {fold+1}/5 ...", flush=True)
        # tabular
        m = make_enet()
        w = dur_weights(y[tr], d[tr])
        # combine with mild Hindi upweight
        w = w * np.where(lang[tr] == "h", 2.0, 1.0)
        m.fit(X[tr], y[tr], clf__sample_weight=w)
        p_tab[te] = m.predict_proba(X[te])[:, 1]
        # temporal
        gru = train_gru(S[tr], y[tr], epochs=30, seed=SEED + fold)
        p_seq[te] = predict_gru(gru, S[te])

    results = []
    for name, p in [
        ("ENET dur+hin", p_tab),
        ("TinyGRU", p_seq),
        ("blend 0.7/0.3", 0.7 * p_tab + 0.3 * p_seq),
        ("blend 0.5/0.5", 0.5 * p_tab + 0.5 * p_seq),
        ("blend 0.3/0.7", 0.3 * p_tab + 0.7 * p_seq),
        ("blend 0.6/0.4", 0.6 * p_tab + 0.4 * p_seq),
        ("max(tab,seq)", np.maximum(p_tab, p_seq)),
    ]:
        me = official_score_g(ge, ye, de, p[lang == "e"])[0] * 1000
        mh = official_score_g(gh, yh, dh, p[lang == "h"])[0] * 1000
        pri = (me + 2 * mh) / 3
        print(f"{name:16s} eng={me:6.0f} hin={mh:6.0f} pri={pri:6.0f}")
        results.append((pri, mh, me, name, p))

    results.sort()
    best = results[0]
    print(f"\nBEST CV: {best[3]} eng={best[2]:.0f} hin={best[1]:.0f}")

    # Also try Hindi-only GRU OOF (hidden test mostly Hindi)
    print("\n--- Hindi-only TinyGRU CV ---")
    ph = np.zeros(len(yh))
    for fold, (tr, te) in enumerate(GroupKFold(5).split(Xh, yh, gh)):
        gru = train_gru(Sh[tr], yh[tr], epochs=35, seed=10 + fold)
        ph[te] = predict_gru(gru, Sh[te])
    mh = official_score_g(gh, yh, dh, ph)[0] * 1000
    print(f"HinOnly GRU hinCV={mh:.0f}")

    # Hindi-only ENET
    ph2 = np.zeros(len(yh))
    for tr, te in GroupKFold(5).split(Xh, yh, gh):
        m = make_enet()
        m.fit(Xh[tr], yh[tr], clf__sample_weight=dur_weights(yh[tr], dh[tr]))
        ph2[te] = m.predict_proba(Xh[te])[:, 1]
    mh2 = official_score_g(gh, yh, dh, ph2)[0] * 1000
    print(f"HinOnly ENET hinCV={mh2:.0f}")

    # Blend Hindi-only
    for a in [0.3, 0.5, 0.7]:
        pb = a * ph2 + (1 - a) * ph
        mh3 = official_score_g(gh, yh, dh, pb)[0] * 1000
        print(f"HinOnly blend tab={a:.1f} hinCV={mh3:.0f}")

    # Refit best multilingual blend on all data and save
    # Prefer best blend by Hindi priority among multilingual
    w_name = best[3]
    if "blend" in w_name:
        # parse alpha
        if "0.7/0.3" in w_name:
            a = 0.7
        elif "0.5/0.5" in w_name:
            a = 0.5
        elif "0.3/0.7" in w_name:
            a = 0.3
        elif "0.6/0.4" in w_name:
            a = 0.6
        else:
            a = 0.7
    elif w_name == "TinyGRU":
        a = 0.0
    elif "max" in w_name:
        a = -1  # special
    else:
        a = 1.0

    tab = make_enet()
    w = dur_weights(y, d) * np.where(lang == "h", 2.0, 1.0)
    tab.fit(X, y, clf__sample_weight=w)
    gru = train_gru(S, y, epochs=35, seed=SEED)
    torch.save({"state_dict": gru.state_dict(), "seq_len": SEQ_LEN, "n_ch": N_CH},
               "eot_gru.pt")
    joblib.dump({
        "pipeline": tab,
        "feature_names": FEATURE_NAMES,
        "winner": w_name,
        "blend_alpha_tab": a,
        "cv_eng_ms": best[2],
        "cv_hin_ms": best[1],
        "has_gru": True,
        "trained_on": "english+hindi",
        "selection": "hindi-priority",
    }, "eot_model.pkl")
    print(f"saved eot_model.pkl + eot_gru.pt (blend_alpha_tab={a})")
    pd.DataFrame([{"name": n, "eng": e, "hin": h, "pri": p}
                  for p, h, e, n, _ in results]).to_csv("temporal_results.csv", index=False)


if __name__ == "__main__":
    main()
