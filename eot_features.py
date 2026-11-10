"""Causal prosodic feature extraction for end-of-turn (EOT) detection.

CAUSALITY (hard requirement of the assignment):
    For a pause at time `pause_start`, every feature is computed ONLY from
    audio in [0, pause_start).  A frame is used only if its entire extent
    ends at or before pause_start (see `frames_before`), so no sample after
    the pause can ever influence a feature.  We never touch `pause_end` of
    the CURRENT pause (that is future information / how the file was cut).
    We DO use the durations of STRICTLY PRIOR pauses (their pause_end is in
    the past), which is legitimate causal context.

Design:
    `compute_frame_features(x, sr)` runs the expensive per-frame analysis
    (energy, a custom vectorised normalised-cross-correlation F0 tracker,
    spectral centroid, ZCR) ONCE over a whole turn.  `pause_features(...)`
    then slices those cached frames up to a pause and turns them into a
    fixed-length feature vector.

No pretrained models or downloaded weights are used anywhere.
"""
import numpy as np

# ---- framing grid -------------------------------------------------------
FRAME_S = 0.032          # 32 ms analysis frame
HOP_S = 0.010            # 10 ms hop
FMIN, FMAX = 65.0, 400.0  # plausible human F0 range (Hz)
VCORR_THR = 0.35         # NCC voicing threshold


# ---- per-turn frame analysis (run once per wav) -------------------------
def compute_frame_features(x, sr):
    """Frame-level acoustic descriptors over the whole turn.

    Returns a dict of 1-D arrays (one value per frame) plus the frame-END
    time of each frame so callers can slice strictly-causally.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    fl = int(round(sr * FRAME_S))
    hp = int(round(sr * HOP_S))
    if len(x) < fl:
        z = np.zeros(0, dtype=np.float32)
        return dict(t_end=z, edb=z, f0=z, vcorr=z, cent=z, zcr=z, tilt=z,
                    voiced=z.astype(bool), mfcc=np.zeros((0, 8), dtype=np.float32))

    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    F = x[idx]                                   # (n, fl)
    # frame END time (right edge) — used for causal slicing
    t_end = (hp * np.arange(n) + fl) / sr

    # --- energy (dB) ---
    rms = np.sqrt(np.mean(F * F, axis=1) + 1e-12)
    edb = 20.0 * np.log10(rms + 1e-12)

    # --- zero-crossing rate ---
    zcr = np.mean(np.abs(np.diff(np.sign(F), axis=1)) > 0, axis=1).astype(np.float32)

    # --- spectral centroid (Hz) via rFFT ---
    win = np.hanning(fl).astype(np.float32)
    Fw = F * win
    mag = np.abs(np.fft.rfft(Fw, axis=1))
    freqs = np.fft.rfftfreq(fl, d=1.0 / sr).astype(np.float32)
    denom = mag.sum(axis=1) + 1e-9
    cent = (mag @ freqs) / denom

    # --- spectral tilt: log-ratio of low-band (<1kHz) to high-band (>2kHz) energy ---
    p = mag ** 2
    lo_band = p[:, freqs < 1000.0].sum(axis=1) + 1e-9
    hi_band = p[:, freqs > 2000.0].sum(axis=1) + 1e-9
    tilt = np.log10(lo_band / hi_band).astype(np.float32)

    # --- MFCC (from-scratch mel filterbank + DCT; no pretrained weights) ---
    mfcc = _mfcc_frames(mag, freqs, sr, n_mels=26, n_mfcc=8)

    # --- F0 via vectorised normalised cross-correlation (custom, causal) ---
    Fz = Fw - Fw.mean(axis=1, keepdims=True)
    lo = int(sr / FMAX)
    hi = min(int(sr / FMIN), fl - 1)
    best_val = np.full(n, -1.0, dtype=np.float32)
    best_lag = np.zeros(n, dtype=np.int64)
    for lag in range(lo, hi + 1):
        a = Fz[:, :fl - lag]
        b = Fz[:, lag:]
        num = np.sum(a * b, axis=1)
        den = np.sqrt((np.sum(a * a, axis=1) + 1e-9) * (np.sum(b * b, axis=1) + 1e-9))
        r = num / den
        upd = r > best_val
        best_val = np.where(upd, r, best_val)
        best_lag = np.where(upd, lag, best_lag)
    voiced = (best_val >= VCORR_THR) & (best_lag > 0)
    f0 = np.where(voiced, sr / np.maximum(best_lag, 1), 0.0).astype(np.float32)
    f0 = _octave_correct(f0, voiced)          # fix halving/doubling errors
    f0 = _median_smooth_voiced(f0, voiced, k=3)

    return dict(t_end=t_end.astype(np.float32), edb=edb.astype(np.float32),
                f0=f0, vcorr=best_val.astype(np.float32),
                cent=cent.astype(np.float32), zcr=zcr, tilt=tilt, voiced=voiced,
                mfcc=mfcc)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def _mfcc_frames(mag, freqs, sr, n_mels=26, n_mfcc=8):
    """Per-frame MFCCs from magnitude spectrum (no external speech models)."""
    n_fft_bins = mag.shape[1]
    f_max = float(freqs[-1]) if len(freqs) else sr / 2.0
    mels = np.linspace(_hz_to_mel(0.0), _hz_to_mel(f_max), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft_bins - 1) * hz / max(f_max, 1e-6)).astype(int)
    bins = np.clip(bins, 0, n_fft_bins - 1)
    fb = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c > l:
            fb[m, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m, c:r] = (r - np.arange(c, r)) / (r - c)
    mel_e = np.maximum(mag @ fb.T, 1e-10)
    log_mel = np.log(mel_e)
    # DCT-II
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct = np.cos(np.pi * (n + 0.5) * k / n_mels).astype(np.float32)
    return (log_mel @ dct.T).astype(np.float32)


def _octave_correct(f0, voiced):
    """Snap frames that are ~2x/0.5x the local voiced median toward it."""
    out = f0.copy()
    vi = np.where(voiced)[0]
    if len(vi) < 5:
        return out
    med = np.median(f0[vi])
    for i in vi:
        if med > 0:
            if f0[i] > 1.6 * med:
                out[i] = f0[i] / 2.0
            elif f0[i] < 0.6 * med:
                out[i] = f0[i] * 2.0
    return out


def _median_smooth_voiced(f0, voiced, k=3):
    out = f0.copy()
    vi = np.where(voiced)[0]
    if len(vi) < k:
        return out
    vals = f0[vi]
    pad = k // 2
    padded = np.pad(vals, pad, mode="edge")
    sm = np.array([np.median(padded[j:j + k]) for j in range(len(vals))], dtype=np.float32)
    out[vi] = sm
    return out


# ---- helpers ------------------------------------------------------------
def frames_before(fr, pause_start):
    """Index of first frame whose END is after pause_start (i.e. use [:k])."""
    return int(np.searchsorted(fr["t_end"], pause_start, side="right"))


def _lin_slope(t, y):
    """Least-squares slope of y vs t (per second); 0 if <2 points."""
    if len(t) < 2:
        return 0.0
    t = t - t.mean()
    denom = np.dot(t, t)
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(t, y - y.mean()) / denom)


def _st(f0, ref):
    """Hz -> semitones relative to ref (>0)."""
    return 12.0 * np.log2(np.maximum(f0, 1e-3) / max(ref, 1e-3))


FEATURE_NAMES = [
    # energy / intensity
    "e_end", "e_end_rel_max", "e_end_rel_mean", "e_slope_300", "e_slope_500",
    "e_std_final", "e_drop_final",
    # pitch (semitones, speaker-normalised by turn-so-far median)
    "f0_end_st", "f0_slope_400", "f0_slope_200", "f0_min_final_st",
    "f0_range_final_st", "f0_delta_end", "f0_std_sofar_st",
    # voicing / duration / rate  (final-lengthening & slowing cues)
    "voiced_frac_final", "voiced_frac_win", "final_voiced_dur",
    "final_voiced_dur_z", "gap_last_speech", "speech_rate_final",
    "n_voiced_runs",
    # spectral
    "cent_end", "cent_slope_final", "zcr_end", "zcr_slope_final",
    # causal context / position
    "pause_start", "pause_index", "prior_pause_mean_dur",
    "run_len_since_prev", "voiced_time_sofar", "energy_trend_turn",
    # boundary-tone shape, creak, tilt, energy fade (added round 2)
    "f0_end_absdev", "f0_flatness_final", "creak_frac", "tilt_end",
    "tilt_slope_final", "e_fade_200", "f0_end_vs_p90", "f0_end_vs_p10",
    # acoustic-completion proxies (AR residual / contour surprise) — round 6+
    "ac_e_rmse", "ac_f0_rmse", "ac_e_surprise", "ac_contour_curl",
    # MFCC mean / delta over last 0.5 s of speech (causal)
    "mfcc1_mean_500", "mfcc2_mean_500", "mfcc3_mean_500", "mfcc4_mean_500",
    "mfcc1_dmean_500", "mfcc2_dmean_500", "mfcc3_dmean_500", "mfcc4_dmean_500",
    # richer speaker/turn pause history (strictly prior pauses only)
    "prior_pause_max_dur", "prior_pause_last_dur", "prior_pause_std_dur",
    "n_prior_pauses", "prior_mean_e_drop", "rel_e_drop",
    "prior_mean_f0_p10", "rel_f0_vs_p10", "ix_x_edrop", "ix_x_fade",
]
N_FEATURES = len(FEATURE_NAMES)


def _ar_rmse(y, n_eval=40):
    """AR(2) residual RMSE on the last n_eval samples; fit on earlier ones."""
    y = np.asarray(y, dtype=float)
    if len(y) < n_eval + 20:
        return np.nan
    Y = y[2:]
    X = np.column_stack([y[1:-1], y[:-2], np.ones(len(y) - 2)])
    te0 = len(Y) - n_eval
    coef, _, _, _ = np.linalg.lstsq(X[:te0], Y[:te0], rcond=None)
    resid = Y[te0:] - X[te0:] @ coef
    return float(np.sqrt(np.mean(resid ** 2)))


def acoustic_completion_feats(fr, pause_start):
    """Causal stand-in for 'is this phrase acoustically finished?'.

    Fit a tiny AR(2) on energy / F0 up to the pause and score how surprising
    the final frames are. Mid-phrase speech is more locally predictable;
    completed phrases often leave a sharper residual / contour curl.
    Uses only frames ending at/before pause_start.
    """
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
    out[0] = _ar_rmse(e, n_eval=min(40, max(10, (last + 1) // 5)))
    if voiced.sum() >= 10:
        idx = np.arange(k)
        f0s = f0.copy()
        f0s[~voiced] = np.interp(idx[~voiced], idx[voiced], f0[voiced])
        out[1] = _ar_rmse(f0s[: last + 1], n_eval=min(40, max(10, (last + 1) // 5)))
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


def pause_features(fr, pause_start, pause_index=0, prior_pause_durs=(),
                   prev_pause_end=None, prior_stats=None):
    """Fixed-length causal feature vector for one pause.

    fr                : dict from compute_frame_features (whole turn)
    pause_start       : seconds (the moment speech stops for this pause)
    pause_index       : # of prior pauses in this turn (causal position)
    prior_pause_durs  : durations of STRICTLY prior pauses (causal)
    prev_pause_end    : end time of the previous pause, or None
    prior_stats       : optional dict with running means of boundary cues from
                        STRICTLY prior pauses in this turn (updated by caller)
    """
    out = np.full(N_FEATURES, np.nan, dtype=np.float32)
    k = frames_before(fr, pause_start)
    if k < 3:                                    # not enough context yet
        _fill_context(out, pause_start, pause_index, prior_pause_durs, prev_pause_end,
                      prior_stats)
        return out

    t = fr["t_end"][:k]
    edb = fr["edb"][:k]
    f0 = fr["f0"][:k]
    voiced = fr["voiced"][:k]
    cent = fr["cent"][:k]
    zcr = fr["zcr"][:k]

    # ---- locate the end of speech (ignore any silence onset before pause) ----
    vi = np.where(voiced)[0]
    last_sp = int(vi[-1]) if len(vi) else k - 1
    t_last = t[last_sp]

    # speaker-normalised pitch reference from voiced pitch so far
    f0v = f0[voiced]
    ref = np.median(f0v) if len(f0v) >= 3 else (np.median(f0v) if len(f0v) else 150.0)
    st_all = _st(f0, ref)

    def win_mask(dt):                # frames within last dt seconds of speech
        return t >= (t_last - dt)

    # ---------- energy ----------
    e_end = float(np.mean(edb[max(0, last_sp - 4):last_sp + 1]))
    out[0] = e_end
    out[1] = e_end - float(np.max(edb))
    ev = edb[voiced]
    out[2] = e_end - (float(np.mean(ev)) if len(ev) else float(np.mean(edb)))
    m3 = win_mask(0.30)
    out[3] = _lin_slope(t[m3], edb[m3])
    m5 = win_mask(0.50)
    out[4] = _lin_slope(t[m5], edb[m5])
    out[5] = float(np.std(edb[m5])) if m5.sum() >= 2 else 0.0
    out[6] = float(np.max(edb[m5])) - e_end if m5.sum() else 0.0

    # ---------- pitch ----------
    vmask_final = voiced & win_mask(0.50)
    st_final = st_all[vmask_final]
    if len(st_final) >= 1:
        out[7] = float(np.mean(st_all[voiced & win_mask(0.12)])) if (voiced & win_mask(0.12)).any() else float(st_final[-1])
        out[10] = float(np.min(st_final))
        out[11] = float(np.ptp(st_final)) if len(st_final) >= 2 else 0.0
    vt4 = voiced & win_mask(0.40)
    out[8] = _lin_slope(t[vt4], st_all[vt4]) if vt4.sum() >= 2 else 0.0
    vt2 = voiced & win_mask(0.20)
    out[9] = _lin_slope(t[vt2], st_all[vt2]) if vt2.sum() >= 2 else 0.0
    # end-region minus mid-region pitch
    vt_end = voiced & win_mask(0.30)
    vt_mid = voiced & (t < (t_last - 0.30)) & (t >= (t_last - 0.90))
    if vt_end.any() and vt_mid.any():
        out[12] = float(np.mean(st_all[vt_end]) - np.mean(st_all[vt_mid]))
    else:
        out[12] = 0.0
    out[13] = float(np.std(st_all[voiced])) if voiced.sum() >= 3 else 0.0

    # ---------- voicing / duration / rate ----------
    out[14] = float(np.mean(voiced[win_mask(0.50)])) if win_mask(0.50).any() else 0.0
    out[15] = float(np.mean(voiced[win_mask(1.00)])) if win_mask(1.00).any() else 0.0
    # final voiced run length (final-syllable lengthening proxy)
    runs = _voiced_runs(voiced, t, last_sp)
    fvd = runs[-1] if runs else 0.0
    out[16] = fvd
    if len(runs) >= 2:
        mu, sd = np.mean(runs[:-1]), np.std(runs[:-1]) + 1e-6
        out[17] = (fvd - mu) / sd
    else:
        out[17] = 0.0
    out[18] = float(pause_start - t_last)        # silence gap already accrued
    # speaking rate: voiced-run onsets per second in last 1.5 s
    m15 = win_mask(1.5)
    on = np.sum((voiced[m15][1:].astype(int) - voiced[m15][:-1].astype(int)) == 1)
    span = max(t[m15][-1] - t[m15][0], 1e-3) if m15.sum() >= 2 else 1.0
    out[19] = on / span
    out[20] = float(len(runs))

    # ---------- spectral ----------
    out[21] = float(np.mean(cent[max(0, last_sp - 4):last_sp + 1]))
    mc = win_mask(0.40)
    out[22] = _lin_slope(t[mc], cent[mc])
    out[23] = float(np.mean(zcr[max(0, last_sp - 4):last_sp + 1]))
    out[24] = _lin_slope(t[mc], zcr[mc])

    # ---------- causal context / position ----------
    out[25] = float(pause_start)
    out[26] = float(pause_index)
    out[27] = float(np.mean(prior_pause_durs)) if len(prior_pause_durs) else 0.0
    out[28] = float(pause_start - prev_pause_end) if prev_pause_end is not None else float(t_last)
    out[29] = float(np.sum(voiced) * HOP_S)
    # global energy trend across the whole turn so far (winding down?)
    out[30] = _lin_slope(t, edb)

    # ---------- round-2: boundary tone shape / creak / tilt / fade ----------
    tilt = fr["tilt"][:k]
    # boundary-tone magnitude: |final pitch dev| — statements fall, questions
    # rise; both mark EOT, while hesitations stay near the median.
    out[31] = abs(out[7]) if not np.isnan(out[7]) else 0.0
    # pitch flatness in final 300ms (low std = level = continuation/hold)
    vt3 = voiced & win_mask(0.30)
    out[32] = float(np.std(st_all[vt3])) if vt3.sum() >= 2 else 0.0
    # creak fraction: final 300ms voiced frames well below speaker median pitch
    if vt3.sum() >= 1:
        out[33] = float(np.mean(st_all[vt3] < -3.0))
    else:
        out[33] = 0.0
    out[34] = float(np.mean(tilt[max(0, last_sp - 4):last_sp + 1]))
    out[35] = _lin_slope(t[mc], tilt[mc])
    # energy fade over the last 200ms of speech
    m2 = win_mask(0.20)
    out[36] = _lin_slope(t[m2], edb[m2]) if m2.sum() >= 2 else 0.0
    # final pitch relative to the speaker's pitch spread so far (percentiles)
    if len(f0v) >= 5:
        p90 = np.percentile(f0v, 90); p10 = np.percentile(f0v, 10)
        f0_end_hz = float(np.mean(f0[voiced & win_mask(0.15)])) if (voiced & win_mask(0.15)).any() else float(f0v[-1])
        out[37] = _st(np.array([f0_end_hz]), p90)[0]     # <0 => below top of range
        out[38] = _st(np.array([f0_end_hz]), p10)[0]     # ~0 => at bottom of range
    else:
        out[37] = 0.0; out[38] = 0.0

    # ---------- acoustic-completion proxies (causal) ----------
    ac = acoustic_completion_feats(fr, pause_start)
    out[39:43] = ac

    # ---------- MFCC mean / delta last 0.5 s ----------
    mfcc = fr.get("mfcc")
    if mfcc is not None and len(mfcc) >= k and m5.sum() >= 3:
        M = mfcc[:k][m5]
        out[43:47] = np.mean(M[:, :4], axis=0)
        if len(M) >= 4:
            dM = np.diff(M[:, :4], axis=0)
            out[47:51] = np.mean(dM, axis=0)
        else:
            out[47:51] = 0.0
    else:
        out[43:51] = 0.0

    # ---------- richer prior-pause / within-turn relative ----------
    pdurs = list(prior_pause_durs)
    out[51] = float(max(pdurs)) if pdurs else 0.0
    out[52] = float(pdurs[-1]) if pdurs else 0.0
    out[53] = float(np.std(pdurs)) if len(pdurs) >= 2 else 0.0
    out[54] = float(len(pdurs))
    ps = prior_stats or {}
    n_ps = float(ps.get("n", 0))
    mean_ed = float(ps.get("mean_e_drop", 0.0))
    mean_fp = float(ps.get("mean_f0_p10", 0.0))
    out[55] = mean_ed
    out[56] = float(out[6] - mean_ed) if n_ps > 0 else 0.0
    out[57] = mean_fp
    out[58] = float(out[38] - mean_fp) if n_ps > 0 else 0.0
    # interactions: late-in-turn × finality (helps demote mid-turn "fake finals")
    out[59] = float(pause_index) * float(out[6])
    out[60] = float(pause_index) * float(out[36])
    return out


def boundary_stats_update(prior_stats, feat_vec):
    """Update running prior-pause boundary means AFTER emitting this pause's feats."""
    if prior_stats is None:
        prior_stats = {}
    n = int(prior_stats.get("n", 0))
    ed = float(feat_vec[6])
    fp = float(feat_vec[38]) if not np.isnan(feat_vec[38]) else 0.0
    mean_ed = float(prior_stats.get("mean_e_drop", 0.0))
    mean_fp = float(prior_stats.get("mean_f0_p10", 0.0))
    prior_stats["mean_e_drop"] = (mean_ed * n + ed) / (n + 1)
    prior_stats["mean_f0_p10"] = (mean_fp * n + fp) / (n + 1)
    prior_stats["n"] = n + 1
    return prior_stats


def _voiced_runs(voiced, t, last_sp, max_gap=2):
    """Durations (s) of contiguous voiced runs up to last_sp (small gaps bridged)."""
    v = voiced[:last_sp + 1].astype(int)
    if v.sum() == 0:
        return []
    runs = []
    i = 0
    n = len(v)
    while i < n:
        if v[i]:
            j = i
            gap = 0
            end = i
            while j + 1 < n:
                if v[j + 1]:
                    end = j + 1
                    gap = 0
                else:
                    gap += 1
                    if gap > max_gap:
                        break
                j += 1
            runs.append(float(t[end] - t[i]) + HOP_S)
            i = end + 1
        else:
            i += 1
    return runs


def _fill_context(out, pause_start, pause_index, prior_pause_durs, prev_pause_end,
                  prior_stats=None):
    out[25] = float(pause_start)
    out[26] = float(pause_index)
    out[27] = float(np.mean(prior_pause_durs)) if len(prior_pause_durs) else 0.0
    out[28] = float(pause_start - prev_pause_end) if prev_pause_end is not None else float(pause_start)
    out[29] = 0.0
    out[30] = 0.0
    pdurs = list(prior_pause_durs)
    out[51] = float(max(pdurs)) if pdurs else 0.0
    out[52] = float(pdurs[-1]) if pdurs else 0.0
    out[53] = float(np.std(pdurs)) if len(pdurs) >= 2 else 0.0
    out[54] = float(len(pdurs))
    ps = prior_stats or {}
    out[55] = float(ps.get("mean_e_drop", 0.0))
    out[56] = 0.0
    out[57] = float(ps.get("mean_f0_p10", 0.0))
    out[58] = 0.0
    out[59] = float(pause_index) * 0.0
    out[60] = float(pause_index) * 0.0
