"""Compare Hindi EOT vs long-hold feature means; find separating dims."""
import numpy as np
import pandas as pd
from train_model import get_lang
from eot_features import FEATURE_NAMES

Xh, yh, gh, dh = get_lang("hindi")
eot = yh == 1
lh = (yh == 0) & (dh > 0.8)
print(f"eot={eot.sum()} longholds={lh.sum()}")

rows = []
for i, name in enumerate(FEATURE_NAMES):
    a = Xh[eot, i]
    b = Xh[lh, i]
    # Welch-ish effect size
    ma, mb = np.nanmean(a), np.nanmean(b)
    sa, sb = np.nanstd(a) + 1e-9, np.nanstd(b) + 1e-9
    d = (ma - mb) / np.sqrt(0.5 * (sa**2 + sb**2))
    # univariate AUC eot > hold
    s = Xh[:, i]
    # handle nans
    mask = eot | lh
    yy = yh[mask]
    ss = s[mask]
    ok = ~np.isnan(ss)
    yy, ss = yy[ok], ss[ok]
    order = np.argsort(ss)
    ranks = np.empty(len(ss)); ranks[order] = np.arange(1, len(ss)+1)
    n1, n0 = yy.sum(), len(yy)-yy.sum()
    au = (ranks[yy==1].sum() - n1*(n1+1)/2) / (n1*n0) if n1 and n0 else np.nan
    rows.append((name, d, au, ma, mb))

rows.sort(key=lambda r: -abs(r[1]))
print("top |effect| EOT-longhold:")
for name, d, au, ma, mb in rows[:20]:
    print(f"  {name:22s} d={d:+.3f} AUC={au:.3f} eot={ma:.3f} lh={mb:.3f}")

rows.sort(key=lambda r: -max(r[2], 1-r[2]) if r[2]==r[2] else -0.5)
print("\ntop |AUC-0.5|:")
for name, d, au, ma, mb in sorted(rows, key=lambda r: -abs(r[2]-0.5))[:15]:
    print(f"  {name:22s} AUC={au:.3f} d={d:+.3f}")

# pause_index of long holds vs eots
print("\npause_index longholds:", Xh[lh, FEATURE_NAMES.index("pause_index")])
print("pause_start longholds:", Xh[lh, FEATURE_NAMES.index("pause_start")])
print("mean pause_index eot vs lh:", Xh[eot, FEATURE_NAMES.index("pause_index")].mean(),
      Xh[lh, FEATURE_NAMES.index("pause_index")].mean())
