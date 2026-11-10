"""Quick Hindi long-hold vs EOT ranking diagnostic."""
import numpy as np
import pandas as pd
from train_model import get_lang, official_score_g, auc

Xe, ye, ge, de = get_lang("english")
Xh, yh, gh, dh = get_lang("hindi")
print("shapes", Xe.shape, Xh.shape)

ph = pd.read_csv("predictions_hindi.csv")
lab = pd.read_csv("eot_handout/eot_data/hindi/labels.csv")
lab["dur"] = lab.pause_end - lab.pause_start
m = ph.merge(lab, on=["turn_id", "pause_index"])
m["dur"] = m.pause_end - m.pause_start

lh = m[(m.label == "hold") & (m.dur > 0.8)].sort_values("p_eot", ascending=False)
eot = m[m.label == "eot"]
print("long holds by p:")
print(lh[["turn_id", "pause_index", "dur", "p_eot"]].to_string())
print("eot p pct", np.percentile(eot.p_eot, [10, 25, 50, 75, 90]))
print("longhold p pct", np.percentile(lh.p_eot, [10, 25, 50, 75, 90]))

y = np.concatenate([np.ones(len(eot)), np.zeros(len(lh))])
s = np.concatenate([eot.p_eot.values, lh.p_eot.values])
order = np.argsort(s)
ranks = np.empty(len(s))
ranks[order] = np.arange(1, len(s) + 1)
n1, n0 = y.sum(), len(y) - y.sum()
print("AUC eot vs longhold", (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# how many EOTs sit below each long hold?
for _, r in lh.iterrows():
    n_below = (eot.p_eot < r.p_eot).sum()
    print(f"  {r.turn_id} p={r.p_eot:.3f} dur={r.dur:.2f} EOTs_below={n_below}")

# counterfactual: zero top-k long holds
p = m.p_eot.values.copy()
yarr = (m.label == "eot").astype(int).values
durs = m.dur.values
groups = m.turn_id.values
for k in [3, 5, 7, 9]:
    idx = lh.index[:k]
    p2 = p.copy()
    # map via turn_id,pause_index
    mask = m.set_index(["turn_id", "pause_index"]).index.isin(
        lh.head(k).set_index(["turn_id", "pause_index"]).index
    )
    # simpler: zero by position in m matching lh head
    keys = set(zip(lh.head(k).turn_id, lh.head(k).pause_index))
    for i, row in m.iterrows():
        if (row.turn_id, row.pause_index) in keys:
            p2[m.index.get_loc(i)] = 0.0
    sc = official_score_g(groups, yarr, durs, p2)
    print(f"zero top-{k} longholds -> {sc[0]*1000:.0f} ms th={sc[2]} d={sc[3]}")

# shipped official
sc0 = official_score_g(groups, yarr, durs, p)
print(f"shipped -> {sc0[0]*1000:.0f} ms th={sc0[2]} d={sc0[3]}")
