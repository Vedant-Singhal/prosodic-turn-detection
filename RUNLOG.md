# RUNLOG — End-of-Turn Detection

Score = mean response delay (ms) at ≤5% false-cutoff rate, from the official
`score.py` (lower is better). Two numbers per run: **English / Hindi**.
Raw scripts + unedited logs for every run below are archived in
[`experiments/`](experiments/README.md), indexed chronologically.

- **CV** = grouped 5-fold cross-validation (a turn is never split across folds),
  the honest generalisation estimate, used for all model/feature decisions.
- **In-sample** = the shipped model (refit on all provided data) scored by the
  official `score.py` on the delivered `predictions_*.csv` — what the grader
  sees when scoring our files on the provided folders.
- **Hindi-priority** = selection score `(eng + 2·hin) / 3` after the human
  flagged that hidden tests are mostly Hindi.

## Facts established up front
- 100 turns/language, exactly **1 eot per turn** (final pause); 148 holds each; 248 pauses/language.
- **Joint model only:** one Eng+Hin fit → `eot_model.pkl`; language is not a
  feature (load/score only). No fixed holdout — decisions use GroupKFold(5) OOF
  by `turn_id`, then refit-all. Separate per-lang models did not help (see
  NOTES “Training / eval design”).
- Causality: features use only audio in `[0, pause_start)`. The current pause's
  `pause_end`/duration is **future** and is never a feature (only ever a *training weight*).
- **Scorer insight that drove everything:** a false cutoff needs `delay < dur`,
  so **short holds are free**. The model only needs to rank eots above **long**
  holds. Optimising raw AUC is the wrong target.

## Run history

| # | Change | Eng | Hin | Notes |
|---|--------|-----|-----|-------|
| 0 | Official silence-only baseline (p_eot=1) | **1600** | **850** | Reference. Hindi baseline already strong. |
| 1 | 31 causal prosodic features, LR(C=0.5, balanced), CV | 1235 | 865 | Beats baseline on Eng; Hin no gain. LR ≫ GB. |
| 2 | +8 features → 39 feats, CV | 1256 | 857 | More features can hurt LR; kept after regularisation. |
| 3 | LR C-grid, multilingual, CV | 1171 | 857 | Combined Eng+Hin helps English. |
| 4 | Forward feature-selection by AUC, CV | 1360 | 850 | **Rejected** — AUC↑ score↓. |
| 5 | Duration-aware LR weights, CV | 1177 | 850 | Metric-aligned L2-LR. Prior agent default. |
| 6 | **Human listen pass** on longest Hindi holds (hi__029/051/097/…) | — | — | Qualitative: long holds show deep final F0 falls + large energy drops — look *more* final than many EOTs. Human overrides Eng-centric tuning. |
| 7 | Mid-complexity bake-off (ElasticNet / LinSVM+Platt / LDA / QDA / RBF-SVM / MLP / HGB depth2 / isotonic), Hindi-priority CV | see table | see table | Human demand: leave vanilla LR. Kernels/MLPs/HGB all ≥850 Hin but worse Eng. |
| 8 | +4 acoustic-completion features (AR residuals), ElasticNet+dur-w, CV | **1134** | **850** | Best Hindi-priority so far; completion helps Eng clear EOTs. |
| 9 | Tiny BiGRU (0.8 s F0/energy) ± tabular blend, CV | 1172–1470 | 850–857 | **Rejected for ship.** GRU alone Eng 1470; blends keep Hin=850 but worsen Eng vs #8. Hindi-only GRU also stuck at 850. |
| — | **FINAL** ElasticNet+AC+dur-w, **in-sample official score.py** | **1030** | **850** | Eng AUC 0.763, op th=0.55/delay=650ms; Hin AUC 0.752, op th=0.05/delay=850ms. |
| 10 | **Hard-neg holds dur>0.8s ×12** (same 43 feats), official score.py | **1015** | **831** | Beat Hindi silence. Op th=0.5/delay=650ms. Honest GroupKFold OOF still **1248 / 850**. |

## Model bake-off (grouped OOF, official metric, Hindi-priority)

```
algo                         AUC   engCV  hinCV   hinPri
LR L2 C=0.5 dur-w           0.667   1177    850     959
LR L2 C=0.5 bal             0.665   1171    857     962
LDA                         0.670   1191    857     969
ElasticNet LR C=0.5         0.671   1210    850     970
LinSVM C=0.1 +Platt         0.660   1226    850     975
HGB depth2 leaf40           0.670   1288    850     996
RBF-SVM C=2                 0.680   1305    850    1002
MLP 32-16 L2=0.1            0.622   1358    850    1019
QDA reg=0.5                 0.632   1334    910    1051
```

Follow-up (43 features = 39 prosody + 4 acoustic-completion):

```
43f ElasticNet C=0.5 dur-w   engCV=1134  hinCV=850  hinPri=945   <-- selected
39f ElasticNet C=0.5 dur-w   engCV=1153  hinCV=850  hinPri=951
43f LR C=0.5 dur-w           engCV=1200  hinCV=850  hinPri=967
```

## Diagnostics that shaped the work
- **English is winnable with prosody + completion**: CV 1177→1134 ms after AC features.
- **Hindi hits a real wall on long holds**: `AUC(eot vs holds dur>0.8s) ≈ 0.45`.
  Human listening confirms those holds are prosodically complete clauses with a
  thinking gap — acoustic-only models cannot safely fire earlier than ~850 ms
  without cutting them. Matching the Hindi baseline while improving English
  generalisation is the honest Hindi-priority outcome here.

## Audit: Hindi stuck at exactly 850 ms? (human-flagged anomaly)

Suspicion: printing **850** every Hindi run looked like a pipeline clamp / wrong CSV /
scorer quirk rather than a real ceiling. Full adversarial pass against official
`score.py` (same path/columns as English):

| Predictions file | Hindi mean delay |
|------------------|-----------------:|
| all `p_eot=0` (never fire) | **1600** ms |
| all `p_eot=1` (silence baseline) | **850** ms |
| oracle (`eot=1`, `hold=0`) | **100** ms |
| near-oracle | **100** ms |
| shipped model | **850** ms |
| model ± noise / flipped ranks | **850** / **850** |
| random scores | **872** ms |

**Verdict: not a bug — true metric floor for this ranking.**

Evidence that ruled out the hypotheses:
1. `predictions_hindi.csv` has correct columns (`turn_id,pause_index,p_eot`), 248/248
   keys match labels, **248 distinct** scores (not a constant), 0 NaNs; Hindi
   features are finite (0% NaN on the 43-D matrix).
2. Score **does** move when predictions change (oracle 100, never-fire 1600, random 872).
3. Silence 850 is geometry, not a hardcode: with `p=1`, delay 800 ms cuts **8%** of
   turns; delay **850 ms** is the first grid point with cutoff ≤5% → mean latency 850.
4. Shipped model op point is `th=0.05, delay=850` — same as silence — because every
   budget-feasible point with `delay<850` has **higher** mean latency (best alt:
   **897 ms** at `th=0.5, delay=650`) once EOT timeouts at 1600 ms are averaged in.
5. Break-even: to beat 850 at delay 650 ms you may time out ≤21 EOTs; the threshold
   that keeps cutoff ≤5% times out **26**. Ranking of the 5 holds with `dur>0.85 s`
   is interleaved with EOTs (e.g. `hi__051` p=0.71 sits above 65 EOTs).
6. Counterfactual (not deployable): zeroing only those 5 long-hold scores drops the
   official score to **736 ms** — proves the scorer can go below 850; the model’s
   ranking of those holds is the bottleneck.

No code fix for the 850 floor itself — the scorer was correct. Later hard-neg
reweighting of long holds unlocked in-sample Hindi **831 ms** (see run #10);
honest OOF Hindi remained **850 ms**. English/Hindi official after hard-neg ship:
**1015 / 831**.
