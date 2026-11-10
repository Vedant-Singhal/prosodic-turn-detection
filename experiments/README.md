# experiments/ — the research trail

Scratch scripts and raw logs kept for auditability. Each ran from the repo root
(`python experiments/<file>` won't work as-is; they imported `eot_features` from `.`).
Chronological — matches the run numbers in `../RUNLOG.md`.

| Stage | Files | What it settled |
|---|---|---|
| EDA | `eda.py` | 100 turns/lang, 1 eot per turn (the final pause), hold-duration geometry → "short holds are free" scorer insight |
| Feature/model rounds 1–5 | `exp.py`, `exp2.py`, `exp3.py`, `metrics.py` | 39 causal prosodic features; LR ≫ gradient boosting; AUC-driven feature selection actively hurts the official score (rejected) |
| Hindi listen pass | `hindi_listen_analysis.py` | Ranked the longest Hindi holds for human listening → they *sound* finished (deep F0 falls, energy drops). Redirected all later work to Hindi risk |
| Model bake-off (round 7) | `bakeoff_models.py`, `bakeoff_log.txt`, `bakeoff_results.csv` | 9 mid-complexity models under Hindi-priority grouped CV; Elastic-Net wins, kernels/MLPs lose on the official metric despite better AUC |
| Acoustic completion (round 8) | `hindi_followup.py`, `followup_*` | +4 AR-residual "phrase finished?" features → Eng CV 1177→1134 |
| Temporal model (round 9) | `train_temporal_blend.py`, `temporal_*` | Tiny BiGRU over last 0.8 s of F0/energy ± blend — rejected, never safely beat Hindi baseline |
| "Always 850" audit | `_diag_holds.py`, `_diag_feats.py`, `hindi_push.py` + logs | Adversarial score.py runs (oracle→100, never-fire→1600, silence→850) prove the floor is real, not a pipeline bug; counterfactual zeroing of 5 long holds → 736 ms shows the ranking is the bottleneck |
| Hard-negative push (round 10) | `hindi_push2/3/4.py`, `_fast_ship.py` + logs/oof csvs | ×12 weight on holds >0.8 s unlocks in-sample Hindi 831 < 850 silence floor; honest OOF stays 850 → shipped with both numbers reported |
