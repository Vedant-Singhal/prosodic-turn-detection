# NOTES

*(Design rationale and per-run history are in RUNLOG.md; this file is the required ≤10-sentence summary.)*

**Signal:** for each pause the model reads 43 strictly-causal features from the speech just before it — 39 prosodic summaries (pitch fall/rise vs the speaker's own range, energy fade, terminal voicing, creak, spectral tilt, final-syllable lengthening) plus 4 acoustic-completion proxies (AR-residual "surprise" on the last energy/F0 frames, a legal stand-in for "is this phrase finished?" without ASR).
An Elastic-Net logistic (C=0.5) with hard-negative weights ×12 on holds longer than 0.8 s beat plain L2-LR, RBF-SVM, small MLPs and shallow trees under Hindi-priority grouped CV, because those alternatives won AUC but lost the official delay metric.
The whole approach follows one scorer fact — a false cutoff needs `delay < dur`, so short holds are free and the model only has to rank true endings above *long* holds, which is why we optimise the official sweep and never raw AUC.
Result: English 1600→1015 ms official (honest GroupKFold OOF 1248) and Hindi 850→831 ms official, though honest OOF Hindi stays at the 850 ms silence floor, and we report both numbers rather than only the flattering one.

**Where it fails:** Hindi's ~9 longest holds are prosodically complete clauses followed by a thinking gap — deep final F0 falls and large energy drops that sound *more* finished than many true endings — so acoustic-only ranking cannot safely demote them fold-to-fold.
We showed this is a real ceiling, not a bug: adversarial `score.py` runs (oracle→100 ms, never-fire→1600 ms, silence→850 ms) confirm the scorer is live, and counterfactually zeroing just those 5 longest-hold scores drops the official score to 736 ms, so the bottleneck is the ranking of 5 pauses.
A tiny BiGRU over the last 0.8 s of pitch/energy was tried as a temporal alternative but never safely beat the Hindi baseline under grouped CV, so the ship stays tabular.

**One more day:** add per-language calibration on top of the single joint model, which was the cheapest un-taken win.
Telling "trailing off to think" from "trailing off because done" ultimately needs the words, and with ASR banned the next legal lever is phone-rate / syllable-nuclei dynamics over a longer context window.
Longer term, collect Hindi turns with deliberate mid-clause vs final pauses so a small temporal model has enough signal to learn that distinction.
