"""EOT inference — REQUIRED deliverable interface.

    python predict.py --data_dir <folder> --out predictions.csv

Loads a SAVED model (eot_model.pkl, trained by train_model.py) and predicts
p_eot for every pause in <folder>/labels.csv. Works on a folder with the same
structure/labels schema it has never seen. Output columns: turn_id,
pause_index, p_eot.

Causality: features come only from audio in [0, pause_start) (see
eot_features.py). No pretrained weights are used — the model is a small
scikit-learn classifier trained from scratch on the provided data.
"""
import os, argparse, csv
import joblib
import numpy as np

from eot_data import build_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=os.path.join(HERE, "eot_model.pkl"))
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    feat_names = bundle.get("feature_names")
    n_feat = len(feat_names) if feat_names is not None else None

    X, _, groups, keys, _ = build_dataset(args.data_dir)
    if n_feat is not None and X.shape[1] != n_feat:
        X = X[:, :n_feat]
    p = pipe.predict_proba(X)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pidx), pi in zip(keys, p):
            w.writerow([tid, pidx, f"{pi:.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
