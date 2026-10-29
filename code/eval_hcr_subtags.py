# code/eval_hcr_subtags.py
"""
Evaluation script for HCR multi-label subtagging.

- Loads the labeled HCR dataset (hcr.csv)
- Computes Sentence-BERT embeddings
- Uses the trained RandomForest MultiOutput model
- Converts probabilities to binary predictions using a threshold
- Reports per-tag Precision/Recall/F1 and macro averages

Default threshold = TAU_TAG from config.py (matches runtime surfacing),
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support

from config import TAU_TAG  # runtime surfacing threshold

ROOT = Path(__file__).resolve().parents[1]  # project root
HCR_CSV = ROOT / "datasets" / "promise" / "hcr.csv"
MODEL_PATH = ROOT / "models" / "hcr_model.pkl"

SUBTAGS = ["AC", "US", "FA", "EX", "TR", "WB", "IN"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=TAU_TAG,
                        help=f"Probability threshold for predicting a tag (default: TAU_TAG={TAU_TAG}).")
    args = parser.parse_args()
    thr = args.threshold

    df = pd.read_csv(HCR_CSV)
    y_true = df[SUBTAGS].values.astype(int)

    # embeddings (normalized for consistency with training)
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    X = sbert.encode(df["RequirementText"].astype(str).tolist(),
                     convert_to_numpy=True, normalize_embeddings=True)

    model = joblib.load(MODEL_PATH)

    # predict probabilities per tag (positive class)
    probs = np.zeros((len(df), len(SUBTAGS)), dtype=float)
    for j, est in enumerate(model.estimators_):
        probs[:, j] = est.predict_proba(X)[:, 1]

    y_pred = (probs >= thr).astype(int)

    print(f"Per-tag metrics (threshold={thr:.2f}):")
    per_tag = []
    for j, tag in enumerate(SUBTAGS):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[:, j], y_pred[:, j], average="binary", zero_division=0
        )
        per_tag.append((p, r, f1))
        print(f"{tag}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    per_tag = np.array(per_tag)
    print("\nMacro avg across tags:")
    print(f"P={per_tag[:,0].mean():.3f}  R={per_tag[:,1].mean():.3f}  F1={per_tag[:,2].mean():.3f}")

if __name__ == "__main__":
    main()
