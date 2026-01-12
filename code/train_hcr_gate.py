"""
train_is_hcr.py
---------------
Train a binary classifier ("Human-Centric Gate") that predicts whether a
requirement text is *human-centric* (HCR) or *non-human-centric*.

Purpose:
    Acts as a lightweight gate before running expensive multi-label
    HCR subtag classification.

Model output:
    ./models/is_hcr.pkl    (Logistic Regression trained on SBERT embeddings)

Datasets used:
    ./datasets/promise/hcr.csv
    ./datasets/promise/Promise+_cleaned.csv
"""

import re
import random
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
HCR_CSV      = "./datasets/promise/hcr.csv"               # Positive (human-centric) examples
PROMISE_CSV  = "./datasets/promise/Promise+_cleaned.csv"  # General requirements (negative pool)
MODEL_PATH   = "./models/is_hcr.pkl"                      # Output model path

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# 1) LOAD DATASETS
# ---------------------------------------------------------------------
hcr  = pd.read_csv(HCR_CSV)       # ~600 labeled human-centric requirements
prom = pd.read_csv(PROMISE_CSV)   # ~3,600 general software requirements

# Ensure text columns are valid strings (handle NaN)
hcr["RequirementText"]  = hcr["RequirementText"].astype(str)
prom["RequirementText"] = prom["RequirementText"].astype(str)

# ---------------------------------------------------------------------
# 2) PREPARE POSITIVE EXAMPLES
# ---------------------------------------------------------------------
pos_texts  = hcr["RequirementText"].dropna().tolist()  # Get all HCR requirement texts
pos_labels = [1] * len(pos_texts)                      # Label them as 1 (human-centric)

# ---------------------------------------------------------------------
# 3) HEURISTIC NEGATIVE SAMPLING
# ---------------------------------------------------------------------
# Define words that commonly signal human-centricity
hc_keywords = r"(fair|bias|equity|inclusive|accessib|explain|transparen|trust|" \
              r"accountab|well[- ]?being|safety|usabil|learnabil|privacy|" \
              r"discriminat|harm|culture)"

def looks_non_hcr(s: str) -> bool:
    """
    Heuristic filter for likely *non-human-centric* requirements.

    Returns True if:
        - Text lacks any human-centric keywords, AND
        - It contains CRUD-style technical verbs (login, update, API, etc.), OR
        - Very short and task-oriented.
    """
    # First check: if it contains HCR keywords, it's likely human-centric
    if re.search(hc_keywords, s, flags=re.I): 
        return False
    
    s_low = s.lower()
    # Look for common CRUD/technical operation keywords
    crudish = any(w in s_low for w in [
        "login","log in","authenticate","export","import","generate report",
        "create","read","update","delete","save","load","upload","download",
        "connect","sync","filter","paginate","sort","render","click","button",
        "field","form","page","screen","api","endpoint","database","table",
        "csv","json","xml","file","token","password"
    ])
    # Also consider very short requirements as likely non-HCR
    return crudish or len(s.split()) <= 14

# Create negative examples using a tiered filtering approach
pool = prom["RequirementText"].tolist()  # All Promise+ requirements

# Tier 1: Requirements that look technical/CRUD and don't have HCR keywords
cand1 = [t for t in pool if looks_non_hcr(t)]

# Tier 2: If not enough candidates, just exclude HCR keywords (less strict)
cand2 = cand1 if len(cand1) >= 200 else [t for t in pool if not re.search(hc_keywords, t, flags=re.I)]

# Tier 3: If still not enough, use all requirements (fallback)
cand3 = cand2 if len(cand2) >= 100 else pool

# Randomly shuffle candidates and select negatives
random.shuffle(cand3)
n_pos = len(pos_texts)
target_neg = max(300, n_pos)  # Want at least 300 negatives, or match positive count
# Select the final negative examples and create labels
neg_texts = cand3[: min(target_neg, len(cand3))]  # Take up to target_neg requirements
neg_labels = [0] * len(neg_texts)                 # Label them as 0 (not human-centric)

# ---------------------------------------------------------------------
# 4) COMBINE DATASETS
# ---------------------------------------------------------------------
print(f"Positives: {len(pos_texts)} | Negatives: {len(neg_texts)} | Total: {len(pos_texts) + len(neg_texts)}")

if len(neg_texts) == 0:
    raise RuntimeError("No negatives found; relax filters or check PROMISE path/columns.")

# ---------------------------------------------------------------------
# 5) FEATURE EXTRACTION (SBERT EMBEDDINGS)
# ---------------------------------------------------------------------
sbert = SentenceTransformer("all-MiniLM-L6-v2")  # Pre-trained model SentenceTransformer provides 384-dimensional semantic embeddings
X_texts = pos_texts + neg_texts    # Combine all requirement texts
y       = pos_labels + neg_labels  # Combine all labels (1=HCR, 0=not HCR)

# ---------------------------------------------------------------------
# 6) TRAIN / TEST SPLIT AND MODEL TRAINING
# ---------------------------------------------------------------------
X = sbert.encode(X_texts, convert_to_numpy=True, normalize_embeddings=True)

# TRAINING: Split data and train binary classifier
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Use logistic regression with balanced class weights to handle potential class imbalance
clf = LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear")
clf.fit(Xtr, ytr)  # Train the model

# ---------------------------------------------------------------------
# 7) EVALUATE MODEL PERFORMANCE
# ---------------------------------------------------------------------
print(classification_report(yte, clf.predict(Xte)))

# ---------------------------------------------------------------------
# 8) SAVE TRAINED MODEL
# ---------------------------------------------------------------------
joblib.dump(clf, MODEL_PATH)
print(f"Saved trained HCR gate model --> {MODEL_PATH}")

# MODEL PURPOSE: This classifier serves as a "gate" in the requirements processing pipeline
# It quickly filters out obviously non-human-centric requirements before running
# more expensive multi-label classification on the remaining candidates

"""
Notes:
------
• The keyword list above is *not* used at runtime.
  It only guides negative-sample selection for balanced training.

• Once trained, the gate model predicts probabilities for "is human-centric?"
  using only learned patterns from semantic embeddings.

• The model is used by `hcr_gate.py` in inference.
"""