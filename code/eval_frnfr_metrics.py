# code/eval_frnfr_metrics.py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA = "./datasets/promise/Promise+_cleaned.csv"
MODEL_DIR = "./models/distilbert_frnfr"
MAX_LEN = 256
BATCH = 32
SEED = 42

def collapse(label: str) -> int:
    u = str(label).strip().upper()
    # Match your training rule:
    # FR if label is F/FR/FUNCTIONAL; otherwise treat as NFR
    return 0 if u in {"F", "FR", "FUNCTIONAL"} else 1  # 0=FR, 1=NFR

def main():
    df = pd.read_csv(DATA)
    df["RequirementText"] = df["RequirementText"].astype(str)
    df["_class_"] = df["_class_"].astype(str)
    df["label"] = df["_class_"].map(collapse)

    # Recreate the same shuffle + split as training
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split = int(0.8 * len(df))
    test_df = df.iloc[split:].copy()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = test_df["RequirementText"].tolist()
    y_true = test_df["label"].to_numpy()

    y_pred = []
    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i:i+BATCH]
        enc = tok(batch_texts, return_tensors="pt", truncation=True,
                  padding=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.append(preds)

    y_pred = np.concatenate(y_pred, axis=0)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(acc, 4))
    print("\nClassification report (FR=0, NFR=1):")
    print(classification_report(y_true, y_pred, target_names=["FR", "NFR"], digits=4))

if __name__ == "__main__":
    main()
