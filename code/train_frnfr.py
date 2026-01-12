"""
train_frnfr.py
--------------
Train a DistilBERT-based classifier to distinguish Functional (FR) vs. 
Non-Functional (NFR) software requirements.

Model output:
    ./models/distilbert_frnfr/

Dataset used:
    ./datasets/promise/Promise+_cleaned.csv
"""

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
DATA = "./datasets/promise/Promise+_cleaned.csv" # Input CSV with requirement text and labels
OUT_DIR   = "./models/distilbert_frnfr"                 # Folder to save trained model
MODEL_NAME = "distilbert-base-uncased"                  # Base pretrained model
MAX_LEN = 256                                           # Max token length for truncation/padding
BATCH_TRAIN = 16                                        # Training batch size
BATCH_EVAL  = 32                                        # Evaluation batch size
EPOCHS = 3                                              # Number of training epochs

# ---------------------------------------------------------------------
# 1) LOAD AND PREPARE DATA
# ---------------------------------------------------------------------
df = pd.read_csv(DATA)
df["RequirementText"] = df["RequirementText"].astype(str)
df["_class_"] = df["_class_"].astype(str)

def collapse(label: str) -> str:
    """
    Normalize original labels to either 'FR' or 'NFR'.
    (Handles variations like 'F', 'Functional', etc.)
    """
    u = label.strip().upper()
    return "FR" if u in {"F", "FR", "FUNCTIONAL"} else "NFR"

# Map text labels → numeric (FR=0, NFR=1)
df["label"] = df["_class_"].map(collapse).map({"FR": 0, "NFR": 1})

# 80/20 random train-test split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
train_df, test_df = df.iloc[:split], df.iloc[split:]

# ---------------------------------------------------------------------
# 2) TOKENIZATION AND DATASET CREATION
# ---------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_texts(texts):
    """Tokenize a list of texts into input IDs + attention masks."""
    return tok(texts, truncation=True, padding="max_length", max_length=MAX_LEN)

# Convert to Hugging Face Datasets format
train_ds = Dataset.from_pandas(
    train_df[["RequirementText", "label"]].rename(columns={"RequirementText": "text"}),
    preserve_index=False
)
test_ds = Dataset.from_pandas(
    test_df[["RequirementText", "label"]].rename(columns={"RequirementText": "text"}),
    preserve_index=False
)

# Apply tokenization to both sets
train_ds = train_ds.map(lambda b: tokenize_texts(b["text"]), batched=True)
test_ds  = test_ds.map(lambda b: tokenize_texts(b["text"]), batched=True)

# Rename for Trainer compatibility
train_ds = train_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")

# Set PyTorch format
cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=cols)
test_ds.set_format(type="torch", columns=cols)

# ---------------------------------------------------------------------
# 3) INITIALIZE MODEL AND TRAINING ARGUMENTS
# ---------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_steps=500,             # Periodic checkpoints
    save_total_limit=1,         # Keep only the latest checkpoint
)

# Build Trainer (evaluation handled manually below for simplicity)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
)

# ---------------------------------------------------------------------
# 4) TRAIN THE MODEL
# ---------------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------------
# 5) MANUAL EVALUATION (ACCURACY + F1)
# ---------------------------------------------------------------------
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loader = DataLoader(test_ds, batch_size=BATCH_EVAL)
probs_all, labels_all = [], []

# Disable gradient computation during inference
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(batch["labels"].cpu().numpy())

probs_all = np.concatenate(probs_all, axis=0)
labels_all = np.concatenate(labels_all, axis=0)
preds = probs_all.argmax(axis=1)

metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

acc = metric_acc.compute(predictions=preds, references=labels_all)["accuracy"]
f1  = metric_f1.compute(predictions=preds, references=labels_all, average="weighted")["f1"]

print({"accuracy": acc, "f1": f1})

# ---------------------------------------------------------------------
# 6) SAVE MODEL + TOKENIZER
# ---------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print("Saved model + tokenizer to", OUT_DIR)
