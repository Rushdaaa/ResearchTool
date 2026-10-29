# code/test_models.py
import joblib
from sentence_transformers import SentenceTransformer

# Load models
hcr_model = joblib.load("./models/hcr_model.pkl")
fr_nfr_model = joblib.load("./models/fr_nfr_model.pkl")

# Load SBERT model for embeddings
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample requirement sentences
sample_reqs = [
    "The system should allow users to reset their passwords via email.",
    "The interface must be intuitive and accessible for visually impaired users.",
    "All user data must be encrypted during transmission and storage.",
    "Stress indicators shall trigger automatic interface simplification modes."
]

print("\n--- FR/NFR Classification ---")
for req in sample_reqs:
    X = sbert_model.encode([req])
    pred = fr_nfr_model.predict(X)[0]
    print(f"Requirement: {req}")
    print(f"Predicted Category (FR/NFR): {pred}\n")

print("\n--- HCR Classification ---")
for req in sample_reqs:
    X = sbert_model.encode([req])
    preds = hcr_model.predict(X)[0]  # multi-output prediction
    tags = ["AC","US","FA","EX","TR","WB","IN"]
    hcr_tags = [tag for tag, val in zip(tags, preds) if val == 1]
    print(f"Requirement: {req}")
    print(f"Predicted HCR Tags: {hcr_tags}\n")
