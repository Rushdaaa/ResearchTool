# code/classifier_hcr.py
# PURPOSE: Multi-label classifier for Human-Centric Requirements (HCR) subtag prediction
# Classifies requirements into 7 HCR categories: AC, US, FA, EX, TR, WB, IN
# Uses trained Random Forest models to predict probability of each subtag independently

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from functools import lru_cache
import joblib

# ---------- TRAINING SECTION (run once to create model) ----------
def train_and_save_hcr_model():
    """
    Training function to create and save the HCR multi-label classifier
    
    Process:
        1. Load HCR dataset with 600 labeled requirements
        2. Convert requirement texts to semantic embeddings
        3. Train separate Random Forest for each of 7 HCR subtags
        4. Save trained model for runtime use
        
    HCR Subtags:
        - AC: Accessibility (screen readers, motor impairments, etc.)
        - US: Usability (ease of use, learnability, etc.)
        - FA: Fairness (bias prevention, discrimination, etc.)
        - EX: Explainability (transparency, interpretability, etc.)
        - TR: Trust (reliability, security, privacy, etc.)
        - WB: Well-being (mental health, stress, burnout, etc.)
        - IN: Inclusivity (cultural diversity, language, etc.)
    """
    # Load dataset with 600 human-centric requirements and their subtag labels
    df = pd.read_csv(ROOT / "datasets" / "promise" / "hcr.csv")
    
    # Initialize SentenceTransformer for semantic embeddings
    # Uses same model as gate classifier for consistency
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Convert requirement texts to 384-dimensional semantic vectors
    X = model.encode(df["RequirementText"].tolist())
    
    # Extract binary labels for each HCR subtag (0=not present, 1=present)
    # Each requirement can have multiple subtags simultaneously
    y = df[["AC","US","FA","EX","TR","WB","IN"]].values
    
    # Create multi-output classifier: one Random Forest per subtag
    # Each forest learns to predict one subtag independently
    multi_clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
    multi_clf.fit(X, y)
    
    # Save trained model to disk for runtime inference
    joblib.dump(multi_clf, ROOT / "models" / "hcr_model.pkl")

# Run training if script is executed directly
if __name__ == "__main__":
    train_and_save_hcr_model()

# ---------- RUNTIME INFERENCE SECTION ----------

# Define the 7 HCR subtag categories in order
SUBTAGS = ["AC","US","FA","EX","TR","WB","IN"]

# Paths
ROOT = Path(__file__).resolve().parents[1]          # RESEARCHTOOL/
HCR_MODEL_PATH = ROOT / "models" / "hcr_model.pkl"

@lru_cache(maxsize=1)
def _sbert():
    """
    Lazy-loaded SentenceTransformer with caching
    
    Returns:
        SentenceTransformer: Pre-trained model for text embeddings
        
    Notes:
        - Uses same model as training for consistency
        - Cached to avoid reloading on every prediction
        - 384-dimensional embeddings optimized for semantic tasks
    """
    return SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def _hcr_model():
    """
    Lazy-loaded HCR classifier with error handling
    
    Returns:
        MultiOutputClassifier: Trained model with 7 Random Forest estimators
        
    Raises:
        FileNotFoundError: If model hasn't been trained yet
        
    Notes:
        - Each estimator predicts one HCR subtag independently
        - Supports multi-label prediction (multiple subtags per requirement)
        - Cached to avoid reloading on every prediction
    """
    if not HCR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing HCR subtag model at {HCR_MODEL_PATH}")
    return joblib.load(HCR_MODEL_PATH)

def hcr_subtag_probabilities(text: str) -> dict:
    """
    Predict probability of each HCR subtag for a requirement text
    
    Args:
        text (str): Requirement text to classify
        
    Returns:
        dict: Mapping of subtag names to probabilities (0.0-1.0)
              Higher values = more likely to belong to that category
              
    Example:
        probs = hcr_subtag_probabilities("The system shall provide screen reader support")
        # Returns: {"AC": 0.95, "US": 0.8, "FA": 0.1, "EX": 0.2, "TR": 0.3, "WB": 0.1, "IN": 0.4}
        
    Process:
        1. Convert text to normalized semantic embedding
        2. Run through each of 7 trained Random Forest classifiers
        3. Extract probability of positive class for each subtag
        4. Return probabilities as dictionary
        
    Notes:
        - Each subtag is predicted independently (multi-label classification)
        - Requirements can have multiple high-probability subtags
        - Probabilities don't need to sum to 1.0
    """
    # Convert requirement text to normalized embedding vector
    emb = _sbert().encode([text], convert_to_numpy=True, normalize_embeddings=True)
    
    # Get the trained multi-output model
    model = _hcr_model()
    
    # Initialize results dictionary
    probs = {}
    
    # Iterate through each subtag and its corresponding Random Forest estimator
    for tag, est in zip(SUBTAGS, model.estimators_):
        # Each estimator is a binary classifier with predict_proba method
        # predict_proba returns [P(negative), P(positive)]
        p = est.predict_proba(emb)[0][1]  # Extract P(positive class)
        probs[tag] = float(p)  # Convert numpy float to Python float
    
    return probs
