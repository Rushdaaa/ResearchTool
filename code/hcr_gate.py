# code/hcr_gate.py
# PURPOSE: Provides a fast binary gate to determine if a requirement text is human-centric
# This acts as a preprocessing filter before running expensive multi-label HCR classification
# Uses the trained model from train_hcr_gate.py to make quick yes/no decisions

import joblib
from functools import lru_cache
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
ROOT = Path(__file__).resolve().parents[1]          # RESEARCHTOOL/
MODEL_PATH = ROOT / "models" / "is_hcr.pkl"

@lru_cache(maxsize=1)
def _sbert():
    """
    Lazy-loaded SentenceTransformer model with caching
    
    Returns:
        SentenceTransformer: Pre-trained model for converting text to embeddings
        
    Notes:
        - Uses lru_cache to ensure model is only loaded once per session
        - Same model used in training (all-MiniLM-L6-v2) for consistency
        - 384-dimensional embeddings optimized for semantic similarity
    """
    return SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def _gate():
    """
    Lazy-loaded gate classifier with caching and error handling
    
    Returns:
        sklearn.LogisticRegression: Trained binary classifier for HCR detection
        
    Raises:
        FileNotFoundError: If the gate model hasn't been trained yet
        
    Notes:
        - Uses lru_cache to load model only once per session
        - Model expects normalized 384-dimensional embeddings as input
        - Returns probabilities for [not_hcr, is_hcr] classes
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing HCR gate model at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def hcr_gate_probability(text: str) -> float:
    """
    Calculate probability that a requirement text is human-centric
    
    Args:
        text (str): Requirement text to analyze
        
    Returns:
        float: Probability between 0.0 and 1.0 that text is human-centric
               Higher values = more likely to be human-centric
               
    Process:
        1. Convert text to semantic embedding using SentenceTransformer
        2. Normalize embedding (consistent with training)
        3. Use trained gate classifier to predict probability
        4. Return probability of positive class (is_hcr==1)
        
    Usage:
        prob = hcr_gate_probability("The system shall ensure user privacy")
        if prob > 0.53:
            # Likely human-centric, proceed with detailed classification
        else:
            # Likely technical/CRUD, skip expensive processing
    """
    # Convert text to normalized embedding (same preprocessing as training)
    emb = _sbert().encode([text], convert_to_numpy=True, normalize_embeddings=True)
    
    # Get probability predictions: [P(not_hcr), P(is_hcr)]
    p = _gate().predict_proba(emb)[0][1]  # Extract P(is_hcr==1)
    
    return float(p)  # Convert numpy float to Python float
