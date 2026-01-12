"""
classifier_frnfr.py
-------------------
Functional / Non-Functional Requirements (FR/NFR) classifier using DistilBERT.

Loads a fine-tuned DistilBERT model from /models/distilbert_frnfr and provides
a helper function `frnfr_predict(text)` that returns the predicted label and
confidence score.
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]          # RESEARCHTOOL/
MODEL_DIR = ROOT / "models" / "distilbert_frnfr"

# --- Globals for lazy-loading ---
_tokenizer: AutoTokenizer = None
_model: AutoModelForSequenceClassification = None


def _load():
    """
    Load the tokenizer and model into memory if not already loaded.
    This avoids reloading the large model on every prediction.
    """
    global _tokenizer, _model

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"""
            ❌ DistilBERT model folder not found at: {MODEL_DIR}
            Train it first (python code/train_frnfr.py)
            or set FRNFR_BACKEND='svm' in code/config.py.
            """
        )

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()


def frnfr_predict(text: str) -> tuple[str, float]:
    """
    Classify a single requirement text as Functional (FR) or Non-Functional (NFR).

    Args:
        text (str): Requirement statement to classify.

    Returns:
        tuple[str, float]: (Predicted label, confidence score)
    """
    _load()

    # Tokenize text
    inputs = _tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=256)

    # Inference (no gradients needed) --> doesn't learn during inference
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Label mapping (0 = FR, 1 = NFR)
    idx = int(probs.argmax())
    label = "FR" if idx == 0 else "NFR"
    conf = float(probs[idx])

    return label, conf
