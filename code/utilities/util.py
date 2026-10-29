"""
Simple similarity-based requirement classifier (no training):
- Loads a sentence embedding model (SBERT: 'all-MiniLM-L6-v2').
- Defines a few hand-picked example sentences for 4 categories.
- For any input text, we:
  * split it into clauses,
  * embed each clause,
  * compute cosine similarity vs. each category’s examples,
  * pick the category whose examples are most similar (if above a threshold),
  * otherwise return 'UNKNOWN'.

This is fast, transparent, and easy to extend by adding more/better examples.
It is NOT trained ML; it’s a nearest-neighbor style “few-shot” similarity check.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load SBERT once. It turns a sentence into a numeric vector ("embedding")
# where semantically similar sentences end up close to each other.
model = SentenceTransformer("all-MiniLM-L6-v2")

# A tiny "reference library" of example requirements per category.
# You can add/edit these to shape behavior quickly.
category_examples = {
    "Functional": [
        "The system must allow users to log in with a username and password.",
        "A user should be able to reset their password via email link.",
        "The platform must support uploading images to user profiles."
    ],
    "Non-functional → Performance": [
        "The system must respond to any query in under 2 seconds.",
        "Page load times should be minimized for better user experience.",
        "The service should handle 500 requests per second."
    ],
    "Non-functional → Security": [
        "All data must be encrypted in transit and at rest.",
        "The system should implement two-factor authentication.",
        "Access should be restricted to authorized personnel only."
    ],
    "Human-Centric": [
        "The interface must be intuitive for first-time users.",
        "Error messages should clearly explain the issue to the user.",
        "The design should be accessible to visually impaired users."
    ]
}

# Embed the example sentences once at import time (performance optimization).
# Now classification only needs to embed the incoming clause(s).
example_embeddings = {}
for category, examples in category_examples.items():
    example_embeddings[category] = model.encode(examples)


def classify_requirement(requirement: str, threshold: float = 0.55):
    """
    Classify a raw requirement string into one of the predefined categories.

    Steps:
      1) Split the text into smaller clauses (easier to tag short, atomic statements).
      2) For each clause:
         - Get its SBERT embedding (vector).
         - For each category, compute cosine similarity to that category’s embedded examples.
         - Take the maximum similarity for each category.
         - Pick the category with the highest max similarity overall.
         - If that best similarity >= threshold, accept it; else label as 'UNKNOWN'.

    Args:
      requirement: raw text (possibly multiple sentences joined with 'and', ';', etc.)
      threshold: how similar a clause must be to some category to accept it
                 (higher threshold = more conservative; lower = more permissive)

    Returns:
      List[str] — a list of category labels, one per clause (in the same order).
    """
    categories = []
    clauses = split_into_clauses(requirement)

    for clause in clauses:
        clause_embedding = model.encode([clause])  # shape (1, dim)

        best_category = None
        best_score = -1.0

        # Compare this clause to each category's examples
        for category, embeddings in example_embeddings.items():
            # cosine_similarity returns a 1 x N array (clause vs N examples)
            sim_scores = cosine_similarity(clause_embedding, embeddings)[0]
            max_score = float(np.max(sim_scores))

            if max_score > best_score:
                best_score = max_score
                best_category = category

        # Apply a minimum similarity threshold to avoid wild guesses
        if best_score >= threshold:
            categories.append(best_category)
        else:
            categories.append("UNKNOWN")

    return categories


def split_into_clauses(text: str):
    """
    Split text into smaller, easier-to-classify chunks.
    - First split on punctuation like '.' and ';'
    - Then split on common coordinators like 'and', 'but', 'or', etc.
    - Trim whitespace and drop empties.

    Note: This is intentionally simple. It trades linguistic perfection for
    robustness and speed during live dictation.
    """
    clauses = re.split(r'[.;]', text)
    refined = []
    for clause in clauses:
        subclauses = re.split(r'\b(?:and|but|or|so|yet|also|while also|while)\b', clause)
        refined.extend(sub.strip() for sub in subclauses if sub.strip())
    return refined
