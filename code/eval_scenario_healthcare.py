"""
code/eval_scenario_healthcare.py

Scenario-based evaluation (Healthcare) using real requirements text
(from the CCHIT dataset, selected manually for coverage).

Prints outputs in a UI-like format:
- clause text
- FR/NFR label + confidence
- HCR gate probability
- surfaced HCR tags using TAU_TAG (same as runtime)

Run:
  python code/eval_scenario_healthcare.py
"""

from config import TAU_HCR, TAU_TAG
from classifier_frnfr import frnfr_predict
from hcr_gate import hcr_gate_probability
from classifier_hcr import hcr_subtag_probabilities

# --- Helpers copied from your main_stream style ---
def normalize_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] not in {".", "!", "?"}:
        s += "."
    return s

def split_into_clauses(text: str):
    """
    Conservative splitter:
    - hard sentence split on punctuation
    - soft split on 'and'/'but'
    - keep clauses >= 5 words (otherwise glue back)
    """
    import re
    chunks = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    out = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        parts = re.split(r"\b(?:and|but)\b", ch, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip(",;: ").strip()
            if len(p.split()) >= 5:
                out.append(p)
            else:
                if out:
                    out[-1] = (out[-1] + " " + p).strip()
                else:
                    out.append(p)
    return out or ([text.strip()] if text.strip() else [])

def classify_text_clause(clause: str) -> dict:
    fr_label, fr_conf = frnfr_predict(clause)
    p_hcr = hcr_gate_probability(clause)

    tags = []
    subtag_probs = {}
    if p_hcr >= TAU_HCR:
        subtag_probs = hcr_subtag_probabilities(clause)
        tags = [t for t, p in subtag_probs.items() if p >= TAU_TAG]

    return {
        "text": normalize_sentence(clause),
        "frnfr": {"label": fr_label, "confidence": fr_conf},
        "hcr": {"p_hcr": p_hcr, "tags": tags, "subtag_probs": subtag_probs},
    }

# --- Healthcare scenario utterances (verbatim “shall” statements) ---
HEALTHCARE_UTTERANCES = [
    "The system shall improve accessibility of online clinical information and results.",
    "The system shall have the capability to capture and store risk, social and medical factors; including, but not limited to the following: Tobacco use and history including number of years and packs per day (PPD), Alcohol use history, Drug use history, Occupational environment, Marital status, Occupation, Religious preference, Ethnicity, Healthcare surrogate with alert, Guardian/legal representative with alert.",
    "The system shall support audit capability.",
    "The system shall timeout after a period of inactivity.",
    "The system shall include a clinical reminder on clinical discrimination.",
    "The system shall be easy to use by improving the user interface to reflect industry standards and principles of usability and user-centered design",
    "The system shall use HL-7 communications protocol for transferring messages among packages",
    "1688,The system shall integrate all components of the patient record to provide comprehensive and intelligent clinical information access and reporting.",
    "The system shall provide an option to so a user can quickly set up orders for frequently placed orders.",
    "The system shall provide OB/GYN progress note templates based on the IHS progress note templates.",
    "The system shall ensure secure electronic messaging with patients.",
    "The system shall have the ability to provide a view of orders by like or comparable type, e.g., all radiology or all lab orders.",
    "The system will abort when the user tries to login if tabs are not assigned to the user in their NEW PERSON file or if the RESTRICT PATIENT SELECTION prompt does not contain a Yes or No.",
]

def main():
    print("\n=== Healthcare Scenario Evaluation (CCHIT-derived utterances) ===")
    print(f"Thresholds: TAU_HCR={TAU_HCR:.2f}, TAU_TAG={TAU_TAG:.2f}\n")

    for u_idx, utterance in enumerate(HEALTHCARE_UTTERANCES, 1):
        print(f"🗣️  Utterance {u_idx}: {utterance}")
        clauses = split_into_clauses(utterance)

        for i, clause in enumerate(clauses, 1):
            r = classify_text_clause(clause)
            fr = r["frnfr"]
            h = r["hcr"]

            print(f"  ▸ clause {i}: {r['text']}")
            print(f"      FR/NFR : {fr['label']} (p={fr['confidence']:.2f})")
            print(f"      HCR    : p={h['p_hcr']:.2f} | tags={h['tags']}")

        print("-" * 64)

if __name__ == "__main__":
    main()
