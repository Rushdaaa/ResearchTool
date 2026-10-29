# code/test_pipeline.py
"""
Hard-coded dataset-driven test for all trained classifiers.
Uses real requirements from PROMISE+ and HCR datasets.
"""

from classifier_frnfr import frnfr_predict
from hcr_gate import hcr_gate_probability
from classifier_hcr import hcr_subtag_probabilities
from config import TAU_HCR, TAU_TAG

print(f"\nTesting models with TAU_HCR={TAU_HCR} and TAU_TAG={TAU_TAG}\n")

# --- PROMISE+: Functional (FR) ---
promise_fr = [
    "The system shall allow the administrator to remove physical locations.",
    "The system shall validate email address existence.",
    "Each repackaged WARC file shall include a user-defined METADATA record.",
]

# --- PROMISE+: Non-Functional (NFR) ---
promise_nfr = [
    "All of the users shall be able to use the system after one day of training.",
    "The interface shall be appealing to callers and supervisors.",
    "98% of the user workstations shall meet the minimum hardware requirements prior to implementation.",
]

# --- HCR: Functional Human-Centric ---
hcr_fr_like = [
    "The system shall allow feedback validation in low-visibility conditions.",
    "Interactive elements shall provide haptic feedback for users with hearing impairments.",
    "Alternative text descriptions shall be generated for all dynamic visual content.",
]

# --- HCR: Non-Functional Human-Centric ---
hcr_nfr_like = [
    "The interface shall provide clear and concise instructions for all user actions.",
    "UI elements shall use culturally inclusive colors, symbols, and language.",
    "All critical AI decisions shall be logged and auditable by authorized personnel.",
]

def run_block(title, texts):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    for s in texts:
        print(f"\n{s}")

        # Step 1: FR/NFR classification
        fr_label, fr_conf = frnfr_predict(s)
        print(f"  → FR/NFR: {fr_label} (confidence={fr_conf:.2f})")

        # Step 2: HCR gate probability
        p_hcr = hcr_gate_probability(s)
        print(f"  → HCR gate probability: {p_hcr:.2f}")

        # Step 3: HCR subtags if above threshold
        if p_hcr >= TAU_HCR:
            subtags = hcr_subtag_probabilities(s)
            tags = [t for t, p in subtags.items() if p >= TAU_TAG]
            print(f"  → HCR tags: {tags}")
            print(f"     subtag probabilities: {subtags}")
        else:
            print("  → Not human-centric (below threshold)")

def main():
    run_block("PROMISE+ • Functional (FR)", promise_fr)
    run_block("PROMISE+ • Non-Functional (NFR)", promise_nfr)
    run_block("HCR • Functional (FR-like)", hcr_fr_like)
    run_block("HCR • Non-Functional (NFR-like)", hcr_nfr_like)
    print("\n✅ Completed test with visible dataset examples.\n")

if __name__ == "__main__":
    main()
