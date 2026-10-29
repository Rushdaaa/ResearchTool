"""
Voice-driven requirements capture (streaming) with live classification.

Pipeline:
  Mic → AssemblyAI streaming → final, formatted turns only →
  clause split → DistilBERT FR/NFR → HCR gate (LR on SBERT) →
  HCR subtags (RF multilabel) → neat print

Notes:
- Loads saved models from ../models.
- Classifies only final turns to reduce flicker/noise.
- Minimal built-in clause splitter to avoid external utilities.
"""

import os
from pathlib import Path
from typing import List, Tuple, Type
from dotenv import load_dotenv

# --- Speech SDK ---
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent, StreamingClient, StreamingClientOptions, StreamingError,
    StreamingEvents, StreamingParameters, StreamingSessionParameters,
    TerminationEvent, TurnEvent
)

# --- Classifiers + thresholds ---
from classifier_frnfr import frnfr_predict                    # DistilBERT FR/NFR
from hcr_gate import hcr_gate_probability                     # LR gate
from classifier_hcr import hcr_subtag_probabilities           # RF multilabel
from config import TAU_HCR, TAU_TAG                           # thresholds

# ------------------------------
# Config / Env
# ------------------------------
HERE = Path(__file__).resolve().parent
load_dotenv()
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not AAI_KEY:
    raise RuntimeError("Missing ASSEMBLYAI_API_KEY in .env")

# ------------------------------
# Small helpers
# ------------------------------
def clear_line():
    print('\r' + ' ' * 100 + '\r', end='', flush=True)

def normalize_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] not in {'.', '!', '?'}:
        s += '.'
    return s

def split_into_clauses(text: str) -> List[str]:
    """
    Very small, safe splitter: sentence boundaries first, then soft split on 'and', 'but'.
    Keeps clauses >= 5 words to avoid over-splitting.
    """
    import re
    # hard sentence splits
    chunks = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    out = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        # soft split on coordinating conjunctions
        parts = re.split(r'\b(?:and|but)\b', ch, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip(",;: ").strip()
            if len(p.split()) >= 5:
                out.append(p)
            else:
                # if too short, glue back to previous if possible
                if out:
                    out[-1] = (out[-1] + " " + p).strip()
                else:
                    out.append(p)
    return out or ([text.strip()] if text.strip() else [])

def classify_text_clause(clause: str) -> dict:
    """
    Run full pipeline on one clause:
      - DistilBERT FR/NFR
      - HCR gate
      - HCR subtags if gate passes
    Returns a dict that can be logged/serialized.
    """
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

def process_final_text(final_text: str) -> dict:
    clauses = split_into_clauses(final_text)
    results = [classify_text_clause(c) for c in clauses]
    return {
        "final_text": final_text,
        "clauses": results,
    }

# Keep last partial to avoid noisy prints
_last_partial = ""

# ------------------------------
# Streaming event handlers
# ------------------------------
def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"session started: {event.id}")
    print("\n🎙️  listening… speak a requirement; press Ctrl+C to stop.\n")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    global _last_partial

    raw = event.transcript or ""
    text = raw.strip()
    if not text:
        return

    # show partials in-place
    if not event.end_of_turn:
        if text != _last_partial:
            print(f"\r📝 partial: {text[:120]}...", end="", flush=True)
            _last_partial = text
        return

    # for finals: ensure formatted casing/punct from AAI
    if not event.turn_is_formatted:
        # request formatting and wait; event loop will deliver formatted final next
        params = StreamingSessionParameters(format_turns=True)
        self.set_params(params)
        return

    # clear the partial line
    clear_line()
    _last_partial = ""

    # final text for this turn
    final_text = text.strip()
    if not final_text:
        return

    print(f"📝 final: {final_text}")
    payload = process_final_text(final_text)

    # split and classify each clause
    clauses = split_into_clauses(final_text)
    for i, clause in enumerate(clauses, 1):
        result = classify_text_clause(clause)
        fr = result["frnfr"]
        hcr = result["hcr"]

        print(f"  ▸ clause {i}: {result['text']}")
        print(f"      FR/NFR : {fr['label']} (p={fr['confidence']:.2f})")
        print(f"      HCR    : p={hcr['p_hcr']:.2f}"
              + (f"  | tags={hcr['tags']}" if hcr['tags'] else "  | tags=[]"))

        # uncomment to see detailed subtag probs:
        # if hcr["subtag_probs"]:
        #     print(f"      subtags: {hcr['subtag_probs']}")

    print("-" * 60)

def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(f"\nsession terminated. audio seconds={event.audio_duration_seconds}")

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"[error] {error}")

# ------------------------------
# entrypoint
# ------------------------------
def main():
    aai.settings.api_key = AAI_KEY

    client = StreamingClient(
        StreamingClientOptions(
            api_key=AAI_KEY,
            api_host="streaming.assemblyai.com"
        )
    )

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate=16000,
            format_turns=True,
        )
    )

    try:
        # microphone stream (16kHz)
        client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
    finally:
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
