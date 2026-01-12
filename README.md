# Voice-Driven Requirements Elicitation Tool

This repository contains the prototype implementation for the paper:

**“A Voice-Driven Requirements Elicitation Tool with Real-Time Classification & Human-Centric Overlay: Proposed for Safety-Critical Systems”**

The tool demonstrates a real-time pipeline for capturing spoken stakeholder requirements, segmenting them into clause-level units, and classifying them using a transformer-based FR/NFR classifier with a gated human-centric (HCR) overlay.

---

## Repository Purpose

This code is provided **for transparency and inspection** in support of the accompanying research paper.  
It is **not intended as a production-ready system**, nor is it expected to run out-of-the-box without configuration and model artifacts.

---

## Core Components

### Runtime Pipeline
- `tool_interface.py`  
  Gradio-based user interface for live sessions and requirement visualization --> this file is ran to open the tool.
  
- `main_stream.py`  
  Streaming speech-to-text orchestration, turn stabilization, clause segmentation, and inference flow.

### Classification Logic
- `classifier_frnfr.py`  
  Transformer-based Functional vs. Non-Functional requirement classifier (DistilBERT).

- `classifier_hcr.py`  
  Human-centric subcategory classifier (multi-label Random Forest).

- `hcr_gate.py`  
  Logistic Regression gate for conditional activation of human-centric reasoning.

### Training Scripts
- `train_frnfr.py`  
  Fine-tuning script for the FR/NFR transformer model using PROMISE+.

- `train_hcr_gate.py`  
  Training script for the human-centric gate and subcategory classifiers.
