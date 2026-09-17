# Scanning Architecture Prototype

Implements the scan pipeline from `scanning-architecture-plan.md` (§2-§6 only —
dashboard/chatbot/nav from §7-§9 are out of scope for this pass) with a
Pywebview desktop GUI.

## Setup

    pip install -r requirements.txt
    # Debian/Ubuntu also needs the tesseract binary:
    sudo apt-get install tesseract-ocr

## Run

    python app.py

## What's implemented

- `backend/frame_gate.py` — §6.1 frame-readiness gate (motion/blur/edge-density)
- `backend/qr_detector.py` — §3.1 QR path (OpenCV QRCodeDetector; swap for ML Kit/ZXing on mobile)
- `backend/ocr_extractor.py` — §3.2 OCR + rule-based fuzzy name matching (Tesseract stand-in for ML Kit Text Recognition v2)
- `backend/embedding.py` — §3.3 visual embedding interface; ships a dev-only histogram embedder, with a `TFLiteEmbedder` stub marked for the real quantized MobileNetV3/EfficientNet-Lite0 model
- `backend/index_store.py` — §3.4 flat similarity index with an ANN swap-in point once catalogs exceed ~250 SKUs
- `backend/fusion_scorer.py` — §3.5 single fused confidence score + auto/candidate/manual decision thresholds
- `backend/pipeline.py` — §2 cascade orchestration + §6.2 debounce/confirm-and-advance
- `backend/db.py` — §4 data model (Product, ProductReferenceImage, ScanLog, StockEvent) in SQLite
- `backend/config.py` — every threshold/weight in one place, per §1
- `gui/index.html` + `app.py` — Pywebview shell: live camera, gate/stage indicators, toast confirmation, top-3 candidate sheet, product/reference-image registration panel

## Known stand-ins (flagged in code)

- `HistogramEmbedder` is a classical-CV placeholder for the CNN embedding the spec calls for — swap in `TFLiteEmbedder` once a quantized model file exists.
- LM-based disambiguation (§3.2) is a stub — the spec lists this as an open v1 decision (§10.3).
- ANN indexing (§3.4) falls back to a flat scan; the swap point is `index_store.SimilarityIndex.query`.
