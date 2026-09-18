# Scanning Architecture Prototype

Implements the scan pipeline from `scanning-architecture-plan.md` (§2-§6 only —
dashboard/chatbot/nav from §7-§9 are out of scope for this pass) with a
Pywebview desktop GUI.

## Setup

    pip install -r requirements.txt
    # Debian/Ubuntu also needs the tesseract binary:
    sudo apt-get install tesseract-ocr

## Run

    python app.py             # scan-architecture debug shell (camera + pipeline internals)
    python app_consumer.py    # consumer shop app (Analytics / Inventory / AI Chat / Options)

Both entry points share the same backend/api.py and the same scan_app.db —
products, reference photos, and sales made in one show up in the other.

## What's implemented

- `backend/frame_gate.py` — §6.1 frame-readiness gate (motion/blur/edge-density)
- `backend/qr_detector.py` — §3.1 QR path (OpenCV QRCodeDetector; swap for ML Kit/ZXing on mobile)
- `backend/ocr_extractor.py` — §3.2 OCR + rule-based fuzzy name matching (Tesseract stand-in for ML Kit Text Recognition v2), degrades gracefully if Tesseract isn't installed
- `backend/embedding.py` — §3.3 visual embedding interface; ships a dev-only histogram embedder, with a `TFLiteEmbedder` stub marked for the real quantized MobileNetV3/EfficientNet-Lite0 model
- `backend/index_store.py` — §3.4 flat similarity index with an ANN swap-in point once catalogs exceed ~250 SKUs
- `backend/fusion_scorer.py` — §3.5 single fused confidence score + auto/candidate/manual decision thresholds
- `backend/pipeline.py` — §2 cascade orchestration + §6.2 debounce/confirm-and-advance
- `backend/db.py` — §4 data model (Product, ProductReferenceImage, ScanLog, StockEvent) plus Sale/SaleItem for checkout, in SQLite
- `backend/config.py` — every threshold/weight in one place, per §1
- `gui/index.html` + `app.py` — debug shell: live camera, gate/stage indicators, toast confirmation, top-3 candidate sheet, product/reference-image registration panel
- `gui/consumer.html` + `app_consumer.py` — consumer shop app:
  - **Analytics** — sample dashboard shell only, not wired up (as requested)
  - **Inventory** — add items with full details, live search, real sales-today strip, low-stock badges
  - **Add item flow** — after saving details, walks the shopkeeper through capturing all 4 reference angles (front/back/left/right); the coaching tip only shows the first time (tracked in `localStorage`), later captures just show the progress dots and angle label
  - **Checkout scanner** (+ button above the tab bar) — full-screen live camera reusing the exact same `process_frame` cascade as the debug shell; every detection surfaces a one-tap "Add" confirmation card rather than auto-adding, builds a running cart, and "End checkout" finalizes the sale (writes Sale/SaleItem, decrements stock) and shows an invoice screen
  - **AI Chat** — Gemini `gemini-3.5-flash-lite` chat with current inventory injected as context each turn; no tool-calling/write actions yet, as requested
  - **Options** — API key entry (device-local), model info, contact support, reset local data

## Known stand-ins (flagged in code)

- `HistogramEmbedder` is a classical-CV placeholder for the CNN embedding the spec calls for — swap in `TFLiteEmbedder` once a quantized model file exists.
- LM-based disambiguation (§3.2) is a stub — the spec lists this as an open v1 decision (§10.3).
- ANN indexing (§3.4) falls back to a flat scan; the swap point is `index_store.SimilarityIndex.query`.
