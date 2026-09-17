"""
§2 Scan Pipeline (Priority Cascade) — orchestrator.

    Camera frame
       -> [gate: §6.1]  only stable/sharp/non-empty frames proceed
       -> 1. QR Detect --match--> direct DB lookup --> DONE (instant)
          | no QR
       -> 2. OCR extraction  \\
       -> 3. Visual embedding / (run in parallel, not sequentially)
       -> 4. Fusion scorer --> auto_confirm / show_candidates / manual

Also implements §6.2's confirm-and-advance / debounce behavior for
continuous scanning mode.
"""

import time
import concurrent.futures

from . import db, config
from .qr_detector import resolve_qr
from .ocr_extractor import extract_text_blocks, normalize_and_match
from .embedding import HistogramEmbedder
from .index_store import SimilarityIndex
from .fusion_scorer import fuse_candidates, decide
from .frame_gate import FrameGate

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class ScanPipeline:
    def __init__(self, embedder=None):
        self.gate = FrameGate()
        self.embedder = embedder or HistogramEmbedder()
        self.index = SimilarityIndex()
        self._last_confirmed = {}  # product_id -> timestamp, for debounce (§6.2)

    # ---- §6.2 debounce ----
    def _debounced(self, product_id):
        last = self._last_confirmed.get(product_id)
        return last is not None and (time.time() - last) < config.DEBOUNCE_SECONDS

    def _mark_confirmed(self, product_id):
        self._last_confirmed[product_id] = time.time()

    def process_frame(self, frame_bgr, force=False):
        """
        Runs one frame through the gate, then (if ready or forced) the full
        cascade. Returns a result dict describing what happened, suitable
        for the GUI layer to render directly.
        """
        gate_result = self.gate.evaluate(frame_bgr)
        if not gate_result["ready"] and not force:
            return {"stage": "gate", "gate": gate_result, "outcome": None}

        # --- Step 1: QR (deterministic, short-circuits everything) ---
        product, qr_payload = resolve_qr(frame_bgr, db.find_product_by_qr_id)
        if product is not None:
            if self._debounced(product["id"]):
                return {"stage": "qr", "gate": gate_result, "outcome": "debounced", "product": product}
            self._mark_confirmed(product["id"])
            db.log_scan(product["id"], "qr", 1.0, resolved_by_user=False)
            self.gate.reset()
            return {
                "stage": "qr",
                "gate": gate_result,
                "outcome": "auto_confirm",
                "product": product,
                "score": 1.0,
            }

        # --- Steps 2 & 3 in parallel (spec explicitly requires this) ---
        ocr_future = _executor.submit(extract_text_blocks, frame_bgr)
        embed_future = _executor.submit(self.embedder.embed, frame_bgr)
        blocks = ocr_future.result()
        query_vector = embed_future.result()

        candidate_products = db.reference_ocr_texts()
        ocr_match_id, ocr_confidence, _ambiguous = normalize_and_match(blocks, candidate_products)
        visual_ranked = self.index.query(query_vector, k=config.TOP_K_CANDIDATES)

        # --- Step 4: fusion scorer ---
        scored = fuse_candidates(ocr_match_id, ocr_confidence, visual_ranked)
        decision, payload, best_score = decide(scored)

        result = {
            "stage": "fusion",
            "gate": gate_result,
            "ocr_match_id": ocr_match_id,
            "ocr_confidence": ocr_confidence,
            "visual_ranked": visual_ranked,
            "score": best_score,
        }

        if decision == "auto_confirm":
            product_id = payload
            if self._debounced(product_id):
                result["outcome"] = "debounced"
                result["product"] = db.get_product(product_id)
                return result
            self._mark_confirmed(product_id)
            db.log_scan(product_id, "auto_fused", best_score, resolved_by_user=False)
            self.gate.reset()
            result["outcome"] = "auto_confirm"
            result["product"] = db.get_product(product_id)

        elif decision == "show_candidates":
            result["outcome"] = "show_candidates"
            result["candidates"] = [
                {**db.get_product(pid), "score": score} for pid, score in payload if db.get_product(pid)
            ]

        else:
            result["outcome"] = "manual"
            result["candidates"] = []

        return result

    def confirm_candidate(self, product_id, score):
        """Called when the shopkeeper taps a candidate in the top-3 sheet (§2 / §6.2)."""
        db.log_scan(product_id, "candidate_pick", score, resolved_by_user=True)
        self._mark_confirmed(product_id)
        self.gate.reset()
        return db.get_product(product_id)

    def register_new_product_scan(self, product_id):
        """Called after 'add as new product' resolves a manual scan (§2)."""
        db.log_scan(product_id, "manual_new", 0.0, resolved_by_user=True)
        self._mark_confirmed(product_id)
        self.gate.reset()
