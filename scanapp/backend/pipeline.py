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
from .qr_detector import resolve_code
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
        self._last_candidates_shown = {}  # top_candidate_id -> timestamp, see config.CANDIDATE_REDISPLAY_COOLDOWN_SECONDS

    # ---- §6.2 debounce ----
    def _debounced(self, product_id):
        last = self._last_confirmed.get(product_id)
        return last is not None and (time.time() - last) < config.DEBOUNCE_SECONDS

    def _mark_confirmed(self, product_id):
        self._last_confirmed[product_id] = time.time()

    def _candidates_recently_shown(self, top_id):
        last = self._last_candidates_shown.get(top_id)
        return last is not None and (time.time() - last) < config.CANDIDATE_REDISPLAY_COOLDOWN_SECONDS

    def _mark_candidates_shown(self, top_id):
        self._last_candidates_shown[top_id] = time.time()

    def process_frame(self, frame_bgr, force=False):
        """
        Runs one frame through the cascade. QR/barcode detection is
        deterministic and near-instant, so it runs on every frame
        regardless of the stability/sharpness gate (§6.1) — requiring a
        held-still frame before even trying to read a code just added
        latency for no benefit, unlike the OCR/visual cascade below, which
        genuinely does better on a settled frame. The gate is only
        consulted once we fall through to that slower path.
        """
        # --- Step 1: QR sticker or real product barcode (deterministic,
        # short-circuits everything, bypasses the frame-readiness gate) ---
        product, code_payload, symbol_type = resolve_code(
            frame_bgr, db.find_product_by_qr_id, db.find_product_by_barcode
        )
        if product is not None:
            match_path = "qr" if symbol_type == "QRCODE" else "barcode"
            if self._debounced(product["id"]):
                return {"stage": match_path, "gate": {"ready": True}, "outcome": "debounced", "product": product}
            self._mark_confirmed(product["id"])
            db.log_scan(product["id"], match_path, 1.0, resolved_by_user=False)
            self.gate.reset()
            return {
                "stage": match_path,
                "gate": {"ready": True},
                "outcome": "auto_confirm",
                "product": product,
                "score": 1.0,
                "code_type": symbol_type,
            }

        # A code was read but doesn't match any known product yet (e.g. a
        # fresh packet whose barcode hasn't been attached to anything in
        # the catalogue) — surface that distinctly so the GUI can offer to
        # attach it to a product, rather than silently falling through to
        # the slower OCR/visual cascade with no explanation. Also bypasses
        # the gate, for the same reason.
        if code_payload is not None and symbol_type != "QRCODE":
            return {
                "stage": "barcode",
                "gate": {"ready": True},
                "outcome": "unknown_barcode",
                "barcode": code_payload,
                "code_type": symbol_type,
            }

        # --- Gate (§6.1): only from here on, since OCR/visual matching
        # genuinely benefits from a held-still, in-focus frame ---
        gate_result = self.gate.evaluate(frame_bgr)
        if not gate_result["ready"] and not force:
            return {"stage": "gate", "gate": gate_result, "outcome": None}

        # --- Steps 2 & 3 in parallel (spec explicitly requires this) ---
        ocr_future = _executor.submit(extract_text_blocks, frame_bgr)
        embed_future = _executor.submit(self.embedder.embed, frame_bgr)
        blocks = ocr_future.result()
        query_vector = embed_future.result()

        candidate_products = db.reference_ocr_texts()
        ocr_match_id, ocr_confidence, _ambiguous, ocr_is_exact = normalize_and_match(blocks, candidate_products)
        visual_ranked = self.index.query(query_vector, k=config.TOP_K_CANDIDATES)

        # --- Step 4: fusion scorer (OCR is the highest-priority signal —
        # an exact OCR match short-circuits the weighted blend, see
        # fusion_scorer.fuse_candidates) ---
        scored = fuse_candidates(ocr_match_id, ocr_confidence, visual_ranked, ocr_is_exact=ocr_is_exact)
        decision, payload, best_score = decide(scored)

        result = {
            "stage": "fusion",
            "gate": gate_result,
            "ocr_match_id": ocr_match_id,
            "ocr_confidence": ocr_confidence,
            "ocr_is_exact": ocr_is_exact,
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
            top_id = payload[0][0] if payload else None
            if top_id and self._candidates_recently_shown(top_id):
                # Same top candidate we just showed (and the shopkeeper
                # didn't pick, or dismissed it) — don't pop it right back
                # up while the object just sits there; wait out the cooldown.
                result["outcome"] = "debounced"
                result["product"] = None
                return result
            if top_id:
                self._mark_candidates_shown(top_id)
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
