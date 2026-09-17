"""
Bridge between the pywebview JS frontend and the Python scan pipeline.
Every method here is exposed to JS as `pywebview.api.<method>(...)`.
"""

import base64
import numpy as np
import cv2

from . import db
from .pipeline import ScanPipeline
from .index_store import SimilarityIndex


def _decode_frame(data_url_or_b64):
    """Accepts a data URL (data:image/jpeg;base64,...) or raw base64 string."""
    if "," in data_url_or_b64[:50]:
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    raw = base64.b64decode(data_url_or_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


class Api:
    def __init__(self):
        db.init_db()
        self.pipeline = ScanPipeline()

    # ---------- Scanning ----------

    def process_frame(self, image_b64):
        frame = _decode_frame(image_b64)
        if frame is None:
            return {"error": "could not decode frame"}
        try:
            result = self.pipeline.process_frame(frame)
        except Exception as e:  # keep the GUI alive even if a stage errors
            return {"error": str(e)}
        return _jsonable(result)

    def confirm_candidate(self, product_id, score):
        product = self.pipeline.confirm_candidate(product_id, float(score))
        return _jsonable(product)

    # ---------- Product management (needed to populate reference data) ----------

    def list_products(self):
        return db.list_products()

    def add_product(self, name, brand, size_variant, qr_enabled):
        pid = db.add_product(name, brand, size_variant, price=0.0, qr_enabled=bool(qr_enabled))
        return {"id": pid}

    def add_reference_image(self, product_id, angle_label, image_b64):
        frame = _decode_frame(image_b64)
        if frame is None:
            return {"error": "could not decode frame"}
        vector = self.pipeline.embedder.embed(frame)
        from .ocr_extractor import extract_text_blocks
        blocks = extract_text_blocks(frame)
        ocr_text = " ".join(b["text"] for b in blocks)
        db.add_reference_image(product_id, angle_label, image_path="", embedding_vector=vector, ocr_text_raw=ocr_text)
        self.pipeline.index.mark_dirty()
        return {"ok": True}

    def register_new_product_scan(self, product_id):
        self.pipeline.register_new_product_scan(product_id)
        return {"ok": True}

    def recent_scans(self):
        return db.recent_scan_logs(limit=20)

    def qr_payload_for(self, product_id):
        """Convenience for generating a test QR sticker (§3.1 payload format)."""
        return f"shopid:{product_id}"


def _jsonable(obj):
    """Recursively strip non-JSON-safe types (e.g. numpy scalars) before returning to JS."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        # NaN/Infinity are not valid JSON and break pywebview's JS bridge.
        return None
    return obj