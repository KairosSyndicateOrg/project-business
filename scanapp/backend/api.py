"""
Bridge between the pywebview JS frontend and the Python scan pipeline.
Every method here is exposed to JS as `pywebview.api.<method>(...)`.
"""

import base64
import os
import time
import numpy as np
import cv2
from dotenv import load_dotenv

from . import db, config
from .pipeline import ScanPipeline
from .index_store import SimilarityIndex

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


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

    def get_gemini_api_key(self):
        return os.getenv("GEMINI_API_KEY", "")

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

    def add_product_full(self, name, brand, size_variant, price, stock_qty, qr_enabled, category, barcode=""):
        """Used by the consumer GUI's Add Item sheet, which collects every field up front."""
        pid = db.add_product(
            name, brand, size_variant,
            price=float(price or 0), qr_enabled=bool(qr_enabled),
            category=category or "", initial_stock=int(stock_qty or 0),
            barcode=(barcode or "").strip(),
        )
        return {"id": pid}

    def detect_barcode(self, image_b64):
        """Used by the "scan a barcode" step in the Add Item flow: decodes
        whatever QR/barcode is in frame (same pyzbar-based detector the
        checkout scanner's process_frame uses) without trying to resolve
        it against a product yet, since the product may not exist."""
        frame = _decode_frame(image_b64)
        if frame is None:
            return {"error": "could not decode frame"}
        from .qr_detector import CodeDetector
        payload, symbol_type = CodeDetector().detect(frame)
        if not payload:
            return {"found": False}
        existing = db.find_product_by_barcode(payload) if symbol_type != "QRCODE" else None
        return {
            "found": True,
            "payload": payload,
            "symbol_type": symbol_type,
            "already_used_by": existing["name"] if existing else None,
        }

    def add_reference_image(self, product_id, angle_label, image_b64):
        frame = _decode_frame(image_b64)
        if frame is None:
            return {"error": "could not decode frame"}
        vector = self.pipeline.embedder.embed(frame)
        from .ocr_extractor import extract_text_blocks
        blocks = extract_text_blocks(frame)
        ocr_text = " ".join(b["text"] for b in blocks)

        # Persist the actual photo to disk — it was previously discarded
        # right after its embedding was computed (image_path was always
        # written as ""), so nothing was left to show/re-derive from later.
        image_path = self._save_reference_image(product_id, angle_label, frame)

        # Store the full per-block OCR output (text/confidence/bbox) alongside
        # the image and its embedding, not just the flattened match string —
        # previously that structured result was computed and then dropped.
        db.add_reference_image(
            product_id, angle_label, image_path=image_path, embedding_vector=vector,
            ocr_text_raw=ocr_text, ocr_blocks=blocks,
        )
        self.pipeline.index.mark_dirty()
        return {"ok": True, "image_path": image_path, "ocr_blocks": _jsonable(blocks)}

    def _save_reference_image(self, product_id, angle_label, frame):
        product_dir = os.path.join(config.REFERENCE_IMAGES_DIR, product_id)
        os.makedirs(product_dir, exist_ok=True)
        filename = f"{angle_label}_{int(time.time() * 1000)}.jpg"
        full_path = os.path.join(product_dir, filename)
        cv2.imwrite(full_path, frame)
        return full_path

    def reference_images_for_product(self, product_id):
        """Returns each saved reference photo's path plus the OCR results
        (concatenated text + full per-block text/conf/bbox) captured from it."""
        return db.reference_images_for_product(product_id)

    def search_products(self, query):
        """Loose name/brand search — used by the AI Chat tool layer to resolve a
        product the shopkeeper mentioned by name into an id before acting on it."""
        return db.find_products_by_name(query)

    def update_stock(self, product_id, delta, reason=""):
        """AI Chat tool: adjust stock by a relative amount (+/-)."""
        db.adjust_stock(product_id, int(delta), event_type="manual_adjustment", source="ai_chat")
        return _jsonable(db.get_product(product_id))

    def set_stock(self, product_id, quantity, reason=""):
        """AI Chat tool: set stock to an absolute amount."""
        return _jsonable(db.set_stock(product_id, int(quantity), source="ai_chat"))

    def update_product(self, product_id, name=None, brand=None, size_variant=None,
                        price=None, qr_enabled=None, category=None, barcode=None):
        """AI Chat tool: edit any product field except stock/adding new items."""
        return _jsonable(db.update_product(
            product_id, name=name, brand=brand, size_variant=size_variant,
            price=price, qr_enabled=qr_enabled, category=category, barcode=barcode,
        ))

    def delete_product(self, product_id):
        """AI Chat tool: remove a product entirely."""
        return db.delete_product(product_id)

    def register_new_product_scan(self, product_id):
        self.pipeline.register_new_product_scan(product_id)
        return {"ok": True}

    def recent_scans(self):
        return db.recent_scan_logs(limit=20)

    def qr_payload_for(self, product_id):
        """Convenience for generating a test QR sticker (§3.1 payload format)."""
        return f"shopid:{product_id}"

    # ---------- Checkout (consumer GUI) ----------

    def start_checkout(self):
        """
        Call when the checkout scanner opens, so an item confirmed in a
        previous session (still inside its debounce window) can be
        re-detected immediately in this new one.
        """
        self.pipeline._last_confirmed = {}
        self.pipeline.gate.reset()
        return {"ok": True}

    def finalize_checkout(self, cart_items):
        """
        cart_items: [{"product_id": str, "quantity": int}, ...] built up
        client-side while scanning. Writes the Sale/SaleItem rows, adjusts
        stock, and returns the invoice for the receipt screen.
        """
        if not cart_items:
            return {"error": "cart is empty"}
        invoice = db.create_sale(cart_items)
        return _jsonable(invoice)

    def today_sales_summary(self):
        return db.today_sales_summary()

    def recent_sales(self):
        return db.recent_sales(limit=20)


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
