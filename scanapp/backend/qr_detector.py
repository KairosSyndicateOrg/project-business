"""
§3.1 QR Path.

Payload convention: `shopid:sku_uuid` — internal product ID only, never
name/price/stock, so printed stickers never go stale on edit.

We use OpenCV's built-in QRCodeDetector so this scaffold has zero extra
native dependencies. Swap in ML Kit Barcode Scanning (Android) for the
real mobile build — it's faster and more angle/lighting tolerant on
budget hardware, per the spec's recommended default. ZXing is the
documented lighter fallback there.
"""

import cv2


class QRDetector:
    def __init__(self):
        self._detector = cv2.QRCodeDetector()

    def detect(self, frame_bgr):
        """Returns the decoded QR payload string, or None if no QR found."""
        data, points, _ = self._detector.detectAndDecode(frame_bgr)
        if data:
            return data
        return None


def resolve_qr(frame_bgr, db_lookup_fn):
    """
    Full QR step of the pipeline (§2, step 1):
      - detect QR
      - if found, look up the product directly by internal ID
      - if the ID isn't found locally (stale sync / corrupted read),
        fall back silently to step 2 (OCR/embedding path) rather than erroring
    Returns: (product_dict_or_None, raw_payload_or_None)
    """
    detector = QRDetector()
    payload = detector.detect(frame_bgr)
    if not payload:
        return None, None
    product = db_lookup_fn(payload)
    if product is None:
        # Stale sync / corrupted read — fall back silently, per spec.
        return None, payload
    return product, payload
