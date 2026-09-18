"""
§3.1 Code path — QR stickers *and* real product barcodes.

Two separate schemes share this one detection step:
  - Our own printed QR stickers: payload `shopid:sku_uuid` — internal
    product ID only, never name/price/stock, so printed stickers never go
    stale on edit. Resolved via db.find_product_by_qr_id.
  - Real-world barcodes already printed on packaging (EAN-13, UPC-A,
    Code128, Code39, ...): resolved by exact match against the
    product.barcode column via db.find_product_by_barcode. These get
    attached to a product either by scanning one while adding the item,
    or by editing the product later.

Uses zxing-cpp (the Python binding for the ZXing-C++ engine) rather than
OpenCV's built-in QRCodeDetector or pyzbar, since ZXing-C++ decodes both
QR codes and the 1D barcode symbologies above *and* ships its native
decoder statically compiled into the pip wheel — `pip install zxing-cpp`
is the entire dependency, no separate system shared library (e.g.
zbar) needs installing.
"""

import cv2
import zxingcpp

# Every symbology worth reading for a retail product. Decoding is cheap
# and an unmatched payload is just ignored downstream, so this errs on
# the side of including anything plausible. These are ORed together into
# one flag set zxingcpp can filter for in a single pass.
_FORMATS = (
    zxingcpp.BarcodeFormat.QRCode
    | zxingcpp.BarcodeFormat.EAN13
    | zxingcpp.BarcodeFormat.EAN8
    | zxingcpp.BarcodeFormat.UPCA
    | zxingcpp.BarcodeFormat.UPCE
    | zxingcpp.BarcodeFormat.Code128
    | zxingcpp.BarcodeFormat.Code39
    | zxingcpp.BarcodeFormat.Code93
    | zxingcpp.BarcodeFormat.ITF
)


class CodeDetector:
    def detect_all(self, frame_bgr):
        """Returns every decoded symbol in the frame as
        [{"payload": str, "symbol_type": str}, ...] (usually 0 or 1 in
        practice, but a frame can contain more than one code). symbol_type
        is the format's enum name upper-cased ("QRCODE", "EAN13", "CODE128", ...)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        out = []
        for result in zxingcpp.read_barcodes(gray, formats=_FORMATS):
            if not result.valid or not result.text:
                continue
            out.append({"payload": result.text, "symbol_type": result.format.name.upper()})
        return out

    def detect(self, frame_bgr):
        """Returns (payload, symbol_type) for the first decoded symbol, or (None, None)."""
        results = self.detect_all(frame_bgr)
        if not results:
            return None, None
        return results[0]["payload"], results[0]["symbol_type"]


def resolve_code(frame_bgr, qr_lookup_fn, barcode_lookup_fn):
    """
    Full code-scan step of the pipeline (§2, step 1):
      - detect any QR code or barcode in the frame
      - a QRCODE symbol is tried against our own sticker scheme first
        (db.find_product_by_qr_id); if that ID isn't found locally (stale
        sync / corrupted read), fall back to treating its payload as a
        plain barcode value instead of erroring
      - any other symbology is looked up directly as a barcode
        (db.find_product_by_barcode)
      - if nothing decodes, or the payload doesn't match any product,
        fall back silently to step 2 (OCR/embedding path) rather than
        erroring, per spec

    Returns: (product_dict_or_None, raw_payload_or_None, symbol_type_or_None)
    """
    detector = CodeDetector()
    payload, symbol_type = detector.detect(frame_bgr)
    if not payload:
        return None, None, None

    if symbol_type == "QRCODE":
        product = qr_lookup_fn(payload)
        if product is not None:
            return product, payload, symbol_type
        # Not one of our stickers (or a stale/corrupted read) -- still
        # worth trying the same payload as a literal barcode value before
        # giving up on this frame's code entirely.
        return barcode_lookup_fn(payload), payload, symbol_type

    return barcode_lookup_fn(payload), payload, symbol_type
