"""
§3.2 OCR Extraction.

Engine: on-device Tesseract here as the desktop stand-in for ML Kit Text
Recognition v2 (the spec's recommended mobile default). Output: raw text
blocks + bounding boxes + per-block confidence.

Normalization step is rule-based (regex + small unit dictionary + fuzzy
match against registered product names) — NOT routed through an LM for
the common case. An LM is reserved, per spec, for the narrow job of
disambiguating between multiple close-scoring fuzzy candidates; that hook
is left as `disambiguate_with_lm` (unimplemented here — v1 open decision
per §10.3).
"""

import re
import os
import shutil
import pytesseract
from rapidfuzz import fuzz, process

from . import config

UNIT_TOKENS = re.compile(r"\b\d+(\.\d+)?\s?(g|kg|ml|l|pcs|pack)\b", re.IGNORECASE)

# Common Windows install locations, checked as a fallback if PATH lookup fails
# (e.g. a terminal opened before the installer updated PATH).
_WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

_TESSERACT_AVAILABLE = None
_WARNED = False


def _tesseract_available():
    global _TESSERACT_AVAILABLE, _WARNED
    if _TESSERACT_AVAILABLE is None:
        found = shutil.which(pytesseract.pytesseract.tesseract_cmd) or shutil.which("tesseract")
        if not found:
            for candidate in _WINDOWS_FALLBACK_PATHS:
                if os.path.isfile(candidate):
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    found = candidate
                    break
        if not found:
            # Last resort: ask pytesseract itself, which shells out directly
            # rather than relying on shutil.which's PATH resolution.
            try:
                pytesseract.get_tesseract_version()
                found = True
            except Exception:
                found = False
        _TESSERACT_AVAILABLE = bool(found)
    if not _TESSERACT_AVAILABLE and not _WARNED:
        _WARNED = True
        print(
            "[ocr_extractor] Tesseract binary not found — OCR step will be skipped "
            "(fusion score falls back to embedding-only).\n"
            "If you just installed it, close and reopen your terminal (PATH changes "
            "don't apply to already-open shells), or hardcode the path here:\n"
            "  pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
        )
    return _TESSERACT_AVAILABLE


def extract_text_blocks(frame_bgr):
    """
    Returns list of dicts: {text, conf, bbox: (x, y, w, h)}
    Mirrors the spec's "raw text blocks + bounding boxes + per-block confidence".
    Returns [] (never raises) if the Tesseract binary isn't installed, so the
    rest of the pipeline (embedding + fusion) still runs.
    """
    if not _tesseract_available():
        return []
    try:
        data = pytesseract.image_to_data(frame_bgr, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        return []
    blocks = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf_raw = data["conf"][i]
        try:
            conf = max(0.0, float(conf_raw)) / 100.0
        except (ValueError, TypeError):
            conf = 0.0
        blocks.append({
            "text": text,
            "conf": conf,
            "bbox": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
        })
    return blocks


def normalize_and_match(blocks, candidate_products, size_dictionary=UNIT_TOKENS):
    """
    Rule-based parser: pulls out probable brand/name/size tokens, then
    fuzzy-matches the combined text against registered product names
    (trigram/Levenshtein-style via rapidfuzz).

    candidate_products: list of {"id", "name", "brand"} from the local DB.

    Returns: (ocr_match_candidate_id: str|None, ocr_confidence: float 0-1,
              ambiguous_candidates: list[(product_id, score)], is_exact: bool)

    is_exact is True when the best fuzzy score clears
    config.OCR_EXACT_MATCH_THRESHOLD — i.e. OCR is confident enough that
    this is a real (near-)literal text match, not just the closest of a
    weak field. The fusion scorer treats that as the highest-priority
    signal and short-circuits the weighted blend (see fusion_scorer.py).
    """
    if not blocks:
        return None, 0.0, [], False

    full_text = " ".join(b["text"] for b in blocks)
    detected_sizes = size_dictionary.findall(full_text)  # noqa: informational, not scored here

    if not candidate_products:
        return None, 0.0, [], False

    choices = {p["id"]: f'{p.get("brand", "")} {p["name"]}'.strip() for p in candidate_products}
    results = process.extract(full_text, choices, scorer=fuzz.token_set_ratio, limit=5)
    # results: list of (matched_string, score, key)
    scored = [(key, score / 100.0) for matched_string, score, key in results]
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return None, 0.0, [], False

    best_id, best_score = scored[0]
    ambiguous = [s for s in scored[1:3] if abs(s[1] - best_score) < 0.08]
    is_exact = best_score >= config.OCR_EXACT_MATCH_THRESHOLD

    return best_id, best_score, ambiguous, is_exact


def disambiguate_with_lm(full_text, ambiguous_candidates):
    """
    Hook only. Per spec §3.2, an LM is reserved for the narrow job of
    resolving ambiguous free text when the rule-based parser returns
    several plausible, similarly-scored candidates. Not wired up in this
    scaffold — see open decision §10.3 (whether it's worth including at all
    for v1).
    """
    raise NotImplementedError("Local LM disambiguation is a v1 open decision (spec §10.3)")
