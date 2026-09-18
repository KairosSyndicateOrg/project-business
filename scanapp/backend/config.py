"""
All tunable values live here (spec §1: "Every threshold is a config value,
not a hardcoded constant"). In production this would be loaded from a
remote-config / local settings table so it's adjustable without an app
update (§3.5) — for this scaffold it's a plain module with sane defaults.
"""

import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scan_app.db")

# Reference photos captured in the 4-angle capture flow (§3.3) are written
# here as JPEGs, one per (product, angle) capture, and the path is what
# gets stored in product_reference_image.image_path. Previously that
# column was left blank and the actual photo was discarded after its
# embedding vector was computed — this directory is where it now lives.
REFERENCE_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_images")

# --- Fusion scorer weights (§3.5) ---
WEIGHTS = {
    "w1_ocr": 0.45,
    "w2_visual": 0.45,
    "w3_size_variant": 0.10,
}

# --- Decision thresholds (§3.5) ---
THRESHOLDS = {
    "auto_threshold": 0.85,
    "candidate_threshold": 0.35,
}

# --- OCR priority override ---
# OCR is treated as the highest-priority signal: if the rule-based
# name/brand match is this close to a perfect string match, we trust it
# outright rather than letting a weak/noisy visual-similarity score drag
# the fused result down. See fusion_scorer.fuse_candidates.
OCR_EXACT_MATCH_THRESHOLD = 0.97   # token_set_ratio (0-1) considered "exact"
OCR_EXACT_MATCH_SCORE = 0.99       # fused score assigned on an exact OCR match

# --- Frame-readiness gate (§6.1) ---
# NOTE (temporary): the sharpness/blur check below was rejecting far too
# many frames in practice, so it's disabled via ENABLE_SHARPNESS_CHECK
# rather than deleted — flip it back to True once the Laplacian threshold
# has been retuned for the real camera hardware.
ENABLE_SHARPNESS_CHECK = False

FRAME_GATE = {
    "motion_diff_threshold": 8.0,      # mean abs pixel diff considered "still"
    "blur_var_threshold": 20.0,        # Laplacian variance below this = too blurry (tune per-camera; see GUI readout)
    "min_edge_density": 0.02,          # fraction of edge pixels needed to consider "object present"
    "stability_frames_required": 3,    # consecutive stable frames before firing pipeline
}

# --- Debounce / cooldown (§6.2) ---
DEBOUNCE_SECONDS = 1.5

# How long after showing the "a few possible matches" sheet for a given
# top candidate we suppress showing it again for that same candidate,
# even though the object is still sitting in frame and still stable.
# Without this, dismissing the sheet without picking anything (or just
# not tapping fast enough) made it pop right back up on the very next
# tick, since nothing else was marking that outcome as "already handled".
CANDIDATE_REDISPLAY_COOLDOWN_SECONDS = 3.0

# --- ANN indexing threshold (§3.4) ---
ANN_INDEX_SKU_THRESHOLD = 250  # above this, switch from flat scan to ANN index

# --- Performance targets in ms, informational only (§5) ---
PERF_TARGETS_MS = {
    "qr_detect": 150,
    "ocr_extraction": 300,
    "embedding_inference": 150,
    "ann_lookup": 100,
    "end_to_end_non_qr": 700,
}

TOP_K_CANDIDATES = 3
