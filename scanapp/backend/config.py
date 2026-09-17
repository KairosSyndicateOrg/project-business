"""
All tunable values live here (spec §1: "Every threshold is a config value,
not a hardcoded constant"). In production this would be loaded from a
remote-config / local settings table so it's adjustable without an app
update (§3.5) — for this scaffold it's a plain module with sane defaults.
"""

import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scan_app.db")

# --- Fusion scorer weights (§3.5) ---
WEIGHTS = {
    "w1_ocr": 0.45,
    "w2_visual": 0.45,
    "w3_size_variant": 0.10,
}

# --- Decision thresholds (§3.5) ---
THRESHOLDS = {
    "auto_threshold": 0.85,
    "candidate_threshold": 0.55,
}

# --- Frame-readiness gate (§6.1) ---
FRAME_GATE = {
    "motion_diff_threshold": 8.0,      # mean abs pixel diff considered "still"
    "blur_var_threshold": 20.0,        # Laplacian variance below this = too blurry (tune per-camera; see GUI readout)
    "min_edge_density": 0.02,          # fraction of edge pixels needed to consider "object present"
    "stability_frames_required": 3,    # consecutive stable frames before firing pipeline
}

# --- Debounce / cooldown (§6.2) ---
DEBOUNCE_SECONDS = 1.5

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