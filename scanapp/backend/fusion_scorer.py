"""
§3.5 Fusion Scorer — single fused confidence score per candidate product,
per spec §1 ("not a waterfall of independent pass/fail checks").

    final_score = w1 * ocr_confidence_normalized
                + w2 * visual_similarity_normalized
                + w3 * size_variant_prior   (optional)

Weights and thresholds are pulled from config.py so they're tunable
without a code change (per spec §1 and §3.5).
"""

from . import config


def score_candidate(ocr_confidence, visual_similarity, size_variant_prior=0.0):
    """All three inputs expected in [0, 1]. Returns a fused score in [0, 1]."""
    w = config.WEIGHTS
    score = (
        w["w1_ocr"] * max(0.0, min(1.0, ocr_confidence))
        + w["w2_visual"] * max(0.0, min(1.0, visual_similarity))
        + w["w3_size_variant"] * max(0.0, min(1.0, size_variant_prior))
    )
    return max(0.0, min(1.0, score))


def fuse_candidates(ocr_match_id, ocr_confidence, visual_ranked, size_prior_fn=None):
    """
    Merge OCR's single best guess with the embedding index's top-k ranked
    list into one set of per-product fused scores.

    ocr_match_id: str|None      — from ocr_extractor.normalize_and_match
    ocr_confidence: float       — 0-1
    visual_ranked: list[(product_id, similarity)] — from index_store.query
    size_prior_fn: optional callable(product_id) -> float in [0,1]

    Returns: list[(product_id, final_score)] sorted descending.
    """
    candidates = {}

    for product_id, similarity in visual_ranked:
        candidates.setdefault(product_id, {"ocr": 0.0, "visual": 0.0})
        candidates[product_id]["visual"] = similarity

    if ocr_match_id:
        candidates.setdefault(ocr_match_id, {"ocr": 0.0, "visual": 0.0})
        candidates[ocr_match_id]["ocr"] = ocr_confidence

    scored = []
    for product_id, parts in candidates.items():
        prior = size_prior_fn(product_id) if size_prior_fn else 0.0
        final = score_candidate(parts["ocr"], parts["visual"], prior)
        scored.append((product_id, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def decide(scored_candidates):
    """
    Applies the decision thresholds (§3.5 / §2):
      score >= auto_threshold        -> ("auto_confirm", product_id, score)
      candidate_threshold <= score   -> ("show_candidates", top-3, best_score)
      score < candidate_threshold    -> ("manual", None, best_score or 0)
    """
    t = config.THRESHOLDS
    if not scored_candidates:
        return ("manual", [], 0.0)

    best_id, best_score = scored_candidates[0]

    if best_score >= t["auto_threshold"]:
        return ("auto_confirm", best_id, best_score)

    if best_score >= t["candidate_threshold"]:
        top3 = scored_candidates[: config.TOP_K_CANDIDATES]
        return ("show_candidates", top3, best_score)

    return ("manual", [], best_score)
