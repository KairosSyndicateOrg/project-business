"""
§3.3 Visual Embedding.

Production target per spec: MobileNetV3-Small or EfficientNet-Lite0,
TFLite/INT8 quantized, output a fixed-length (e.g. 256-d) embedding vector,
compared via cosine similarity — NOT classical CV (SIFT/ORB/histogram),
because keypoint matching is brittle to lighting/angle/background shifts.

This scaffold defines the same interface (`Embedder.embed(frame) -> vector`)
so the real quantized CNN can be dropped in without touching the pipeline
or fusion scorer. The bundled `HistogramEmbedder` is a CPU-only, dependency-
free stand-in for local development ONLY — the spec explicitly calls out
its brittleness, so it must not ship as-is; swap in `TFLiteEmbedder` (stub
below) once a converted model file is available.
"""

import numpy as np
import cv2


class Embedder:
    """Interface every embedding backend implements."""

    dim = 256

    def embed(self, frame_bgr):
        raise NotImplementedError


class HistogramEmbedder(Embedder):
    """
    Dev-only stand-in. Produces a fixed-length vector from a concatenated
    HSV color histogram + downsampled grayscale patch, L2-normalized so
    cosine similarity behaves sanely. This is explicitly the kind of
    classical-CV approach the spec says to avoid for production — kept
    here only so the pipeline is runnable end-to-end without a model file.
    """

    dim = 256

    def embed(self, frame_bgr):
        img = cv2.resize(frame_bgr, (96, 96))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 12], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # 192-d

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8)).flatten().astype(np.float32) / 255.0  # 64-d

        vec = np.concatenate([hist, small])[: self.dim]
        if vec.shape[0] < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.shape[0]))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()


class TFLiteEmbedder(Embedder):
    """
    Stub for the real production backend: load a quantized
    MobileNetV3-Small / EfficientNet-Lite0 .tflite model and run inference
    per spec §3.3. Not implemented in this scaffold — requires a converted
    model artifact and tflite-runtime, tracked as follow-up work.
    """

    def __init__(self, model_path):
        raise NotImplementedError(
            "Drop a quantized .tflite model at model_path and implement "
            "interpreter load + invoke here to go from dev stand-in to prod."
        )


def cosine_similarity(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_similarity(query_vector, reference_vectors, k=1):
    """
    reference_vectors: list of (product_id, vector)
    Returns best (product_id, similarity) — "max or top-k average across a
    product's stored angles" per spec §3.3; this groups by product_id and
    takes the max across that product's stored angles, then returns the
    single best product.
    """
    best_by_product = {}
    for product_id, vec in reference_vectors:
        sim = cosine_similarity(query_vector, vec)
        if product_id not in best_by_product or sim > best_by_product[product_id]:
            best_by_product[product_id] = sim

    ranked = sorted(best_by_product.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
