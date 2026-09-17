"""
§6.1 Frame Processing Loop / frame-readiness gate.

Runs cheap checks on every incoming frame so the expensive pipeline
(QR/OCR/embedding) only fires on a small fraction of frames, not all of
them at 30fps (§6.3 performance note):

  - Motion/stability check: only proceed once frame-to-frame diff is low
    (shopkeeper has stopped moving the product), and only after N
    consecutive stable frames.
  - Blur check (Laplacian variance): reject if too blurry.
  - Coarse "object of interest" check: edge-density in the central region,
    to skip empty/background frames.
"""

import cv2
import numpy as np

from . import config


class FrameGate:
    def __init__(self):
        self._prev_gray = None
        self._stable_count = 0

    def reset(self):
        self._prev_gray = None
        self._stable_count = 0

    def _motion_diff(self, gray):
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            # No previous frame yet, or its size doesn't match this one
            # (camera resolution can renegotiate between frames) — treat
            # as maximum diff rather than crashing cv2.absdiff on a shape
            # mismatch.
            return 255.0
        diff = cv2.absdiff(gray, self._prev_gray)
        return float(np.mean(diff))

    def _blur_variance(self, gray):
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _edge_density_central(self, gray):
        h, w = gray.shape
        cy0, cy1 = int(h * 0.25), int(h * 0.75)
        cx0, cx1 = int(w * 0.25), int(w * 0.75)
        central = gray[cy0:cy1, cx0:cx1]
        edges = cv2.Canny(central, 60, 150)
        return float(np.count_nonzero(edges)) / edges.size

    def evaluate(self, frame_bgr):
        """
        Returns a dict describing gate state for UI debugging, with
        'ready': bool indicating whether the frame should be pushed into
        the full scan pipeline (§2).
        """
        # Normalize to a fixed size first: the incoming frame's resolution
        # can drift slightly between calls (camera renegotiation, canvas
        # resize), and the motion-diff step requires two frames of
        # identical shape.
        frame_bgr = cv2.resize(frame_bgr, (320, 240))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        cfg = config.FRAME_GATE
        motion = self._motion_diff(gray)
        blur_var = self._blur_variance(gray)
        edge_density = self._edge_density_central(gray)

        is_still = motion <= cfg["motion_diff_threshold"]
        is_sharp = blur_var >= cfg["blur_var_threshold"]
        has_object = edge_density >= cfg["min_edge_density"]

        if is_still and is_sharp and has_object:
            self._stable_count += 1
        else:
            self._stable_count = 0

        self._prev_gray = gray

        ready = self._stable_count >= cfg["stability_frames_required"]

        return {
            "ready": ready,
            "motion": motion,
            "blur_var": blur_var,
            "edge_density": edge_density,
            "is_still": is_still,
            "is_sharp": is_sharp,
            "has_object": has_object,
            "stable_count": self._stable_count,
        }