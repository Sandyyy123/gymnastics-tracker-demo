"""
MediaPipe Holistic / Pose wrapper.
Returns (bbox, landmarks, confidence) per detected person.
Falls back to last valid pose when confidence drops.
"""
from __future__ import annotations
import numpy as np
import cv2

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


class PoseEstimator:
    """
    Wraps MediaPipe Pose for single-person landmark detection and
    bounding-box extraction.

    Parameters
    ----------
    min_detection_confidence : float
        Minimum detection confidence (0-1). Lower = more detections, more noise.
    min_tracking_confidence : float
        Minimum tracking confidence (0-1).
    fallback_frames : int
        How many frames to return the last valid pose when confidence drops.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        fallback_frames: int = 5,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.fallback_frames = fallback_frames

        self._last_valid: dict | None = None
        self._fallback_remaining: int = 0

        if _MP_AVAILABLE:
            mp_pose = mp.solutions.pose
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self._pose = None

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> dict | None:
        """
        Run pose estimation on a BGR frame.

        Returns a dict with keys:
            bbox   : (x, y, w, h) in pixels
            landmarks : list of (x, y, z, visibility) in normalized coords
            confidence: float

        Returns None if no person detected and fallback exhausted.
        """
        h, w = frame.shape[:2]

        if self._pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                xs = [lm.x for lm in lms]
                ys = [lm.y for lm in lms]
                vis = [lm.visibility for lm in lms]
                conf = float(np.mean(vis))

                if conf >= self.min_detection_confidence:
                    x1 = max(0, int(min(xs) * w))
                    y1 = max(0, int(min(ys) * h))
                    x2 = min(w, int(max(xs) * w))
                    y2 = min(h, int(max(ys) * h))
                    bbox = (x1, y1, x2 - x1, y2 - y1)

                    out = {
                        "bbox": bbox,
                        "landmarks": [(lm.x, lm.y, lm.z, lm.visibility) for lm in lms],
                        "confidence": conf,
                    }
                    self._last_valid = out
                    self._fallback_remaining = self.fallback_frames
                    return out

        # Confidence dropped or MediaPipe not available - use fallback
        if self._last_valid is not None and self._fallback_remaining > 0:
            self._fallback_remaining -= 1
            return {**self._last_valid, "fallback": True}

        return None

    # ------------------------------------------------------------------
    def close(self):
        if self._pose is not None:
            self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
