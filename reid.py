"""
Lightweight re-identification module using HSV color histograms.
Each track maintains a gallery of recent appearance embeddings.
Matching uses cosine similarity.
"""
import numpy as np
import cv2


def _hsv_histogram(crop: np.ndarray, bins=(16, 8, 8)) -> np.ndarray:
    """Compute a normalized HSV histogram embedding from a BGR crop."""
    if crop is None or crop.size == 0:
        return np.zeros(bins[0] * bins[1] * bins[2], dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [bins[0], bins[1], bins[2]],
        [0, 180, 0, 256, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors."""
    return float(np.dot(a, b))


class AppearanceGallery:
    """Stores up to `maxlen` recent embeddings for a single track ID."""

    def __init__(self, track_id: int, maxlen: int = 10):
        self.track_id = track_id
        self.maxlen = maxlen
        self._embeddings: list[np.ndarray] = []

    def update(self, embedding: np.ndarray) -> None:
        self._embeddings.append(embedding)
        if len(self._embeddings) > self.maxlen:
            self._embeddings.pop(0)

    def mean_embedding(self) -> np.ndarray | None:
        if not self._embeddings:
            return None
        return np.mean(self._embeddings, axis=0)


class ReIDModule:
    """
    Manages per-track appearance galleries.
    Call `update` each frame for matched tracks.
    Call `match` when IoU matching fails to find the best gallery match.
    """

    def __init__(self, sim_threshold: float = 0.72, gallery_maxlen: int = 10):
        self.sim_threshold = sim_threshold
        self.gallery_maxlen = gallery_maxlen
        self._galleries: dict[int, AppearanceGallery] = {}

    # ------------------------------------------------------------------
    def _get_or_create(self, track_id: int) -> AppearanceGallery:
        if track_id not in self._galleries:
            self._galleries[track_id] = AppearanceGallery(track_id, self.gallery_maxlen)
        return self._galleries[track_id]

    def update(self, track_id: int, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """Extract embedding from `bbox` region of `frame` and store it."""
        x, y, w, h = bbox
        crop = frame[y : y + h, x : x + w]
        embedding = _hsv_histogram(crop)
        self._get_or_create(track_id).update(embedding)

    def match(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        candidate_ids: list[int],
    ) -> int | None:
        """
        Compare the appearance of `bbox` region against galleries of `candidate_ids`.
        Returns the best matching track ID if similarity >= threshold, else None.
        """
        x, y, w, h = bbox
        crop = frame[y : y + h, x : x + w]
        query_emb = _hsv_histogram(crop)

        best_id, best_sim = None, -1.0
        for tid in candidate_ids:
            gallery = self._galleries.get(tid)
            if gallery is None:
                continue
            mean_emb = gallery.mean_embedding()
            if mean_emb is None:
                continue
            sim = cosine_similarity(query_emb, mean_emb)
            if sim > best_sim:
                best_sim, best_id = sim, tid

        if best_sim >= self.sim_threshold:
            return best_id
        return None

    def remove_track(self, track_id: int) -> None:
        self._galleries.pop(track_id, None)
