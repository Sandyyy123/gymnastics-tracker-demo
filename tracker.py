"""
Multi-object tracker combining:
  - Kalman filter motion prediction (filterpy)
  - Hungarian algorithm assignment (scipy)
  - IoU primary matching + Re-ID secondary matching
  - Track state machine: tentative -> active -> lost -> dead
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from reid import ReIDModule

try:
    from filterpy.kalman import KalmanFilter
    _FILTERPY = True
except ImportError:
    _FILTERPY = False


# -----------------------------------------------------------------------
# Kalman filter factory: state = [cx, cy, w, h, vx, vy, vw, vh]
# -----------------------------------------------------------------------
def _make_kalman() -> "KalmanFilter | None":
    if not _FILTERPY:
        return None
    kf = KalmanFilter(dim_x=8, dim_z=4)
    # State transition
    kf.F = np.eye(8)
    for i in range(4):
        kf.F[i, i + 4] = 1.0
    # Measurement
    kf.H = np.eye(4, 8)
    kf.R *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    return kf


def _iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x,y,w,h) boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# -----------------------------------------------------------------------
class Track:
    _next_id = 1

    def __init__(self, bbox: tuple, reid: ReIDModule):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox          # current (x,y,w,h)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = "tentative"  # tentative | active | lost | dead
        self._reid = reid

        # Init Kalman
        self._kf = _make_kalman()
        if self._kf is not None:
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            self._kf.x[:4] = np.array([[cx], [cy], [bbox[2]], [bbox[3]]])

    # ------------------------------------------------------------------
    def predict(self) -> tuple:
        """Advance Kalman and return predicted bbox."""
        if self._kf is not None:
            self._kf.predict()
            cx, cy, w, h = (self._kf.x[:4].flatten())
            w = max(1, w); h = max(1, h)
            self.bbox = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: tuple) -> None:
        """Update track with a matched detection."""
        self.bbox = bbox
        if self._kf is not None:
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            self._kf.update(np.array([[cx], [cy], [bbox[2]], [bbox[3]]]))
        self.hits += 1
        self.time_since_update = 0
        if self.state == "tentative" and self.hits >= 3:
            self.state = "active"
        elif self.state == "lost":
            self.state = "active"

    def mark_lost(self) -> None:
        self.state = "lost"

    def mark_dead(self) -> None:
        self.state = "dead"


# -----------------------------------------------------------------------
class MultiObjectTracker:
    """
    Two-stage matcher:
      Stage 1 - IoU matching against predicted bboxes.
      Stage 2 - Re-ID appearance matching for unmatched detections vs lost tracks.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_age: int = 30,
        reid_sim_threshold: float = 0.72,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_age = max_lost_age
        self._tracks: list[Track] = []
        self._reid = ReIDModule(sim_threshold=reid_sim_threshold)

    # ------------------------------------------------------------------
    def update(
        self,
        detections: list[tuple],
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Parameters
        ----------
        detections : list of (x, y, w, h) bboxes from current frame.
        frame      : BGR frame (used for Re-ID embedding updates).

        Returns
        -------
        List of dicts: {track_id, bbox, state}
        """
        # Step 1: predict all tracks
        for t in self._tracks:
            t.predict()

        active = [t for t in self._tracks if t.state in ("tentative", "active")]
        lost   = [t for t in self._tracks if t.state == "lost"]

        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[int] = set()

        # ------ Stage 1: IoU matching (active tracks vs detections) ------
        if active and detections:
            iou_matrix = np.zeros((len(active), len(detections)))
            for ti, track in enumerate(active):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _iou(track.bbox, det)

            cost = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost)

            for ri, ci in zip(row_ind, col_ind):
                if iou_matrix[ri, ci] >= self.iou_threshold:
                    active[ri].update(detections[ci])
                    if frame is not None:
                        self._reid.update(active[ri].track_id, frame, detections[ci])
                    matched_track_ids.add(active[ri].track_id)
                    if ci in unmatched_dets:
                        unmatched_dets.remove(ci)

        # Mark unmatched active tracks as lost
        for t in active:
            if t.track_id not in matched_track_ids:
                t.mark_lost()

        # ------ Stage 2: Re-ID matching (lost tracks vs remaining dets) ------
        if lost and unmatched_dets and frame is not None:
            lost_ids = [t.track_id for t in lost]
            still_unmatched = []
            for di in unmatched_dets:
                matched_id = self._reid.match(frame, detections[di], lost_ids)
                if matched_id is not None:
                    track = next(t for t in lost if t.track_id == matched_id)
                    track.update(detections[di])
                    self._reid.update(matched_id, frame, detections[di])
                    matched_track_ids.add(matched_id)
                else:
                    still_unmatched.append(di)
            unmatched_dets = still_unmatched

        # Create new tracks for remaining unmatched detections
        for di in unmatched_dets:
            t = Track(detections[di], self._reid)
            if frame is not None:
                self._reid.update(t.track_id, frame, detections[di])
            self._tracks.append(t)

        # Kill stale lost tracks
        for t in self._tracks:
            if t.state == "lost" and t.time_since_update > self.max_lost_age:
                t.mark_dead()
                self._reid.remove_track(t.track_id)

        # Remove dead tracks
        self._tracks = [t for t in self._tracks if t.state != "dead"]

        return [
            {"track_id": t.track_id, "bbox": t.bbox, "state": t.state}
            for t in self._tracks
            if t.state in ("tentative", "active")
        ]

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self._tracks if t.state == "active")
