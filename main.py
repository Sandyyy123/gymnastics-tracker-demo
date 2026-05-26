"""
Entry point for the gymnastics multi-object tracker.

Usage:
    python main.py --input path/to/video.mp4 [--output out.mp4] [--demo]

If --demo is passed (or no input file exists), a synthetic video with
randomly moving colored rectangles is generated for testing.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import cv2

from tracker import MultiObjectTracker
from pose_estimator import PoseEstimator


# -----------------------------------------------------------------------
def _generate_demo_frame(
    shape: tuple[int, int, int],
    objects: list[dict],
    frame_idx: int,
) -> tuple[np.ndarray, list[tuple]]:
    """Produce a synthetic frame with moving rectangles (demo mode)."""
    h, w, c = shape
    frame = np.zeros((h, w, c), dtype=np.uint8)

    detections = []
    for obj in objects:
        # Simple sinusoidal motion
        cx = int(obj["cx"] + obj["rx"] * np.sin(frame_idx * obj["fx"] + obj["px"]))
        cy = int(obj["cy"] + obj["ry"] * np.sin(frame_idx * obj["fy"] + obj["py"]))
        bw, bh = obj["bw"], obj["bh"]
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)
        color = obj["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        detections.append((x1, y1, x2 - x1, y2 - y1))

    return frame, detections


def _make_demo_objects(n: int = 3, w: int = 1280, h: int = 720) -> list[dict]:
    rng = np.random.default_rng(42)
    colors = [(255, 100, 50), (50, 200, 100), (100, 100, 255),
              (200, 200, 50), (200, 50, 200)]
    objects = []
    for i in range(n):
        objects.append({
            "cx": int(rng.integers(150, w - 150)),
            "cy": int(rng.integers(150, h - 150)),
            "rx": int(rng.integers(80, 180)),
            "ry": int(rng.integers(50, 130)),
            "fx": rng.uniform(0.02, 0.07),
            "fy": rng.uniform(0.02, 0.07),
            "px": rng.uniform(0, 2 * np.pi),
            "py": rng.uniform(0, 2 * np.pi),
            "bw": int(rng.integers(60, 120)),
            "bh": int(rng.integers(100, 180)),
            "color": colors[i % len(colors)],
        })
    return objects


# -----------------------------------------------------------------------
def _draw_tracks(frame: np.ndarray, track_results: list[dict]) -> np.ndarray:
    out = frame.copy()
    for tr in track_results:
        x, y, w, h = tr["bbox"]
        tid = tr["track_id"]
        color = ((tid * 67) % 256, (tid * 113) % 256, (tid * 193) % 256)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out, f"ID:{tid}", (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return out


# -----------------------------------------------------------------------
def run(input_path: str | None, output_path: str | None, demo: bool) -> None:
    tracker = MultiObjectTracker(iou_threshold=0.3, max_lost_age=30)
    pose_est = PoseEstimator(min_detection_confidence=0.45)

    use_demo = demo or not input_path
    cap = None if use_demo else cv2.VideoCapture(input_path)
    fps_target = 30
    W, H = 1280, 720

    if cap is not None:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_target = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_target, (W, H))

    demo_objs = _make_demo_objects(3, W, H) if use_demo else None
    frame_idx = 0
    id_switch_count = 0
    t0 = time.time()

    print(f"[INFO] Starting {'demo' if use_demo else 'video'} mode ...")

    while True:
        if use_demo:
            if frame_idx >= 300:  # 10s at 30fps
                break
            frame, detections = _generate_demo_frame((H, W, 3), demo_objs, frame_idx)
        else:
            ok, frame = cap.read()
            if not ok:
                break
            # Use pose estimator when MediaPipe is available
            pose_result = pose_est.process(frame)
            if pose_result and "bbox" in pose_result:
                detections = [pose_result["bbox"]]
            else:
                detections = []

        track_results = tracker.update(detections, frame)
        annotated = _draw_tracks(frame, track_results)

        elapsed = time.time() - t0
        fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(
            annotated, f"FPS: {fps:.1f}  Tracks: {len(track_results)}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        if writer:
            writer.write(annotated)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx:4d} | FPS {fps:5.1f} | Active tracks: {len(track_results)}")

        frame_idx += 1

    # Cleanup
    pose_est.close()
    if cap:
        cap.release()
    if writer:
        writer.release()

    total_time = time.time() - t0
    print(f"[DONE] {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.1f} avg FPS)")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gymnastics multi-object tracker")
    parser.add_argument("--input", type=str, default=None, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path (optional)")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo without a video")
    args = parser.parse_args()
    run(args.input, args.output, args.demo)
