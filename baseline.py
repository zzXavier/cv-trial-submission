"""Baseline auto-reframe: landscape video -> 9:16 vertical video (1080x1920).

Idea: run MoveNet single-pose on every full frame, take the mean x of the
confident keypoints, and cut a vertical crop centred on that x. MoveNet
single-pose locks onto whichever person is most prominent -- it has no idea
which player is the target in targets.json. No smoothing, no tracking, no
training: deliberately naive so there is room to improve.

Usage:
    python baseline.py clips/my_clip.mp4 out/my_clip_vertical.mp4
    python baseline.py clips/my_clip.mp4 out/my_clip_vertical.mp4 --log out/my_clip_baseline.json
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # force CPU even if a GPU exists
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"  # 192x192 input
IN_SIZE, OUT_W, OUT_H, MIN_SCORE = 192, 1080, 1920, 0.2


def mean_keypoint_x(movenet, frame_bgr):
    """Return the mean x (in source pixels) of confident keypoints, or None."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # MoveNet wants a square int32 image; letterbox so the aspect ratio is kept.
    inp = tf.image.resize_with_pad(rgb[None], IN_SIZE, IN_SIZE)
    kps = movenet(tf.cast(inp, tf.int32))["output_0"].numpy()[0, 0]  # (17, 3): y, x, score
    xs = kps[kps[:, 2] > MIN_SCORE, 1]
    if xs.size == 0:
        return None
    # Undo the letterbox: normalised padded coords -> source pixel coords.
    scale = IN_SIZE / max(h, w)
    pad_x = (IN_SIZE - w * scale) / 2
    return float((xs.mean() * IN_SIZE - pad_x) / scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src"), ap.add_argument("dst")
    ap.add_argument("--log", help="optional JSON path for per-frame crop positions")
    args = ap.parse_args()

    movenet = hub.load(MODEL_URL).signatures["serving_default"]
    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = min(w, round(h * 9 / 16))  # widest 9:16 window that fits the frame height
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    out = cv2.VideoWriter(args.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))

    cx, log, t0 = w / 2, [], time.perf_counter()  # start centred; hold the last good x
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = mean_keypoint_x(movenet, frame)
        cx = cx if x is None else x
        left = int(np.clip(cx - crop_w / 2, 0, w - crop_w))  # keep the window inside the frame
        out.write(cv2.resize(frame[:, left:left + crop_w], (OUT_W, OUT_H)))
        log.append({"frame_index": len(log), "timestamp_s": round(len(log) / fps, 3),
                    "pose_found": x is not None, "crop_x": left, "crop_w": crop_w})
    cap.release(), out.release()
    secs = time.perf_counter() - t0
    if args.log:
        with open(args.log, "w") as f:
            json.dump(log, f, indent=1)
    print(f"wrote {len(log)} frames to {args.dst} in {secs:.1f}s "
          f"({secs / max(len(log) / fps, 1e-9):.1f}s per second of video)")


if __name__ == "__main__":
    main()
