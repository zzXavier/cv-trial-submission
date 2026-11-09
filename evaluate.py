"""Compare source-coordinate crops against independent sparse body annotations.

Run ``python evaluate.py`` after generating the baseline and all three tracker logs.
Regular and diagnostic samples remain separate. Raw motion uses every logged
frame; additional within-shot metrics exclude transitions into camera resets.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean


def rectangle(x, y, w, h):
    values = (x, y, w, h)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError(f"Non-finite or non-numeric rectangle: {values}")
    if w <= 0 or h <= 0:
        raise ValueError(f"Non-positive rectangle dimensions: {values}")
    return values


def evaluate_clip(logs, references, source_w, source_h, frame_count=None,
                  exclude_initialization=True):
    """Return per-group accuracy, per-sample evidence and full-clip crop motion.

    Center error is horizontal, divided by crop width. Containment uses both axes.
    A center exactly on the rectangle boundary is counted as inside. Body retention
    is intersection area / annotated body-box area (an approximation to body pixels).
    Baseline logs omit crop_y/crop_h, interpreted as 0/source_h. Frame 0 labels
    initialize the target and are excluded from accuracy by default.
    An optional boolean camera_reset marks an engine-confirmed camera boundary,
    not independent shot ground truth. Only within-shot motion excludes the
    transition into that frame; raw motion and accuracy retain it. Manual anchor
    labels must be kept separate from independent accuracy evaluation labels.
    """
    if not logs or not references or source_w <= 0 or source_h <= 0:
        raise ValueError("Non-empty logs/references and positive source dimensions required")
    if frame_count is not None and len(logs) != frame_count:
        raise ValueError(f"Log/source frame count mismatch: {len(logs)} != {frame_count}")
    crops, camera_resets = [], []
    for expected, row in enumerate(logs):
        index = row.get("frame_index")
        if type(index) is not int or index != expected:
            raise ValueError(f"Log frame alignment: expected {expected}, got {index}")
        camera_reset = row.get("camera_reset", False)
        if type(camera_reset) is not bool:
            raise ValueError(f"camera_reset must be boolean at frame {index}")
        box = rectangle(row["crop_x"], row.get("crop_y", 0),
                        row["crop_w"], row.get("crop_h", source_h))
        x, y, w, h = box
        if x < 0 or y < 0 or x + w > source_w or y + h > source_h:
            raise ValueError(f"Crop outside source at frame {index}: {box}")
        crops.append(box)
        camera_resets.append(camera_reset)

    samples, seen = [], set()
    for ref in references:
        index, group = ref.get("frame_index"), ref.get("sample")
        if type(index) is not int or index < 0 or index >= len(crops):
            raise ValueError(f"Missing or invalid reference frame {index} in log")
        if index in seen:
            raise ValueError(f"Duplicate reference frame {index}")
        if group not in ("regular", "diagnostic"):
            raise ValueError(f"Unknown reference sample group: {group}")
        if type(ref.get("visible")) is not bool:
            raise ValueError(f"Reference visibility must be boolean at frame {index}")
        seen.add(index)
        sample = dict(frame_index=index, sample=group, visible=ref["visible"])
        if ref["visible"]:
            bx, by, bw, bh = rectangle(ref["x"], ref["y"], ref["w"], ref["h"])
            x, y, w, h = crops[index]
            cx, cy = bx + bw / 2, by + bh / 2
            intersection_w = max(0, min(x + w, bx + bw) - max(x, bx))
            intersection_h = max(0, min(y + h, by + bh) - max(y, by))
            sample.update(center_error=abs(cx - (x + w / 2)) / w,
                          center_outside=not (x <= cx <= x + w and y <= cy <= y + h),
                          body_retained_fraction=intersection_w * intersection_h / (bw * bh))
        samples.append(sample)

    excluded = sum(s["frame_index"] == 0 for s in samples) if exclude_initialization else 0
    if exclude_initialization:
        samples = [s for s in samples if s["frame_index"] != 0]
    groups = {}
    for group in ("regular", "diagnostic"):
        selected = [s for s in samples if s["sample"] == group]
        visible = [s for s in selected if s["visible"]]
        errors = [s["center_error"] for s in visible]
        retained = [s["body_retained_fraction"] for s in visible]
        groups[group] = dict(reference_count=len(selected), visible_count=len(visible),
            mean_center_error=fmean(errors) if errors else None,
            max_center_error=max(errors) if errors else None,
            center_outside_fraction=fmean(s["center_outside"] for s in visible) if visible else None,
            mean_body_retained_fraction=fmean(retained) if retained else None,
            min_body_retained_fraction=min(retained) if retained else None,
            body_clipped_over_5pct_count=sum(v < 0.95 - 1e-12 for v in retained))
    # Horizontal crop-center displacement; normalize each step by its current width.
    steps = [abs((b[0] + b[2] / 2) - (a[0] + a[2] / 2))
             for a, b in zip(crops, crops[1:])]
    normalized = [step / box[2] for step, box in zip(steps, crops[1:])]
    within_steps = [step for step, reset in zip(steps, camera_resets[1:]) if not reset]
    within_normalized = [step for step, reset in zip(normalized, camera_resets[1:]) if not reset]
    motion = dict(total_movement_px=sum(steps),
                  mean_abs_frame_step_normalized=fmean(normalized) if normalized else 0,
                  max_frame_step_normalized=max(normalized, default=0),
                  camera_reset_count=sum(camera_resets[1:]),
                  within_shot_total_movement_px=sum(within_steps),
                  within_shot_mean_abs_frame_step_normalized=fmean(within_normalized) if within_normalized else 0,
                  within_shot_max_frame_step_normalized=max(within_normalized, default=0))
    return dict(frame_count=len(logs), excluded_initialization_reference_count=excluded,
                groups=groups, motion=motion, samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("references.json"))
    parser.add_argument("--clips-dir", type=Path, default=Path("clips"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("out"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/improved"))
    args = parser.parse_args()
    import cv2  # Evaluation needs video metadata only; no tracker or TensorFlow imports.

    references = json.loads(args.references.read_text(encoding="utf-8"))
    if not references.get("clips"):
        raise ValueError("Reference file has no clips")
    report = dict(reference_file=str(args.references),
                  reference_metadata=references.get("metadata", {}),
                  notes=["Accuracy uses sparse independent boxes, not tracker estimates.",
                         "Frame 0 references initialize the target and are excluded from accuracy; each mode reports the excluded count.",
                         "Regular and diagnostic samples are not pooled.",
                         "Motion is horizontal crop-center movement over all consecutive frames; a frozen crop scores zero.",
                         "Within-shot motion excludes transitions into camera_reset=True frames; reset counts exclude frame 0. Raw motion still includes those transitions.",
                         "camera_reset is an engine-confirmed boundary for an explicit manual anchor or confirmed nonlocal relocation, not independent shot ground truth.",
                         "Manual anchor labels must be kept separate from independent accuracy evaluation labels; accuracy exclusion remains limited to frame 0 by default.",
                         "Body retention measures annotated box area, not body segmentation; >5% clipping allows approximate labels."],
                  clips={})
    for stem, refs in references["clips"].items():
        path = args.clips_dir / f"{stem}.mp4"
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open source video: {path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        results = {}
        for mode in ("baseline", "direct", "smooth", "deadzone"):
            directory = args.baseline_dir if mode == "baseline" else args.output_dir
            log_path = directory / f"{stem}_{mode}.json"
            logs = json.loads(log_path.read_text(encoding="utf-8"))
            results[mode] = evaluate_clip(logs, refs, width, height, count)
        report["clips"][stem] = dict(source_width=width, source_height=height,
                                     source_frame_count=count, source_fps=fps, modes=results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "metrics.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("clip mode group n mean_error max_error outside retained clipped>5% mean_step max_step travel_px")
    for stem, clip in report["clips"].items():
        for mode, result in clip["modes"].items():
            motion = result["motion"]
            for group, metric in result["groups"].items():
                values = [metric[k] for k in ("mean_center_error", "max_center_error",
                          "center_outside_fraction", "mean_body_retained_fraction")]
                formatted = " ".join("n/a" if v is None else f"{v:.4f}" for v in values)
                print(f"{stem} {mode} {group} {metric['visible_count']} {formatted} "
                      f"{metric['body_clipped_over_5pct_count']} "
                      f"{motion['mean_abs_frame_step_normalized']:.5f} "
                      f"{motion['max_frame_step_normalized']:.5f} {motion['total_movement_px']:.1f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
