"""Bounded MoveNet search with a frozen, anchor-bound shirt-color reference.

This is a conservative clothing heuristic, not person re-identification. Similar
clothes, illumination changes, occlusion and a single-pose model missing people
remain limitations. A match needs temporal confirmation by the caller; ambiguous
or missing matches should retain the last trusted state or use a manual anchor.
"""

import cv2
import numpy as np


MAX_APPEARANCE_DISTANCE = 0.32
MIN_MATCH_MARGIN = 0.08
TORSO = [5, 6, 12, 11]


def _pose_box(frame, keypoints):
    """Validate a complete torso and return a padded, source-bounded body box."""
    pose = np.asarray(keypoints, dtype=float)
    if pose.shape != (17, 3) or not np.isfinite(pose).all():
        return None
    visible = pose[:, 2] > 0.25
    if visible.sum() < 6 or not visible[TORSO].all():
        return None
    height, width = frame.shape[:2]
    torso = pose[TORSO, :2]
    if (torso < 0).any() or (torso >= [width, height]).any():
        return None
    shoulders, hips = pose[5:7, :2].mean(axis=0), pose[11:13, :2].mean(axis=0)
    length = float(np.linalg.norm(shoulders - hips))
    shoulder_width = float(np.linalg.norm(pose[5, :2] - pose[6, :2]))
    hip_width = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
    if length < 8 or max(shoulder_width, hip_width) > 2 * length:
        return None
    points = pose[visible, :2]
    center = (shoulders + hips) / 2
    if (np.abs(points - center).max(axis=0) > [3 * length, 4 * length]).any():
        return None
    low = np.maximum(points.min(axis=0) - 0.12 * length, 0)
    high = np.minimum(points.max(axis=0) + 0.12 * length, [width, height])
    if (high <= low).any():
        return None
    return np.r_[low, high - low].tolist()


def _descriptor(frame, pose, anchor_box=None):
    if _pose_box(frame, pose) is None:
        return None
    # Inset the shoulder/hip quadrilateral to reduce skin and background pixels.
    torso = np.asarray(pose, dtype=float)[TORSO, :2]
    shoulders, hips = torso[:2].mean(axis=0), torso[2:].mean(axis=0)
    axis = hips - shoulders
    length = np.linalg.norm(axis)
    if min(np.linalg.norm(torso[0] - torso[1]),
           np.linalg.norm(torso[2] - torso[3])) < 0.1 * length:
        # Side views can project left/right joints almost onto one another.
        # Sample a narrow strip along the visible torso, retaining confidence
        # checks and clipping to the first-frame anchor when constructing it.
        across = np.array([-axis[1], axis[0]]) * 0.08
        center = (shoulders + hips) / 2
        torso = np.array([center - 0.35 * axis - across,
                          center - 0.35 * axis + across,
                          center + 0.35 * axis + across,
                          center + 0.35 * axis - across])
    else:
        torso = torso.mean(axis=0) + 0.7 * (torso - torso.mean(axis=0))
    polygon = np.rint(torso).astype(np.int32)
    height, width = frame.shape[:2]
    lower, upper = np.array([0, 0]), np.array([width - 1, height - 1])
    if anchor_box is not None:
        lower = np.maximum(lower, np.ceil(anchor_box[:2]).astype(int))
        upper = np.minimum(upper, np.ceil(anchor_box[:2] + anchor_box[2:]).astype(int) - 1)
    polygon = np.clip(polygon, lower, upper).astype(np.int32)
    x, y, w, h = cv2.boundingRect(polygon)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon - [x, y], 255)
    if np.count_nonzero(mask) < 24:
        return None
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    # Hue is unstable on gray/white or nearly black fabric.
    hsv[:, :, 0][hsv[:, :, 1] < 40] = 0
    hsv[:, :, :2][hsv[:, :, 2] < 35] = 0
    hs = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256])
    # Gentle hue smoothing avoids a hard discontinuity at histogram-bin edges.
    hs = 0.5 * hs + 0.25 * (np.roll(hs, 1, axis=0) + np.roll(hs, -1, axis=0))
    value = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).ravel()
    value = np.convolve(value, [0.25, 0.5, 0.25], mode='same')
    return (hs.ravel() / hs.sum()).astype(np.float32), (value / value.sum()).astype(np.float32)


class TargetAppearance:
    """Freeze shirt histograms only when all four torso corners bind to an anchor."""

    max_distance = MAX_APPEARANCE_DISTANCE

    def __init__(self, frame, pose, anchor_box):
        box = np.asarray(anchor_box, dtype=float)
        height, width = frame.shape[:2]
        if (box.shape != (4,) or not np.isfinite(box).all()
                or (box[:2] < 0).any() or (box[2:] <= 0).any()
                or (box[:2] + box[2:] > [width, height]).any()):
            raise ValueError('Appearance anchor must be a valid source-frame box')
        descriptor = _descriptor(frame, pose, anchor_box=box)
        if descriptor is None:
            raise ValueError('Appearance reference needs a valid visible torso')
        torso = np.asarray(pose)[TORSO, :2]
        if (torso < box[:2]).any() or (torso > box[:2] + box[2:]).any():
            raise ValueError('Appearance reference torso is outside the target anchor')
        self._reference = descriptor

    def distance(self, frame, pose):
        """Return color distance in [0, 1], or infinity for an unusable torso."""
        descriptor = _descriptor(frame, pose)
        if descriptor is None:
            return float('inf')
        distances = [cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
                     for a, b in zip(self._reference, descriptor)]
        return float(0.6 * distances[0] + 0.4 * distances[1])


def _search_rois(frame, reference_height):
    if not np.isfinite(reference_height) or reference_height <= 0:
        raise ValueError('reference_height must be finite and positive')
    height, width = frame.shape[:2]
    full_side = float(max(width, height))
    rois = [((width - full_side) / 2, (height - full_side) / 2, full_side)]
    side = min(full_side, max(64.0, 2.2 * reference_height, full_side / 2.5))
    xs = np.linspace(min(side, width) / 2, width - min(side, width) / 2, 3)
    ys = np.linspace(min(side, height) / 2, height - min(side, height) / 2, 3)
    for cy in ys:
        for cx in xs:
            roi = (float(cx - side / 2), float(cy - side / 2), float(side))
            if roi not in rois:
                rois.append(roi)
    return rois


def _same_person(first, second):
    a, b = first['pose'], second['pose']
    center_a, center_b = a[TORSO, :2].mean(axis=0), b[TORSO, :2].mean(axis=0)
    height_a, height_b = first['box'][3], second['box'][3]
    return (0.67 <= height_a / height_b <= 1.5
            and np.linalg.norm(center_a - center_b) <= max(8, 0.12 * min(height_a, height_b)))


def find_target(frame, infer, appearance, reference_height):
    """Run at most ten ROIs and return a unique color-compatible candidate.

    ``infer(frame, (left, top, side))`` must return original-frame x/y/score
    keypoints. Lower ``score`` is better. Distinct-person ambiguity is tested
    after deduplication, with no preference for the previous target position.
    Neither this function nor distance() modifies the appearance reference.
    """
    candidates = []
    for roi in _search_rois(frame, reference_height):
        pose = np.asarray(infer(frame, roi), dtype=float)
        box = _pose_box(frame, pose)
        if box is None:
            continue
        score = appearance.distance(frame, pose)
        if np.isfinite(score):
            candidates.append(dict(pose=pose.copy(), box=box, score=float(score), roi=list(roi)))
    distinct = []
    for candidate in sorted(candidates, key=lambda row: row['score']):
        if not any(_same_person(candidate, other) for other in distinct):
            distinct.append(candidate)
    summaries = [dict(box=c['box'], score=c['score']) for c in distinct]
    selected = None
    if not distinct:
        reason = 'no_candidates'
    elif distinct[0]['score'] > MAX_APPEARANCE_DISTANCE:
        reason = 'appearance_mismatch'
    elif len(distinct) > 1 and distinct[1]['score'] - distinct[0]['score'] < MIN_MATCH_MARGIN:
        reason = 'ambiguous'
    else:
        reason, selected = 'matched', distinct[0]
    return dict(candidate=selected, reason=reason, candidates=summaries)
