"""Local pose tracking and recovery across video edits."""

from collections import deque
import copy
import math

import cv2
import numpy as np


def map_pose(keypoints, roi):
    """Map normalized MoveNet y/x/score into original-image x/y/score."""
    left, top, side = roi
    result = np.asarray(keypoints, dtype=float)[:, [1, 0, 2]].copy()
    result[:, :2] = result[:, :2] * side + [left, top]
    return result


class PoseTracker:
    def __init__(self, initial_box, frame_size, fps):
        x, y, w, h = np.asarray(initial_box, dtype=float)
        width, height = frame_size
        if not np.isfinite([x, y, w, h, fps]).all() or fps <= 0:
            raise ValueError('target and fps must be finite, with positive fps')
        if min(w, h) <= 0 or min(x, y) < 0 or x + w > width or y + h > height:
            raise ValueError('target box must be positive and inside the source frame')
        self.fps = fps
        self.frame_size = np.array(frame_size, dtype=float)
        self.trusted_center = np.array([x + w / 2, y + h / 2])
        self.roi_center = self.trusted_center.copy()
        self.side = max(w, h) * 1.6
        self.body_height = h
        self.torso_size = h * 0.25
        self.missed = 0
        self.pending_center = None

    @property
    def roi(self):
        # Bounded expansion covers brief misses without letting a bad pose steer it.
        expansion = 1 + min(self.missed / (self.fps * 0.5), 0.5)
        side = self.side * expansion
        return (*map(float, self.roi_center - side / 2), float(side))

    def _hold(self, reason, count, pending=None):
        self.missed += 1
        self.pending_center = pending
        return {'status': 'recovering' if pending is not None else 'lost',
                'reason': reason, 'target_x': None, 'target_box': None,
                'valid_keypoints': count}

    def update(self, keypoints):
        pose = np.asarray(keypoints, dtype=float)
        if pose.shape != (17, 3) or not np.isfinite(pose).all():
            return self._hold('invalid_pose', 0)
        visible = pose[:, 2] > 0.25
        count = int(visible.sum())
        if count < 6 or not visible[5:7].any() or not visible[11:13].any():
            return self._hold('weak_torso', count)

        shoulders = pose[5:7, :2][visible[5:7]].mean(axis=0)
        hips = pose[11:13, :2][visible[11:13]].mean(axis=0)
        center = (shoulders + hips) / 2
        torso_size = float(np.linalg.norm(shoulders - hips))
        scale_ratio = torso_size / self.torso_size
        if not 0.35 <= scale_ratio <= 2.8:
            return self._hold('scale_jump', count)
        allowance = self.body_height * min(0.3 + self.missed / self.fps, 0.8)
        if np.linalg.norm(center - self.trusted_center) > allowance:
            return self._hold('position_jump', count)

        points = pose[visible, :2]
        # Check whole-pose geometry as well as its torso, including stray head points.
        if np.max(np.abs(points[:, 0] - center[0])) > self.body_height * 0.9:
            return self._hold('inconsistent_body', count)
        if np.max(np.abs(points[:, 1] - center[1])) > self.body_height * 1.1:
            return self._hold('inconsistent_body', count)

        if self.missed:
            consistent = (self.pending_center is not None and
                          np.linalg.norm(center - self.pending_center) < self.body_height * 0.25)
            if not consistent:
                return self._hold('confirm_recovery', count, center)

        low = np.maximum(points.min(axis=0) - self.body_height * 0.04, 0)
        high = np.minimum(points.max(axis=0) + self.body_height * 0.04, self.frame_size)
        box = np.r_[low, np.maximum(high - low, 1)]
        body_range = float(np.max(np.abs(points - hips)))
        desired_side = max(4 * torso_size, 2.4 * body_range, 64)
        desired_side = np.clip(desired_side, self.side * 0.75, self.side * 1.3)
        self.side = float(0.8 * self.side + 0.2 * desired_side)
        self.roi_center = hips.copy()
        self.trusted_center = center.copy()
        self.torso_size = 0.8 * self.torso_size + 0.2 * torso_size
        # Partial visibility must not teach the tracker that a torso is a full body.
        if visible[:5].any() and visible[15:17].any():
            self.body_height = 0.9 * self.body_height + 0.1 * float(box[3])
        self.missed = 0
        self.pending_center = None
        return {'status': 'tracking', 'reason': 'accepted',
                'target_x': float(center[0]), 'target_box': box.tolist(),
                'valid_keypoints': count}


class ShotDetector:
    """Propose cuts from thumbnail changes after compensating for exposure shifts."""

    def __init__(self, threshold=0.06):
        if not 0 < threshold <= 1:
            raise ValueError('threshold must be in (0, 1]')
        self.threshold = float(threshold)
        self.previous = None

    def update(self, frame):
        """Consume one original uint8 BGR frame and return a cut proposal."""
        if (frame is None or frame.ndim != 3 or frame.shape[2] != 3
                or frame.dtype != np.uint8 or frame.size == 0):
            raise ValueError('frame must be a nonempty uint8 BGR image')
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        current = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        previous = self.previous
        self.previous = current
        if previous is None:
            return {'cut': False, 'score': 0.0}

        difference = current - previous
        brightness_shift = float(np.median(difference[:, :, 0]))
        luma = np.abs(difference[:, :, 0] - brightness_shift)
        chroma = np.mean(np.abs(difference[:, :, 1:]), axis=2)
        change = np.maximum(luma, chroma) / 255.0
        score = float(np.mean(change))
        changed_fraction = float(np.mean(change >= 0.10))
        return {'cut': score >= self.threshold and changed_fraction >= 0.35, 'score': score}


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


def validate_anchors(rows, frame_size, frame_count):
    if not isinstance(rows, list):
        raise ValueError('Shot anchors must be a list of frame-indexed boxes')
    result = {}
    for row in rows:
        index = row.get('frame_index')
        if type(index) is not int or not 0 < index < frame_count or index in result:
            raise ValueError('Anchor frames must be unique and between 1 and the last source frame')
        box = [row[key] for key in ('x', 'y', 'w', 'h')]
        PoseTracker(box, frame_size, 30)  # Reuse finite/in-bounds geometry validation.
        result[index] = box
    return result


class ShotTracker:
    """Track one player and revise buffered observations after a confirmed cut."""

    def __init__(self, initial_box, frame_size, fps, infer, *, anchors=None, detector=None):
        self.frame_size, self.fps, self.infer = frame_size, fps, infer
        self.tracker = PoseTracker(initial_box, frame_size, fps)
        self.anchor_box = list(initial_box)
        self.anchors = anchors or {}
        self.appearance = None
        self.detector = detector or ShotDetector()
        self.pending = None
        self.unconfirmed_shot = False
        self.history = deque(maxlen=max(4, math.ceil(fps * 0.2) + 2))
        self.loss_trigger = max(2, round(fps * 0.1))
        self.search_stride = max(1, round(fps * 0.5))
        self.last_search = -self.search_stride
        self.search_attempts = 0
        self.boundary_floor = 0

    def _observe(self, tracker, frame):
        roi = tracker.roi
        pose = self.infer(frame, roi)
        state = tracker.update(pose)
        state['roi'] = list(roi)
        return pose, state

    def _bind_appearance(self, frame, pose):
        try:
            self.appearance = TargetAppearance(frame, pose, self.anchor_box)
        except ValueError:
            self.appearance = None

    def _confirm(self, frame):
        pending = self.pending
        probe = copy.deepcopy(pending['tracker'])
        pose, state = self._observe(probe, frame)
        if (state['status'] != 'tracking'
                or self.appearance.distance(frame, pose) > self.appearance.max_distance
                or np.linalg.norm(probe.trusted_center - pending['tracker'].trusted_center)
                >= pending['tracker'].body_height * 0.25):
            self.pending = None
            return None

        pending['row'].update(pending['state'])
        pending['row']['reason'] = 'relocation_confirmed'
        earliest = pending['row']
        reverse_tracker = copy.deepcopy(pending['tracker'])
        # Re-observe buffered frames, using their own pixels and poses. Stop at
        # any trusted observation, explicit boundary, or failed verification.
        for old_frame, row in reversed(self.history):
            if row['frame_index'] >= pending['index']:
                continue
            if row['frame_index'] < self.boundary_floor or row['target_x'] is not None:
                break
            old_pose, old_state = self._observe(reverse_tracker, old_frame)
            if (old_state['status'] != 'tracking'
                    or self.appearance.distance(old_frame, old_pose) > self.appearance.max_distance):
                break
            row.update(old_state)
            row['reason'] = 'relocation_backfill'
            earliest = row
        distance = np.linalg.norm(pending['tracker'].trusted_center - self.tracker.trusted_center)
        # An unconfirmed visual signal alone never moves the camera. For a
        # confirmed same-position match, continue normal camera smoothing.
        if distance > self.tracker.body_height * 0.3:
            earliest['camera_reset'] = True
            earliest['reset_reason'] = ('cut_relocation' if pending['cut']
                                        else 'relocation_after_loss')
        self.tracker = probe
        self.pending = None
        self.unconfirmed_shot = False
        self.search_attempts = 0
        state['reason'] = 'relocation_confirmed'
        return state

    def _reacquire(self, frame, index, state, cut):
        search_frame, search_row = frame, state
        if not self.unconfirmed_shot:
            # Start at the first buffered miss, before the loss threshold.
            for image, row in reversed(self.history):
                if row['target_x'] is not None or row['frame_index'] < self.boundary_floor:
                    break
                search_frame, search_row = image, row
        result = find_target(search_frame, self.infer, self.appearance, self.tracker.body_height)
        if result['candidate'] is None and search_row is not state:
            result = find_target(frame, self.infer, self.appearance, self.tracker.body_height)
            search_row = state

        self.last_search = index
        self.search_attempts += 1
        state.update(search_reason=result['reason'], search_candidates=result['candidates'])
        candidate = result['candidate']
        if candidate is None:
            return
        proposed = PoseTracker(candidate['box'], self.frame_size, self.fps)
        observed = proposed.update(candidate['pose'])
        if observed['status'] != 'tracking':
            return

        search_index = search_row.get('frame_index', index)
        observed['roi'] = candidate.get('roi', list(proposed.roi))
        self.pending = {'tracker': proposed, 'state': observed,
                        'index': search_index, 'row': search_row, 'cut': cut}
        search_row.update(status='recovering', reason='confirm_relocation', target_x=None, target_box=None)
        if search_index == index:
            return
        # Use the same confirmation path for a buffered frame or the current frame.
        for image, row in (*self.history, (frame, state)):
            if row.get('frame_index', index) <= search_index:
                continue
            if self.pending is None:
                _, replacement = self._observe(self.tracker, image)
            else:
                replacement = self._confirm(image)
                if replacement is None:
                    return
            row.update(replacement)

    def update(self, frame, index):
        if self.history and index != self.history[-1][1]['frame_index'] + 1:
            raise ValueError('Frames must be supplied in order')
        signal = self.detector.update(frame)
        cut = bool(signal['cut'])
        if cut:
            self.pending = None
            self.unconfirmed_shot = True
            self.boundary_floor = index
            if index - self.last_search >= self.search_stride:
                self.search_attempts = 0

        manual = index in self.anchors
        if index == 0 or manual:
            if manual:
                self.anchor_box = list(self.anchors[index])
                self.tracker = PoseTracker(self.anchor_box, self.frame_size, self.fps)
            self.pending = None
            self.unconfirmed_shot = False
            self.boundary_floor = index
            self.search_attempts = 0
            pose, state = self._observe(self.tracker, frame)
            self._bind_appearance(frame, pose)
            # The explicit anchor locates the target even if the pose is weak.
            state['camera_reset'] = True
            state['reset_reason'] = 'manual_anchor' if manual else 'initial_target'
            state['reset_box'] = self.anchor_box.copy()
        else:
            state = self._confirm(frame) if self.pending else None
            if state is None:
                if self.unconfirmed_shot:
                    state = self.tracker._hold('unconfirmed_shot', 0)
                    state['roi'] = list(self.tracker.roi)
                else:
                    pose, state = self._observe(self.tracker, frame)
                    if state['status'] == 'tracking':
                        self.search_attempts = 0
                        if self.appearance is None:
                            self._bind_appearance(frame, pose)

                can_search = (self.appearance is not None and self.search_attempts < 12
                              and index - self.last_search >= self.search_stride)
                needs_search = cut or self.tracker.missed >= self.loss_trigger
                if can_search and needs_search:
                    self._reacquire(frame, index, state, cut)

        state.setdefault('camera_reset', False)
        state.update(frame_index=index, timestamp_s=round(index / self.fps, 6),
                     cut_proposed=cut, cut_score=float(signal['score']),
                     appearance_available=self.appearance is not None)
        self.history.append((frame, state))
        return state
