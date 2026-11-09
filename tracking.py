"""Single-target state and pose validation in original-image coordinates."""

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
