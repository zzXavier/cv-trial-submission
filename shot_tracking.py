"""Transactional relocation across edits, around the existing local pose tracker.

Returned state dictionaries can be revised when the next frame confirms a
candidate. Render only after the inference pass has finished.
"""
from collections import deque
import copy
import math

import numpy as np

from appearance_search import TargetAppearance, find_target
from shot_detection import ShotDetector
from tracking import PoseTracker


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

    def _confirm(self, frame, index):
        pending = self.pending
        probe = copy.deepcopy(pending['tracker'])
        pose, state = self._observe(probe, frame)
        if (state['status'] != 'tracking'
                or self.appearance.distance(frame, pose) > self.appearance.max_distance
                or np.linalg.norm(probe.trusted_center - pending['tracker'].trusted_center)
                >= pending['tracker'].body_height * 0.25):
            self.pending = None
            return None

        old_center = self.tracker.trusted_center.copy()
        old_height = self.tracker.body_height
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
            reverse_probe = copy.deepcopy(reverse_tracker)
            old_pose, old_state = self._observe(reverse_probe, old_frame)
            if (old_state['status'] != 'tracking'
                    or self.appearance.distance(old_frame, old_pose) > self.appearance.max_distance):
                break
            row.update(old_state)
            row['reason'] = 'relocation_backfill'
            earliest = row
            reverse_tracker = reverse_probe

        moved_far = np.linalg.norm(pending['tracker'].trusted_center - old_center) > old_height * 0.3
        # An unconfirmed visual signal alone never moves the camera. For a
        # confirmed same-position match, continue normal camera smoothing.
        if moved_far:
            earliest['camera_reset'] = True
            earliest['reset_reason'] = ('cut_relocation' if pending['cut']
                                        else 'relocation_after_loss')
        self.tracker = probe
        self.pending = None
        self.unconfirmed_shot = False
        self.search_attempts = 0
        state['reason'] = 'relocation_confirmed'
        return state

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
            state = self._confirm(frame, index) if self.pending else None
            if state is None:
                if self.unconfirmed_shot:
                    state = self.tracker._hold('unconfirmed_shot', 0)
                    state['roi'] = list(self.tracker.roi)
                else:
                    probe = copy.deepcopy(self.tracker)
                    pose, state = self._observe(probe, frame)
                    distance = (self.appearance.distance(frame, pose)
                                if self.appearance is not None and state['status'] == 'tracking'
                                else math.inf)
                    # Pose rotation can change the visible cloth sample. Color
                    # is a gate for nonlocal reacquisition, not a veto on an
                    # otherwise continuous local pose.
                    self.tracker = probe
                    if math.isfinite(distance):
                        state['appearance_distance'] = float(distance)
                if state['status'] == 'tracking':
                    self.search_attempts = 0
                    if self.appearance is None:
                        self._bind_appearance(frame, pose)

                can_search = (self.appearance is not None and self.search_attempts < 12
                              and index - self.last_search >= self.search_stride)
                needs_search = cut or self.tracker.missed >= self.loss_trigger
                if can_search and needs_search:
                    search_frame, search_index, search_row = frame, index, state
                    if not self.unconfirmed_shot:
                        # The edit starts at the first missing observation, not
                        # at the frame that finally reaches the loss threshold.
                        for buffered_frame, buffered_row in reversed(self.history):
                            if (buffered_row['target_x'] is not None
                                    or buffered_row['frame_index'] < self.boundary_floor):
                                break
                            search_frame, search_index, search_row = (
                                buffered_frame, buffered_row['frame_index'], buffered_row)
                    result = find_target(search_frame, self.infer, self.appearance, self.tracker.body_height)
                    # If the boundary itself is blurred/occluded, also try the
                    # current frame. Thus one attempt uses at most 20 ROIs.
                    if result['candidate'] is None and search_index != index:
                        result = find_target(frame, self.infer, self.appearance, self.tracker.body_height)
                        search_frame, search_index, search_row = frame, index, state
                    self.last_search = index
                    self.search_attempts += 1
                    state['search_reason'] = result['reason']
                    state['search_candidates'] = result['candidates']
                    candidate = result['candidate']
                    if candidate is not None:
                        proposed = PoseTracker(candidate['box'], self.frame_size, self.fps)
                        observed = proposed.update(candidate['pose'])
                        if observed['status'] == 'tracking':
                            observed['roi'] = candidate.get('roi', list(proposed.roi))
                            self.pending = {'tracker': proposed, 'state': observed,
                                            'index': search_index, 'row': search_row, 'cut': cut}
                            search_row.update(status='recovering', reason='confirm_relocation',
                                              target_x=None, target_box=None)
                            confirmed = False
                            for buffered_frame, buffered_row in self.history:
                                if buffered_row['frame_index'] <= search_index:
                                    continue
                                if not confirmed:
                                    replacement = self._confirm(buffered_frame, buffered_row['frame_index'])
                                    if replacement is None:
                                        break
                                    confirmed = True
                                else:
                                    _, replacement = self._observe(self.tracker, buffered_frame)
                                buffered_row.update(replacement)
                            if confirmed:
                                _, replacement = self._observe(self.tracker, frame)
                                state.update(replacement)
                            elif self.pending is not None and search_index < index:
                                # At lower fps the loss threshold is two frames:
                                # the next observation is this frame, not history.
                                replacement = self._confirm(frame, index)
                                if replacement is not None:
                                    state.update(replacement)

        state.setdefault('camera_reset', False)
        state.update(frame_index=index, timestamp_s=round(index / self.fps, 6),
                     cut_proposed=cut, cut_score=float(signal['score']),
                     appearance_available=self.appearance is not None)
        self.history.append((frame, state))
        return state
