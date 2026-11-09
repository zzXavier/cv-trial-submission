"""Regression tests for tracking, camera movement and evaluation."""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from evaluate import evaluate_clip
from track import CropController, build_crops, validate_replay
from tracking import PoseTracker, ShotDetector, ShotTracker, TargetAppearance, find_target, map_pose, validate_anchors


def person(dx=0.0, scale=1.0):
    # A stationary full-body pose, in original-image x/y coordinates.
    xy = np.array([
        [300, 120], [295, 115], [305, 115], [290, 120], [310, 120],
        [280, 150], [320, 150], [270, 190], [330, 190], [260, 220],
        [340, 220], [285, 220], [315, 220], [285, 280], [315, 280],
        [285, 340], [315, 340],
    ], dtype=float)
    xy = (xy - [300, 220]) * scale + [300 + dx, 220]
    return np.column_stack([xy, np.full(17, 0.8)])


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PoseTracker((250, 100, 100, 250), (1280, 720), 30)

    def test_weak_isolated_joint_cannot_move_target(self):
        pose = person(350)
        pose[:, 2] = 0.1
        pose[13, 2] = 0.26
        result = self.tracker.update(pose)
        self.assertIsNone(result['target_x'])
        self.assertEqual(self.tracker.trusted_center[0], 300)

    def test_remote_person_does_not_update_trusted_roi(self):
        self.assertEqual(self.tracker.update(person())['status'], 'tracking')
        center = self.tracker.trusted_center.copy()
        result = self.tracker.update(person(400, 0.4))
        self.assertIsNone(result['target_x'])
        np.testing.assert_array_equal(self.tracker.trusted_center, center)

    def test_short_loss_requires_consistent_recovery(self):
        self.tracker.update(person())
        self.tracker.update(np.zeros((17, 3)))
        self.assertIsNone(self.tracker.update(person(8))['target_x'])
        result = self.tracker.update(person(10))
        self.assertEqual(result['status'], 'tracking')
        self.assertAlmostEqual(result['target_x'], 310)

    def test_local_input_mapping_preserves_original_coordinates(self):
        pose = np.tile([0.25, 0.75, 0.9], (17, 1))  # normalized y/x/score
        result = map_pose(pose, (100, 200, 400))
        np.testing.assert_allclose(result[0], [400, 300, 0.9])

    def test_partial_occlusion_does_not_shrink_full_body_scale(self):
        self.tracker.update(person())
        partial = person()
        partial[:, 2] = 0.1
        partial[[5, 6, 7, 8, 11, 12], 2] = 0.8
        for _ in range(20):
            self.tracker.update(partial)
        result = self.tracker.update(person())
        self.assertEqual(result['status'], 'tracking')

    def test_invalid_target_box_rejected(self):
        for box in [(0, 0, 0, 10), (-1, 0, 100, 100), (1200, 0, 200, 100)]:
            with self.assertRaises(ValueError):
                PoseTracker(box, (1280, 720), 30)


def cut_scene():
    y, x = np.indices((180, 320))
    palette = np.array([
        [50, 80, 140], [180, 70, 60], [60, 170, 100], [140, 130, 65],
    ], dtype=np.uint8)
    return palette[(x // 40 + y // 30) % len(palette)]


class ShotDetectionTests(unittest.TestCase):
    def test_first_and_repeated_frames_do_not_propose_cuts(self):
        detector = ShotDetector()
        for _ in range(10):
            result = detector.update(cut_scene())
            self.assertFalse(result['cut'])
            self.assertEqual(result['score'], 0.0)

    def test_gradual_camera_motion_does_not_propose_cuts(self):
        detector = ShotDetector()
        original = cut_scene()
        for offset in range(0, 25, 2):
            moved = cv2.warpAffine(
                original, np.float32([[1, 0, offset], [0, 1, offset / 2]]),
                (320, 180), borderMode=cv2.BORDER_REFLECT,
            )
            self.assertFalse(detector.update(moved)['cut'])

    def test_hard_cut_is_proposed_once_then_new_shot_is_stable(self):
        detector = ShotDetector()
        detector.update(cut_scene())
        replacement = cv2.flip(cut_scene(), -1)
        result = detector.update(replacement)
        self.assertTrue(result['cut'])
        self.assertGreater(result['score'], 0)
        self.assertFalse(detector.update(replacement)['cut'])

    def test_uniform_brightness_shift_is_not_a_cut(self):
        detector = ShotDetector()
        original = cut_scene()
        detector.update(original)
        self.assertFalse(detector.update(original + np.uint8(40))['cut'])
        self.assertFalse(detector.update(original)['cut'])

    def test_color_change_is_not_removed_as_brightness_shift(self):
        detector = ShotDetector()
        detector.update(np.full((180, 320, 3), [180, 50, 50], np.uint8))
        result = detector.update(np.full((180, 320, 3), [50, 50, 180], np.uint8))
        self.assertTrue(result['cut'])

    def test_small_foreground_move_is_not_a_cut(self):
        detector = ShotDetector()
        first, second = cut_scene(), cut_scene()
        cv2.rectangle(first, (100, 80), (115, 130), (10, 10, 10), -1)
        cv2.rectangle(second, (150, 80), (165, 130), (10, 10, 10), -1)
        detector.update(first)
        self.assertFalse(detector.update(second)['cut'])


def appearance_pose(x, y=200):
    xy = np.array([[0, -90], [-5, -95], [5, -95], [-10, -90], [10, -90],
                   [-25, -60], [25, -60], [-35, -25], [35, -25],
                   [-45, 10], [45, 10], [-20, 10], [20, 10],
                   [-20, 65], [20, 65], [-20, 120], [20, 120]], dtype=float)
    return np.column_stack([xy + [x, y], np.full(17, .9)])


def clothing_scene(people, shape=(600, 1000, 3)):
    frame = np.full(shape, 100, dtype=np.uint8)
    for pose, color in people:
        cv2.fillConvexPoly(frame, np.rint(pose[[5, 6, 12, 11], :2]).astype(np.int32), color)
    return frame


class AppearanceSearchTests(unittest.TestCase):
    def setUp(self):
        self.initial = appearance_pose(150)
        self.initial_frame = clothing_scene([(self.initial, (20, 30, 220))])
        self.appearance = TargetAppearance(self.initial_frame, self.initial,
                                           [90, 95, 120, 240])

    def test_first_template_cannot_bind_a_person_outside_the_anchor(self):
        with self.assertRaises(ValueError):
            TargetAppearance(self.initial_frame, appearance_pose(600), [90, 95, 120, 240])

    def test_weak_torso_cannot_create_or_match_a_template(self):
        weak = self.initial.copy()
        weak[5, 2] = .1
        with self.assertRaises(ValueError):
            TargetAppearance(self.initial_frame, weak, [90, 95, 120, 240])
        self.assertTrue(np.isinf(self.appearance.distance(self.initial_frame, weak)))

    def test_confident_side_profile_can_seed_and_match_after_a_cut(self):
        # Perspective can collapse left/right joints despite a visible shirt.
        profile = appearance_pose(150)
        profile[5, :2], profile[6, :2] = [154, 140], [156, 140]
        profile[11, :2], profile[12, :2] = [145, 210], [146, 210]
        initial = np.full((600, 1000, 3), 100, dtype=np.uint8)
        shirt = np.array([[145, 140], [165, 140], [155, 210], [135, 210]])
        cv2.fillConvexPoly(initial, shirt, (20, 30, 220))
        appearance = TargetAppearance(initial, profile, [90, 95, 120, 240])
        distant = profile.copy()
        distant[:, 0] += 700
        changed = np.full_like(initial, 100)
        cv2.fillConvexPoly(changed, shirt + [700, 0], (20, 30, 220))
        result = find_target(changed, lambda image, roi: distant, appearance, 240)
        self.assertIsNotNone(result['candidate'])
        self.assertLess(result['candidate']['score'], appearance.max_distance)

    def test_distant_matching_clothes_win_over_nearby_different_clothes(self):
        distractor, target = appearance_pose(150), appearance_pose(850)
        frame = clothing_scene([(distractor, (220, 30, 20)), (target, (20, 30, 220))])
        calls = []

        def infer(image, roi):
            calls.append(roi)
            left, top, side = roi
            cx = left + side / 2
            return distractor if abs(cx - 150) < abs(cx - 850) else target

        result = find_target(frame, infer, self.appearance, 240)
        self.assertEqual(result['reason'], 'matched')
        self.assertAlmostEqual(result['candidate']['pose'][5, 0], 825)
        self.assertLessEqual(len(calls), 10)
        # Source-space searches cover the complete image, including frame corners.
        for x, y in [(0, 0), (999, 0), (0, 599), (999, 599)]:
            self.assertTrue(any(l <= x <= l + s and t <= y <= t + s for l, t, s in calls))
        json.dumps(result['candidates'], allow_nan=False)

    def test_same_clothes_on_two_distinct_people_are_ambiguous(self):
        left, right = appearance_pose(150), appearance_pose(850)
        frame = clothing_scene([(left, (20, 30, 220)), (right, (20, 30, 220))])
        result = find_target(frame, lambda image, roi: left if roi[0] + roi[2] / 2 < 500 else right,
                             self.appearance, 240)
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['reason'], 'ambiguous')
        self.assertEqual(len(result['candidates']), 2)

    def test_unrelated_single_candidate_is_refused(self):
        other = appearance_pose(850)
        frame = clothing_scene([(other, (220, 30, 20))])
        result = find_target(frame, lambda image, roi: other, self.appearance, 240)
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['reason'], 'appearance_mismatch')

    def test_repeated_detections_of_one_person_are_not_false_ambiguity(self):
        target = appearance_pose(850)
        frame = clothing_scene([(target, (20, 30, 220))])
        result = find_target(frame, lambda image, roi: target, self.appearance, 240)
        self.assertIsNotNone(result['candidate'])
        self.assertEqual(len(result['candidates']), 1)
        self.assertTrue(all(np.isfinite(result['candidate']['box'])))

    def test_rejected_matches_cannot_reteach_the_frozen_reference(self):
        other = appearance_pose(850)
        frame = clothing_scene([(other, (220, 30, 20))])
        before = self.appearance.distance(frame, other)
        for _ in range(10):
            find_target(frame, lambda image, roi: other, self.appearance, 240)
        self.assertEqual(self.appearance.distance(frame, other), before)
        self.assertAlmostEqual(self.appearance.distance(self.initial_frame, self.initial), 0)

    def test_inconsistent_or_empty_poses_do_not_become_candidates(self):
        malformed = self.initial.copy()
        malformed[0, 0] = 950
        for pose in (np.zeros((17, 3)), malformed, np.zeros((3, 3))):
            with self.subTest(pose=pose.shape):
                result = find_target(self.initial_frame, lambda image, roi: pose,
                                     self.appearance, 240)
                self.assertIsNone(result['candidate'])
                self.assertEqual(result['reason'], 'no_candidates')


class Detector:
    def __init__(self, cuts=()):
        self.cuts = cuts
        self.index = -1

    def update(self, frame):
        self.index += 1
        return {'cut': self.index in self.cuts, 'score': 0.2 if self.index in self.cuts else 0.0}


class Appearance:
    max_distance = 0.45

    def __init__(self, frame, pose, anchor_box):
        self.anchor = list(anchor_box)

    def distance(self, frame, pose):
        return 0.0


class ShotTrackingTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch('tracking.TargetAppearance', Appearance)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.search_patch = patch('tracking.find_target')
        self.search = self.search_patch.start()
        self.addCleanup(self.search_patch.stop)
        self.search.side_effect = lambda frame, infer, appearance, height: {
            'candidate': {'pose': self.actual_pose(frame), 'box': self.box(frame), 'score': 0.0},
            'reason': 'matched', 'candidates': []}

    def frame(self, x):
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[0, 0, :2] = [x // 256, x % 256]
        return frame

    def target_x(self, frame):
        return int(frame[0, 0, 0]) * 256 + int(frame[0, 0, 1])

    def actual_pose(self, frame):
        return person(self.target_x(frame) - 300)

    def box(self, frame):
        return [self.target_x(frame) - 50, 100, 100, 250]

    def infer(self, frame, roi):
        x = self.target_x(frame)
        left, top, side = roi
        return self.actual_pose(frame) if left <= x <= left + side else np.zeros((17, 3))

    def tracker(self, cuts=(), anchors=None):
        return ShotTracker((250, 100, 100, 250), (1280, 720), 30,
                           self.infer, anchors=anchors, detector=Detector(cuts))

    def test_continuous_track_does_not_search_or_reset_camera(self):
        tracker = self.tracker()
        rows = [tracker.update(self.frame(300 + i), i) for i in range(12)]
        self.assertTrue(all(r['status'] == 'tracking' for r in rows))
        self.assertFalse(any(r.get('camera_reset') for r in rows[1:]))
        self.search.assert_not_called()

    def test_clothing_sample_change_does_not_override_continuous_local_geometry(self):
        tracker = self.tracker()
        tracker.update(self.frame(300), 0)
        with patch.object(Appearance, 'distance', return_value=0.9):
            row = tracker.update(self.frame(302), 1)
        self.assertEqual(row['target_x'], 302)
        self.search.assert_not_called()

    def test_cut_waits_for_confirmation_and_revises_actual_boundary_observation(self):
        tracker = self.tracker(cuts=(1,))
        tracker.update(self.frame(300), 0)
        boundary = tracker.update(self.frame(900), 1)
        self.assertIsNone(boundary['target_x'])
        self.assertEqual(tracker.tracker.trusted_center[0], 300)
        confirmed = tracker.update(self.frame(905), 2)
        self.assertEqual(boundary['target_x'], 900)
        self.assertTrue(boundary['camera_reset'])
        self.assertEqual(confirmed['target_x'], 905)
        self.assertFalse(confirmed['camera_reset'])

    def test_unmatched_or_ambiguous_search_cannot_replace_target(self):
        self.search.side_effect = None
        self.search.return_value = {'candidate': None, 'reason': 'ambiguous', 'candidates': []}
        tracker = self.tracker(cuts=(1,))
        tracker.update(self.frame(300), 0)
        row = tracker.update(self.frame(900), 1)
        self.assertIsNone(row['target_x'])
        self.assertFalse(row['camera_reset'])
        self.assertEqual(tracker.tracker.trusted_center[0], 300)
        self.assertEqual(row['search_reason'], 'ambiguous')

    def test_old_location_person_cannot_bypass_unconfirmed_cut_matching(self):
        self.search.side_effect = None
        self.search.return_value = {'candidate': None, 'reason': 'ambiguous', 'candidates': []}
        tracker = self.tracker(cuts=(1,))
        tracker.update(self.frame(300), 0)
        rows = [tracker.update(self.frame(300), i) for i in range(1, 6)]
        self.assertTrue(all(row['target_x'] is None for row in rows))

    def test_same_background_jump_uses_loss_fallback_and_observed_backfill(self):
        tracker = self.tracker()
        rows = [tracker.update(self.frame(x), i)
                for i, x in enumerate([300, 900, 902, 904, 906, 908])]
        self.assertTrue(self.search.called)
        self.assertEqual(rows[1]['target_x'], 900)
        self.assertTrue(rows[1]['camera_reset'])
        self.assertEqual(rows[2]['target_x'], 902)
        self.assertEqual(rows[-1]['target_x'], 908)
        self.assertEqual(self.target_x(self.search.call_args.args[0]), 900)

    def test_loss_recovery_uses_current_frame_as_confirmation_at_lower_fps(self):
        tracker = ShotTracker((250, 100, 100, 250), (1280, 720), 20,
                              self.infer, detector=Detector())
        rows = [tracker.update(self.frame(x), i) for i, x in enumerate([300, 900, 902, 904])]
        self.assertEqual([row['target_x'] for row in rows], [300, 900, 902, 904])
        self.assertTrue(rows[1]['camera_reset'])

    def test_new_cut_cancels_unconfirmed_candidate(self):
        tracker = self.tracker(cuts=(1, 2))
        tracker.update(self.frame(300), 0)
        first = tracker.update(self.frame(900), 1)
        tracker.update(self.frame(300), 2)
        for i in range(3, 19):
            tracker.update(self.frame(300), i)
        self.assertIsNone(first['target_x'])
        self.assertEqual(tracker.tracker.trusted_center[0], 300)

    def test_manual_anchor_overrides_old_location_and_rebuilds_appearance(self):
        tracker = self.tracker(anchors={1: [850, 100, 100, 250]})
        tracker.update(self.frame(300), 0)
        row = tracker.update(self.frame(900), 1)
        self.assertTrue(row['camera_reset'])
        self.assertEqual(row['target_x'], 900)
        self.assertEqual(row['reset_reason'], 'manual_anchor')
        self.assertEqual(tracker.appearance.anchor[0], 850)

    def test_search_frequency_and_attempts_are_bounded(self):
        self.search.side_effect = None
        self.search.return_value = {'candidate': None, 'reason': 'no_match', 'candidates': []}
        tracker = self.tracker()
        tracker.update(self.frame(300), 0)
        for i in range(1, 301):
            tracker.update(self.frame(900), i)
        self.assertGreater(self.search.call_count, 0)
        self.assertLessEqual(tracker.search_attempts, 12)
        self.assertLessEqual(self.search.call_count, 24)  # Boundary + current frame per attempt.

    def test_anchor_validation_rejects_duplicate_out_of_range_and_invalid_boxes(self):
        for anchors in ([{'frame_index': 100, 'x': 1, 'y': 1, 'w': 10, 'h': 10}],
                        [{'frame_index': 2, 'x': -1, 'y': 1, 'w': 10, 'h': 10}],
                        [{'frame_index': 2, 'x': 1, 'y': 1, 'w': 10, 'h': 10}] * 2):
            with self.subTest(anchors=anchors), self.assertRaises(ValueError):
                validate_anchors(anchors, (1280, 720), 100)


class CropControllerTests(unittest.TestCase):
    def test_direct_centers_target_and_clamps_to_source(self):
        camera = CropController(1920, 1080, 30, 960, mode="direct")
        self.assertEqual(camera.crop_w, 608)
        self.assertEqual(camera.update(1000), 696)
        self.assertEqual(camera.update(-200), 0)
        self.assertEqual(camera.update(2400), 1312)

    def test_deadzone_holds_small_target_jitter(self):
        camera = CropController(1920, 1080, 30, 960)
        positions = [camera.update(x) for x in [960, 981, 930, 1000, 945] * 10]
        self.assertEqual(set(positions), {656})

    def test_deadzone_moves_only_excess_beyond_zone(self):
        camera = CropController(1920, 1080, 30, 960)
        for _ in range(120):
            position = camera.update(1100)
        self.assertAlmostEqual(position, 705, delta=1)

    def test_sustained_motion_follows_with_bounded_steps(self):
        camera = CropController(1920, 1080, 30, 600)
        positions = [camera.update(600 + frame * 5) for frame in range(121)]
        self.assertGreater(positions[-1] - positions[0], 450)
        self.assertLessEqual(max(b - a for a, b in zip(positions, positions[1:])), 21)
        self.assertLess(abs(1200 - positions[-1] - 304), 150)

    def test_smooth_mode_centers_where_deadzone_holds(self):
        smooth = CropController(1920, 1080, 30, 960, mode="smooth")
        deadzone = CropController(1920, 1080, 30, 960)
        for _ in range(90):
            smooth_x = smooth.update(1000)
            deadzone_x = deadzone.update(1000)
        self.assertEqual(smooth_x, 696)
        self.assertEqual(deadzone_x, 656)

    def test_fast_lateral_motion_leaves_room_for_extended_action(self):
        for direction in (-1, 1):
            camera = CropController(2400, 1080, 30, 1200)
            previous = round(camera.x)
            for frame in range(1, 31):
                target = 1200 + direction * frame * 15
                position = camera.update(target, (target - 100, 400, 200, 400))
                self.assertLessEqual(abs(position - previous), 21)
                # A hand-held object extends 100 px beyond the body on the
                # moving side. This reproduces the late crop found in video QA.
                action_edge = target + direction * 200
                clearance = (position + camera.crop_w - action_edge if direction > 0
                             else action_edge - position)
                self.assertGreaterEqual(clearance, 10)
                previous = position

    def test_missing_target_holds_without_drift(self):
        for mode in ("direct", "smooth", "deadzone"):
            with self.subTest(mode=mode):
                camera = CropController(1920, 1080, 30, 600, mode=mode)
                position = camera.update(1000)
                self.assertEqual([camera.update(None) for _ in range(60)], [position] * 60)

    def test_target_returning_to_deadzone_stops_immediately(self):
        camera = CropController(1920, 1080, 30, 600)
        for _ in range(10):
            position = camera.update(1200)
        stopped = [camera.update(position + 304) for _ in range(30)]
        self.assertEqual(set(stopped), {position})

    def test_same_elapsed_time_has_comparable_motion_at_different_fps(self):
        for mode in ("smooth", "deadzone"):
            endpoints = []
            for fps in (24, 30, 60):
                camera = CropController(1920, 1080, fps, 600, mode=mode)
                for frame in range(fps):
                    position = camera.update(600 + 300 * (frame + 1) / fps)
                endpoints.append(position)
            self.assertLessEqual(max(endpoints) - min(endpoints), 6)

    def test_all_modes_stay_in_bounds_and_allow_narrow_source(self):
        for mode in ("direct", "smooth", "deadzone"):
            camera = CropController(1920, 1080, 30, -100, mode=mode)
            for target in [-1000] * 60 + [3000] * 180:
                self.assertTrue(0 <= camera.update(target) <= 1312)
            narrow = CropController(320, 1080, 30, 160, mode=mode)
            self.assertEqual(narrow.crop_w, 320)
            self.assertEqual(narrow.update(1000), 0)

    def test_body_box_eventually_has_margin_without_instant_jump(self):
        camera = CropController(1920, 1080, 30, 960)
        first = camera.update(960, (900, 400, 400, 500))
        self.assertLessEqual(first - 656, 21)
        for _ in range(120):
            position = camera.update(960, (900, 400, 400, 500))
        self.assertLessEqual(position, 880)
        self.assertGreaterEqual(position + 608, 1320)

    def test_contained_limb_variation_does_not_move_camera(self):
        camera = CropController(1920, 1080, 30, 960)
        positions = [camera.update(960, box) for box in
                     [(830, 400, 260, 500), (780, 390, 360, 520)] * 30]
        self.assertEqual(set(positions), {656})

    def test_oversized_body_does_not_displace_stable_target(self):
        camera = CropController(1920, 1080, 30, 960)
        self.assertEqual(camera.update(960, (300, 200, 1100, 700)), 656)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"fps": 0}, {"fps": float("nan")}, {"width": 0},
                       {"height": -1}, {"mode": "other"}, {"response_s": 0},
                       {"max_speed": 0}, {"deadzone": 0.5}):
            options = dict(width=1920, height=1080, fps=30, initial_x=960)
            options.update(kwargs)
            with self.subTest(options=kwargs), self.assertRaises(ValueError):
                CropController(**options)


class PipelineTests(unittest.TestCase):
    def test_confirmed_boundary_resets_camera_without_pan_across_empty_frame(self):
        states = [{'frame_index': 0, 'target_x': 300, 'target_box': [250, 100, 100, 250]},
                  {'frame_index': 1, 'target_x': 900, 'target_box': [850, 100, 100, 250],
                   'camera_reset': True}]
        rows = build_crops(states, (1280, 720), 30, [250, 100, 100, 250], 'deadzone')
        self.assertAlmostEqual(rows[1]['crop_x'] + rows[1]['crop_w'] / 2, 900, delta=1)
        states[1]['camera_reset'] = False
        continuous = build_crops(states, (1280, 720), 30, [250, 100, 100, 250], 'deadzone')
        self.assertLessEqual(abs(continuous[1]['crop_x'] - continuous[0]['crop_x']), 14)

    def test_explicit_anchor_positions_camera_when_pose_is_temporarily_weak(self):
        states = [{'frame_index': 0, 'target_x': None, 'target_box': None,
                   'camera_reset': True, 'reset_box': [850, 100, 100, 250]}]
        row = build_crops(states, (1280, 720), 30, [250, 100, 100, 250], 'deadzone')[0]
        self.assertAlmostEqual(row['crop_x'] + row['crop_w'] / 2, 900, delta=1)

    def test_replay_rejects_old_format_changed_source_or_changed_anchors(self):
        metadata = dict(schema_version=2, source_sha256='abc', initial_box=[1, 2, 3, 4], anchors=[])
        validate_replay({**metadata, 'frames': []}, metadata)
        for altered in ({'schema_version': 1}, {'source_sha256': 'def'},
                        {'anchors': [{'frame_index': 9}]}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                validate_replay({**metadata, **altered, 'frames': []}, metadata)


def reference(frame=0, x=40, y=10, w=20, h=20, sample="regular", visible=True):
    return dict(frame_index=frame, x=x, y=y, w=w, h=h, sample=sample, visible=visible)


def crop(frame=0, x=0, y=0, w=100, h=100):
    return dict(frame_index=frame, crop_x=x, crop_y=y, crop_w=w, crop_h=h)


class EvaluationTests(unittest.TestCase):
    def evaluate(self, *args, **kwargs):
        # Geometry fixtures often contain just frame 0; explicitly opt in for them.
        return evaluate_clip(*args, exclude_initialization=False, **kwargs)

    def test_initialization_frame_is_excluded_by_default_and_reported(self):
        result = evaluate_clip([crop(0), crop(1)],
                               [reference(), reference(1, x=100)], 200, 100)
        self.assertEqual(result["groups"]["regular"]["reference_count"], 1)
        self.assertEqual(result["excluded_initialization_reference_count"], 1)
        self.assertAlmostEqual(result["groups"]["regular"]["mean_center_error"], 0.6)
        self.assertEqual([s["frame_index"] for s in result["samples"]], [1])

    def test_crop_center_error_uses_independent_reference(self):
        logs = [crop(0), crop(1)]
        logs[0]["target_x"] = 999  # Tracker output must never become ground truth.
        result = self.evaluate(logs, [reference(), reference(1, x=110)], 200, 100)
        regular = result["groups"]["regular"]
        self.assertAlmostEqual(regular["mean_center_error"], 0.35)
        self.assertAlmostEqual(regular["max_center_error"], 0.7)
        self.assertEqual(regular["center_outside_fraction"], 0.5)
        self.assertEqual(regular["mean_body_retained_fraction"], 0.5)
        self.assertEqual(regular["body_clipped_over_5pct_count"], 1)

    def test_offset_crop_uses_both_source_axes_for_body_containment(self):
        result = self.evaluate([crop(x=100, y=50)],
                               [reference(x=90, y=40, w=40, h=40)], 300, 200)
        sample = result["samples"][0]
        self.assertAlmostEqual(sample["center_error"], 0.4)
        self.assertAlmostEqual(sample["body_retained_fraction"], 0.5625)
        self.assertFalse(sample["center_outside"])

    def test_vertical_center_outside_is_reported_even_with_zero_horizontal_error(self):
        result = self.evaluate([crop(y=50, h=50)], [reference(y=0)], 100, 100)
        sample = result["samples"][0]
        self.assertEqual(sample["center_error"], 0)
        self.assertTrue(sample["center_outside"])
        self.assertEqual(sample["body_retained_fraction"], 0)

    def test_diagnostic_oversampling_is_not_pooled_with_regular_frames(self):
        result = self.evaluate([crop(0), crop(1), crop(2)],
                               [reference(), reference(1, x=100, sample="diagnostic"),
                                reference(2, x=100, sample="diagnostic")], 200, 100)
        self.assertEqual(result["groups"]["regular"]["mean_center_error"], 0)
        self.assertEqual(result["groups"]["regular"]["visible_count"], 1)
        self.assertEqual(result["groups"]["diagnostic"]["visible_count"], 2)
        self.assertAlmostEqual(result["groups"]["diagnostic"]["mean_center_error"], 0.6)

    def test_motion_measures_all_consecutive_frames_including_unlabelled_ones(self):
        result = self.evaluate([crop(0), crop(1, x=20), crop(2, x=10)],
                               [reference()], 200, 100)
        self.assertEqual(result["motion"]["total_movement_px"], 30)
        self.assertAlmostEqual(result["motion"]["mean_abs_frame_step_normalized"], 0.15)
        self.assertAlmostEqual(result["motion"]["max_frame_step_normalized"], 0.2)
        self.assertEqual(result["motion"]["camera_reset_count"], 0)
        self.assertEqual(result["motion"]["within_shot_total_movement_px"], 30)
        self.assertAlmostEqual(result["motion"]["within_shot_mean_abs_frame_step_normalized"], 0.15)
        self.assertAlmostEqual(result["motion"]["within_shot_max_frame_step_normalized"], 0.2)

    def test_reset_keeps_raw_jump_but_excludes_only_transition_into_reset(self):
        logs = [crop(0), crop(1, x=20), crop(2, x=420),
                crop(3, x=430), crop(4, x=428)]
        logs[0]["camera_reset"] = True
        logs[2]["camera_reset"] = True
        result = self.evaluate(logs, [reference(), reference(2, x=450)], 600, 100)
        motion = result["motion"]
        self.assertEqual(motion["total_movement_px"], 432)
        self.assertAlmostEqual(motion["mean_abs_frame_step_normalized"], 1.08)
        self.assertAlmostEqual(motion["max_frame_step_normalized"], 4.0)
        self.assertEqual(motion["camera_reset_count"], 1)  # Frame 0 is not a transition.
        self.assertEqual(motion["within_shot_total_movement_px"], 32)
        self.assertAlmostEqual(motion["within_shot_mean_abs_frame_step_normalized"], 0.32 / 3)
        self.assertAlmostEqual(motion["within_shot_max_frame_step_normalized"], 0.2)
        self.assertEqual([s["frame_index"] for s in result["samples"]], [0, 2])

    def test_all_transitions_reset_have_zero_within_shot_motion(self):
        logs = [crop(0), crop(1, x=400)]
        logs[1]["camera_reset"] = True
        motion = self.evaluate(logs, [reference()], 600, 100)["motion"]
        self.assertEqual(motion["total_movement_px"], 400)
        self.assertEqual(motion["camera_reset_count"], 1)
        self.assertEqual(motion["within_shot_total_movement_px"], 0)
        self.assertEqual(motion["within_shot_mean_abs_frame_step_normalized"], 0)
        self.assertEqual(motion["within_shot_max_frame_step_normalized"], 0)

    def test_camera_reset_flag_must_be_boolean_when_present(self):
        for value in (0, 1, "true", None, []):
            log = crop()
            log["camera_reset"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "camera_reset"):
                self.evaluate([log], [reference()], 100, 100)

    def test_baseline_defaults_to_full_source_height(self):
        result = self.evaluate([dict(frame_index=0, crop_x=0, crop_w=100)],
                               [reference(y=80)], 100, 100)
        self.assertEqual(result["samples"][0]["body_retained_fraction"], 1)
        self.assertEqual(result["motion"]["total_movement_px"], 0)

    def test_missing_label_frame_fails_instead_of_silently_dropping_it(self):
        with self.assertRaisesRegex(ValueError, "reference.*frame.*2"):
            self.evaluate([crop()], [reference(2)], 200, 100)

    def test_missing_duplicate_or_reordered_log_frames_are_rejected(self):
        for logs in ([crop(1)], [crop(0), crop(2)], [crop(0), crop(0)],
                     [crop(1), crop(0)]):
            with self.subTest(logs=logs), self.assertRaises(ValueError):
                self.evaluate(logs, [reference()], 200, 100)

    def test_truncated_log_is_rejected_when_source_frame_count_is_known(self):
        with self.assertRaisesRegex(ValueError, "frame count"):
            self.evaluate([crop()], [reference()], 200, 100, frame_count=2)

    def test_duplicate_references_and_unknown_groups_are_rejected(self):
        for refs in ([reference(), reference()], [reference(sample="other")], []):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                self.evaluate([crop()], refs, 200, 100)

    def test_invalid_geometry_is_rejected(self):
        for log in (crop(w=0), crop(x=-1), crop(x=150), crop(w=float("nan"))):
            with self.subTest(log=log), self.assertRaises(ValueError):
                self.evaluate([log], [reference()], 200, 100)
        with self.assertRaises(ValueError):
            self.evaluate([crop()], [reference(h=0)], 200, 100)

    def test_invisible_labels_are_explicitly_counted_and_not_given_fake_scores(self):
        result = self.evaluate([crop()], [reference(visible=False)], 200, 100)
        group = result["groups"]["regular"]
        self.assertEqual(group["reference_count"], 1)
        self.assertEqual(group["visible_count"], 0)
        self.assertIsNone(group["mean_center_error"])


if __name__ == "__main__":
    unittest.main()
