import unittest
from unittest.mock import patch

import numpy as np

from test_tracking import person
from shot_tracking import ShotTracker, validate_anchors


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
        self.patch = patch('shot_tracking.TargetAppearance', Appearance)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.search_patch = patch('shot_tracking.find_target')
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


if __name__ == '__main__':
    unittest.main()
