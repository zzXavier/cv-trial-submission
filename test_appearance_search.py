"""Synthetic identity checks; only the expensive pose model is substituted."""

import json
from pathlib import Path
import unittest

import cv2
import numpy as np

if Path(__file__).with_name('appearance_search.py').exists():
    from appearance_search import TargetAppearance, find_target
else:
    TargetAppearance = find_target = None


def person(x, y=200):
    xy = np.array([[0, -90], [-5, -95], [5, -95], [-10, -90], [10, -90],
                   [-25, -60], [25, -60], [-35, -25], [35, -25],
                   [-45, 10], [45, 10], [-20, 10], [20, 10],
                   [-20, 65], [20, 65], [-20, 120], [20, 120]], dtype=float)
    return np.column_stack([xy + [x, y], np.full(17, .9)])


def scene(people, shape=(600, 1000, 3)):
    frame = np.full(shape, 100, dtype=np.uint8)
    for pose, color in people:
        cv2.fillConvexPoly(frame, np.rint(pose[[5, 6, 12, 11], :2]).astype(np.int32), color)
    return frame


class AppearanceSearchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(TargetAppearance, 'Appearance search is not implemented')
        self.initial = person(150)
        self.initial_frame = scene([(self.initial, (20, 30, 220))])
        self.appearance = TargetAppearance(self.initial_frame, self.initial,
                                           [90, 95, 120, 240])

    def test_first_template_cannot_bind_a_person_outside_the_anchor(self):
        with self.assertRaises(ValueError):
            TargetAppearance(self.initial_frame, person(600), [90, 95, 120, 240])

    def test_weak_torso_cannot_create_or_match_a_template(self):
        weak = self.initial.copy()
        weak[5, 2] = .1
        with self.assertRaises(ValueError):
            TargetAppearance(self.initial_frame, weak, [90, 95, 120, 240])
        self.assertTrue(np.isinf(self.appearance.distance(self.initial_frame, weak)))

    def test_confident_side_profile_can_seed_and_match_after_a_cut(self):
        # Perspective can collapse left/right joints despite a visible shirt.
        profile = person(150)
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
        distractor, target = person(150), person(850)
        frame = scene([(distractor, (220, 30, 20)), (target, (20, 30, 220))])
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
        left, right = person(150), person(850)
        frame = scene([(left, (20, 30, 220)), (right, (20, 30, 220))])
        result = find_target(frame, lambda image, roi: left if roi[0] + roi[2] / 2 < 500 else right,
                             self.appearance, 240)
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['reason'], 'ambiguous')
        self.assertEqual(len(result['candidates']), 2)

    def test_unrelated_single_candidate_is_refused(self):
        other = person(850)
        frame = scene([(other, (220, 30, 20))])
        result = find_target(frame, lambda image, roi: other, self.appearance, 240)
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['reason'], 'appearance_mismatch')

    def test_repeated_detections_of_one_person_are_not_false_ambiguity(self):
        target = person(850)
        frame = scene([(target, (20, 30, 220))])
        result = find_target(frame, lambda image, roi: target, self.appearance, 240)
        self.assertIsNotNone(result['candidate'])
        self.assertEqual(len(result['candidates']), 1)
        self.assertTrue(all(np.isfinite(result['candidate']['box'])))

    def test_rejected_matches_cannot_reteach_the_frozen_reference(self):
        other = person(850)
        frame = scene([(other, (220, 30, 20))])
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


if __name__ == '__main__':
    unittest.main()
