import unittest

import numpy as np

from tracking import PoseTracker, map_pose


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


if __name__ == '__main__':
    unittest.main()
