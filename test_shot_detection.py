import unittest

import cv2
import numpy as np

from shot_detection import ShotDetector


def scene():
    y, x = np.indices((180, 320))
    palette = np.array([
        [50, 80, 140], [180, 70, 60], [60, 170, 100], [140, 130, 65],
    ], dtype=np.uint8)
    return palette[(x // 40 + y // 30) % len(palette)]


class ShotDetectionTests(unittest.TestCase):
    def test_first_and_repeated_frames_do_not_propose_cuts(self):
        detector = ShotDetector()
        for _ in range(10):
            result = detector.update(scene())
            self.assertFalse(result['cut'])
            self.assertEqual(result['score'], 0.0)

    def test_gradual_camera_motion_does_not_propose_cuts(self):
        detector = ShotDetector()
        original = scene()
        for offset in range(0, 25, 2):
            moved = cv2.warpAffine(
                original, np.float32([[1, 0, offset], [0, 1, offset / 2]]),
                (320, 180), borderMode=cv2.BORDER_REFLECT,
            )
            self.assertFalse(detector.update(moved)['cut'])

    def test_hard_cut_is_proposed_once_then_new_shot_is_stable(self):
        detector = ShotDetector()
        detector.update(scene())
        replacement = cv2.flip(scene(), -1)
        result = detector.update(replacement)
        self.assertTrue(result['cut'])
        self.assertGreater(result['score'], 0)
        self.assertFalse(detector.update(replacement)['cut'])

    def test_uniform_brightness_shift_is_not_a_cut(self):
        detector = ShotDetector()
        original = scene()
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
        first, second = scene(), scene()
        cv2.rectangle(first, (100, 80), (115, 130), (10, 10, 10), -1)
        cv2.rectangle(second, (150, 80), (165, 130), (10, 10, 10), -1)
        detector.update(first)
        self.assertFalse(detector.update(second)['cut'])


if __name__ == '__main__':
    unittest.main()
