"""Behavioral checks for the horizontal portrait crop camera."""

import unittest

from crop_control import CropController


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


if __name__ == "__main__":
    unittest.main()
