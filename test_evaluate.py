"""Hand-computed checks for independent crop evaluation."""
import unittest
from evaluate import evaluate_clip


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
