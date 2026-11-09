import unittest

from track import build_crops, validate_replay


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


if __name__ == '__main__':
    unittest.main()
