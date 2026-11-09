"""Cheap cut proposals from consecutive ORIGINAL BGR frames.

This is a visual discontinuity signal, not an identity decision. Same-camera
edits affecting only a small player can remain below its global thresholds;
the caller must also handle persistent tracking failure and confirm proposals.
"""

import cv2
import numpy as np


class ShotDetector:
    """Compare blurred thumbnails while ignoring additive brightness shifts.

    ``score`` is the mean normalized per-pixel change, after compensating the
    median luminance shift. Chroma is retained so a color-only cut is visible.
    A proposal also requires at least 35% of pixels to change by 0.10, which
    suppresses isolated player movement and small camera translations.

    Uniform brightness changes, including a featureless black/white change,
    are deliberately not sufficient evidence. A spatially complex flash can
    still propose a cut; temporal/identity confirmation belongs to the caller.
    """

    def __init__(self, threshold=0.06):
        if not 0 < threshold <= 1:
            raise ValueError('threshold must be in (0, 1]')
        self.threshold = float(threshold)
        self.previous = None

    def update(self, frame):
        """Consume one original uint8 BGR frame and return a cut proposal."""
        if (frame is None or frame.ndim != 3 or frame.shape[2] != 3
                or frame.dtype != np.uint8 or frame.size == 0):
            raise ValueError('frame must be a nonempty uint8 BGR image')
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        current = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        previous = self.previous
        self.previous = current
        if previous is None:
            return {
                'cut': False, 'score': 0.0, 'changed_fraction': 0.0,
                'brightness_shift': 0.0,
            }

        difference = current - previous
        brightness_shift = float(np.median(difference[:, :, 0]))
        luma = np.abs(difference[:, :, 0] - brightness_shift)
        chroma = np.mean(np.abs(difference[:, :, 1:]), axis=2)
        change = np.maximum(luma, chroma) / 255.0
        score = float(np.mean(change))
        changed_fraction = float(np.mean(change >= 0.10))
        return {
            'cut': bool(score >= self.threshold and changed_fraction >= 0.35),
            'score': score,
            'changed_fraction': changed_fraction,
            'brightness_shift': brightness_shift,
        }
