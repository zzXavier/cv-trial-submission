"""Small, frame-rate-aware horizontal camera for a fixed 9:16 crop."""

import math


class CropController:
    """Track a source-frame target x coordinate with a bounded portrait crop.

    ``initial_x`` is the initial target/camera center, not the crop's left edge.
    ``x`` retains the floating-point left edge; update returns its rounded value.
    ``deadzone=0.15`` means the central 35%-65% of the crop holds the camera.
    ``response_s`` is the exponential response time constant; ``max_speed`` is
    measured in crop widths per second. Smooth and deadzone share both limits.
    Near a body edge, the dead zone shrinks and the response time halves to
    leave space for fast actions. The speed limit still applies.
    Body boxes are source-frame (x, y, width, height). Their margin is a soft
    goal, subject to the same motion limits; impossible boxes are ignored.
    """

    def __init__(self, width, height, fps, initial_x, mode="deadzone", *,
                 deadzone=0.15, response_s=0.2, max_speed=1.0):
        if any(not math.isfinite(v) or v <= 0 for v in
               (width, height, fps, response_s, max_speed)):
            raise ValueError("Dimensions, fps, response_s and max_speed must be positive and finite")
        if int(width) != width or int(height) != height:
            raise ValueError("Frame dimensions must be integers")
        if not math.isfinite(initial_x) or not 0 <= deadzone < 0.5:
            raise ValueError("initial_x must be finite and deadzone must be in [0, 0.5)")
        if mode not in ("direct", "smooth", "deadzone"):
            raise ValueError("mode must be direct, smooth or deadzone")
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.crop_w = min(self.width, round(self.height * 9 / 16))
        self.mode, self.deadzone = mode, deadzone
        self.response_s, self.max_speed = response_s, max_speed
        self.x = self._bounded(initial_x - self.crop_w / 2)
        self.velocity = 0.0  # Actual camera speed in source pixels per second.
        self._max_step = max_speed * self.crop_w / self.fps

    def _bounded(self, x):
        return min(max(float(x), 0.0), self.width - self.crop_w)

    def update(self, target_x: float | None, body_box=None) -> int:
        """Advance one frame; a missing target holds position and stops motion."""
        if target_x is None or not math.isfinite(target_x):
            self.velocity = 0.0
            return round(self.x)

        valid_body = (body_box is not None
                      and all(math.isfinite(v) for v in body_box)
                      and 0 < body_box[2] <= self.crop_w and body_box[3] > 0)
        zone_scale = 1.0
        if valid_body:
            left, top, body_w, body_h = body_box
            clearance = min(left - self.x, self.x + self.crop_w - left - body_w)
            # Start reacting with 30% of the crop left beside the body; by
            # 15%, prioritize following. This leaves room beyond wrists/feet.
            zone_scale = max(0.0, min(1.0, (clearance / self.crop_w - 0.15) / 0.15))
        alpha = -math.expm1(-1 / (self.fps * self.response_s * (0.5 + 0.5 * zone_scale)))

        desired = target_x - self.crop_w / 2
        if self.mode == "deadzone":
            error = desired - self.x
            excess = max(0.0, abs(error) - self.deadzone * zone_scale * self.crop_w)
            desired = self.x + math.copysign(excess, error)

        if valid_body:
            left, top, body_w, body_h = body_box
            margin = min(0.04 * self.crop_w, (self.crop_w - body_w) / 2)
            low = left + body_w + margin - self.crop_w
            high = left - margin
            # Only request the minimum correction to fit the body. Never
            # hard-clamp camera position to noisy limb-derived box edges.
            desired = min(max(desired, low), high)

        desired = self._bounded(desired)
        previous = self.x
        if self.mode == "direct":
            self.x = desired
        else:
            step = (desired - self.x) * alpha
            self.x = self._bounded(self.x + max(-self._max_step, min(step, self._max_step)))
        self.velocity = (self.x - previous) * self.fps
        return round(self.x)
