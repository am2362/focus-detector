"""Escalation state machine. No hardware or camera dependencies, so it is easy to test."""


class Escalator:
    """Turns a stream of 'distracted or not' readings into a score (0..1) and a level (0..4).

    The score rises while the user is distracted (after a short grace period) and
    falls when they return. Levels use hysteresis so they don't flicker.
    """

    # score thresholds for entering levels 1..4
    UP = (0.05, 0.30, 0.60, 0.85)
    # score must drop this far below the entry threshold to leave a level
    HYST = 0.05

    def __init__(self, grace=1.5, rise_time=20.0, fall_time=6.0):
        self.grace = grace            # seconds of distraction before escalation starts
        self.rise_rate = 1.0 / rise_time   # score per second while distracted
        self.fall_rate = 1.0 / fall_time   # score per second while focused
        self.score = 0.0
        self.level = 0
        self._distracted_for = 0.0

    def update(self, distracted, dt):
        if distracted:
            self._distracted_for += dt
            if self._distracted_for >= self.grace:
                self.score = min(1.0, self.score + self.rise_rate * dt)
        else:
            self._distracted_for = 0.0
            self.score = max(0.0, self.score - self.fall_rate * dt)
        self._update_level()
        return self.level

    def _update_level(self):
        lvl = self.level
        while lvl < 4 and self.score >= self.UP[lvl]:
            lvl += 1
        while lvl > 0 and self.score <= self.UP[lvl - 1] - self.HYST:
            lvl -= 1
        self.level = lvl

    def reset(self):
        self.score = 0.0
        self.level = 0
        self._distracted_for = 0.0
