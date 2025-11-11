"""classifier.py
A lightweight rule-based classifier that uses face + upper-body metrics over a short temporal window
to classify into the three expressions, plus NEUTRAL.

Labels: 'messi_yell', 'mom_homeless_laugh', 'bark_lilnasx', 'neutral'

This scaffold is intentionally explicit and tunable. If you later want a learned classifier,
plug a small model here and use the same API.
"""

from collections import deque
from typing import Optional, Dict, Any
import time


class ExpressionClassifier:
    def __init__(self, window_seconds: float = 0.6, fps_assume: int = 30):
        # time window to smooth features across frames
        self.win = window_seconds
        self.buffer = deque()
        self.last_time = None

        # thresholds (tweak as needed)
        self.MAR_YELL_THRESHOLD = 0.45   # mouth aspect ratio for wide open mouth
        self.EAR_CLOSED_THRESHOLD = 0.18 # eye aspect ratio indicates closed eyes
        self.BARK_HEAD_MOVEMENT = 15.0   # pixels of rapid nose oscillation
        self.MOTION_LOW_THRESHOLD = 5.0  # global motion threshold to decide "neutral"

        # classification hold (frames) to avoid jitter
        self.debounce_seconds = 0.25
        self.last_label = "neutral"
        self.last_label_time = 0

        # store last nose x for movement detection
        self.prev_nose_x = None

    def update(self, metrics: Dict[str, Any], timestamp: float) -> str:
        """metrics: dict with keys 'face' and 'upper' from pose_utils
        returns: label string
        """
        # push into buffer
        self.buffer.append((timestamp, metrics))
        # pop old
        while len(self.buffer) > 0 and (timestamp - self.buffer[0][0]) > self.win:
            self.buffer.popleft()

        label = self._classify_from_buffer()

        # debounce: require label hold for debounce_seconds before switching
        now = timestamp
        if label != self.last_label:
            if now - self.last_label_time >= self.debounce_seconds:
                self.last_label = label
                self.last_label_time = now
            else:
                # keep previous label
                label = self.last_label
        else:
            self.last_label_time = now

        return label

    def _classify_from_buffer(self) -> str:
        # aggregate simple stats over buffer
        ears = []
        mars = []
        nose_xs = []
        arm_exts = []

        for t, m in self.buffer:
            if m is None:
                continue
            f = m.get("face")
            u = m.get("upper")
            if f:
                ears.append(f.get("ear", 0.0))
                mars.append(f.get("mar", 0.0))
                nose_xs.append(f.get("nose_xy", (0.0, 0.0))[0])
            if u:
                arm_exts.append(u.get("left_arm_ext", 0.0) + u.get("right_arm_ext", 0.0))

        # compute simple aggregates
        mean_ear = sum(ears) / len(ears) if ears else 1.0
        mean_mar = sum(mars) / len(mars) if mars else 0.0
        mean_arm_ext = sum(arm_exts) / len(arm_exts) if arm_exts else 0.0

        nose_movement = 0.0
        if len(nose_xs) >= 2:
            nose_movement = max(nose_xs) - min(nose_xs)

        # Heuristics:
        # 1) "messi_yell": mouth very open (MAR large)
        if mean_mar > self.MAR_YELL_THRESHOLD:
            return "messi_yell"

        # 2) "mom_homeless_laugh": eyes closed (or near closed) and subtle mouth/clench signals
        # use EAR small AND arm extension not excessive
        if mean_ear < self.EAR_CLOSED_THRESHOLD and mean_arm_ext < 150:
            return "mom_homeless_laugh"

        # 3) "bark_lilnasx": repeated/large nose x oscillation / head movement AND arms possibly raised
        if nose_movement > self.BARK_HEAD_MOVEMENT or mean_arm_ext > 250:
            return "bark_lilnasx"

        # If no significant changes: neutral
        return "neutral"
