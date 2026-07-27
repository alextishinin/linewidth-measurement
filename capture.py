"""Shared container for one digitized sweep.

Lives in its own module (free of picosdk imports) so the simulator, the
analysis self-test, and CI can run on machines without the Pico driver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Capture:
    t: np.ndarray            # seconds, 0 = trigger (= ramp start)
    pd: np.ndarray           # channel A volts (photodiode amplifier out)
    monitor: np.ndarray | None  # channel B volts (ramp/10) or None
    dt: float
    triggered: bool          # False if the auto-trigger timeout fired
    clipped: bool            # ADC over-range on channel A
