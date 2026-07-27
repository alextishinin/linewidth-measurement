"""Hardware simulators so the whole pipeline runs without a laser.

The simulated laser is a multi-mode HeNe: three longitudinal modes spaced
435 MHz (a ~34 cm tube), each far narrower than the SA210's 67 MHz
instrument function, seen through an ideal confocal Fabry-Perot with
finesse 150 (exact Airy transmission). The optical frequency drifts slowly
so the peaks walk across the sweep like a real free-running laser.
"""
from __future__ import annotations

import time

import numpy as np

import config
from capture import Capture

_RNG = np.random.default_rng()


class SimulatedSA201B:
    """Same surface as sa201b.SA201B, backed by plain attributes."""

    port = "SIM"

    def __init__(self):
        self.amplitude_v = 30.0
        self.dc_offset_v = 0.0
        self.risetime_step = 0
        self.sweep_expand_index = 0
        self.pd_gain_index = 0
        self.trigger_percent = 50
        self.sawtooth = True
        self.waveform_enabled = True

    def identify(self) -> str:
        return "SIMULATED SA201B"

    def rise_time_s(self) -> float:
        frac = self.risetime_step / config.RISETIME_STEPS
        base = config.RISETIME_MIN_S + frac * (config.RISETIME_MAX_S -
                                               config.RISETIME_MIN_S)
        return base * config.SWEEP_EXPANSION_FACTORS[self.sweep_expand_index]

    def apply_scan_settings(self, amplitude_v=30.0, dc_offset_v=0.0,
                            risetime_step=0, sweep_expand_index=0,
                            pd_gain_index=0) -> None:
        self.amplitude_v = amplitude_v
        self.dc_offset_v = dc_offset_v
        self.risetime_step = risetime_step
        self.sweep_expand_index = sweep_expand_index
        if pd_gain_index is not None:
            self.pd_gain_index = pd_gain_index

    def close(self) -> None:
        pass


class SimulatedScope:
    """Same surface as pico5000a.Pico5000A, producing synthetic sweeps."""

    variant = "SIM-5242D"
    serial = "SIM0000"
    use_monitor = True

    FINESSE = 150.0
    MODES = [(0.0, 1.0), (435e6, 0.55), (870e6, 0.18)]  # (offset Hz, rel amp)
    VOLTS_PER_FSR = 10.0     # SA210 manual: a 0-20 V sawtooth covers ~2 FSR
    PEAK_VOLTS = 3.2         # tallest peak at the photodiode amplifier output
    NOISE_V = 0.004
    SWEEP_NONLINEARITY = 0.03

    def __init__(self, controller: SimulatedSA201B):
        self.ctrl = controller
        self._dt = None
        self._n = 0
        self._n_pre = 0
        self._t0 = time.monotonic()

    def open(self) -> None:
        pass

    def configure_window(self, window_s: float,
                         dt_target_s: float = config.DEFAULT_DT_S,
                         pre_trigger_fraction: float = 0.02) -> None:
        self._dt = dt_target_s
        self._n = max(int(round(window_s / dt_target_s)), 1000)
        self._n_pre = int(self._n * pre_trigger_fraction)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def window_s(self) -> float:
        return self._n * self._dt

    def capture(self, timeout_s: float = 5.0) -> Capture:
        rise = self.ctrl.rise_time_s()
        time.sleep(min(rise + 0.001, 0.05))
        t = (np.arange(self._n) - self._n_pre) * self._dt

        # Piezo scan: optical frequency swept linearly (plus a little
        # quadratic nonlinearity) at hz_per_s across the rising ramp.
        n_fsr = self.ctrl.amplitude_v / self.VOLTS_PER_FSR
        hz_per_s = n_fsr * config.FSR_HZ / rise
        x = np.clip(t, 0.0, rise)
        freq = hz_per_s * (x + self.SWEEP_NONLINEARITY * x**2 / rise)

        # Slow drift + per-sweep jitter of the laser vs. the cavity comb.
        wall = time.monotonic() - self._t0
        offset = (1.7e9 + 400e6 * np.sin(2 * np.pi * wall / 45.0)
                  + _RNG.normal(0.0, 15e6))

        pd = np.zeros_like(t)
        coeff = (2.0 * self.FINESSE / np.pi) ** 2
        for mode_off, amp in self.MODES:
            nu = freq - offset - mode_off
            pd += amp / (1.0 + coeff * np.sin(np.pi * nu / config.FSR_HZ) ** 2)
        pd *= self.PEAK_VOLTS
        gain_scale = {0: 1.0, 1: 10.0, 2: 100.0}[self.ctrl.pd_gain_index]
        pd = np.clip(pd * gain_scale, 0.0, 5.0)

        rising = (t >= 0) & (t <= rise)
        pd[~rising] = 0.0                      # SA201B blanking on the retrace
        pd += _RNG.normal(0.0, self.NOISE_V, self._n) + 0.008

        monitor = np.zeros_like(t)
        ramp_v = (self.ctrl.dc_offset_v + self.ctrl.amplitude_v * x / rise) / 10.0
        monitor[rising] = ramp_v[rising]
        fall = (t > rise) & (t <= rise + 0.001)
        if fall.any():
            top = (self.ctrl.dc_offset_v + self.ctrl.amplitude_v) / 10.0
            monitor[fall] = top * (1.0 - (t[fall] - rise) / 0.001)
        monitor += _RNG.normal(0.0, 0.002, self._n)

        return Capture(t=t, pd=pd, monitor=monitor, dt=self._dt,
                       triggered=True, clipped=bool(pd.max() >= 4.999))

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
