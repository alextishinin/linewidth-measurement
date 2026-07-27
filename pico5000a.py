"""PicoScope 5000D block-mode acquisition for the Fabry-Perot spectrum.

Channel plan (verified against a PicoScope 5242D):
  * Channel A  <- SA201B "PD AMPLIFIER OUT" (0..5 V spectrum signal)
  * Channel B  <- SA201B "MONITOR OUT" (ramp/10, 0..4.5 V) -- optional but
                  recommended; used to measure the true ramp duration.
  * EXT        <- SA201B "TRIGGER OUT" (5 V TTL; rising edge = ramp start)

With B enabled the scope runs in 15-bit mode (two channels); with
--single-channel it runs 16-bit. Either way the LSB is microvolts -- far below
the SA201B's own output noise -- so both are effectively noiseless here.
"""
from __future__ import annotations

import atexit
import ctypes
import os
import time

import numpy as np

import config
from capture import Capture

if os.path.isdir(config.PICO_SDK_LIB):  # make ps5000a.dll findable
    os.add_dll_directory(config.PICO_SDK_LIB)
    os.environ["PATH"] = config.PICO_SDK_LIB + os.pathsep + os.environ.get("PATH", "")

from picosdk.ps5000a import ps5000a as ps  # noqa: E402
from picosdk.functions import assert_pico_ok  # noqa: E402

_POWER_STATUS = {282, 286}  # USB-power notifications, not errors on 2ch units


class PicoScopeError(RuntimeError):
    pass


class Pico5000A:
    """Block-mode wrapper tailored to this measurement."""

    def __init__(self, use_monitor_channel: bool = True):
        self.use_monitor = use_monitor_channel
        self.handle = ctypes.c_int16()
        self._range_a = ps.PS5000A_RANGE["PS5000A_5V"]
        self._range_b = ps.PS5000A_RANGE["PS5000A_5V"]
        self._max_adc = ctypes.c_int16()
        self._timebase = None
        self._dt = None
        self._n_samples = 0
        self._n_pre = 0
        self._buf_a = None
        self._buf_b = None
        self.variant = "?"
        self.serial = "?"
        self._closed = True      # becomes False in open()
        self._abort = False

    # ------------------------------------------------------------------ open
    def open(self) -> None:
        res_name = ("PS5000A_DR_15BIT" if self.use_monitor
                    else "PS5000A_DR_16BIT")
        resolution = ps.PS5000A_DEVICE_RESOLUTION[res_name]
        status = ps.ps5000aOpenUnit(ctypes.byref(self.handle), None, resolution)
        if status in _POWER_STATUS:
            status = ps.ps5000aChangePowerSource(self.handle, status)
        assert_pico_ok(status)
        self._closed = False
        # Safety net: whatever way the process ends (unhandled exception,
        # sys.exit, normal return), release the unit so the driver never
        # keeps it claimed for the next launch. close() is idempotent.
        atexit.register(self.close)

        info = ctypes.create_string_buffer(64)
        req = ctypes.c_int16()
        ps.ps5000aGetUnitInfo(self.handle, info, 64, ctypes.byref(req), 3)
        self.variant = info.value.decode()
        ps.ps5000aGetUnitInfo(self.handle, info, 64, ctypes.byref(req), 4)
        self.serial = info.value.decode()

        coupling = ps.PS5000A_COUPLING["PS5000A_DC"]
        cha = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
        chb = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
        assert_pico_ok(ps.ps5000aSetChannel(self.handle, cha, 1, coupling,
                                            self._range_a, 0))
        assert_pico_ok(ps.ps5000aSetChannel(self.handle, chb,
                                            1 if self.use_monitor else 0,
                                            coupling, self._range_b, 0))
        assert_pico_ok(ps.ps5000aMaximumValue(self.handle,
                                              ctypes.byref(self._max_adc)))

        threshold = int(config.EXT_TRIGGER_LEVEL_V /
                        config.EXT_TRIGGER_FULLSCALE_V * 32767)
        assert_pico_ok(ps.ps5000aSetSimpleTrigger(
            self.handle, 1, ps.PS5000A_CHANNEL["PS5000A_EXTERNAL"], threshold,
            ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"], 0,
            config.AUTO_TRIGGER_MS))

    # ------------------------------------------------------------- configure
    def configure_window(self, window_s: float,
                         dt_target_s: float = config.DEFAULT_DT_S,
                         pre_trigger_fraction: float = 0.02) -> None:
        """Choose a timebase >= dt_target and size the capture buffers."""
        timebase, dt = self._pick_timebase(dt_target_s)
        n = int(round(window_s / dt))
        n = max(n, 1000)
        self._timebase, self._dt, self._n_samples = timebase, dt, n
        self._n_pre = int(n * pre_trigger_fraction)
        self._buf_a = (ctypes.c_int16 * n)()
        self._buf_b = (ctypes.c_int16 * n)() if self.use_monitor else None
        none_mode = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]
        cha = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
        assert_pico_ok(ps.ps5000aSetDataBuffer(
            self.handle, cha, ctypes.byref(self._buf_a), n, 0, none_mode))
        if self.use_monitor:
            chb = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_B"]
            assert_pico_ok(ps.ps5000aSetDataBuffer(
                self.handle, chb, ctypes.byref(self._buf_b), n, 0, none_mode))

    def _pick_timebase(self, dt_target_s: float) -> tuple[int, float]:
        """Walk timebase numbers until the interval reaches the target."""
        interval_ns = ctypes.c_float()
        max_samples = ctypes.c_int32()
        best = None
        for tb in range(1, 20000):
            status = ps.ps5000aGetTimebase2(
                self.handle, tb, 1000, ctypes.byref(interval_ns),
                ctypes.byref(max_samples), 0)
            if status != 0:
                continue  # timebase not available at this resolution
            dt = interval_ns.value * 1e-9
            best = (tb, dt)
            if dt >= dt_target_s * 0.999:
                return best
        if best is None:
            raise PicoScopeError("No valid timebase found")
        return best

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def window_s(self) -> float:
        return self._n_samples * self._dt

    def request_abort(self) -> None:
        """Make an in-flight capture() return promptly (used at shutdown so
        long sweeps can never outlive the closing application)."""
        self._abort = True

    # --------------------------------------------------------------- capture
    def capture(self, timeout_s: float = 5.0) -> Capture:
        if self._timebase is None:
            raise PicoScopeError("configure_window() must be called first")
        self._abort = False
        n_post = self._n_samples - self._n_pre
        t_start = time.monotonic()
        assert_pico_ok(ps.ps5000aRunBlock(
            self.handle, self._n_pre, n_post, self._timebase, None, 0,
            None, None))
        ready = ctypes.c_int16(0)
        while not ready.value:
            if self._abort:
                ps.ps5000aStop(self.handle)
                raise PicoScopeError("capture aborted (shutdown)")
            assert_pico_ok(ps.ps5000aIsReady(self.handle, ctypes.byref(ready)))
            if time.monotonic() - t_start > timeout_s:
                ps.ps5000aStop(self.handle)
                raise PicoScopeError("Capture timed out")
            time.sleep(0.002)
        elapsed = time.monotonic() - t_start

        n = ctypes.c_uint32(self._n_samples)
        overflow = ctypes.c_int16()
        none_mode = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]
        assert_pico_ok(ps.ps5000aGetValues(
            self.handle, 0, ctypes.byref(n), 1, none_mode, 0,
            ctypes.byref(overflow)))

        scale_a = 5.0 / self._max_adc.value  # volts per count on the 5 V range
        pd = np.frombuffer(self._buf_a, dtype=np.int16,
                           count=n.value).astype(np.float64) * scale_a
        monitor = None
        if self.use_monitor:
            monitor = np.frombuffer(self._buf_b, dtype=np.int16,
                                    count=n.value).astype(np.float64) * scale_a
        t = (np.arange(n.value) - self._n_pre) * self._dt
        # If the auto-trigger fired, the block completes only after the full
        # window + timeout; a real trigger completes in about window + ~0 s.
        triggered = elapsed < (self.window_s + config.AUTO_TRIGGER_MS / 1e3) * 0.9
        return Capture(t=t, pd=pd, monitor=monitor, dt=self._dt,
                       triggered=triggered, clipped=bool(overflow.value & 0b01))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            ps.ps5000aStop(self.handle)
        finally:
            ps.ps5000aCloseUnit(self.handle)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
