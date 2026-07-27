"""Serial driver for the Thorlabs SA201B Scanning Fabry-Perot Controller.

Protocol (verified against hardware, firmware v1.2):
  * 115200 baud, 8N1, no flow control, commands terminated with CR ('\r').
  * Replies look like b'\rAmplitude: 30.00\r>' -- text then a '>' prompt.
  * Unknown input yields b'\rUNKNOWN COMMAND!'.

Only the documented command set from the SA201B manual (TTN298338-D02,
section 5.7.5) is used.
"""
from __future__ import annotations

import re
import threading
import time

import serial
from serial.tools import list_ports

import config

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


class SA201BError(RuntimeError):
    pass


def find_port() -> str | None:
    """Return the COM port of the first attached SA201B, or None."""
    for p in list_ports.comports():
        if p.vid == config.SA201B_VID and p.pid == config.SA201B_PID:
            return p.device
    return None


class SA201B:
    """Thin, robust wrapper around the SA201B command-line interface."""

    def __init__(self, port: str | None = None, timeout: float = 0.6):
        port = port or find_port()
        if port is None:
            raise SA201BError(
                "No SA201B found (USB VID 0x1313 / PID 0x100A). "
                "Is the controller powered on and connected via USB?")
        self.port = port
        self._ser = serial.Serial(port, baudrate=config.SA201B_BAUD,
                                  bytesize=8, parity="N", stopbits=1,
                                  timeout=0.05)
        self._timeout = timeout
        self._lock = threading.Lock()   # serializes access across threads
        time.sleep(0.1)
        self._ser.reset_input_buffer()

    # ------------------------------------------------------------- low level
    def _transact(self, command: str) -> str:
        """Send one command, return the reply text (prompt stripped)."""
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write(command.encode("ascii") + b"\r")
            deadline = time.monotonic() + self._timeout
            chunks = bytearray()
            while time.monotonic() < deadline:
                chunks += self._ser.read(256)
                if chunks.endswith(b">"):
                    break
        text = chunks.decode("ascii", errors="replace")
        return text.rstrip(">").replace("\r", "\n").strip()

    def query(self, keyword: str) -> str:
        reply = self._transact(keyword + "?")
        if "UNKNOWN" in reply.upper():
            raise SA201BError(f"Device rejected query '{keyword}?': {reply!r}")
        return reply

    def query_number(self, keyword: str) -> float:
        reply = self.query(keyword)
        # Replies are 'Name: value [units]'; take the first number after the
        # colon (falls back to first number anywhere).
        tail = reply.split(":", 1)[-1]
        m = _NUM.search(tail) or _NUM.search(reply)
        if not m:
            raise SA201BError(f"Could not parse number from {reply!r}")
        return float(m.group())

    def set(self, keyword: str, value) -> None:
        if isinstance(value, float):
            value = f"{value:.2f}"
        reply = self._transact(f"{keyword}={value}")
        if "UNKNOWN" in reply.upper():
            raise SA201BError(f"Device rejected '{keyword}={value}': {reply!r}")

    def set_verified(self, keyword: str, value: float, tol: float = 0.05) -> None:
        self.set(keyword, value)
        back = self.query_number(keyword)
        if abs(back - float(value)) > tol:
            raise SA201BError(
                f"Setting '{keyword}={value}' did not stick (device reports {back})")

    # ------------------------------------------------------------ identity
    def identify(self) -> str:
        return self.query("ID")

    # ------------------------------------------------------------ parameters
    @property
    def amplitude_v(self) -> float:
        return self.query_number("amplitude")

    @amplitude_v.setter
    def amplitude_v(self, volts: float) -> None:
        self.set_verified("amplitude", float(volts))

    @property
    def dc_offset_v(self) -> float:
        return self.query_number("dcoffset")

    @dc_offset_v.setter
    def dc_offset_v(self, volts: float) -> None:
        self.set_verified("dcoffset", float(volts))

    @property
    def risetime_step(self) -> int:
        return int(self.query_number("risetime"))

    @risetime_step.setter
    def risetime_step(self, step: int) -> None:
        self.set_verified("risetime", int(step))

    @property
    def sweep_expand_index(self) -> int:
        return int(self.query_number("sweepexpand"))

    @sweep_expand_index.setter
    def sweep_expand_index(self, index: int) -> None:
        self.set_verified("sweepexpand", int(index))

    @property
    def pd_gain_index(self) -> int:
        return int(self.query_number("pdgain"))

    @pd_gain_index.setter
    def pd_gain_index(self, index: int) -> None:
        self.set_verified("pdgain", int(index))

    @property
    def trigger_percent(self) -> int:
        return int(self.query_number("trigpercent"))

    @trigger_percent.setter
    def trigger_percent(self, percent: int) -> None:
        self.set_verified("trigpercent", int(percent))

    @property
    def sawtooth(self) -> bool:
        return int(self.query_number("mode")) == 1

    @sawtooth.setter
    def sawtooth(self, enabled: bool) -> None:
        self.set_verified("mode", 1 if enabled else 0)

    @property
    def waveform_enabled(self) -> bool:
        return int(self.query_number("wfenable")) == 1

    @waveform_enabled.setter
    def waveform_enabled(self, enabled: bool) -> None:
        self.set_verified("wfenable", 1 if enabled else 0)

    # ------------------------------------------------------------ derived
    def rise_time_s(self) -> float:
        """Estimated sweep rise time from the step and expansion settings."""
        frac = self.risetime_step / config.RISETIME_STEPS
        base = config.RISETIME_MIN_S + frac * (config.RISETIME_MAX_S -
                                               config.RISETIME_MIN_S)
        return base * config.SWEEP_EXPANSION_FACTORS[self.sweep_expand_index]

    def apply_scan_settings(self, amplitude_v: float = 30.0,
                            dc_offset_v: float = 0.0,
                            risetime_step: int = 0,
                            sweep_expand_index: int = 0,
                            pd_gain_index: int | None = 0) -> None:
        """Configure a standard linewidth scan (sawtooth, full amplitude)."""
        self.sawtooth = True
        self.sweep_expand_index = sweep_expand_index
        self.risetime_step = risetime_step
        self.amplitude_v = amplitude_v
        self.dc_offset_v = dc_offset_v
        if pd_gain_index is not None:
            self.pd_gain_index = pd_gain_index
        self.waveform_enabled = True

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
