"""Live laser linewidth measurement.

Chain: laser -> SA210 Fabry-Perot -> SA201B controller -> PicoScope 5242D -> here.

Run:  python linewidth_live.py   (or run.bat)

Drag with the left mouse button on any graph to zoom into that region
(auto-scaling pauses for that graph until you press its "reset view" button).

Keys in the plot window (ignored while typing in the wavelength box):
  r  run one sweep (single mode)      m  toggle live/single mode
  g  cycle PD amplifier gain          a  toggle auto-gain
  e  export displayed data to CSV    s  snapshot (PNG + raw CSV)
  d  dark/light theme                v  reset all zoomed views
  left/right  nudge DC offset        p  pause display   q  quit
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import os
import queue
import sys
import threading
import time

import numpy as np

import config
import analysis as ana

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
EXPORT_DIR = os.path.join(HERE, "exports")
SETTINGS_PATH = os.path.join(HERE, "settings.json")


def _ms_to_step(ms: float) -> int:
    lo = config.RISETIME_MIN_S * 1e3
    hi = config.RISETIME_MAX_S * 1e3
    step = int(round((ms - lo) / (hi - lo) * config.RISETIME_STEPS))
    return max(0, min(config.RISETIME_STEPS, step))


def load_saved_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_args(args) -> None:
    """Fill unset CLI options from settings.json, then built-in defaults.

    Precedence: explicit CLI flag > last session's saved value > default.
    """
    saved = load_saved_settings()

    def pick(cli_value, key, builtin):
        if cli_value is not None:
            return cli_value
        return saved.get(key, builtin)

    args.wavelength_nm = float(pick(args.wavelength_nm, "wavelength_nm", 1064.0))
    args.theme = pick(args.theme, "theme", "dark")
    args.amplitude = float(pick(args.amplitude, "amplitude_v", 30.0))
    args.offset = float(pick(args.offset, "offset_v", 0.0))
    args.sweep_expand = int(pick(args.sweep_expand, "expand_idx", 0))
    if args.risetime_step is None:
        ms = saved.get("sweep_ms")
        args.risetime_step = _ms_to_step(float(ms)) if ms is not None else 0
    if args.pdgain is None:
        if saved.get("gain_auto", True):
            args.pdgain = "auto"
        else:
            args.pdgain = str(int(saved.get("gain_idx", 0)))
    mode = saved.get("span_mode", "auto")
    args.span_mode = mode if mode in ("auto", "full", "manual") else "auto"
    span = saved.get("span_manual")
    try:
        args.span_manual = (float(span[0]), float(span[1]))
    except Exception:
        args.span_manual = (-1000.0, 1000.0)


def _is_ctrl_error(exc) -> bool:
    """True for exceptions that mean the SA201B serial link is gone."""
    if isinstance(exc, ValueError):
        return False
    try:
        import serial
        from sa201b import SA201BError
        if isinstance(exc, (serial.SerialException, SA201BError)):
            return True
    except Exception:
        pass
    return isinstance(exc, OSError)


def _decimate_envelope(x, y, max_bins: int = 1200):
    """Reduce a trace to a min/max envelope for fast plotting.

    Rendering 20k+ points per frame is what makes the whole UI lag; two
    points (bin min and max) per bin preserves every peak and the noise
    floor exactly as the eye would see them at screen resolution.
    """
    n = len(x)
    if n <= 2 * max_bins:
        return x, y
    stride = n // max_bins
    m = (n // stride) * stride
    yb = y[:m].reshape(-1, stride)
    xs = np.repeat(x[:m].reshape(-1, stride).mean(axis=1), 2)
    ys = np.empty(2 * (m // stride), dtype=y.dtype)
    ys[0::2] = yb.min(axis=1)
    ys[1::2] = yb.max(axis=1)
    return xs, ys


# --------------------------------------------------------------------- setup
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--single", action="store_true",
                   help="start in single-sweep mode (also switchable in-app "
                        "via the Mode button)")
    p.add_argument("--port", default=None, help="SA201B COM port (default: autodetect)")
    p.add_argument("--no-controller", action="store_true",
                   help="don't talk to the SA201B (use its touchscreen instead)")
    p.add_argument("--single-channel", action="store_true",
                   help="PD only on ch A at 16 bit (MONITOR OUT not wired to ch B)")
    p.add_argument("--amplitude", type=float, default=None,
                   help="ramp amplitude in V (default 30 -> ~3 FSR; persisted)")
    p.add_argument("--offset", type=float, default=None,
                   help="DC offset in V (persisted)")
    p.add_argument("--risetime-step", type=int, default=None,
                   help="SA201B rise-time step 0..200 (0 = 10 ms sweep; "
                        "persisted as sweep ms)")
    p.add_argument("--sweep-expand", type=int, default=None, choices=range(7),
                   help="sweep expansion index 0..6 (1x..100x; persisted)")
    p.add_argument("--pdgain", default=None, choices=["auto", "0", "1", "2"],
                   help="PD amplifier gain index, or auto (persisted)")
    p.add_argument("--rise-ms", type=float, default=None,
                   help="sweep rise time in ms when not using the controller")
    p.add_argument("--dt-us", type=float, default=0.5,
                   help="target sample interval in microseconds")
    p.add_argument("--window-ms", type=float, default=None,
                   help="capture window in ms (default: auto from rise time)")
    p.add_argument("--avg", type=int, default=1,
                   help="average N consecutive triggered sweeps (default 1)")
    p.add_argument("--wavelength-nm", type=float, default=None,
                   help="laser wavelength in nm for the wavelength-units "
                        "display (editable in the app; persisted)")
    p.add_argument("--theme", choices=["dark", "light"], default=None,
                   help="UI color theme (default dark; toggle in-app; "
                        "persisted)")
    p.add_argument("--fsr-ghz", type=float, default=config.FSR_HZ / 1e9,
                   help="interferometer FSR in GHz (SA210 = 10)")
    p.add_argument("--instrument-mhz", type=float,
                   default=config.INSTRUMENT_RES_HZ / 1e6,
                   help="instrument resolution in MHz (SA210 spec = 67)")
    p.add_argument("--history-s", type=float, default=180.0,
                   help="linewidth trend window in seconds")
    p.add_argument("--no-log", action="store_true", help="disable CSV logging")
    return p.parse_args()


def open_hardware(args):
    """Return (controller_or_None, scope, rise_time_s). Raises if no scope."""
    ctrl = None
    rise = (args.rise_ms or 10.0) / 1e3
    if not args.no_controller:
        try:
            from sa201b import SA201B
            ctrl = SA201B(port=args.port)
            print(f"[SA201B] {ctrl.identify()}  on {ctrl.port}")
            ctrl.apply_scan_settings(
                amplitude_v=args.amplitude, dc_offset_v=args.offset,
                risetime_step=args.risetime_step,
                sweep_expand_index=args.sweep_expand,
                pd_gain_index=None if args.pdgain == "auto" else int(args.pdgain))
            if args.pdgain == "auto" and ctrl.pd_gain_index != 0:
                ctrl.pd_gain_index = 0
            rise = ctrl.rise_time_s()
            print(f"[SA201B] sawtooth {args.amplitude:g} V, offset {args.offset:g} V, "
                  f"rise time ~{rise * 1e3:.1f} ms")
        except Exception as exc:
            print(f"[SA201B] WARNING: {exc}\n"
                  f"         Continuing without controller; set the ramp on the "
                  f"touchscreen and pass --rise-ms if not 10.")
            ctrl = None

    try:
        from pico5000a import Pico5000A
        scope = Pico5000A(use_monitor_channel=not args.single_channel)
        scope.open()
    except Exception:
        if ctrl is not None:
            ctrl.close()
        raise
    bits = 15 if not args.single_channel else 16
    print(f"[scope] PicoScope {scope.variant} s/n {scope.serial} ({bits}-bit mode)")
    return ctrl, scope, rise


# --------------------------------------------------------------- acquisition
class Acquirer(threading.Thread):
    """Owns the scope. Captures continuously (live mode) or only when a
    one-shot request arrives (single mode); otherwise the scope sits idle."""

    def __init__(self, scope, rise_s: float, dt_s: float,
                 window_s: float | None, n_avg: int, continuous: bool = True):
        super().__init__(daemon=True)
        self.scope = scope
        self.rise_s = rise_s
        self.window_fixed = window_s is not None
        self.window_s = window_s or self._auto_window(rise_s)
        self.n_avg = max(1, n_avg)
        self.latest = collections.deque(maxlen=max(3, self.n_avg))
        self.status = "starting"
        self.sweep_period_s = None
        self.stop_flag = threading.Event()
        self.continuous = threading.Event()
        if continuous:
            self.continuous.set()
        self._oneshot = 0
        self._oneshot_lock = threading.Lock()
        self._pending_window = None
        self._last_trig_time = None
        scope.configure_window(self.window_s, dt_target_s=dt_s)

    @staticmethod
    def _auto_window(rise_s: float) -> float:
        return rise_s * 1.25 + 0.002

    # one-shot bookkeeping (single mode) ---------------------------------
    def request_oneshot(self, n: int = 1) -> None:
        """Ask for n captures (one Run-once click = one averaged result)."""
        with self._oneshot_lock:
            self._oneshot = max(self._oneshot, max(1, n))

    def _oneshot_pending(self) -> bool:
        with self._oneshot_lock:
            return self._oneshot > 0

    def _consume_oneshot(self) -> None:
        with self._oneshot_lock:
            if self._oneshot > 0:
                self._oneshot -= 1

    def request_window(self, window_s: float) -> None:
        """Ask the acquisition thread to resize the capture window (applied
        between captures; scope calls stay on the acquisition thread)."""
        self._pending_window = float(window_s)

    def shutdown(self, timeout_s: float = 8.0) -> None:
        """Stop acquisition and guarantee the scope is released, even if a
        long capture is in flight or the thread has already died."""
        self.stop_flag.set()
        abort = getattr(self.scope, "request_abort", None)
        if abort is not None:
            try:
                abort()
            except Exception:
                pass
        if self.is_alive():          # join() raises if never started
            self.join(timeout=timeout_s)
        try:
            self.scope.close()      # idempotent; no-op if the thread got it
        except Exception:
            pass

    def run(self):
        try:
            self._run_loop()
        finally:
            # the scope MUST be released on every exit path (including the
            # consecutive-error bailout), or the driver keeps the unit
            # claimed and the next launch fails with PICO_NOT_FOUND
            try:
                self.scope.close()
            except Exception:
                pass

    def _run_loop(self):
        consecutive_errors = 0
        while not self.stop_flag.is_set():
            if not self.continuous.is_set() and not self._oneshot_pending():
                self._last_trig_time = None   # sweep-rate stat would be stale
                time.sleep(0.02)              # idle: no captures armed
                continue
            if self._pending_window is not None:
                w, self._pending_window = self._pending_window, None
                self.window_s = w
                self.rise_s = (w - 0.002) / 1.25
                # long sweeps (100x expansion) get a coarser dt to keep the
                # sample count near 2M; still thousands of points per peak
                dt_t = max(config.DEFAULT_DT_S, w / 2_000_000)
                try:
                    self.scope.configure_window(w, dt_target_s=dt_t)
                    self.latest.clear()       # old lengths can't be averaged
                except Exception as exc:
                    self.status = f"window reconfig failed: {exc}"
            try:
                cap = self.scope.capture(timeout_s=self.window_s + 3.0)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                self.status = f"capture error: {exc}"
                if consecutive_errors > 5:
                    self.status = f"scope lost: {exc} — close and restart"
                    return
                time.sleep(0.5)
                continue

            if cap.triggered:
                now = time.monotonic()
                if self._last_trig_time is not None:
                    self.sweep_period_s = now - self._last_trig_time
                self._last_trig_time = now
                self.status = "ok"
            else:
                self.status = ("no trigger — is SA201B TRIGGER OUT connected "
                               "to the scope EXT input and the ramp enabled?")
            self.latest.append((time.time(), cap))
            if not self.continuous.is_set():
                self._consume_oneshot()
            self._maybe_adapt_window(cap)

    def _maybe_adapt_window(self, cap):
        """Track the real ramp length using the MONITOR OUT channel."""
        if self.window_fixed or cap.monitor is None or not cap.triggered:
            return
        mon = cap.monitor
        if float(np.max(mon) - np.min(mon)) < 0.5:
            return
        measured_rise = float(cap.t[int(np.argmax(mon))])
        if measured_rise < 0.001:
            return
        target = self._auto_window(measured_rise)
        if abs(target - self.window_s) / self.window_s > 0.12:
            self.window_s = target
            self.rise_s = measured_rise
            self.scope.configure_window(self.window_s, dt_target_s=self.scope.dt)

    def newest(self):
        return self.latest[-1] if self.latest else None

    def averaged(self):
        """Average the last n_avg triggered captures (same length only).

        The returned timestamp is the arrival time of the newest capture, so
        callers can tell whether anything new has actually arrived.
        """
        if not self.latest:
            return None
        ts_newest, newest_cap = self.latest[-1]
        caps = [c for (_ts, c) in list(self.latest) if c.triggered][-self.n_avg:]
        if not caps:
            return (ts_newest, newest_cap)
        ref = caps[-1]
        if self.n_avg == 1 or len(caps) == 1:
            return (ts_newest, ref)
        stack = [c.pd for c in caps if len(c.pd) == len(ref.pd)]
        mons = []
        if ref.monitor is not None:
            mons = [c.monitor for c in caps
                    if c.monitor is not None and len(c.monitor) == len(ref.monitor)]
        avg = ref.__class__(t=ref.t, pd=np.mean(stack, axis=0),
                            monitor=(np.mean(mons, axis=0) if mons else ref.monitor),
                            dt=ref.dt, triggered=True, clipped=ref.clipped)
        return (ts_newest, avg)


# ---------------------------------------------------------------------- app
class LiveApp:
    GAIN_NAMES = {0: "10k V/A", 1: "100k V/A", 2: "1M V/A"}

    def __init__(self, args):
        resolve_args(args)               # idempotent; covers direct callers
        self.args = args
        self.fsr_hz = args.fsr_ghz * 1e9
        self.instrument_hz = args.instrument_mhz * 1e6
        self.mode = "single" if args.single else "live"
        self._armed = False                  # single mode: wait for Run once
        self._arm_time = 0.0
        self._last_single_stamp = None
        self.wavelength_nm = float(args.wavelength_nm)
        self.theme_name = args.theme
        self._box_guard = False
        self._manual_gain_idx = 0 if args.pdgain == "auto" else int(args.pdgain)
        self.align_mode = False
        # graph 2 x-range: "auto" (fit-driven), "full" (one whole FSR), or
        # "manual" (user min/max, clamped to +-FSR/2)
        self.zoom_span_mode = getattr(args, "span_mode", None) or "auto"
        self.zoom_span_manual = tuple(getattr(args, "span_manual", None)
                                      or (-1000.0, 1000.0))
        self.scan_amplitude = float(args.amplitude)
        self.scan_offset = float(args.offset)
        span_ms = (config.RISETIME_MAX_S - config.RISETIME_MIN_S) * 1e3
        self.scan_sweep_ms = (config.RISETIME_MIN_S * 1e3 +
                              args.risetime_step / config.RISETIME_STEPS * span_ms)
        self.scan_expand_idx = int(args.sweep_expand)
        self.ctrl, self.scope, self.rise_s = open_hardware(args)
        self.acq = self._new_acquirer()
        # auto_gain is user *intent*; a missing controller only gates its
        # effect, so intent survives USB dropouts and reconnects
        self.auto_gain = args.pdgain == "auto"
        self._ctrl_retry_at = 0.0
        self._ctrl_reconnecting = False
        self._ctrl_restored = False
        # controller writes take ~0.5 s over serial (set + verify); they run
        # on this worker so button clicks and text boxes respond instantly
        self._ctrl_q = queue.Queue()
        self._ctrl_err = None            # (message, monotonic time)
        threading.Thread(target=self._ctrl_worker, daemon=True).start()
        self._fsr_hist = collections.deque(maxlen=20)
        self._last_err_hz = None
        self._gain_cooldown_until = 0.0
        # cached copy of the SA201B gain index -- querying the device takes
        # ~100 ms of serial I/O, far too slow for the per-frame UI path
        self._gain_cache = 0 if args.pdgain == "auto" else int(args.pdgain)
        self._gain_busy = False
        self.paused = False
        self.history = collections.deque()   # (wall_time, linewidth_hz)
        self.t_start = time.time()
        self.last_result = None
        self._last_processed = None
        self.log_path = None
        self._log_file = None
        self._log_writer = None
        if not args.no_log:
            os.makedirs(LOG_DIR, exist_ok=True)
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(LOG_DIR, f"session_{stamp}.csv")
            self._log_file = open(self.log_path, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(
                ["unix_time", "iso_time", "linewidth_hz", "linewidth_direct_hz",
                 "deconvolved_hz", "finesse", "fsr_period_s", "hz_per_s",
                 "peak_v", "n_modes", "pd_gain_index", "wavelength_nm",
                 "linewidth_pm", "linewidth_err_hz", "transverse_frac",
                 "flags"])
        self._build_figure()

    def _new_acquirer(self) -> Acquirer:
        a = self.args
        return Acquirer(self.scope, self.rise_s, a.dt_us * 1e-6,
                        None if a.window_ms is None else a.window_ms / 1e3,
                        a.avg, continuous=(self.mode == "live"))

    # ------------------------------------------------------------- figure
    def _build_figure(self):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.rcParams.update({
            "figure.facecolor": config.COL_PAGE,
            "axes.facecolor": config.COL_SURFACE,
            "axes.edgecolor": config.COL_AXIS,
            "axes.labelcolor": config.COL_INK2,
            "axes.titlecolor": config.COL_INK,
            "axes.grid": True,
            "grid.color": config.COL_GRID,
            "grid.linewidth": 0.8,
            "xtick.color": config.COL_MUTED,
            "ytick.color": config.COL_MUTED,
            "xtick.labelcolor": config.COL_INK2,
            "ytick.labelcolor": config.COL_INK2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            # keep matplotlib's built-in shortcuts off our keys
            "keymap.grid": [],
            "keymap.pan": [],
            "keymap.save": ["ctrl+s"],
        })
        self.fig, (self.ax_sweep, self.ax_zoom, self.ax_trend) = plt.subplots(
            3, 1, figsize=(11.8, 8.4), height_ratios=[1.1, 1.7, 1.0])
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("SA210 laser linewidth")
            try:    # dedicated-station default: start maximized ('f' = fullscreen)
                self.fig.canvas.manager.window.state("zoomed")
            except Exception:
                pass
        self.fig.subplots_adjust(left=0.075, right=0.70, top=0.94,
                                 bottom=0.075, hspace=0.52)
        for ax in (self.ax_sweep, self.ax_zoom, self.ax_trend):
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        self.ax_sweep.set_title("Full sweep (photodiode amplifier out)",
                                loc="left", fontsize=10)
        self.ax_sweep.set_xlabel("time along ramp (ms)", fontsize=9)
        self.ax_sweep.set_ylabel("signal (V)", fontsize=9)
        (self.ln_sweep,) = self.ax_sweep.plot(
            [], [], color=config.COL_SERIES1, lw=1.0)
        (self.mk_peaks,) = self.ax_sweep.plot(
            [], [], linestyle="none", marker="v", ms=6,
            markerfacecolor="none", markeredgecolor=config.COL_MUTED)

        self.ax_zoom.set_title("Main peak, frequency calibrated", loc="left",
                               fontsize=10)
        self.ax_zoom.set_xlabel("optical frequency offset (MHz)", fontsize=9)
        self.ax_zoom.set_ylabel("signal (V)", fontsize=9)
        (self.ln_zoom,) = self.ax_zoom.plot(
            [], [], color=config.COL_SERIES1, lw=1.6, label="measured")
        (self.ln_fit,) = self.ax_zoom.plot(
            [], [], color=config.COL_SERIES2, lw=2.0, ls="--",
            label="Lorentzian fit")
        self.ax_zoom.legend(loc="upper right", frameon=False, fontsize=9,
                            labelcolor=config.COL_INK2)

        self.ax_trend.set_title("Linewidth (FWHM) history", loc="left",
                                fontsize=10)
        self.ax_trend.set_xlabel("elapsed time (s)", fontsize=9)
        self.ax_trend.set_ylabel("FWHM (MHz)", fontsize=9)
        (self.ln_trend,) = self.ax_trend.plot(
            [], [], color=config.COL_SERIES1, lw=1.4, marker="o", ms=2.5,
            markerfacecolor=config.COL_SERIES1)

        # --- drag-to-zoom state (works while data keeps streaming) ---------
        from matplotlib.patches import Rectangle
        self._zoom_axes = (self.ax_sweep, self.ax_zoom, self.ax_trend)
        # per axis AND per dimension, so a wide flat drag zooms x only
        self._user_zoom = {(id(a), d): False
                           for a in self._zoom_axes for d in ("x", "y")}
        self._auto_lims = {}       # what auto-scaling wants, for "reset view"
        self._drag = None
        self._zoom_rect = {}
        for ax in self._zoom_axes:
            rect = Rectangle((0, 0), 0, 0, visible=False, animated=True,
                             facecolor=config.COL_SERIES1, alpha=0.20,
                             edgecolor=config.COL_SERIES1, lw=1.0, zorder=5)
            ax.add_patch(rect)
            self._zoom_rect[id(ax)] = rect

        x0 = 0.725
        self.txt_head = self.fig.text(x0, 0.90, "—", fontsize=26,
                                      color=config.COL_INK, fontweight="bold")
        self.txt_sub = self.fig.text(x0, 0.865, "", fontsize=10,
                                     color=config.COL_INK2)
        self.txt_status = self.fig.text(x0, 0.835, "", fontsize=9.0,
                                        color=config.COL_CRITICAL, va="top",
                                        wrap=True)
        self.txt_stats = self.fig.text(x0, 0.755, "", fontsize=9.5,
                                       color=config.COL_INK2, va="top",
                                       linespacing=1.5)
        from matplotlib.widgets import Button, TextBox

        class _PatchedTextBox(TextBox):
            # matplotlib 3.11 bug: TextBox._resize is wrapped by a decorator
            # that expects mouse events, so plain ResizeEvents (no .inaxes)
            # raise AttributeError on every window resize. An undecorated
            # override restores the intended stop-typing behavior.
            def _resize(self, event):
                self.stop_typing()

        def _style(widget, wax):
            widget.label.set_fontsize(9)
            widget.label.set_color(config.COL_INK)
            for spine in wax.spines.values():
                spine.set_color(config.COL_AXIS)

        def _make_box(x, y, w, label, initial, callback, h=0.042):
            bax = self.fig.add_axes([x, y, w, h])
            box = _PatchedTextBox(bax, label, initial=initial,
                                  color=config.COL_SURFACE,
                                  hovercolor="#edece6")
            box.label.set_fontsize(8.5)
            box.label.set_color(config.COL_INK2)
            box.text_disp.set_fontsize(9)
            box.text_disp.set_color(config.COL_INK)
            for spine in bax.spines.values():
                spine.set_color(config.COL_AXIS)
            box.on_submit(callback)
            return box

        self.box_wavelength = _make_box(
            x0 + 0.14, 0.375, 0.07, "λ nm (100–5000)  ",
            f"{self.wavelength_nm:g}", self._on_wavelength)
        self.box_amplitude = _make_box(
            x0 + 0.075, 0.317, 0.04, "ampl V (1–30) ",
            f"{self.scan_amplitude:g}", self._on_amplitude_box)
        self.box_offset = _make_box(
            x0 + 0.20, 0.317, 0.04, "offs V (0–15) ",
            f"{self.scan_offset:g}", self._on_offset_box)
        self.box_sweep = _make_box(
            x0 + 0.095, 0.259, 0.04, "sweep ms (10–100) ",
            f"{self.scan_sweep_ms:g}", self._on_sweep_box)

        self.EXPAND_LABELS = [f"{f}×" for f in config.SWEEP_EXPANSION_FACTORS]
        self.combo_expand = self._add_combo(
            x0 + 0.155, 0.259, 0.085, self.EXPAND_LABELS,
            self._on_expand_combo)
        self.txt_expand_label = self.fig.text(
            x0 + 0.152, 0.312, "expand", fontsize=7.5,
            color=config.COL_MUTED, va="top")

        self.GAIN_LABELS = ["Auto", "10k", "100k", "1M"]
        self.txt_gain_label = self.fig.text(
            x0, 0.2225, "PD gain (V/A)", fontsize=9,
            color=config.COL_INK2, va="center")
        self.combo_gain = self._add_combo(
            x0 + 0.115, 0.201, 0.125, self.GAIN_LABELS, self._on_gain_combo)

        ax_run = self.fig.add_axes([x0, 0.143, 0.115, 0.045])
        ax_mode = self.fig.add_axes([x0 + 0.125, 0.143, 0.115, 0.045])
        self.btn_run = Button(ax_run, "Run once", color=config.COL_SURFACE,
                              hovercolor="#edece6")
        self.btn_mode = Button(ax_mode, "", color=config.COL_SURFACE,
                               hovercolor="#edece6")
        _style(self.btn_run, ax_run)
        _style(self.btn_mode, ax_mode)
        self.btn_run.on_clicked(self._on_run_once)
        self.btn_mode.on_clicked(self._on_toggle_mode)

        ax_exp = self.fig.add_axes([x0, 0.085, 0.115, 0.045])
        self.btn_export = Button(ax_exp, "Export data",
                                 color=config.COL_SURFACE,
                                 hovercolor="#edece6")
        _style(self.btn_export, ax_exp)
        self.btn_export.on_clicked(self._on_export)
        ax_align = self.fig.add_axes([x0 + 0.125, 0.085, 0.115, 0.045])
        self.btn_align = Button(ax_align, "Align: off",
                                color=config.COL_SURFACE,
                                hovercolor="#edece6")
        _style(self.btn_align, ax_align)
        self.btn_align.on_clicked(self._on_toggle_align)

        ax_theme = self.fig.add_axes([x0 + 0.185, 0.952, 0.055, 0.038])
        self.btn_theme = Button(ax_theme, "", color=config.COL_SURFACE,
                                hovercolor="#edece6")
        _style(self.btn_theme, ax_theme)
        self.btn_theme.on_clicked(self._on_toggle_theme)

        # one "reset view" button per graph, tucked above its top-right corner
        self._reset_buttons = {}
        for ax in self._zoom_axes:
            pos = ax.get_position()
            bax = self.fig.add_axes([pos.x1 - 0.068, pos.y1 + 0.007,
                                     0.068, 0.026])
            b = Button(bax, "reset view", color=config.COL_SURFACE,
                       hovercolor="#edece6")
            _style(b, bax)
            b.label.set_fontsize(7.5)
            b.on_clicked(lambda _e, a=ax: self._reset_view(a))
            self._reset_buttons[id(ax)] = b

        # graph 2 x-range controls, on the same row as its reset button
        zpos = self.ax_zoom.get_position()
        zy = zpos.y1 + 0.007
        half = self.fsr_hz / 2e6
        self.txt_span_label = self.fig.text(
            0.283, zy + 0.019,
            f"x-range (MHz, −{half:g}…{half:g}):", fontsize=7.5,
            color=config.COL_MUTED, va="center")
        self.SPAN_LABELS = ["Auto", f"Full {self.args.fsr_ghz:g} GHz",
                            "Manual"]
        self.combo_span = self._add_combo(0.400, zy, 0.085, self.SPAN_LABELS,
                                          self._on_span_combo)
        self.box_span_min = _make_box(0.507, zy, 0.038, "min ",
                                      f"{self.zoom_span_manual[0]:g}",
                                      self._on_span_min, h=0.026)
        self.box_span_max = _make_box(0.567, zy, 0.038, "max ",
                                      f"{self.zoom_span_manual[1]:g}",
                                      self._on_span_max, h=0.026)
        for b in (self.box_span_min, self.box_span_max):
            b.label.set_fontsize(7.5)
            b.text_disp.set_fontsize(8)

        for evt, cb in (("button_press_event", self._on_zoom_press),
                        ("motion_notify_event", self._on_zoom_motion),
                        ("button_release_event", self._on_zoom_release)):
            self.fig.canvas.mpl_connect(evt, cb)

        self._sync_mode_button()
        self._sync_gain_combo()
        self._sync_expand_combo()
        self._sync_span_widgets()
        self.txt_keys = self.fig.text(
            x0, 0.045,
            "r run · m mode · t align · g gain · a auto\n"
            "e export · s snap · d theme · v reset views\n"
            "p pause · q quit   ·   drag on a graph to zoom",
            fontsize=8, color=config.COL_MUTED, va="top")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        if self.mode == "single":
            self.txt_stats.set_text('single mode — press "Run once" (or r)\n'
                                    "to capture a sweep")

        # Blitting: the full figure takes >100 ms to draw, which is what made
        # the UI lag. Dynamic artists are marked animated and blitted over a
        # cached background; a full draw happens only when axis limits move.
        self._animated = [self.ln_sweep, self.mk_peaks, self.ln_zoom,
                          self.ln_fit, self.ln_trend, self.txt_head,
                          self.txt_sub, self.txt_stats, self.txt_status,
                          *self._zoom_rect.values()]
        for a in self._animated:
            a.set_animated(True)
        # Widget axes are re-drawn from live state on every blit; otherwise a
        # blit restores the cached background and wipes hover highlights the
        # instant a new sweep is rendered.
        self._theme_buttons = [self.btn_run, self.btn_mode, self.btn_export,
                               self.btn_align, self.btn_theme,
                               *self._reset_buttons.values()]
        self._theme_boxes = [self.box_wavelength, self.box_amplitude,
                             self.box_offset, self.box_sweep,
                             self.box_span_min, self.box_span_max]
        self._widget_axes = [w.ax for w in
                             (*self._theme_buttons, *self._theme_boxes)]
        self._bg = None
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self._apply_theme(self.theme_name)
        self._sync_theme_button()

    def _on_draw(self, _event):
        """After any full draw: cache the background, re-add dynamic artists."""
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for a in self._animated:
            self.fig.draw_artist(a)

    def _lim(self, ax, axis, lo, hi, exact=False):
        """Set axis limits only when actually needed; returns True if changed.

        Hysteresis (grow at once, shrink only when data uses <55% of the
        range) keeps limits — and therefore the cached background — stable.
        An axis the user has zoomed into is left alone: the target is only
        recorded, so "reset view" knows where to go back to.
        """
        key = (id(ax), axis)
        cur = ax.get_xlim() if axis == "x" else ax.get_ylim()
        if not exact:
            span = hi - lo
            if span <= 0:
                return False
            lo_pad, hi_pad = lo - 0.1 * span, hi + 0.1 * span
        else:
            lo_pad, hi_pad = lo, hi

        if self._user_zoom.get(key):
            self._auto_lims[key] = (lo_pad, hi_pad)   # remember, don't apply
            return False

        if exact:
            if abs(cur[0] - lo) < 1e-12 and abs(cur[1] - hi) < 1e-12:
                self._auto_lims[key] = cur
                return False
        else:
            inside = lo >= cur[0] - 1e-12 and hi <= cur[1] + 1e-12
            fills = (hi - lo) > 0.55 * (cur[1] - cur[0])
            if inside and fills:
                self._auto_lims[key] = cur
                return False
        (ax.set_xlim if axis == "x" else ax.set_ylim)(lo_pad, hi_pad)
        self._auto_lims[key] = (lo_pad, hi_pad)
        return True

    # ------------------------------------------------------- drag-to-zoom
    def _toolbar_busy(self) -> bool:
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(tb is not None and getattr(tb, "mode", ""))

    def _on_zoom_press(self, event):
        if (event.button != 1 or self._toolbar_busy()
                or event.inaxes not in self._zoom_axes
                or event.xdata is None or event.ydata is None):
            return
        self._drag = (event.inaxes, event.xdata, event.ydata,
                      event.x, event.y)
        rect = self._zoom_rect[id(event.inaxes)]
        rect.set_bounds(event.xdata, event.ydata, 0, 0)
        rect.set_visible(True)

    def _on_zoom_motion(self, event):
        if self._drag is None:
            return
        ax, x0, y0, _px, _py = self._drag
        if event.xdata is None or event.ydata is None:
            return          # cursor left the axes; keep the last rectangle
        rect = self._zoom_rect[id(ax)]
        rect.set_bounds(min(x0, event.xdata), min(y0, event.ydata),
                        abs(event.xdata - x0), abs(event.ydata - y0))
        self._blit()        # live rubber band, independent of the data timer

    def _on_zoom_release(self, event):
        if self._drag is None:
            return
        ax, x0, y0, px, py = self._drag
        self._drag = None
        self._zoom_rect[id(ax)].set_visible(False)
        x1 = x0 if event.xdata is None else event.xdata
        y1 = y0 if event.ydata is None else event.ydata
        dx_px = abs((event.x if event.x is not None else px) - px)
        dy_px = abs((event.y if event.y is not None else py) - py)

        # A drag that is wide but flat zooms x only, and vice versa, so you
        # can rescale one dimension without disturbing the other.
        MIN_PX = 8
        zoomed = False
        if dx_px >= MIN_PX and x1 != x0:
            ax.set_xlim(min(x0, x1), max(x0, x1))
            self._user_zoom[(id(ax), "x")] = True
            zoomed = True
        if dy_px >= MIN_PX and y1 != y0:
            ax.set_ylim(min(y0, y1), max(y0, y1))
            self._user_zoom[(id(ax), "y")] = True
            zoomed = True
        if zoomed:
            print(f"[zoom] {ax.get_title(loc='left') or 'graph'}: "
                  f"zoomed — auto-scaling paused until reset")
        self._sync_reset_buttons()
        self._bg = None                 # ticks changed: rebuild background
        self.fig.canvas.draw_idle()

    def _reset_view(self, ax, redraw=True):
        for d in ("x", "y"):
            key = (id(ax), d)
            if self._user_zoom.get(key):
                self._user_zoom[key] = False
                lims = self._auto_lims.get(key)
                if lims:
                    (ax.set_xlim if d == "x" else ax.set_ylim)(*lims)
        self._sync_reset_buttons()
        if redraw:
            self._bg = None
            self.fig.canvas.draw_idle()

    def _reset_all_views(self):
        for ax in self._zoom_axes:
            self._reset_view(ax, redraw=False)
        self._bg = None
        self.fig.canvas.draw_idle()

    def _sync_reset_buttons(self):
        """Highlight the reset button of every graph that is zoomed."""
        T = config.THEMES[self.theme_name]
        for ax in self._zoom_axes:
            active = any(self._user_zoom.get((id(ax), d))
                         for d in ("x", "y"))
            b = self._reset_buttons[id(ax)]
            face = T["HOVER"] if active else T["SURFACE"]
            b.color = face
            b.ax.set_facecolor(face)
            b.label.set_color(T["INK"] if active else T["MUTED"])

    def _blit(self):
        """Repaint the dynamic artists over the cached background."""
        canvas = self.fig.canvas
        if self._bg is None:
            canvas.draw()
            return
        canvas.restore_region(self._bg)
        for a in self._animated:
            self.fig.draw_artist(a)
        for wax in self._widget_axes:
            self.fig.draw_artist(wax)
        canvas.blit(self.fig.bbox)

    # ---------------------------------------------------------------- theme
    def _on_toggle_theme(self, _event=None):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self._apply_theme(self.theme_name)
        self._sync_theme_button()

    def _sync_theme_button(self):
        # the button names the theme you would switch TO
        self.btn_theme.label.set_text(
            "Light" if self.theme_name == "dark" else "Dark")
        self.fig.canvas.draw_idle()

    def _apply_theme(self, name):
        """Repaint every element of the UI in the given palette."""
        T = config.THEMES[name]
        self.fig.set_facecolor(T["PAGE"])
        for ax in (self.ax_sweep, self.ax_zoom, self.ax_trend):
            ax.set_facecolor(T["SURFACE"])
            for spine in ax.spines.values():
                spine.set_color(T["AXIS"])
            ax.tick_params(colors=T["MUTED"], labelcolor=T["INK2"])
            ax.xaxis.label.set_color(T["INK2"])
            ax.yaxis.label.set_color(T["INK2"])
            ax.title.set_color(T["INK"])
            ax.grid(True, color=T["GRID"], linewidth=0.8)
        for ln in (self.ln_sweep, self.ln_zoom, self.ln_trend):
            ln.set_color(T["SERIES1"])
        self.ln_trend.set_markerfacecolor(T["SERIES1"])
        self.ln_fit.set_color(T["SERIES2"])
        self.mk_peaks.set_markeredgecolor(T["MUTED"])
        self.ax_zoom.legend(loc="upper right", frameon=False, fontsize=9,
                            labelcolor=T["INK2"])
        self.txt_head.set_color(T["INK"])
        self.txt_sub.set_color(T["INK2"])
        self.txt_stats.set_color(T["INK2"])
        self.txt_status.set_color(T["CRITICAL"])
        self.txt_keys.set_color(T["MUTED"])
        self.txt_gain_label.set_color(T["INK2"])
        self.txt_expand_label.set_color(T["MUTED"])
        self.txt_span_label.set_color(T["MUTED"])
        for b in self._theme_buttons:
            b.color = T["SURFACE"]
            b.hovercolor = T["HOVER"]
            b.ax.set_facecolor(T["SURFACE"])
            b.label.set_color(T["INK"])
            for spine in b.ax.spines.values():
                spine.set_color(T["AXIS"])
        for rect in self._zoom_rect.values():
            rect.set_facecolor(T["SERIES1"])
            rect.set_edgecolor(T["SERIES1"])
        self._sync_reset_buttons()
        for box in self._theme_boxes:
            box.color = T["SURFACE"]
            box.hovercolor = T["HOVER"]
            box.ax.set_facecolor(T["SURFACE"])
            box.label.set_color(T["INK2"])
            box.text_disp.set_color(T["INK"])
            try:
                box.cursor.set_color(T["INK"])
            except Exception:
                pass
            for spine in box.ax.spines.values():
                spine.set_color(T["AXIS"])
        self._style_combos(T)
        self._bg = None                  # cached background is now stale
        self.fig.canvas.draw_idle()

    def _style_combos(self, T):
        """Native ttk dropdowns need Tk-level styling (incl. the popup list)."""
        try:
            import tkinter.ttk as ttk
            style = ttk.Style()
            style.theme_use("clam")      # the only stock theme that obeys
            style.configure("LW.TCombobox",  # field/arrow color settings
                            fieldbackground=T["SURFACE"],
                            background=T["SURFACE"],
                            foreground=T["INK"],
                            arrowcolor=T["INK"],
                            bordercolor=T["AXIS"],
                            lightcolor=T["SURFACE"],
                            darkcolor=T["SURFACE"])
            style.map("LW.TCombobox",
                      fieldbackground=[("readonly", T["SURFACE"])],
                      foreground=[("readonly", T["INK"])],
                      selectbackground=[("readonly", T["SURFACE"])],
                      selectforeground=[("readonly", T["INK"])])
            for combo in (self.combo_gain, self.combo_expand):
                combo.configure(style="LW.TCombobox")
                pd = combo.tk.call("ttk::combobox::PopdownWindow", combo)
                combo.tk.call(f"{pd}.f.l", "configure",
                              "-background", T["SURFACE"],
                              "-foreground", T["INK"],
                              "-selectbackground", T["SERIES1"],
                              "-selectforeground", "#ffffff")
        except Exception as exc:
            print(f"[theme] dropdown styling skipped: {exc}")

    def _add_combo(self, x_fig, y_fig, w_fig, values, callback):
        """Native Tk dropdown overlaid on the canvas at figure coordinates
        (matplotlib has no combobox; the TkAgg canvas hosts real widgets)."""
        import tkinter.ttk as ttk
        tkc = self.fig.canvas.get_tk_widget()
        combo = ttk.Combobox(tkc, values=values, state="readonly",
                             font=("Segoe UI", 9))
        combo.place(relx=x_fig, rely=1.0 - y_fig, relwidth=w_fig, anchor="sw")

        def _selected(_event):
            callback(combo.get())
            tkc.focus_set()      # hand the keyboard back to the plot
        combo.bind("<<ComboboxSelected>>", _selected)
        return combo

    # ---------------------------------------------------------------- modes
    def _sync_mode_button(self):
        self.btn_mode.label.set_text(
            "Mode: Live" if self.mode == "live" else "Mode: Single")
        self.fig.canvas.draw_idle()

    def _on_run_once(self, _event=None):
        if self.mode != "single":
            self.mode = "single"
            self.acq.continuous.clear()
            self._sync_mode_button()
        self._armed = True
        self._arm_time = time.time()
        self.acq.request_oneshot(self.args.avg)   # scope captures, then idles

    def _on_toggle_mode(self, _event=None):
        self.mode = "live" if self.mode == "single" else "single"
        if self.mode == "single":
            self._armed = False              # scope idle until Run once
            self.acq.continuous.clear()
        else:
            self.acq.continuous.set()
        self._sync_mode_button()

    # ------------------------------------------------------------ text boxes
    def _norm_box(self, box, text):
        """Rewrite a TextBox without re-triggering its submit callback."""
        self._box_guard = True
        try:
            box.set_val(text)
        finally:
            self._box_guard = False

    def _typing(self) -> bool:
        return any(getattr(b, "capturekeystrokes", False)
                   for b in self._theme_boxes)

    def _on_wavelength(self, text):
        if self._box_guard:
            return
        try:
            v = float(str(text).strip())
            if not (100.0 <= v <= 5000.0):
                raise ValueError
            self.wavelength_nm = v
            print(f"[λ] wavelength set to {v:g} nm")
        except ValueError:
            self.txt_status.set_text(
                f"bad wavelength {text!r} — keeping {self.wavelength_nm:g} nm")
        self._norm_box(self.box_wavelength, f"{self.wavelength_nm:g}")

    # ------------------------------------------------- graph 2 x-range
    def _sync_span_widgets(self):
        idx = {"auto": 0, "full": 1, "manual": 2}[self.zoom_span_mode]
        self.combo_span.set(self.SPAN_LABELS[idx])
        self._norm_box(self.box_span_min, f"{self.zoom_span_manual[0]:g}")
        self._norm_box(self.box_span_max, f"{self.zoom_span_manual[1]:g}")

    def _apply_span_mode(self, mode):
        self.zoom_span_mode = mode
        # an explicit x-range choice supersedes any drag-zoom on that graph
        self._user_zoom[(id(self.ax_zoom), "x")] = False
        self._sync_span_widgets()
        self._sync_reset_buttons()
        self._bg = None
        self.fig.canvas.draw_idle()

    def _on_span_combo(self, label):
        mode = ("auto", "full", "manual")[self.SPAN_LABELS.index(label)]
        self._apply_span_mode(mode)
        print(f"[graph2] x-range mode: {mode}")

    def _on_span_min(self, text):
        self._set_span_edge(text, "min")

    def _on_span_max(self, text):
        self._set_span_edge(text, "max")

    def _set_span_edge(self, text, which):
        if self._box_guard:
            return
        half = self.fsr_hz / 2e6
        lo, hi = self.zoom_span_manual
        try:
            v = float(str(text).strip())
            if not (-half <= v <= half):
                raise ValueError(f"valid range -{half:g} to {half:g} MHz")
            lo, hi = (v, hi) if which == "min" else (lo, v)
            if hi - lo < 1.0:
                raise ValueError("max must exceed min by at least 1 MHz")
            self.zoom_span_manual = (lo, hi)
            print(f"[graph2] x-range {lo:g} .. {hi:g} MHz (manual)")
            self._apply_span_mode("manual")     # typing implies manual
            return
        except Exception as exc:
            self.txt_status.set_text(f"x-range not set: {exc}")
        self._sync_span_widgets()

    # --------------------------------------------------------- scan controls
    def _push_window(self):
        rise = (self.scan_sweep_ms / 1e3 *
                config.SWEEP_EXPANSION_FACTORS[self.scan_expand_idx])
        self.acq.request_window(rise * 1.25 + 0.002)

    def _no_ctrl_hint(self):
        self.txt_status.set_text(
            "no SA201B USB — set it on the touchscreen (display will follow)")

    def _on_amplitude_box(self, text):
        if self._box_guard:
            return
        try:
            v = float(str(text).strip())
            if not (1.0 <= v <= 30.0):
                raise ValueError("valid range 1-30 V")
            self.scan_amplitude = v
            self._ctrl_do(f"amplitude -> {v:g} V",
                          lambda c: setattr(c, "amplitude_v", v))
        except ValueError as exc:
            self.txt_status.set_text(f"amplitude not set: {exc}")
        self._norm_box(self.box_amplitude, f"{self.scan_amplitude:g}")

    def _on_offset_box(self, text):
        if self._box_guard:
            return
        try:
            v = float(str(text).strip())
            if not (0.0 <= v <= 15.0):
                raise ValueError("valid range 0-15 V")
            self.scan_offset = v
            self._ctrl_do(f"DC offset -> {v:g} V",
                          lambda c: setattr(c, "dc_offset_v", v))
        except ValueError as exc:
            self.txt_status.set_text(f"offset not set: {exc}")
        self._norm_box(self.box_offset, f"{self.scan_offset:g}")

    def _on_sweep_box(self, text):
        if self._box_guard:
            return
        lo = config.RISETIME_MIN_S * 1e3
        hi = config.RISETIME_MAX_S * 1e3
        try:
            ms = float(str(text).strip())
            if not (lo <= ms <= hi):
                raise ValueError(f"valid range {lo:g}-{hi:g} ms (at 1x)")
            step = _ms_to_step(ms)
            self.scan_sweep_ms = lo + step / config.RISETIME_STEPS * (hi - lo)
            self._push_window()
            self._ctrl_do(f"sweep -> {self.scan_sweep_ms:g} ms at 1x "
                          f"(step {step})",
                          lambda c: setattr(c, "risetime_step", step))
        except ValueError as exc:
            self.txt_status.set_text(f"sweep time not set: {exc}")
        self._norm_box(self.box_sweep, f"{self.scan_sweep_ms:g}")

    def _sync_expand_combo(self):
        self.combo_expand.set(self.EXPAND_LABELS[self.scan_expand_idx])

    def _set_expand(self, idx):
        self.scan_expand_idx = idx
        self._push_window()
        self._ctrl_do(
            f"sweep expansion -> {config.SWEEP_EXPANSION_FACTORS[idx]}x",
            lambda c: setattr(c, "sweep_expand_index", idx))
        self._sync_expand_combo()

    def _on_expand_combo(self, label):
        self._set_expand(self.EXPAND_LABELS.index(label))

    # ------------------------------------------------------- alignment mode
    def _on_toggle_align(self, _event=None):
        self.align_mode = not self.align_mode
        want_saw = not self.align_mode
        self._ctrl_do("triangle scan (alignment)" if self.align_mode
                      else "sawtooth scan (measurement)",
                      lambda c: setattr(c, "sawtooth", want_saw))
        self.btn_align.label.set_text(
            "Align: TRI" if self.align_mode else "Align: off")
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------------- gain
    def _apply_gain_choice(self, idx):
        """idx 0 = Auto; 1..3 = manual gain index (idx - 1)."""
        if self.ctrl is None:
            self.txt_status.set_text(
                "SA201B USB not connected — gain control unavailable")
        elif idx == 0:
            self.auto_gain = True
            self._gain_cooldown_until = 0.0
            print("[gain] auto-gain ON")
        else:
            g = idx - 1
            self._gain_cache = g
            self._manual_gain_idx = g
            self.auto_gain = False
            self._ctrl_do(f"manual gain -> {self.GAIN_NAMES[g]}",
                          lambda c: setattr(c, "pd_gain_index", g))
        self._sync_gain_combo()

    def _on_gain_combo(self, label):
        self._apply_gain_choice(self.GAIN_LABELS.index(label))

    def _sync_gain_combo(self):
        self.combo_gain.set(
            "Auto" if (self.auto_gain or self.ctrl is None)
            else self.GAIN_LABELS[self._manual_gain_idx + 1])

    # --------------------------------------------------------------- export
    def _on_export(self, _event=None):
        res = self.last_result
        if res is None or res.t is None:
            self.txt_status.set_text("nothing to export yet")
            return
        os.makedirs(EXPORT_DIR, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        lam = self.wavelength_nm
        wrote = []

        sweep_path = os.path.join(EXPORT_DIR, f"sweep_{stamp}.csv")
        with open(sweep_path, "w", newline="") as fh:
            fh.write("# SA210 / SA201B / PicoScope 5242D — one sweep\n")
            fh.write(f"# exported: "
                     f"{_dt.datetime.now().isoformat(timespec='seconds')}\n")
            fh.write(f"# wavelength_nm: {lam:g}\n")
            fh.write(f"# fsr_hz: {self.fsr_hz:g}\n")
            if res.hz_per_s:
                fh.write(f"# hz_per_s: {res.hz_per_s:.6g}\n")
            if res.linewidth_hz:
                wv, wu = ana.wavelength_width(res.linewidth_hz, lam)
                fh.write(f"# fwhm_mhz: {res.linewidth_hz / 1e6:.4f}\n")
                if self._last_err_hz:
                    fh.write(f"# fwhm_err_mhz_1sigma: "
                             f"{self._last_err_hz / 1e6:.4f}\n")
                fh.write(f"# fwhm_wavelength: {wv:.4g} {wu}\n")
            if res.finesse:
                fh.write(f"# effective_finesse: {res.finesse:.1f}\n")
            if res.transverse_frac is not None:
                fh.write(f"# transverse_frac: {res.transverse_frac:.3f}\n")
            if self.ctrl is not None:
                fh.write(f"# pd_gain: {self.GAIN_NAMES[self._gain_cache]}\n")
            fh.write(f"# mode: {self.mode}\n")
            w = csv.writer(fh)
            w.writerow(["time_s", "signal_v", "freq_offset_mhz", "fit_v"])
            fit_map = {}
            if res.fit_t is not None:
                fit_map = {round(float(tt), 12): float(fv)
                           for tt, fv in zip(res.fit_t, res.fit_v)}
            calibrated = bool(res.hz_per_s and res.fit_center_s is not None)
            for tt, vv in zip(res.t, res.v):
                f_off = (f"{(tt - res.fit_center_s) * res.hz_per_s / 1e6:.6f}"
                         if calibrated else "")
                fv = fit_map.get(round(float(tt), 12))
                w.writerow([f"{tt:.9f}", f"{vv:.6f}", f_off,
                            "" if fv is None else f"{fv:.6f}"])
        wrote.append(sweep_path)

        if self.history:
            hist_path = os.path.join(EXPORT_DIR, f"history_{stamp}.csv")
            with open(hist_path, "w", newline="") as fh:
                fh.write(f"# linewidth history — wavelength_nm: {lam:g}\n")
                w = csv.writer(fh)
                w.writerow(["iso_time", "elapsed_s", "linewidth_mhz",
                            "linewidth_pm"])
                for wall, lw in self.history:
                    w.writerow([
                        _dt.datetime.fromtimestamp(wall)
                        .isoformat(timespec="seconds"),
                        f"{wall - self.t_start:.2f}",
                        f"{lw / 1e6:.4f}",
                        f"{ana.delta_lambda_m(lw, lam) * 1e12:.6f}"])
            wrote.append(hist_path)

        names = ", ".join(os.path.basename(p) for p in wrote)
        self.txt_status.set_text(f"exported: {names}")
        print("[export] " + " | ".join(wrote))

    # ---------------------------------------------------------------- keys
    def _on_key(self, event):
        if self._typing():
            return      # user is typing in one of the input boxes
        if event.key == "q":
            self.plt.close(self.fig)
        elif event.key == "p":
            self.paused = not self.paused
        elif event.key == "r":
            self._on_run_once()
        elif event.key == "m":
            self._on_toggle_mode()
        elif event.key == "t":
            self._on_toggle_align()
        elif event.key == "d":
            self._on_toggle_theme()
        elif event.key == "v":
            self._reset_all_views()
        elif event.key == "s":
            self._snapshot()
        elif event.key == "e":
            self._on_export()
        elif event.key == "a":
            if self.ctrl is not None:
                self.auto_gain = not self.auto_gain
                if not self.auto_gain:
                    self._manual_gain_idx = self._gain_cache
                print(f"[gain] auto-gain {'ON' if self.auto_gain else 'OFF'}")
                self._sync_gain_combo()
        elif event.key == "g" and self.ctrl is not None:
            self._apply_gain_choice(((self._gain_cache + 1) % 3) + 1)
        elif event.key in ("left", "right") and self.ctrl is not None:
            delta = 0.25 if event.key == "right" else -0.25
            new = float(np.clip(self.scan_offset + delta, 0.0, 15.0))
            self.scan_offset = new
            self._norm_box(self.box_offset, f"{new:g}")
            self._ctrl_do(f"DC offset -> {new:.2f} V",
                          lambda c: setattr(c, "dc_offset_v", new))

    def _snapshot(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        png = os.path.join(LOG_DIR, f"snapshot_{stamp}.png")
        for a in self._animated:     # ensure traces render into the file
            a.set_animated(False)
        try:
            self.fig.savefig(png, dpi=150, facecolor=self.fig.get_facecolor())
        finally:
            for a in self._animated:
                a.set_animated(True)
            self._bg = None          # print renderer invalidated the cache
            self.fig.canvas.draw_idle()
        res = self.last_result
        if res is not None and res.t is not None:
            csv_path = os.path.join(LOG_DIR, f"snapshot_{stamp}.csv")
            with open(csv_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time_s", "signal_v"])
                w.writerows(zip(res.t, res.v))
        print(f"[snapshot] saved {png}")

    # ------------------------------------------------------------- settings
    def save_settings(self) -> None:
        """Persist the user's choices for the next launch."""
        data = {
            "wavelength_nm": self.wavelength_nm,
            "theme": self.theme_name,
            "amplitude_v": self.scan_amplitude,
            "offset_v": self.scan_offset,
            "sweep_ms": self.scan_sweep_ms,
            "expand_idx": self.scan_expand_idx,
            "gain_auto": bool(self.auto_gain),
            "gain_idx": int(self._manual_gain_idx),
            "span_mode": self.zoom_span_mode,
            "span_manual": list(self.zoom_span_manual),
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            print(f"[settings] saved to {os.path.basename(SETTINGS_PATH)}")
        except Exception as exc:
            print(f"[settings] could not save: {exc}")

    # ---------------------------------------------------------- uncertainty
    def _uncertainty_hz(self, res):
        """1-sigma uncertainty on the linewidth, in Hz.

        Two independent contributions, added in quadrature:
          * the Lorentzian fit's own covariance on the width, and
          * the frequency calibration, whose scale (Hz per second of sweep)
            is set by the FSR peak spacing and jitters sweep to sweep. Its
            relative error propagates directly into the width.
        """
        if not res.ok or not res.linewidth_hz or not res.fsr_period_s:
            return None
        self._fsr_hist.append(res.fsr_period_s)
        rel_cal = 0.0
        if len(self._fsr_hist) >= 3:
            arr = np.fromiter(self._fsr_hist, dtype=float)
            mean = float(arr.mean())
            if mean > 0:
                # standard error of the mean period -> relative scale error
                rel_cal = float(arr.std(ddof=1)) / mean / np.sqrt(len(arr))
        err_cal = res.linewidth_hz * rel_cal
        err_fit = 0.0
        if res.fit_fwhm_err_s and res.hz_per_s:
            err_fit = res.fit_fwhm_err_s * res.hz_per_s
        total = float(np.hypot(err_fit, err_cal))
        return total if np.isfinite(total) and total > 0 else None

    # -------------------------------------------- async controller writes
    def _ctrl_do(self, label, fn):
        """Queue a controller write; the UI state is already updated
        optimistically, and a reconnect re-applies everything anyway."""
        if self.ctrl is None:
            self._no_ctrl_hint()
            return
        self._ctrl_q.put((label, fn))

    def _ctrl_worker(self):
        while True:
            label, fn = self._ctrl_q.get()
            ctrl = self.ctrl
            if ctrl is None:
                self._ctrl_q.task_done()
                continue
            try:
                fn(ctrl)
                print(f"[scan] {label}")
            except Exception as exc:
                print(f"[ctrl] {label} failed: {exc}")
                self._ctrl_err = (f"{label} failed: {exc}", time.monotonic())
                if _is_ctrl_error(exc):
                    self._note_ctrl_failure(exc)
            finally:
                self._ctrl_q.task_done()

    # ------------------------------------------------- SA201B auto-reconnect
    def _note_ctrl_failure(self, exc) -> None:
        """Serial link died: drop the controller and let the watchdog retry."""
        if self.ctrl is None:
            return
        print(f"[SA201B] connection lost: {exc}")
        try:
            self.ctrl.close()
        except Exception:
            pass
        self.ctrl = None
        self._ctrl_retry_at = time.monotonic() + 2.0

    def _ctrl_watchdog(self) -> None:
        """Called every UI tick; re-attaches the SA201B when it comes back."""
        if self._ctrl_restored:
            self._ctrl_restored = False
            self._sync_gain_combo()      # UI updates on the UI thread only
        if (self.ctrl is not None or self.args.no_controller
                or self._ctrl_reconnecting
                or time.monotonic() < self._ctrl_retry_at):
            return
        self._ctrl_reconnecting = True

        def _worker():
            try:
                from sa201b import SA201B
                ctrl = SA201B(port=self.args.port)
                ctrl.apply_scan_settings(
                    amplitude_v=self.scan_amplitude,
                    dc_offset_v=self.scan_offset,
                    risetime_step=_ms_to_step(self.scan_sweep_ms),
                    sweep_expand_index=self.scan_expand_idx,
                    pd_gain_index=self._gain_cache)
                ctrl.sawtooth = not self.align_mode
                self.ctrl = ctrl
                self._ctrl_restored = True
                print(f"[SA201B] reconnected on {ctrl.port}; "
                      f"settings re-applied")
            except Exception:
                self._ctrl_retry_at = time.monotonic() + 4.0
            finally:
                self._ctrl_reconnecting = False

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------ auto gain
    def _auto_gain_step(self, res):
        """Decide on a gain change; apply it on a worker thread so the
        ~0.5 s serial round-trip never blocks the UI."""
        if (not self.auto_gain or self.ctrl is None or res is None
                or self._gain_busy):
            return
        now = time.monotonic()
        if now < self._gain_cooldown_until:
            return
        g = self._gain_cache
        if res.saturating and g > 0:
            target, why = g - 1, "saturating"
        elif res.weak and not res.saturating and g < 2:
            target, why = g + 1, "weak signal"
        else:
            return
        self._gain_cooldown_until = now + 4.0
        self._gain_busy = True

        def _worker():
            try:
                self.ctrl.pd_gain_index = target
                self._gain_cache = target
                print(f"[gain] {why} -> {self.GAIN_NAMES[target]}")
            except Exception as exc:
                print(f"[gain] auto-gain failed: {exc}")
                if _is_ctrl_error(exc):
                    self._note_ctrl_failure(exc)
            finally:
                self._gain_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ log
    def _log(self, res):
        if self._log_writer is None or res is None:
            return
        flags = ";".join(f for f, on in
                         [("saturating", res.saturating), ("weak", res.weak)]
                         if on) or "ok" if res.ok else res.message
        now = time.time()
        self._log_writer.writerow([
            f"{now:.3f}", _dt.datetime.now().isoformat(timespec="seconds"),
            f"{res.linewidth_hz:.6g}" if res.linewidth_hz else "",
            f"{res.linewidth_direct_hz:.6g}" if res.linewidth_direct_hz else "",
            f"{res.deconvolved_hz:.6g}" if res.deconvolved_hz is not None else "",
            f"{res.finesse:.1f}" if res.finesse else "",
            f"{res.fsr_period_s:.6g}" if res.fsr_period_s else "",
            f"{res.hz_per_s:.6g}" if res.hz_per_s else "",
            f"{np.max(res.v):.4f}" if res.v is not None else "",
            len(res.mode_offsets_hz),
            self._gain_cache if self.ctrl is not None else "",
            f"{self.wavelength_nm:g}",
            (f"{ana.delta_lambda_m(res.linewidth_hz, self.wavelength_nm) * 1e12:.6g}"
             if res.linewidth_hz else ""),
            f"{self._last_err_hz:.6g}" if self._last_err_hz else "",
            (f"{res.transverse_frac:.3f}"
             if res.transverse_frac is not None else ""),
            flags])
        if int(now) % 5 == 0:
            self._log_file.flush()

    # ---------------------------------------------------------------- frame
    def _update(self, _frame):
        if self.paused or self.acq is None:
            return
        self._ctrl_watchdog()      # re-attach the SA201B if its USB returns
        item = self.acq.averaged()
        if item is None or item[0] == self._last_processed:
            return
        if self.mode == "single":
            if not self._armed or item[0] < self._arm_time:
                return      # frozen, or waiting for a sweep newer than the click
        self._last_processed = item[0]
        wall, cap = item
        if self.mode == "single":
            self._armed = False
            self._last_single_stamp = _dt.datetime.fromtimestamp(
                wall).strftime("%H:%M:%S")

        t, v, ramp_end = ana.trim_to_rising_ramp(cap, self.acq.rise_s)
        # nominal piezo calibration (~10 V/FSR) as a soft prior so the FSR
        # search can tell a half-FSR transverse comb from the real spacing
        expected_T = None
        if self.scan_amplitude > 0:
            expected_T = (self.acq.rise_s * config.VOLTS_PER_FSR
                          / self.scan_amplitude)
        res = ana.analyze_sweep(t, v, fsr_hz=self.fsr_hz,
                                instrument_hz=self.instrument_hz,
                                expected_fsr_period_s=expected_T)
        self.last_result = res
        self._auto_gain_step(res)
        self._last_err_hz = self._uncertainty_hz(res)
        if not self.align_mode:      # alignment sweeps don't pollute the log
            self._log(res)

        changed = self._bg is None

        # ---- full sweep panel (envelope-decimated for fast redraws)
        sx, sy = _decimate_envelope(t * 1e3, v)
        self.ln_sweep.set_data(sx, sy)
        self.mk_peaks.set_data(res.peak_times * 1e3,
                               res.peak_heights + 0.06 * max(1e-3, np.max(v)))
        changed |= self._lim(self.ax_sweep, "x", t[0] * 1e3, t[-1] * 1e3,
                             exact=True)
        top = max(0.5, float(np.max(v)) * 1.25)
        changed |= self._lim(self.ax_sweep, "y", -0.04 * top, top)

        # ---- zoom panel
        if res.ok and res.hz_per_s and res.fit_center_s is not None:
            f_off = (t - res.fit_center_s) * res.hz_per_s / 1e6
            half_fsr = self.fsr_hz / 2e6
            if self.zoom_span_mode == "full":
                xlo, xhi = -half_fsr, half_fsr
            elif self.zoom_span_mode == "manual":
                xlo, xhi = self.zoom_span_manual
            else:
                biggest_mode = max((abs(m) for m in res.mode_offsets_hz),
                                   default=0.0) / 1e6
                lw_mhz = res.linewidth_hz / 1e6
                span = float(np.clip(
                    max(12 * lw_mhz, 1.3 * biggest_mode + 300),
                    250, 0.45 * self.fsr_hz / 1e6))
                for q in (250.0, 500.0, 1000.0, 2000.0, 4500.0):
                    if span <= q:    # quantize so the limits rarely move
                        span = q
                        break
                xlo, xhi = -span, span
            sel = (f_off >= xlo) & (f_off <= xhi)
            zx, zy = _decimate_envelope(f_off[sel], v[sel])
            self.ln_zoom.set_data(zx, zy)
            if res.fit_t is not None:
                self.ln_fit.set_data(
                    (res.fit_t - res.fit_center_s) * res.hz_per_s / 1e6,
                    res.fit_v)
            else:
                self.ln_fit.set_data([], [])
            changed |= self._lim(self.ax_zoom, "x", xlo, xhi, exact=True)
            ztop = max(0.2, float(np.max(v[sel])) * 1.2) if sel.any() else 1.0
            changed |= self._lim(self.ax_zoom, "y", -0.04 * ztop, ztop)

        # ---- trend panel
        if res.ok and res.linewidth_hz and not self.align_mode:
            self.history.append((wall, res.linewidth_hz))
        if self.mode == "live":     # in single mode old runs stay on screen
            cutoff = time.time() - self.args.history_s
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()
        if self.history:
            xs = np.array([w - self.t_start for w, _ in self.history])
            ys = np.array([lw / 1e6 for _, lw in self.history])
            self.ln_trend.set_data(xs, ys)
            if self.mode == "live":
                # quantize the scroll to 10 s steps so the background (and
                # tick labels) only regenerate occasionally, not every frame
                right = max(10.0, np.ceil(xs[-1] / 10.0) * 10.0)
                left = max(0.0, right - self.args.history_s)
                changed |= self._lim(self.ax_trend, "x", left, right,
                                     exact=True)
            else:
                changed |= self._lim(self.ax_trend, "x",
                                     max(0.0, xs[0] - 5.0),
                                     max(10.0, xs[-1] + 5.0))
            lo, hi = float(ys.min()), float(ys.max())
            pad = max(2.0, 0.15 * (hi - lo))
            changed |= self._lim(self.ax_trend, "y", max(0.0, lo - pad),
                                 hi + pad)

        # ---- readouts
        if self.align_mode:
            peak_v = float(np.max(v)) if len(v) else 0.0
            self.txt_head.set_text(f"{peak_v:.3f} V")
            sub = "alignment — maximize peak height (triangle scan)"
            if res.transverse_frac is not None:
                sub = (f"alignment — peak height up, transverse "
                       f"{res.transverse_frac * 100:.0f}% down")
            self.txt_sub.set_text(sub)
        elif res.ok and res.linewidth_hz:
            err = self._last_err_hz
            head = f"{res.linewidth_hz / 1e6:.1f}"
            if err:
                head += f" ± {err / 1e6:.1f}"
            self.txt_head.set_text(head + " MHz")
            wl_v, wl_u = ana.wavelength_width(res.linewidth_hz,
                                              self.wavelength_nm)
            note = ("instrument-limited"
                    if res.linewidth_hz < 1.35 * self.instrument_hz
                    else f"est. laser ≈ {res.deconvolved_hz / 1e6:.0f} MHz "
                         f"(67 MHz removed)")
            self.txt_sub.set_text(
                f"= {wl_v:.3g} {wl_u} @ {self.wavelength_nm:g} nm — {note}")
        else:
            self.txt_head.set_text("—")
            self.txt_sub.set_text("")

        stats = []
        if self.mode == "single" and self._last_single_stamp:
            stats.append(f"single sweep captured {self._last_single_stamp}")
        if self._last_err_hz and res.linewidth_hz:
            frac = 100.0 * self._last_err_hz / res.linewidth_hz
            stats.append(f"uncertainty: ±{self._last_err_hz / 1e6:.2f} MHz "
                         f"({frac:.1f}%, 1σ)")
        if res.linewidth_direct_hz:
            stats.append(f"half-max width: {res.linewidth_direct_hz / 1e6:.1f} MHz")
        if res.fit_r2 is not None:
            stats.append(f"fit R²: {res.fit_r2:.3f}")
        if res.finesse:
            stats.append(f"effective finesse: {res.finesse:.0f}  (spec >150)")
        if res.fsr_period_s:
            stats.append(f"FSR spacing: {res.fsr_period_s * 1e3:.2f} ms "
                         f"→ {res.hz_per_s / 1e12:.2f} GHz/ms")
        n_modes = len(res.mode_offsets_hz)
        if n_modes > 1:
            spacings = np.diff(sorted(res.mode_offsets_hz)) / 1e6
            stats.append(f"modes in one FSR: {n_modes} "
                         f"(spacing {', '.join(f'{s:.0f}' for s in spacings)} MHz)")
        elif res.ok:
            stats.append("modes in one FSR: 1 (single-frequency)")
        if res.transverse_frac is not None:
            frac = res.transverse_frac
            tag = ("good" if frac < 0.05 else
                   "fair" if frac < 0.20 else "poor")
            stats.append(f"transverse modes: {frac * 100:.0f}% of main "
                         f"(alignment {tag})")
        if self.ctrl is not None:
            stats.append(f"PD gain: {self.GAIN_NAMES[self._gain_cache]}"
                         f"{'  [auto]' if self.auto_gain else ''}")
        factor = config.SWEEP_EXPANSION_FACTORS[self.scan_expand_idx]
        scan_line = (f"scan: {self.scan_amplitude:g} V · "
                     f"{self.scan_sweep_ms:g} ms × {factor}")
        if self.scan_offset:
            scan_line += f" · offs {self.scan_offset:g} V"
        stats.append(scan_line)
        if self.align_mode:
            stats.append("ALIGNMENT MODE — triangle scan, not logged")
        if self.mode == "live" and self.acq.sweep_period_s:
            stats.append(f"sweep rate: {1.0 / self.acq.sweep_period_s:.1f} Hz"
                         f"   window: {self.acq.window_s * 1e3:.1f} ms")
        if self.log_path:
            stats.append(f"log: {os.path.basename(self.log_path)}")
        self.txt_stats.set_text("\n".join(stats))

        warn = []
        if not res.ok:
            warn.append(res.message)
        if res.saturating:
            warn.append("! signal saturating — lower PD gain / input power")
        if cap.clipped:
            warn.append("! scope ADC clipped")
        if res.transverse_frac is not None and res.transverse_frac > 0.5:
            warn.append("! strong transverse modes — improve alignment "
                        "(minimize the half-FSR peaks)")
        if self._ctrl_err is not None:
            msg, when = self._ctrl_err
            if time.monotonic() - when < 8.0:
                warn.append(f"! {msg}")
            else:
                self._ctrl_err = None
        if self.ctrl is None and not self.args.no_controller:
            warn.append("! SA201B USB disconnected — retrying...")
        if self.acq.status not in ("ok", "starting"):
            warn.append(self.acq.status)
        self.txt_status.set_text("\n".join(warn))

        # ---- render: blit the dynamic artists; full draw only when the
        # axis furniture (limits/ticks) actually changed
        if changed or self._bg is None:
            self.fig.canvas.draw()       # _on_draw refreshes the background
        else:
            self._blit()

    # ------------------------------------------------------------------ run
    def run(self):
        self.acq.start()
        # A plain canvas timer instead of FuncAnimation: FuncAnimation forces
        # a full ~100 ms canvas redraw on every tick even when nothing
        # changed, which saturates the Tk event loop and lags the mouse.
        # Blit frames cost ~15-25 ms, so ~7 Hz still leaves the event loop
        # mostly idle for mouse and widget traffic.
        self._timer = self.fig.canvas.new_timer(interval=150)
        self._timer.add_callback(self._update, 0)
        self._timer.start()
        try:
            self.plt.show()
        finally:
            try:
                self._timer.stop()
            except Exception:
                pass
            self.save_settings()
            self.acq.shutdown()          # aborts in-flight captures, closes scope
            if self.ctrl is not None:
                self.ctrl.close()
            if self._log_file is not None:
                self._log_file.close()
                print(f"[log] {self.log_path}")
            print("[exit] scope released")


def main():
    args = parse_args()
    resolve_args(args)
    print("=" * 64)
    print("SA210 / SA201B / PicoScope 5242D - live laser linewidth")
    print("=" * 64)
    app = None
    for attempt in range(1, 13):        # ~60 s: USB can be slow after boot
        try:
            app = LiveApp(args)
            break
        except Exception as exc:
            print(f"[startup] attempt {attempt}/12 failed: {exc}")
            if attempt == 12:
                print("\n[fatal] could not start.")
                print("  * Is the PicoScope connected via USB (and not open "
                      "in another program, e.g. PicoScope 7)?")
                print("  * Is the SA201B powered on with its USB connected?")
                return 2
            print("[startup] retrying in 5 s ...")
            time.sleep(5)
    print("[ui] close the window or press q to stop")
    app.run()


if __name__ == "__main__":
    sys.exit(main())
