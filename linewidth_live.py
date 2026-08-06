"""Live laser linewidth measurement.

Chain: laser -> SA210 Fabry-Perot -> SA201B controller -> PicoScope 5242D -> here.

Run:  python linewidth_live.py   (or run.bat)

UI: PySide6 + pyqtgraph. Drag with the left mouse button on any graph to
zoom into that box (auto-ranging pauses for that graph until its reset
button, or the v key, restores it). Scroll wheel zooms; dragging an axis
zooms that axis only.

Keys (ignored while typing in an input box):
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
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import analysis as ana
import config

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
EXPORT_DIR = os.path.join(HERE, "exports")
SETTINGS_PATH = os.path.join(HERE, "settings.json")
ICON_PATH = os.path.join(HERE, "icon.ico")


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
    p.add_argument("--median", type=int, default=11,
                   help="report the rolling median of N sweeps in live mode "
                        "(default 11 ~ 1.6 s; 1 = raw per-sweep values)")
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


# -------------------------------------------------------------------- window
class MainWindow(QtWidgets.QWidget):
    """Top-level window; delegates keys and the close event to the app."""

    def __init__(self, app_logic):
        super().__init__()
        self._app = app_logic

    def keyPressEvent(self, ev):
        if not self._app._on_key(ev.key()):
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._app._cleanup()
        ev.accept()


# ---------------------------------------------------------------------- app
class LiveApp:
    GAIN_NAMES = {0: "10k V/A", 1: "100k V/A", 2: "1M V/A"}
    GAIN_LABELS = ["Auto", "10k", "100k", "1M"]
    PLOTS = ("sweep", "peak", "trend")

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
        self._manual_gain_idx = 0 if args.pdgain == "auto" else int(args.pdgain)
        self.align_mode = False
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
        self._ctrl_q = queue.Queue()
        self._ctrl_err = None            # (message, monotonic time)
        threading.Thread(target=self._ctrl_worker, daemon=True).start()
        self._fsr_hist = collections.deque(maxlen=20)
        self._last_err_hz = None
        self._prev_fit_center = None     # sticky peak tracking across sweeps
        self._lw_recent = collections.deque(maxlen=max(1, args.median))
        self._lw_scatter_hz = None
        self._disp_lw_hz = None
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
        self._cleaned_up = False

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
                ["unix_time", "iso_time", "linewidth_hz",
                 "linewidth_median_hz", "linewidth_direct_hz",
                 "deconvolved_hz", "finesse", "fsr_period_s", "hz_per_s",
                 "peak_v", "n_modes", "pd_gain_index", "wavelength_nm",
                 "linewidth_pm", "linewidth_err_hz", "transverse_frac",
                 "flags"])
        self._build_ui()

    def _new_acquirer(self) -> Acquirer:
        a = self.args
        return Acquirer(self.scope, self.rise_s, a.dt_us * 1e-6,
                        None if a.window_ms is None else a.window_ms / 1e3,
                        a.avg, continuous=(self.mode == "live"))

    # -------------------------------------------------------------------- ui
    def _build_ui(self):
        self._qapp = QtWidgets.QApplication.instance() \
            or QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True)

        self.win = MainWindow(self)
        self.win.setWindowTitle("SA210 laser linewidth")
        if os.path.exists(ICON_PATH):
            self.win.setWindowIcon(QtGui.QIcon(ICON_PATH))

        root = QtWidgets.QHBoxLayout(self.win)
        root.setContentsMargins(8, 8, 8, 8)

        def make_edit(initial, handler, width=70):
            ed = QtWidgets.QLineEdit(initial)
            ed.setFixedWidth(width)
            ed.editingFinished.connect(lambda: handler(ed.text()))
            return ed

        # ---- x-range controls (live beside graph 2) ---------------------
        half = self.fsr_hz / 2e6
        self.SPAN_LABELS = ["Auto", f"Full {self.args.fsr_ghz:g} GHz",
                            "Manual"]
        self.combo_span = QtWidgets.QComboBox()
        self.combo_span.addItems(self.SPAN_LABELS)
        self.combo_span.activated.connect(
            lambda i: self._on_span_combo(self.SPAN_LABELS[i]))
        self.ed_span_min = make_edit(f"{self.zoom_span_manual[0]:g}",
                                     self._on_span_min, 56)
        self.ed_span_max = make_edit(f"{self.zoom_span_manual[1]:g}",
                                     self._on_span_max, 56)
        self.lbl_span = QtWidgets.QLabel(f"x-range (±{half:g} MHz)")

        # ---- plots, each with a header bar holding its own controls -----
        self.lbl_titles = {}
        self.btns_reset = {}
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(4)

        def plot_block(name, title, extra=()):
            header = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(title)
            self.lbl_titles[name] = lbl
            header.addWidget(lbl)
            header.addStretch(1)
            for w in extra:
                header.addWidget(w)
            btn = QtWidgets.QPushButton("reset view")
            btn.setFixedWidth(88)
            btn.clicked.connect(lambda _c=False, n=name: self._reset_view(n))
            self.btns_reset[name] = btn
            header.addWidget(btn)
            pw = pg.PlotWidget()
            block = QtWidgets.QVBoxLayout()
            block.setSpacing(2)
            block.addLayout(header)
            block.addWidget(pw)
            return block, pw

        blk, self.p_sweep = plot_block(
            "sweep", "Full sweep (photodiode amplifier out)")
        left.addLayout(blk, stretch=11)
        blk, self.p_peak = plot_block(
            "peak", "Main peak, frequency calibrated",
            (self.lbl_span, self.combo_span,
             self.ed_span_min, self.ed_span_max))
        left.addLayout(blk, stretch=17)
        blk, self.p_trend = plot_block(
            "trend", "Linewidth (FWHM) history")
        left.addLayout(blk, stretch=10)
        root.addLayout(left, stretch=1)
        self._plots = {"sweep": self.p_sweep, "peak": self.p_peak,
                       "trend": self.p_trend}

        self.p_sweep.setLabel("bottom", "time along ramp (ms)")
        self.p_sweep.setLabel("left", "signal (V)")
        self.p_peak.setLabel("bottom", "optical frequency offset (MHz)")
        self.p_peak.setLabel("left", "signal (V)")
        self.p_trend.setLabel("bottom", "elapsed time (s)")
        self.p_trend.setLabel("left", "FWHM (MHz)")
        for p in (self.p_sweep, self.p_peak, self.p_trend):
            for axname in ("left", "bottom"):
                # units are spelled out in the labels; no "(x0.001)" scaling
                p.getAxis(axname).enableAutoSIPrefix(False)

        self._legend = self.p_peak.addLegend(offset=(-10, 8))
        self.ln_sweep = self.p_sweep.plot()
        self.mk_peaks = pg.ScatterPlotItem(symbol="t", size=9,
                                           brush=None)
        self.p_sweep.addItem(self.mk_peaks)
        self.ln_zoom = self.p_peak.plot(name="measured")
        self.ln_fit = self.p_peak.plot(name="Lorentzian fit")
        self.ln_trend = self.p_trend.plot(symbol="o", symbolSize=4,
                                          symbolPen=None)
        for curve in (self.ln_sweep, self.ln_zoom):
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)

        self._user_zoom = {n: False for n in self.PLOTS}
        self._auto_range = {}
        for name, p in self._plots.items():
            vb = p.getViewBox()
            vb.setMouseMode(pg.ViewBox.RectMode)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.25)
            vb.sigRangeChangedManually.connect(
                lambda _mask, n=name: self._on_manual_zoom(n))

        # ---- side panel -------------------------------------------------
        side = QtWidgets.QVBoxLayout()
        side.setSpacing(6)
        panel = QtWidgets.QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(360)
        root.addWidget(panel)

        top_row = QtWidgets.QHBoxLayout()
        self.lbl_head = QtWidgets.QLabel("—")
        f = self.lbl_head.font()
        f.setPointSize(22)
        f.setBold(True)
        self.lbl_head.setFont(f)
        top_row.addWidget(self.lbl_head, stretch=1)
        self.btn_theme = QtWidgets.QPushButton("")
        self.btn_theme.setFixedWidth(64)
        self.btn_theme.clicked.connect(self._on_toggle_theme)
        top_row.addWidget(self.btn_theme,
                          alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        side.addLayout(top_row)

        self.lbl_sub = QtWidgets.QLabel("")
        self.lbl_sub.setWordWrap(True)
        side.addWidget(self.lbl_sub)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        side.addWidget(self.lbl_status)
        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        side.addWidget(self.lbl_stats)
        side.addStretch(1)

        def add_row(*widgets):
            row = QtWidgets.QHBoxLayout()
            for w in widgets:
                row.addWidget(w)
            side.addLayout(row)
            return row

        lam_lbl = QtWidgets.QLabel("laser λ nm (100–5000)")
        self.ed_wavelength = make_edit(f"{self.wavelength_nm:g}",
                                       self._on_wavelength)
        add_row(lam_lbl, self.ed_wavelength)

        self.ed_amplitude = make_edit(f"{self.scan_amplitude:g}",
                                      self._on_amplitude_box, 56)
        self.ed_offset = make_edit(f"{self.scan_offset:g}",
                                   self._on_offset_box, 56)
        add_row(QtWidgets.QLabel("ampl V (1–30)"), self.ed_amplitude,
                QtWidgets.QLabel("offs V (0–15)"), self.ed_offset)

        self.ed_sweep = make_edit(f"{self.scan_sweep_ms:g}",
                                  self._on_sweep_box, 56)
        self.EXPAND_LABELS = [f"{f}×" for f in config.SWEEP_EXPANSION_FACTORS]
        self.combo_expand = QtWidgets.QComboBox()
        self.combo_expand.addItems(self.EXPAND_LABELS)
        self.combo_expand.activated.connect(
            lambda i: self._on_expand_combo(self.EXPAND_LABELS[i]))
        add_row(QtWidgets.QLabel("sweep ms (10–100)"), self.ed_sweep,
                QtWidgets.QLabel("expand"), self.combo_expand)

        self.combo_gain = QtWidgets.QComboBox()
        self.combo_gain.addItems(self.GAIN_LABELS)
        self.combo_gain.activated.connect(
            lambda i: self._on_gain_combo(self.GAIN_LABELS[i]))
        add_row(QtWidgets.QLabel("PD gain (V/A)"), self.combo_gain)

        self.btn_run = QtWidgets.QPushButton("Run once")
        self.btn_run.clicked.connect(self._on_run_once)
        self.btn_mode = QtWidgets.QPushButton("")
        self.btn_mode.clicked.connect(self._on_toggle_mode)
        add_row(self.btn_run, self.btn_mode)
        self.btn_export = QtWidgets.QPushButton("Export data")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_align = QtWidgets.QPushButton("Align: off")
        self.btn_align.clicked.connect(self._on_toggle_align)
        add_row(self.btn_export, self.btn_align)

        self.lbl_keys = QtWidgets.QLabel(
            "r run · m mode · t align · g gain · a auto\n"
            "e export · s snap · d theme · v reset views\n"
            "p pause · q quit · drag on a graph to zoom")
        side.addWidget(self.lbl_keys)

        self._sync_mode_button()
        self._sync_gain_combo()
        self._sync_expand_combo()
        self._sync_span_widgets()
        if self.mode == "single":
            self.lbl_stats.setText('single mode — press "Run once" (or r) '
                                   "to capture a sweep")
        self._apply_theme(self.theme_name)
        self._sync_theme_button()

    # ----------------------------------------------------------- keys/zoom
    def _on_key(self, key) -> bool:
        K = QtCore.Qt.Key
        if key == K.Key_Q:
            self.win.close()
        elif key == K.Key_P:
            self.paused = not self.paused
        elif key == K.Key_R:
            self._on_run_once()
        elif key == K.Key_M:
            self._on_toggle_mode()
        elif key == K.Key_T:
            self._on_toggle_align()
        elif key == K.Key_D:
            self._on_toggle_theme()
        elif key == K.Key_S:
            self._snapshot()
        elif key == K.Key_E:
            self._on_export()
        elif key == K.Key_V:
            self._reset_all_views()
        elif key == K.Key_A:
            if self.ctrl is not None:
                self.auto_gain = not self.auto_gain
                if not self.auto_gain:
                    self._manual_gain_idx = self._gain_cache
                print(f"[gain] auto-gain {'ON' if self.auto_gain else 'OFF'}")
                self._sync_gain_combo()
        elif key == K.Key_G:
            if self.ctrl is not None:
                self._apply_gain_choice(((self._gain_cache + 1) % 3) + 1)
        elif key in (K.Key_Left, K.Key_Right):
            if self.ctrl is not None:
                delta = 0.25 if key == K.Key_Right else -0.25
                new = float(np.clip(self.scan_offset + delta, 0.0, 15.0))
                self.scan_offset = new
                self.ed_offset.setText(f"{new:g}")
                self._ctrl_do(f"DC offset -> {new:.2f} V",
                              lambda c: setattr(c, "dc_offset_v", new))
        else:
            return False
        return True

    def _on_manual_zoom(self, name):
        if not self._user_zoom[name]:
            print(f"[zoom] {name}: zoomed — auto-ranging paused until reset")
        self._user_zoom[name] = True
        self._sync_reset_buttons()

    def _reset_view(self, name):
        self._user_zoom[name] = False
        rng = self._auto_range.get(name)
        if rng is not None:
            vb = self._plots[name].getViewBox()
            vb.setXRange(*rng[0], padding=0)
            vb.setYRange(*rng[1], padding=0)
        self._sync_reset_buttons()

    def _reset_all_views(self):
        for name in self.PLOTS:
            self._reset_view(name)

    def _sync_reset_buttons(self):
        for name, b in self.btns_reset.items():
            f = b.font()
            f.setBold(self._user_zoom[name])
            b.setFont(f)

    # ---------------------------------------------------------------- theme
    def _on_toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self._apply_theme(self.theme_name)
        self._sync_theme_button()

    def _sync_theme_button(self):
        # the button names the theme you would switch TO
        self.btn_theme.setText(
            "Light" if self.theme_name == "dark" else "Dark")

    def _apply_theme(self, name):
        T = config.THEMES[name]
        self.win.setStyleSheet(f"""
            QWidget {{ background-color: {T['PAGE']}; color: {T['INK']}; }}
            QLineEdit, QComboBox {{
                background-color: {T['SURFACE']}; color: {T['INK']};
                border: 1px solid {T['AXIS']}; padding: 2px 4px; }}
            QComboBox QAbstractItemView {{
                background-color: {T['SURFACE']}; color: {T['INK']};
                selection-background-color: {T['SERIES1']};
                selection-color: #ffffff; }}
            QPushButton {{
                background-color: {T['SURFACE']}; color: {T['INK']};
                border: 1px solid {T['AXIS']}; padding: 5px 10px; }}
            QPushButton:hover {{ background-color: {T['HOVER']}; }}
        """)
        self.lbl_sub.setStyleSheet(f"color: {T['INK2']};")
        self.lbl_status.setStyleSheet(f"color: {T['CRITICAL']};")
        self.lbl_stats.setStyleSheet(f"color: {T['INK2']};")
        self.lbl_keys.setStyleSheet(f"color: {T['MUTED']};")

        for p in self._plots.values():
            p.setBackground(T["PAGE"])
            p.getViewBox().setBackgroundColor(T["SURFACE"])
            for axname in ("left", "bottom"):
                ax = p.getAxis(axname)
                ax.setPen(pg.mkPen(T["AXIS"]))
                ax.setTextPen(pg.mkPen(T["INK2"]))
        for lbl in self.lbl_titles.values():
            lbl.setStyleSheet(f"color: {T['INK']}; font-weight: 600;")
        self.ln_sweep.setPen(pg.mkPen(T["SERIES1"], width=1))
        self.ln_zoom.setPen(pg.mkPen(T["SERIES1"], width=2))
        self.ln_fit.setPen(pg.mkPen(T["SERIES2"], width=2,
                                    style=QtCore.Qt.PenStyle.DashLine))
        self.ln_trend.setPen(pg.mkPen(T["SERIES1"], width=1.5))
        self.ln_trend.setSymbolBrush(pg.mkBrush(T["SERIES1"]))
        self.mk_peaks.setPen(pg.mkPen(T["MUTED"]))
        if self._legend is not None:
            self._legend.setLabelTextColor(T["INK2"])

    # ---------------------------------------------------------------- modes
    def _sync_mode_button(self):
        self.btn_mode.setText(
            "Mode: Live" if self.mode == "live" else "Mode: Single")

    def _on_run_once(self):
        if self.mode != "single":
            self.mode = "single"
            self.acq.continuous.clear()
            self._sync_mode_button()
        self._armed = True
        self._arm_time = time.time()
        self.acq.request_oneshot(self.args.avg)   # scope captures, then idles

    def _on_toggle_mode(self):
        self.mode = "live" if self.mode == "single" else "single"
        if self.mode == "single":
            self._armed = False              # scope idle until Run once
            self.acq.continuous.clear()
        else:
            self.acq.continuous.set()
        self._sync_mode_button()

    # ----------------------------------------------------------- wavelength
    def _on_wavelength(self, text):
        try:
            v = float(str(text).strip())
            if not (100.0 <= v <= 5000.0):
                raise ValueError
            if v != self.wavelength_nm:
                self.wavelength_nm = v
                print(f"[λ] wavelength set to {v:g} nm")
        except ValueError:
            self.lbl_status.setText(
                f"bad wavelength {text!r} — keeping {self.wavelength_nm:g} nm")
        self.ed_wavelength.setText(f"{self.wavelength_nm:g}")

    # ------------------------------------------------- graph 2 x-range
    def _sync_span_widgets(self):
        idx = {"auto": 0, "full": 1, "manual": 2}[self.zoom_span_mode]
        self.combo_span.setCurrentIndex(idx)
        self.ed_span_min.setText(f"{self.zoom_span_manual[0]:g}")
        self.ed_span_max.setText(f"{self.zoom_span_manual[1]:g}")

    def _apply_span_mode(self, mode):
        self.zoom_span_mode = mode
        # an explicit x-range choice supersedes any drag-zoom on that graph
        self._user_zoom["peak"] = False
        self._sync_span_widgets()
        self._sync_reset_buttons()

    def _on_span_combo(self, label):
        mode = ("auto", "full", "manual")[self.SPAN_LABELS.index(label)]
        self._apply_span_mode(mode)
        print(f"[graph2] x-range mode: {mode}")

    def _on_span_min(self, text):
        self._set_span_edge(text, "min")

    def _on_span_max(self, text):
        self._set_span_edge(text, "max")

    def _set_span_edge(self, text, which):
        half = self.fsr_hz / 2e6
        lo, hi = self.zoom_span_manual
        try:
            v = float(str(text).strip())
            if not (-half <= v <= half):
                raise ValueError(f"valid range -{half:g} to {half:g} MHz")
            new_lo, new_hi = (v, hi) if which == "min" else (lo, v)
            if new_hi - new_lo < 1.0:
                raise ValueError("max must exceed min by at least 1 MHz")
            if (new_lo, new_hi) != self.zoom_span_manual:
                self.zoom_span_manual = (new_lo, new_hi)
                print(f"[graph2] x-range {new_lo:g} .. {new_hi:g} MHz (manual)")
                self._apply_span_mode("manual")     # typing implies manual
                return
        except Exception as exc:
            self.lbl_status.setText(f"x-range not set: {exc}")
        self._sync_span_widgets()

    # --------------------------------------------------------- scan controls
    def _push_window(self):
        rise = (self.scan_sweep_ms / 1e3 *
                config.SWEEP_EXPANSION_FACTORS[self.scan_expand_idx])
        self.acq.request_window(rise * 1.25 + 0.002)

    def _no_ctrl_hint(self):
        self.lbl_status.setText(
            "no SA201B USB — set it on the touchscreen (display will follow)")

    def _on_amplitude_box(self, text):
        try:
            v = float(str(text).strip())
            if not (1.0 <= v <= 30.0):
                raise ValueError("valid range 1-30 V")
            if v != self.scan_amplitude:
                self.scan_amplitude = v
                self._ctrl_do(f"amplitude -> {v:g} V",
                              lambda c: setattr(c, "amplitude_v", v))
        except ValueError as exc:
            self.lbl_status.setText(f"amplitude not set: {exc}")
        self.ed_amplitude.setText(f"{self.scan_amplitude:g}")

    def _on_offset_box(self, text):
        try:
            v = float(str(text).strip())
            if not (0.0 <= v <= 15.0):
                raise ValueError("valid range 0-15 V")
            if v != self.scan_offset:
                self.scan_offset = v
                self._ctrl_do(f"DC offset -> {v:g} V",
                              lambda c: setattr(c, "dc_offset_v", v))
        except ValueError as exc:
            self.lbl_status.setText(f"offset not set: {exc}")
        self.ed_offset.setText(f"{self.scan_offset:g}")

    def _on_sweep_box(self, text):
        lo = config.RISETIME_MIN_S * 1e3
        hi = config.RISETIME_MAX_S * 1e3
        try:
            ms = float(str(text).strip())
            if not (lo <= ms <= hi):
                raise ValueError(f"valid range {lo:g}-{hi:g} ms (at 1x)")
            step = _ms_to_step(ms)
            achieved = lo + step / config.RISETIME_STEPS * (hi - lo)
            if achieved != self.scan_sweep_ms:
                self.scan_sweep_ms = achieved
                self._push_window()
                self._ctrl_do(f"sweep -> {self.scan_sweep_ms:g} ms at 1x "
                              f"(step {step})",
                              lambda c: setattr(c, "risetime_step", step))
        except ValueError as exc:
            self.lbl_status.setText(f"sweep time not set: {exc}")
        self.ed_sweep.setText(f"{self.scan_sweep_ms:g}")

    def _sync_expand_combo(self):
        self.combo_expand.setCurrentIndex(self.scan_expand_idx)

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
    def _on_toggle_align(self):
        self.align_mode = not self.align_mode
        want_saw = not self.align_mode
        self._ctrl_do("triangle scan (alignment)" if self.align_mode
                      else "sawtooth scan (measurement)",
                      lambda c: setattr(c, "sawtooth", want_saw))
        self.btn_align.setText(
            "Align: TRI" if self.align_mode else "Align: off")

    # ----------------------------------------------------------------- gain
    def _apply_gain_choice(self, idx):
        """idx 0 = Auto; 1..3 = manual gain index (idx - 1)."""
        if self.ctrl is None:
            self.lbl_status.setText(
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
        val = ("Auto" if (self.auto_gain or self.ctrl is None)
               else self.GAIN_LABELS[self._manual_gain_idx + 1])
        self.combo_gain.setCurrentIndex(self.GAIN_LABELS.index(val))

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

    # ------------------------------------------------------------------ log
    def _log(self, res):
        if self._log_writer is None or res is None:
            return
        flags = ";".join(f for f, on in
                         [("saturating", res.saturating), ("weak", res.weak),
                          ("cal-fallback", res.calibration_fallback)]
                         if on) or "ok" if res.ok else res.message
        now = time.time()
        self._log_writer.writerow([
            f"{now:.3f}", _dt.datetime.now().isoformat(timespec="seconds"),
            f"{res.linewidth_hz:.6g}" if res.linewidth_hz else "",
            (f"{self._disp_lw_hz:.6g}"
             if getattr(self, "_disp_lw_hz", None) else ""),
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

    # --------------------------------------------------------------- export
    def _on_export(self):
        res = self.last_result
        if res is None or res.t is None:
            self.lbl_status.setText("nothing to export yet")
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
        self.lbl_status.setText(f"exported: {names}")
        print("[export] " + " | ".join(wrote))

    def _snapshot(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        png = os.path.join(LOG_DIR, f"snapshot_{stamp}.png")
        self.win.grab().save(png)
        res = self.last_result
        if res is not None and res.t is not None:
            csv_path = os.path.join(LOG_DIR, f"snapshot_{stamp}.csv")
            with open(csv_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time_s", "signal_v"])
                w.writerows(zip(res.t, res.v))
        print(f"[snapshot] saved {png}")

    # ---------------------------------------------------------------- frame
    def _update(self, _frame=0):
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
        # nominal piezo calibration as a soft prior so the FSR search can
        # tell a half-FSR transverse comb from the real spacing. The piezo
        # moves lambda/4 per FSR, so the volts-per-FSR figure scales with
        # the wavelength entered in the lambda box.
        expected_T = None
        if self.scan_amplitude > 0:
            v_per_fsr = (config.VOLTS_PER_FSR * self.wavelength_nm
                         / config.VOLTS_PER_FSR_REF_NM)
            expected_T = self.acq.rise_s * v_per_fsr / self.scan_amplitude
        robust_T = None
        if len(self._fsr_hist) >= 5:
            robust_T = float(np.median(self._fsr_hist))
        res = ana.analyze_sweep(t, v, fsr_hz=self.fsr_hz,
                                instrument_hz=self.instrument_hz,
                                expected_fsr_period_s=expected_T,
                                prefer_time_s=self._prev_fit_center,
                                robust_fsr_period_s=robust_T)
        if res.ok and res.fit_center_s is not None:
            self._prev_fit_center = res.fit_center_s
        self.last_result = res
        self._auto_gain_step(res)
        self._last_err_hz = self._uncertainty_hz(res)

        # rolling-median linewidth: single sweeps sample jitter; the median
        # over ~1.6 s is the reported value in live mode (raw in single mode)
        self._lw_scatter_hz = None
        disp_lw = res.linewidth_hz
        if res.ok and res.linewidth_hz and not self.align_mode:
            self._lw_recent.append(res.linewidth_hz)
        if len(self._lw_recent) >= 5:
            arr = np.fromiter(self._lw_recent, dtype=float)
            med = float(np.median(arr))
            self._lw_scatter_hz = float(
                1.4826 * np.median(np.abs(arr - med)))
            if self.mode == "live" and res.ok and res.linewidth_hz:
                disp_lw = med
            if self._lw_scatter_hz and self._last_err_hz is not None:
                se_med = 1.253 * self._lw_scatter_hz / np.sqrt(len(arr))
                self._last_err_hz = float(np.hypot(self._last_err_hz, se_med))
        self._disp_lw_hz = disp_lw

        if not self.align_mode:      # alignment sweeps don't pollute the log
            self._log(res)

        # ---- full sweep panel
        t_ms = t * 1e3
        self.ln_sweep.setData(t_ms, v)
        top = max(0.5, float(np.max(v)) * 1.25)
        if len(res.peak_times):
            self.mk_peaks.setData(x=res.peak_times * 1e3,
                                  y=res.peak_heights
                                  + 0.06 * max(1e-3, float(np.max(v))))
        else:
            self.mk_peaks.setData(x=[], y=[])
        self._auto_range["sweep"] = ((float(t_ms[0]), float(t_ms[-1])),
                                     (-0.04 * top, top))
        if not self._user_zoom["sweep"]:
            vb = self.p_sweep.getViewBox()
            vb.setXRange(t_ms[0], t_ms[-1], padding=0)
            vb.setYRange(-0.04 * top, top, padding=0)

        # ---- calibrated peak panel
        if res.ok and res.hz_per_s and res.fit_center_s is not None:
            f_off = (t - res.fit_center_s) * res.hz_per_s / 1e6
            self.ln_zoom.setData(f_off, v)
            if res.fit_t is not None:
                self.ln_fit.setData(
                    (res.fit_t - res.fit_center_s) * res.hz_per_s / 1e6,
                    res.fit_v)
            else:
                self.ln_fit.setData([], [])
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
                xlo, xhi = -span, span
            sel = (f_off >= xlo) & (f_off <= xhi)
            ztop = max(0.2, float(np.max(v[sel])) * 1.2) if sel.any() else 1.0
            self._auto_range["peak"] = ((xlo, xhi), (-0.04 * ztop, ztop))
            if not self._user_zoom["peak"]:
                vb = self.p_peak.getViewBox()
                vb.setXRange(xlo, xhi, padding=0)
                vb.setYRange(-0.04 * ztop, ztop, padding=0)

        # ---- trend panel (live mode waits out the median warm-up so the
        # first seconds can't seed the history with raw jitter samples)
        median_ready = (self.mode != "live"
                        or self._lw_recent.maxlen <= 1
                        or len(self._lw_recent) >= 5)
        if res.ok and res.linewidth_hz and not self.align_mode \
                and median_ready:
            self.history.append((wall, self._disp_lw_hz))
        if self.mode == "live":     # in single mode old runs stay on screen
            cutoff = time.time() - self.args.history_s
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()
        if self.history:
            xs = np.array([w - self.t_start for w, _ in self.history])
            ys = np.array([lw / 1e6 for _, lw in self.history])
            self.ln_trend.setData(xs, ys)
            if self.mode == "live":
                tx = (max(0.0, xs[-1] - self.args.history_s),
                      max(10.0, xs[-1] * 1.02))
            else:
                tx = (max(0.0, xs[0] - 5.0), max(10.0, xs[-1] + 5.0))
            lo, hi = float(ys.min()), float(ys.max())
            pad = max(2.0, 0.15 * (hi - lo))
            ty = (max(0.0, lo - pad), hi + pad)
            self._auto_range["trend"] = (tx, ty)
            if not self._user_zoom["trend"]:
                vb = self.p_trend.getViewBox()
                vb.setXRange(*tx, padding=0)
                vb.setYRange(*ty, padding=0)

        # ---- readouts
        if self.align_mode:
            peak_v = float(np.max(v)) if len(v) else 0.0
            self.lbl_head.setText(f"{peak_v:.3f} V")
            sub = "alignment — maximize peak height (triangle scan)"
            if res.transverse_frac is not None:
                sub = (f"alignment — peak height up, transverse "
                       f"{res.transverse_frac * 100:.0f}% down")
            self.lbl_sub.setText(sub)
        elif res.ok and res.linewidth_hz:
            show = self._disp_lw_hz or res.linewidth_hz
            err = self._last_err_hz
            head = f"{show / 1e6:.1f}"
            if err:
                head += f" ± {err / 1e6:.1f}"
            self.lbl_head.setText(head + " MHz")
            wl_v, wl_u = ana.wavelength_width(show, self.wavelength_nm)
            deconv = max(show - self.instrument_hz, 0.0)
            note = ("instrument-limited"
                    if show < 1.35 * self.instrument_hz
                    else f"est. laser ≈ {deconv / 1e6:.0f} MHz "
                         f"(67 MHz removed)")
            self.lbl_sub.setText(
                f"= {wl_v:.3g} {wl_u} @ {self.wavelength_nm:g} nm — {note}")
        else:
            self.lbl_head.setText("—")
            self.lbl_sub.setText("")

        stats = []
        if self.mode == "single" and self._last_single_stamp:
            stats.append(f"single sweep captured {self._last_single_stamp}")
        if self._lw_scatter_hz:
            stats.append(f"sweep-to-sweep scatter: "
                         f"±{self._lw_scatter_hz / 1e6:.2f} MHz (jitter, "
                         f"median of {self._lw_recent.maxlen})")
        if res.linewidth_direct_hz:
            stats.append(f"half-max width: "
                         f"{res.linewidth_direct_hz / 1e6:.1f} MHz")
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
            stats.append(f"sweep rate: {1.0 / self.acq.sweep_period_s:.0f} Hz"
                         f"   window: {self.acq.window_s * 1e3:.0f} ms")
        if self.log_path:
            stats.append(f"log: {os.path.basename(self.log_path)}")
        self.lbl_stats.setText("\n".join(stats))

        warn = []
        if not res.ok:
            warn.append(res.message)
        if res.saturating:
            warn.append("! signal saturating — lower PD gain / input power")
        if cap.clipped:
            warn.append("! scope ADC clipped")
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
        self.lbl_status.setText("\n".join(warn))

    # ------------------------------------------------------------------ run
    def _cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        tmr = getattr(self, "_timer", None)
        if tmr is not None:
            tmr.stop()
        self.save_settings()
        self.acq.shutdown()          # aborts in-flight captures, closes scope
        if self.ctrl is not None:
            self.ctrl.close()
        if self._log_file is not None:
            self._log_file.close()
            print(f"[log] {self.log_path}")
        print("[exit] scope released")

    def run(self):
        self.acq.start()
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(120)
        self.win.showMaximized()
        print("[ui] close the window or press q to stop")
        try:
            self._qapp.exec()
        finally:
            self._cleanup()


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
    app.run()


if __name__ == "__main__":
    sys.exit(main())
