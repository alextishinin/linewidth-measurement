"""Turn one Fabry-Perot sweep into a calibrated linewidth measurement.

Physics
-------
The SA201B sweeps the SA210 cavity length with a sawtooth, so laser lines
appear as transmission peaks along the time axis. The same laser line repeats
every free spectral range (FSR = 10 GHz for the SA210). The time between two
repeats of the *same* line therefore corresponds to exactly one FSR, which
calibrates the time axis into optical frequency with no knowledge of the
piezo, the ramp slope, or the wavelength.

The measured peak shape is the laser lineshape convolved with the instrument
function (near-Lorentzian Airy peak, FWHM = FSR/finesse = 67 MHz spec). For
Lorentzian shapes the widths add, so a crude deconvolved laser width is
(measured - instrument); for a laser far narrower than 67 MHz the measured
width simply *is* the instrument function.

Pipeline: baseline -> peak detection -> FSR period (autocorrelation candidate,
confirmed by a same-height partner peak) -> Lorentzian fit of the tallest
peak -> widths in Hz.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import config


@dataclass
class SweepAnalysis:
    ok: bool = False
    message: str = ""
    # raw trace (possibly trimmed to the rising ramp)
    t: np.ndarray | None = None
    v: np.ndarray | None = None
    baseline: float = 0.0
    # detected peaks
    peak_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    peak_heights: np.ndarray = field(default_factory=lambda: np.empty(0))
    # calibration
    fsr_period_s: float | None = None      # sweep time per 10 GHz
    hz_per_s: float | None = None
    # main-peak fit
    fit_t: np.ndarray | None = None        # Lorentzian model curve (t, v)
    fit_v: np.ndarray | None = None
    fit_center_s: float | None = None
    fit_fwhm_s: float | None = None
    fit_r2: float | None = None
    fwhm_direct_s: float | None = None     # half-max crossings, fit-free
    # results in frequency units
    linewidth_hz: float | None = None
    linewidth_direct_hz: float | None = None
    deconvolved_hz: float | None = None
    finesse: float | None = None
    mode_offsets_hz: list[float] = field(default_factory=list)
    # health flags
    saturating: bool = False
    weak: bool = False


_C_M_PER_S = 299_792_458.0


def delta_lambda_m(delta_nu_hz: float, lambda_nm: float) -> float:
    """Frequency width -> wavelength width (meters): dl = lambda^2 * dnu / c."""
    return (lambda_nm * 1e-9) ** 2 * delta_nu_hz / _C_M_PER_S


def wavelength_width(delta_nu_hz: float, lambda_nm: float) -> tuple[float, str]:
    """Convert a frequency width to wavelength units with a tidy auto-unit.

    Returns (value, unit) with unit nm/pm/fm chosen so value >= 1 where
    possible (e.g. 3 GHz @ 1064 nm -> (11.3, 'pm'); 67 MHz -> (253, 'fm')).
    """
    d = delta_lambda_m(delta_nu_hz, lambda_nm)
    v = d * 1e9
    if v >= 1.0:
        return v, "nm"
    v = d * 1e12
    if v >= 1.0:
        return v, "pm"
    return d * 1e15, "fm"


def _lorentzian(t, amp, t0, gamma, offset):
    return offset + amp * gamma**2 / ((t - t0) ** 2 + gamma**2)


def _parabolic_apex(t, v, i):
    """Refine a peak position with a 3-point parabola around sample i."""
    if i <= 0 or i >= len(v) - 1:
        return t[i]
    denom = v[i - 1] - 2.0 * v[i] + v[i + 1]
    if denom == 0:
        return t[i]
    shift = 0.5 * (v[i - 1] - v[i + 1]) / denom
    return t[i] + np.clip(shift, -1.0, 1.0) * (t[1] - t[0])


def _direct_fwhm(t, v, i_peak, baseline):
    """FWHM from interpolated half-maximum crossings around sample i_peak."""
    half = baseline + 0.5 * (v[i_peak] - baseline)
    i = i_peak
    while i > 0 and v[i] > half:
        i -= 1
    if v[i] > half:
        return None
    left = np.interp(half, [v[i], v[i + 1]], [t[i], t[i + 1]])
    j = i_peak
    while j < len(v) - 1 and v[j] > half:
        j += 1
    if v[j] > half:
        return None
    right = np.interp(half, [v[j], v[j - 1]], [t[j], t[j - 1]])
    return right - left


def trim_to_rising_ramp(cap, rise_time_estimate_s: float):
    """Restrict a Capture to the rising-ramp portion of the sweep.

    Uses the MONITOR OUT ramp on channel B when it is wired (>0.5 V swing);
    otherwise trusts the trigger plus the SA201B rise-time estimate.
    """
    t, v = cap.t, cap.pd
    ramp_end = None
    if cap.monitor is not None:
        mon = cap.monitor
        swing = float(np.percentile(mon, 99.5) - np.percentile(mon, 0.5))
        if swing > 0.5:
            i_top = int(np.argmax(mon))
            if t[i_top] > 0.2 * rise_time_estimate_s:
                ramp_end = t[i_top]
    if ramp_end is None:
        ramp_end = 0.95 * rise_time_estimate_s
    guard = 0.01 * ramp_end
    sel = (t >= guard) & (t <= ramp_end - guard)
    if sel.sum() < 100:
        sel = t >= 0
    return t[sel], v[sel], ramp_end


def analyze_sweep(t: np.ndarray, v: np.ndarray,
                  fsr_hz: float = config.FSR_HZ,
                  instrument_hz: float = config.INSTRUMENT_RES_HZ,
                  min_signal_v: float = 0.05) -> SweepAnalysis:
    out = SweepAnalysis(t=t, v=v)
    n = len(v)
    if n < 500:
        out.message = "trace too short"
        return out
    dt = t[1] - t[0]

    # ---------------------------------------------------------------- signal
    baseline = float(np.percentile(v, 10))
    out.baseline = baseline
    vv = v - baseline
    lower = vv[vv <= np.percentile(vv, 50)]
    noise = float(1.4826 * np.median(np.abs(lower))) or 1e-4
    vmax = float(vv.max())
    out.saturating = bool(v.max() > 4.85)
    out.weak = bool(vmax < 0.35)
    if vmax < max(8 * noise, min_signal_v):
        out.message = "no signal — check laser alignment into the SA210"
        return out

    # ----------------------------------------------------------- find peaks
    thr = max(6 * noise, 0.05 * vmax)
    distance = max(int(3e-6 / dt), 5)
    peaks, props = find_peaks(vv, height=thr, prominence=thr,
                              distance=distance, width=2)
    if len(peaks) == 0:
        out.message = "no peaks found"
        return out
    heights = props["peak_heights"]
    order = np.argsort(heights)[::-1]
    peaks, heights = peaks[order], heights[order]
    widths_s = props["widths"][order] * dt
    apex_t = np.array([_parabolic_apex(t, vv, i) for i in peaks])
    out.peak_times = apex_t
    out.peak_heights = heights + baseline
    i_main = 0
    t_main = apex_t[i_main]

    # ------------------------------------------------- FSR period candidates
    # np.correlate is O(n^2): keep the decimated trace small. Lag precision
    # is refined by the same-mode partner peak match afterwards anyway.
    dec = max(1, n // 2048)
    x = vv[::dec] - vv[::dec].mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] if ac[0] > 0 else 1.0
    lo = max(int(0.05 * len(x)), 4)
    hi = int(0.95 * len(x))
    lag_peaks, lag_props = find_peaks(ac[lo:hi], height=0.10, prominence=0.05)
    candidates = [((lp + lo) * dec * dt, h)
                  for lp, h in zip(lag_peaks, lag_props["peak_heights"])]
    candidates.sort(key=lambda c: -c[1])

    def partner_near(t_target, min_height_frac=0.65):
        tol = max(0.07 * abs(t_target - t_main), 20 * dt)
        best = None
        for tt, hh in zip(apex_t, heights):
            if abs(tt - t_target) < tol and hh >= min_height_frac * heights[i_main]:
                if best is None or abs(tt - t_target) < abs(best - t_target):
                    best = tt
        return best

    fsr_period = None
    for lag_s, _score in candidates[:6]:
        for sign in (+1, -1):
            partner = partner_near(t_main + sign * lag_s)
            if partner is not None:
                fsr_period = abs(partner - t_main)
                break
        if fsr_period:
            break
    if fsr_period is None and candidates:
        fsr_period = candidates[0][0]   # autocorrelation only, unconfirmed

    # ------------------------------------------------------ main-peak width
    gap = np.inf
    for tt in apex_t[1:]:
        gap = min(gap, abs(tt - t_main))
    w0 = max(widths_s[i_main], 4 * dt)
    half_win = min(6 * w0, 0.45 * gap if np.isfinite(gap) else 6 * w0)
    half_win = max(half_win, 8 * dt)
    sel = np.abs(t - t_main) <= half_win
    ts, vs = t[sel], v[sel]

    i_peak_local = int(np.argmin(np.abs(ts - t_main)))
    out.fwhm_direct_s = _direct_fwhm(ts, vs, int(np.argmax(vs)), baseline)

    try:
        p0 = [heights[i_main], t_main, w0 / 2, baseline]
        lb = [0.2 * heights[i_main], ts[0], dt / 2, -1.0]
        ub = [3.0 * heights[i_main], ts[-1], (ts[-1] - ts[0]), 2.0]
        popt, _ = curve_fit(_lorentzian, ts, vs, p0=p0, bounds=(lb, ub),
                            maxfev=4000)
        model = _lorentzian(ts, *popt)
        ss_res = float(np.sum((vs - model) ** 2))
        ss_tot = float(np.sum((vs - vs.mean()) ** 2)) or 1.0
        out.fit_r2 = 1.0 - ss_res / ss_tot
        out.fit_t, out.fit_v = ts, model
        out.fit_center_s = float(popt[1])
        out.fit_fwhm_s = float(2.0 * popt[2])
    except Exception:
        out.fit_fwhm_s = None

    fwhm_s = out.fit_fwhm_s if (out.fit_fwhm_s and (out.fit_r2 or 0) > 0.7) \
        else out.fwhm_direct_s
    if fwhm_s is None:
        out.message = "could not measure the peak width"
        return out

    # ---------------------------------------------------------- calibration
    if fsr_period is None:
        out.message = ("only one FSR visible — raise the SA201B amplitude "
                       "(30 V shows ~3 FSR) so the peak spacing can "
                       "calibrate the axis")
        return out
    out.fsr_period_s = float(fsr_period)
    out.hz_per_s = fsr_hz / fsr_period
    out.linewidth_hz = fwhm_s * out.hz_per_s
    if out.fwhm_direct_s:
        out.linewidth_direct_hz = out.fwhm_direct_s * out.hz_per_s
    out.deconvolved_hz = max(out.linewidth_hz - instrument_hz, 0.0)
    out.finesse = fsr_hz / out.linewidth_hz

    # ------------------------------------------------------- mode structure
    win_lo, win_hi = t_main - 0.02 * fsr_period, t_main + 0.90 * fsr_period
    offsets = [(tt - t_main) * out.hz_per_s
               for tt in apex_t if win_lo <= tt <= win_hi]
    out.mode_offsets_hz = sorted(offsets)

    out.ok = True
    out.message = "ok"
    return out
