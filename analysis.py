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
    fit_fwhm_err_s: float | None = None    # 1-sigma from the fit covariance
    fit_r2: float | None = None
    fwhm_direct_s: float | None = None     # half-max crossings, fit-free
    # results in frequency units
    linewidth_hz: float | None = None
    linewidth_direct_hz: float | None = None
    deconvolved_hz: float | None = None
    finesse: float | None = None
    mode_offsets_hz: list[float] = field(default_factory=list)
    # transverse-mode contamination: height of the strongest peak found at
    # half-FSR positions relative to the main peak (0 = clean TEM00,
    # ~1 = as tall as the fundamental). None when it cannot be assessed.
    transverse_frac: float | None = None
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


def _nearest_peak(apex_t, heights, t_target, tol):
    """(time, height) of the detected peak nearest t_target within tol."""
    best = None
    best_d = tol
    for tt, hh in zip(apex_t, heights):
        d = abs(tt - t_target)
        if d <= best_d:
            best, best_d = (float(tt), float(hh)), d
    return best


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
                  min_signal_v: float = 0.05,
                  expected_fsr_period_s: float | None = None) -> SweepAnalysis:
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

    # ------------------------------------------- FSR period candidate set
    # Two complementary sources, because each fails somewhere the other
    # doesn't:
    #  * spacings between TALL peaks adjacent to the main one -- immune to
    #    piezo chirp and to weak satellites (longitudinal side modes,
    #    transverse contamination), but blind to pattern structure;
    #  * autocorrelation lags -- see the repeating pattern of multimode
    #    clusters, but chirp smears the long lags.
    # Every candidate is also entered doubled, so a half-FSR transverse comb
    # offers its true FSR as a hypothesis.
    w0_main = widths_s[i_main]
    main_h = heights[i_main]
    cand = set()

    tall_t = np.sort(apex_t[heights >= 0.7 * main_h])
    i_t = int(np.argmin(np.abs(tall_t - t_main)))
    if i_t > 0:
        cand.add(float(t_main - tall_t[i_t - 1]))
    if i_t < len(tall_t) - 1:
        cand.add(float(tall_t[i_t + 1] - t_main))
    if 0 < i_t < len(tall_t) - 1:
        cand.add(float((tall_t[i_t + 1] - tall_t[i_t - 1]) / 2))

    dec = max(1, n // 2048)
    x = vv[::dec] - vv[::dec].mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] if ac[0] > 0 else 1.0
    lag_step = dec * dt
    lo = max(4, min(int(3 * w0_main / lag_step), int(0.15 * len(x))))
    hi = int(0.95 * len(x))
    if hi > lo + 4:
        lag_peaks, lag_props = find_peaks(ac[lo:hi], height=0.10,
                                          prominence=0.05)
        by_score = sorted(zip(lag_props["peak_heights"], lag_peaks),
                          reverse=True)
        for _score, lp in by_score[:4]:
            cand.add(float((lp + lo) * lag_step))
    for T in list(cand):
        cand.add(2.0 * T)

    def _partner(T):
        tol = max(0.15 * T, 6 * w0_main)
        best = None
        for sign in (+1, -1):
            p = _nearest_peak(apex_t, heights, t_main + sign * T, tol)
            if p is not None and p[1] >= 0.3 * main_h and \
                    (best is None or p[1] > best[1]):
                best = p
        return best

    def _support(T):
        """Fraction of tall peaks near the main one explained by comb T."""
        tol = max(0.15 * T, 6 * w0_main)
        seen = ok = 0
        for tt in tall_t:
            d = abs(tt - t_main)
            if d < 0.5 * tol or d > 2.2 * T:
                continue
            seen += 1
            k = round(d / T)
            if k >= 1 and abs(d - k * T) <= tol * k:
                ok += 1
        return ok / seen if seen else 0.0

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
        popt, pcov = curve_fit(_lorentzian, ts, vs, p0=p0, bounds=(lb, ub),
                               maxfev=4000)
        model = _lorentzian(ts, *popt)
        ss_res = float(np.sum((vs - model) ** 2))
        ss_tot = float(np.sum((vs - vs.mean()) ** 2)) or 1.0
        out.fit_r2 = 1.0 - ss_res / ss_tot
        out.fit_t, out.fit_v = ts, model
        out.fit_center_s = float(popt[1])
        out.fit_fwhm_s = float(2.0 * popt[2])
        try:
            var = float(pcov[2, 2])
            if np.isfinite(var) and var >= 0:
                out.fit_fwhm_err_s = float(2.0 * np.sqrt(var))
        except Exception:
            pass
    except Exception:
        out.fit_fwhm_s = None

    fwhm_s = out.fit_fwhm_s if (out.fit_fwhm_s and (out.fit_r2 or 0) > 0.7) \
        else out.fwhm_direct_s
    if fwhm_s is None:
        out.message = "could not measure the peak width"
        return out

    # ---------------------------------------------------------- calibration
    # Filter the candidates by physics, then select:
    #   * guard: a period implying finesse far beyond the instrument spec is
    #     impossible (the ruler is wrong, not the laser), and the FSR must
    #     comfortably exceed the measured linewidth;
    #   * a same-comb partner peak must actually exist one period away;
    #   * the piezo prior (~10 V per FSR on the SA210) picks among the
    #     survivors -- it is the only thing that can tell a half-FSR comb of
    #     EQUAL-height transverse modes from the real spacing;
    #   * without a prior, the candidate explaining the most tall peaks wins
    #     (ties resolve to the smaller period: over-reading the linewidth is
    #     safer than inventing resolution).
    finesse_max = 1.5 * fsr_hz / instrument_hz
    valid = []
    for T in sorted(cand):
        if T <= 0 or T / fwhm_s > finesse_max or T < 2.5 * fwhm_s:
            continue
        if _partner(T) is None:
            continue
        if valid and T <= 1.05 * valid[-1]:
            continue                    # merge near-duplicates
        valid.append(T)
    if not valid:
        out.message = ("only one FSR visible — raise the SA201B amplitude "
                       "(30 V shows ~3 FSR) so the peak spacing can "
                       "calibrate the axis")
        return out
    if expected_fsr_period_s:
        fsr_period = min(valid,
                         key=lambda T: abs(np.log(T / expected_fsr_period_s)))
    else:
        fsr_period = max(valid, key=lambda T: (_support(T), -T))

    # refinement: averaging the left and right neighbour spacing cancels the
    # linear piezo-chirp term at the analyzed peak
    tolr = max(0.15 * fsr_period, 6 * w0_main)
    lft = _nearest_peak(apex_t, heights, t_main - fsr_period, tolr)
    rgt = _nearest_peak(apex_t, heights, t_main + fsr_period, tolr)
    if lft is not None and lft[1] < 0.3 * main_h:
        lft = None
    if rgt is not None and rgt[1] < 0.3 * main_h:
        rgt = None
    if lft is not None and rgt is not None:
        fsr_period = (rgt[0] - lft[0]) / 2
    elif lft is not None:
        fsr_period = t_main - lft[0]
    elif rgt is not None:
        fsr_period = rgt[0] - t_main

    out.fsr_period_s = float(fsr_period)
    out.hz_per_s = fsr_hz / fsr_period
    out.linewidth_hz = fwhm_s * out.hz_per_s
    if out.fwhm_direct_s:
        out.linewidth_direct_hz = out.fwhm_direct_s * out.hz_per_s
    out.deconvolved_hz = max(out.linewidth_hz - instrument_hz, 0.0)
    out.finesse = fsr_hz / out.linewidth_hz

    # -------------------------------------------- transverse contamination
    # strongest peak at a half-FSR position, relative to the main peak
    half_tol = max(0.12 * fsr_period / 2, 6 * w0_main)
    trans = 0.0
    for k in (-1.5, -0.5, 0.5, 1.5):
        p = _nearest_peak(apex_t, heights, t_main + k * fsr_period, half_tol)
        if p is not None:
            trans = max(trans, p[1] / main_h)
    out.transverse_frac = float(trans)

    # ------------------------------------------------------- mode structure
    # peaks at half-FSR positions are cavity transverse modes, not laser
    # modes -- keep them out of the longitudinal-mode listing
    win_lo, win_hi = t_main - 0.02 * fsr_period, t_main + 0.90 * fsr_period
    offsets = []
    for tt in apex_t:
        if not (win_lo <= tt <= win_hi):
            continue
        frac_pos = (tt - t_main) / fsr_period
        dist_to_int = abs(frac_pos - round(frac_pos))
        if trans > 0 and abs(dist_to_int - 0.5) < 0.12:
            continue        # half-FSR position: transverse, not a laser mode
        offsets.append((tt - t_main) * out.hz_per_s)
    out.mode_offsets_hz = sorted(offsets)

    out.ok = True
    out.message = "ok"
    return out
