"""Self-test: run the analysis pipeline against the simulator and check
that it recovers the known truth. Run:  python test_analysis.py

Needs no hardware and no instrument drivers (numpy + scipy only).
"""
from __future__ import annotations

import numpy as np

import analysis as ana
import config
from simulator import SimulatedSA201B, SimulatedScope


def make_scope(modes=None, transverse=0.0, nonlinearity=0.03):
    ctrl = SimulatedSA201B()
    scope = SimulatedScope(ctrl)
    if modes is not None:
        scope.MODES = modes
    scope.TRANSVERSE_FRAC = transverse
    scope.SWEEP_NONLINEARITY = nonlinearity
    scope.configure_window(0.0145, dt_target_s=5e-7)
    return ctrl, scope


def run_sweeps(ctrl, scope, n=8, use_prior=True):
    rise = ctrl.rise_time_s()
    expected_T = (rise * config.VOLTS_PER_FSR / ctrl.amplitude_v
                  if use_prior else None)
    results = []
    for _ in range(n):
        cap = scope.capture()
        t, v, _ = ana.trim_to_rising_ramp(cap, rise)
        res = ana.analyze_sweep(t, v, expected_fsr_period_s=expected_T)
        assert res.ok, f"analysis failed: {res.message}"
        results.append(res)
    return results


def median_of(results, attr):
    return float(np.median([getattr(r, attr) for r in results]))


def main() -> int:
    width_true = config.FSR_HZ / SimulatedScope.FINESSE   # 66.7 MHz Airy FWHM

    # ------------------------------------------------ 1. multimode (legacy)
    ctrl, scope = make_scope()
    rise = ctrl.rise_time_s()
    hz_per_s_true = (ctrl.amplitude_v / config.VOLTS_PER_FSR
                     * config.FSR_HZ / rise)
    fsr_period_true = config.FSR_HZ / hz_per_s_true
    res_mm = run_sweeps(ctrl, scope)
    lw = median_of(res_mm, "linewidth_hz")
    fsr_p = median_of(res_mm, "fsr_period_s")
    mode_counts = [len(r.mode_offsets_hz) for r in res_mm]
    errs = [r.fit_fwhm_err_s * r.hz_per_s for r in res_mm
            if r.fit_fwhm_err_s is not None]
    print(f"[multimode]  linewidth {lw / 1e6:.1f} MHz "
          f"(truth {width_true / 1e6:.1f}), FSR period {fsr_p * 1e3:.3f} ms "
          f"(truth {fsr_period_true * 1e3:.3f}), modes {mode_counts}")
    assert abs(lw - width_true) / width_true < 0.15, "linewidth off by >15%"
    assert abs(fsr_p - fsr_period_true) / fsr_period_true < 0.06, \
        "FSR period off by >6%"
    assert np.median(mode_counts) >= 2, "multi-mode structure not resolved"
    trans_mm = median_of(res_mm, "transverse_frac")
    assert trans_mm < 0.10, f"clean multimode flagged transverse {trans_mm}"

    assert errs, "fit produced no width uncertainty"
    err = float(np.median(errs))
    print(f"[multimode]  fit uncertainty {err / 1e6:.2f} MHz "
          f"({100 * err / lw:.1f}%)")
    assert 0 < err < 0.2 * lw, f"implausible fit uncertainty {err:.3g} Hz"

    # -------------------------------------- 2. clean single-frequency laser
    ctrl, scope = make_scope(modes=[(0.0, 1.0)])
    res_cl = run_sweeps(ctrl, scope)
    lw = median_of(res_cl, "linewidth_hz")
    trans = median_of(res_cl, "transverse_frac")
    print(f"[clean]      linewidth {lw / 1e6:.1f} MHz, "
          f"transverse {trans * 100:.0f}%, "
          f"finesse {median_of(res_cl, 'finesse'):.0f}")
    assert abs(lw - width_true) / width_true < 0.15
    assert trans < 0.05, f"clean cavity shows transverse {trans}"

    # ---- 3. the bench case: equal-height transverse comb + strong chirp
    # (six equal peaks at FSR/2 spacing; this used to miscalibrate by 2.5x)
    ctrl, scope = make_scope(modes=[(0.0, 1.0)], transverse=1.0,
                             nonlinearity=0.20)
    res_tr = run_sweeps(ctrl, scope)
    lw = median_of(res_tr, "linewidth_hz")
    trans = median_of(res_tr, "transverse_frac")
    fin = median_of(res_tr, "finesse")
    print(f"[transverse] linewidth {lw / 1e6:.1f} MHz "
          f"(truth {width_true / 1e6:.1f}), transverse {trans * 100:.0f}%, "
          f"finesse {fin:.0f}")
    assert 0.75 * width_true < lw < 1.35 * width_true, \
        f"half-FSR comb miscalibrated the linewidth: {lw / 1e6:.1f} MHz"
    assert trans > 0.6, f"equal transverse comb not detected: {trans}"
    assert fin <= 1.5 * config.FSR_HZ / config.INSTRUMENT_RES_HZ + 1, \
        f"impossible finesse {fin}"
    n_modes_tr = int(np.median([len(r.mode_offsets_hz) for r in res_tr]))
    assert n_modes_tr == 1, \
        f"transverse peaks leaked into the mode list ({n_modes_tr})"

    # Peak selection must be deterministic and edge-avoiding: with six
    # near-identical peaks, sweep-to-sweep scatter should be tight (the old
    # tallest-peak lottery produced +-6% telegraph jumps from one-sided
    # edge calibrations).
    ctrl, scope = make_scope(modes=[(0.0, 1.0)], transverse=1.0,
                             nonlinearity=0.20)
    rise = ctrl.rise_time_s()
    expected_T = rise * config.VOLTS_PER_FSR / ctrl.amplitude_v
    lws, centers = [], []
    prev_center = None
    for _ in range(12):
        cap = scope.capture()
        tt, vw, _ = ana.trim_to_rising_ramp(cap, rise)
        r = ana.analyze_sweep(tt, vw, expected_fsr_period_s=expected_T,
                              prefer_time_s=prev_center)
        assert r.ok
        prev_center = r.fit_center_s
        lws.append(r.linewidth_hz)
        centers.append(r.fit_center_s)
        # never an edge peak while interior candidates exist
        assert r.peak_times.min() < r.fit_center_s < r.peak_times.max(), \
            "edge peak selected for the linewidth fit"
    spread = float(np.std(lws) / np.mean(lws))
    print(f"[stability]  12 sweeps: {np.mean(lws) / 1e6:.1f} MHz, "
          f"relative std {spread * 100:.1f}%")
    assert spread < 0.05, f"linewidth telegraphing: {spread * 100:.1f}% std"

    # ------------------------- 4. moderate, alternating transverse comb
    ctrl, scope = make_scope(modes=[(0.0, 1.0)], transverse=0.4,
                             nonlinearity=0.10)
    res_alt = run_sweeps(ctrl, scope)
    lw = median_of(res_alt, "linewidth_hz")
    trans = median_of(res_alt, "transverse_frac")
    print(f"[alt 40%]    linewidth {lw / 1e6:.1f} MHz, "
          f"transverse {trans * 100:.0f}%")
    assert 0.75 * width_true < lw < 1.35 * width_true
    assert 0.25 < trans < 0.6, f"transverse fraction off: {trans}"

    # -------------------------------- robust-period fallback (spike guard)
    ctrl, scope = make_scope(modes=[(0.0, 1.0)], transverse=1.0,
                             nonlinearity=0.20)
    rise = ctrl.rise_time_s()
    expected_T = rise * config.VOLTS_PER_FSR / ctrl.amplitude_v
    cap = scope.capture()
    tt, vw, _ = ana.trim_to_rising_ramp(cap, rise)
    base = ana.analyze_sweep(tt, vw, expected_fsr_period_s=expected_T)
    assert base.ok and not base.calibration_fallback
    # median far from this sweep's own period -> fallback must engage
    forced = ana.analyze_sweep(tt, vw, expected_fsr_period_s=expected_T,
                               robust_fsr_period_s=base.fsr_period_s * 1.2)
    assert forced.calibration_fallback, "fallback did not engage"
    assert abs(forced.hz_per_s - config.FSR_HZ /
               (base.fsr_period_s * 1.2)) < 1e-3 * forced.hz_per_s
    # median consistent with this sweep -> no fallback, own period used
    agree = ana.analyze_sweep(tt, vw, expected_fsr_period_s=expected_T,
                              robust_fsr_period_s=base.fsr_period_s * 1.01)
    assert not agree.calibration_fallback
    print("[robust]     period fallback engages at >4% deviation, "
          "stays off when consistent")

    # ------------------------------------------------- helpers & edge cases
    v, u = ana.wavelength_width(3e9, 1064.0)
    assert u == "pm" and abs(v - 11.33) < 0.1, (v, u)
    v, u = ana.wavelength_width(67e6, 1064.0)
    assert u == "fm" and abs(v - 253.0) < 5.0, (v, u)
    v, u = ana.wavelength_width(500e9, 1064.0)
    assert u == "nm" and abs(v - 1.888) < 0.02, (v, u)
    print("wavelength unit helper: ok (3 GHz -> 11.3 pm, 67 MHz -> 253 fm)")

    ctrl, scope = make_scope()
    cap = scope.capture()
    t, v = cap.t, cap.pd
    t, v, _ = ana.trim_to_rising_ramp(cap, ctrl.rise_time_s())
    res_weak = ana.analyze_sweep(t, (v - np.median(v)) * 0.002 + np.median(v))
    assert not res_weak.ok and "signal" in res_weak.message
    print("weak-signal path: ok ->", res_weak.message)

    print("\nALL ANALYSIS TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
