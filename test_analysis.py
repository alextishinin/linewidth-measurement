"""Self-test: run the analysis pipeline against the simulator and check
that it recovers the known truth. Run:  python test_analysis.py
"""
from __future__ import annotations

import numpy as np

import analysis as ana
import config
from simulator import SimulatedSA201B, SimulatedScope


def main() -> int:
    ctrl = SimulatedSA201B()
    scope = SimulatedScope(ctrl)
    scope.configure_window(0.0145, dt_target_s=5e-7)

    # Truth from the simulator's construction:
    rise = ctrl.rise_time_s()
    hz_per_s_true = (ctrl.amplitude_v / scope.VOLTS_PER_FSR) * config.FSR_HZ / rise
    fsr_period_true = config.FSR_HZ / hz_per_s_true
    width_true = config.FSR_HZ / scope.FINESSE          # 66.7 MHz Airy FWHM

    lws, fsrs, mode_counts = [], [], []
    for _ in range(8):
        cap = scope.capture()
        t, v, _ = ana.trim_to_rising_ramp(cap, rise)
        res = ana.analyze_sweep(t, v)
        assert res.ok, f"analysis failed: {res.message}"
        lws.append(res.linewidth_hz)
        fsrs.append(res.fsr_period_s)
        mode_counts.append(len(res.mode_offsets_hz))
        assert res.fit_r2 is None or res.fit_r2 > 0.9, f"poor fit R2={res.fit_r2}"

    lw = np.median(lws)
    fsr_p = np.median(fsrs)
    print(f"linewidth: median {lw / 1e6:.1f} MHz  (truth {width_true / 1e6:.1f}, "
          f"spread {np.std(lws) / 1e6:.1f})")
    print(f"FSR period: {fsr_p * 1e3:.3f} ms  (truth {fsr_period_true * 1e3:.3f})")
    print(f"modes found per sweep: {mode_counts}  (truth 3)")

    assert abs(lw - width_true) / width_true < 0.15, "linewidth off by >15%"
    assert abs(fsr_p - fsr_period_true) / fsr_period_true < 0.06, "FSR period off by >6%"
    assert np.median(mode_counts) >= 2, "multi-mode structure not resolved"

    # Wavelength-unit conversion helper.
    v, u = ana.wavelength_width(3e9, 1064.0)
    assert u == "pm" and abs(v - 11.33) < 0.1, (v, u)
    v, u = ana.wavelength_width(67e6, 1064.0)
    assert u == "fm" and abs(v - 253.0) < 5.0, (v, u)
    v, u = ana.wavelength_width(500e9, 1064.0)
    assert u == "nm" and abs(v - 1.888) < 0.02, (v, u)
    print("wavelength unit helper: ok (3 GHz -> 11.3 pm, 67 MHz -> 253 fm)")

    # Weak-signal path must degrade gracefully, not crash.
    cap = scope.capture()
    t, v, _ = ana.trim_to_rising_ramp(cap, rise)
    res_weak = ana.analyze_sweep(t, (v - np.median(v)) * 0.002 + np.median(v))
    assert not res_weak.ok and "signal" in res_weak.message
    print("weak-signal path: ok ->", res_weak.message)

    print("\nALL ANALYSIS TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
