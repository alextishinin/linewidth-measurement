"""Shared constants for the SA210 / SA201B / PicoScope linewidth system."""

# ---------------------------------------------------------------- instrument
FSR_HZ = 10.0e9              # SA210 series free spectral range (confocal, c/4d)
INSTRUMENT_RES_HZ = 67.0e6   # SA210 spec resolution (FSR / finesse, finesse >150)

# ---------------------------------------------------------------- SA201B (USB serial)
SA201B_VID = 0x1313          # Thorlabs
SA201B_PID = 0x100A
SA201B_BAUD = 115200

# Rise-time step mapping (remote command range 0..200 maps linearly onto the
# 0.01 s .. 0.1 s rise-time range at 1x sweep expansion, per SA201B manual).
RISETIME_MIN_S = 0.010
RISETIME_MAX_S = 0.100
RISETIME_STEPS = 200
SWEEP_EXPANSION_FACTORS = [1, 2, 5, 10, 20, 50, 100]
PD_GAINS_V_PER_A = {0: 1e4, 1: 1e5, 2: 1e6}

# ---------------------------------------------------------------- PicoScope
PICO_SDK_LIB = r"C:\Program Files\Pico Technology\SDK\lib"
EXT_TRIGGER_LEVEL_V = 1.5    # SA201B TRIGGER OUT is 5 V TTL; rising edge = ramp start
EXT_TRIGGER_FULLSCALE_V = 5.0
DEFAULT_DT_S = 5.0e-7        # 2 MS/s: ~45 samples across a 67 MHz feature at 10 ms sweep
AUTO_TRIGGER_MS = 1000       # capture something even with no trigger connected

# ---------------------------------------------------------------- palette (dataviz)
# Light values also serve as creation-time defaults; the app repaints
# everything through THEMES at startup and on toggle.
COL_SURFACE = "#fcfcfb"
COL_PAGE = "#f9f9f7"
COL_INK = "#0b0b0b"
COL_INK2 = "#52514e"
COL_MUTED = "#898781"
COL_GRID = "#e1e0d9"
COL_AXIS = "#c3c2b7"
COL_SERIES1 = "#2a78d6"      # measured trace / trend
COL_SERIES2 = "#eb6834"      # Lorentzian fit overlay
COL_CRITICAL = "#d03b3b"     # warnings
COL_GOOD = "#006300"

# Both palettes are the validated light/dark steps of the same design
# system (series hues re-stepped for the dark surface, not just inverted).
THEMES = {
    "light": dict(
        SURFACE="#fcfcfb", PAGE="#f9f9f7", INK="#0b0b0b", INK2="#52514e",
        MUTED="#898781", GRID="#e1e0d9", AXIS="#c3c2b7",
        SERIES1="#2a78d6", SERIES2="#eb6834", CRITICAL="#d03b3b",
        GOOD="#006300", HOVER="#edece6",
    ),
    "dark": dict(
        SURFACE="#1a1a19", PAGE="#0d0d0d", INK="#ffffff", INK2="#c3c2b7",
        MUTED="#898781", GRID="#2c2c2a", AXIS="#383835",
        SERIES1="#3987e5", SERIES2="#d95926", CRITICAL="#d03b3b",
        GOOD="#0ca30c", HOVER="#30302e",
    ),
}
