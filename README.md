# Laser linewidth measurement — SA210-8B + SA201B + PicoScope 5242D

Live measurement of a CW laser's linewidth using a Thorlabs scanning
Fabry-Pérot interferometer, with a PicoScope as the digitizer — a complete
Python application with a live display, self-calibrating frequency axis,
Lorentzian fitting, auto-gain, and CSV export.

[![tests](https://github.com/alextishinin/linewidth-measurement/actions/workflows/tests.yml/badge.svg)](https://github.com/alextishinin/linewidth-measurement/actions/workflows/tests.yml)
![platform](https://img.shields.io/badge/platform-Windows-blue)
![python](https://img.shields.io/badge/python-3.11%2B-green)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Hardware

| Instrument | Role |
|---|---|
| Thorlabs **SA210-8B** scanning Fabry-Pérot (FSR 10 GHz, resolution 67 MHz, 820–1275 nm) | the interferometer |
| Thorlabs **SA201B** controller | piezo ramp generator + photodiode amplifier (USB-controlled) |
| Pico Technology **PicoScope 5242D** | digitizer for the spectrum signal (any 5000D-series model should work) |
| 3× BNC cables + the SMA→BNC cable shipped with the SA210 | wiring below |

Other SA210/SA200-series cavities work too — pass your FSR with `--fsr-ghz`.
Swapping wavelength bands (e.g. an SA210-12B for 1550 nm) needs no code
changes: set the laser wavelength in the in-app λ box — it feeds both the
Δλ display **and** the piezo calibration prior, which scales with λ because
the mirror moves λ/4 per free spectral range.

## Installation

1. Windows 10/11 with Python 3.12+ (`winget install Python.Python.3.12`).
2. `pip install -r requirements.txt`
   (numpy, scipy, matplotlib, pyserial, picosdk).
3. Install **PicoSDK 64-bit** from
   [picotech.com/downloads](https://www.picotech.com/downloads) — provides
   `ps5000a.dll` (default location `C:\Program Files\Pico Technology\SDK`,
   which `config.py` points at).
4. Plug in the SA201B (USB) and the PicoScope (USB 3), wire as below, and run
   `run.bat` (edit the Python path inside if yours differs) or
   `python linewidth_live.py`.

## How it works

The SA201B drives the SA210's piezo with a sawtooth, sweeping the cavity
length. Each time the cavity comes into resonance with the laser, a
transmission peak appears on the photodiode. The same laser line repeats every
free spectral range, so the time between two repeats equals exactly **10 GHz**
— that calibrates the time axis into optical frequency with no assumptions
about wavelength or ramp slope. The program then fits a Lorentzian to the
tallest peak; its FWHM in calibrated units is the measured linewidth.

Finding that repeat spacing is less trivial than it sounds: the piezo's
nonlinearity chirps the peak spacing across the sweep, and a misaligned
confocal cavity adds **odd transverse modes exactly halfway between the
fundamentals** (a comb at FSR/2 that can look identical to the real one).
The calibration therefore builds candidate periods from tall-peak spacings
and autocorrelation, rejects any that imply a finesse beyond ~1.5× the
instrument spec (an impossibly *good* answer means the ruler is wrong),
requires an actual partner peak one period away, and picks among the
survivors using the piezo's nominal ~10 V/FSR as a soft prior. The spacing
is then measured as the average of the left and right neighbour gaps, which
cancels the chirp to first order at the analyzed peak. The analyzed peak
itself is chosen deterministically — a competitive interior peak nearest the
sweep center (or the previously tracked one) — because with several
near-equal peaks, "the tallest" is a per-sweep noise lottery, and edge peaks
only permit a one-sided, chirp-biased calibration.

**Resolution floor:** the measured width is the laser lineshape *convolved*
with the instrument function (~67 MHz). A laser much narrower than 67 MHz
shows ≈67 MHz regardless — that reading means "narrower than the instrument",
and the effective finesse readout (FSR / measured width ≥ 150) confirms the
interferometer is performing to spec. For broader lasers the program also
shows a deconvolved estimate (measured − 67 MHz, valid for Lorentzian-ish
shapes).

**Uncertainty:** the headline reads `78.3 ± 2.1 MHz`. The 1σ error bar
combines, in quadrature, the Lorentzian fit's own covariance on the width,
the frequency-calibration error (the standard error of the FSR peak spacing
over the last 20 sweeps), and the standard error of the rolling median. It
is recorded in the CSV log and in exported files.

**Jitter and the rolling median.** A single sweep samples the laser-cavity
frequency jitter during the ~µs peak transit, so individual widths scatter
by a few percent with occasional outlier sweeps. In live mode the app
therefore reports the **median of the last 11 sweeps** (~1.6 s;
`--median N` to change, `--median 1` for raw), which is spike-free, while
the per-sweep value is still logged (`linewidth_hz`) next to the reported
one (`linewidth_median_hz`) — and the observed sweep-to-sweep scatter is
itself displayed, since it is a direct measure of the jitter. Single-sweep
mode always shows the individual sweep's value.

## Wiring

| From | To | Cable |
|---|---|---|
| SA210 attached cable (piezo) | SA201B rear **OUTPUT** (0–45 V) | its own BNC |
| SA210 detector SMA | SA201B rear **PD AMPLIFIER IN** | SMA→BNC (supplied) |
| SA201B front **PD AMPLIFIER OUT** | PicoScope **Channel A** | BNC coax |
| SA201B front **TRIGGER OUT** | PicoScope **EXT** | BNC coax |
| SA201B rear **MONITOR OUT** | PicoScope **Channel B** | BNC coax (recommended) |
| SA201B USB-B + PicoScope USB-B | this PC | USB |

Channel B is optional but recommended: the program uses the ramp monitor to
measure the true sweep duration each capture. Without it, pass
`--single-channel` (the scope then runs 16-bit instead of 15-bit).

## Optical setup (from the SA210 manual)

* Mount the SA210 in a Ø1" kinematic mount (KM100), beam ~1 mm.
* Fold mirror into the input iris; f = 100 mm lens focused at the cavity
  center, ~25 mm past the front flange.
* To align: close the input iris, open the back iris, scan running, center the
  beam; watch the live "Full sweep" panel while tweaking tip/tilt until peaks
  appear, then maximize their height and symmetry. Sharp, tall, non-split
  peaks = good alignment.

## Running

There is one command — double-click `run.bat` (or `python linewidth_live.py`).
Both instruments must be connected; if the scope can't be opened the program
exits with a clear message. Inside the app:

* **Mode button** — switches between **Live** streaming and **Single** sweep.
* **Run once button** — in single mode, captures and displays exactly one
  fresh sweep, then freezes.

`python test_analysis.py` runs the analysis self-test (it uses the synthetic
source in `simulator.py`; the measurement app itself is hardware-only).
CLI flags (below) exist as optional startup presets for scripting.

On startup the program configures the SA201B over USB (sawtooth, 30 V
amplitude ≈ 3 FSR, 10 ms sweep, blanking on) and then continuously:
captures one sweep per trigger → finds peaks → calibrates Hz/s from the
10 GHz peak spacing → fits the main peak → updates the plots and the CSV log
in `logs\`.

**Live vs. single-sweep mode.** By default the scope captures and the display
streams continuously. The **Mode** button (or `m`) switches to single-sweep
mode, where the scope is *idle* — no data is acquired at all (the PicoScope
LED stops flashing). Each click of **Run once** (or `r`) arms exactly one
capture (or N sweeps with `--avg N`): the sweep is taken, analyzed, displayed,
and acquisition stops again until the next run. Handy for documenting discrete
measurements — each run adds one point to the history panel, and old points
don't expire. Start directly in this mode with `--single`.

### Useful options

Options marked *(persisted)* default to whatever you last used in the app —
see **Settings are remembered** below.

| Option | Meaning |
|---|---|
| `--single` | start in single-sweep mode |
| `--wavelength-nm 1064` | laser wavelength for the Δλ display *(persisted)* |
| `--theme dark` | `dark` (default) or `light` *(persisted)* |
| `--amplitude 30` | ramp volts, 30 V ≈ 3 FSR on the SA210 *(persisted)* |
| `--risetime-step 0` | 0..200 → 10..100 ms sweep *(persisted)* |
| `--sweep-expand 0` | expansion index 0..6 = 1×..100× *(persisted)* |
| `--pdgain auto` | photodiode amp gain 10k/100k/1M V/A, or `0`/`1`/`2` fixed *(persisted)* |
| `--avg 4` | average 4 triggered sweeps before analysis |
| `--single-channel` | MONITOR OUT not wired to channel B |
| `--no-controller` | leave the SA201B alone (touchscreen control) |
| `--dt-us 0.5` | sample interval (0.5 µs ≈ 45 samples across 67 MHz) |
| `--window-ms` / `--rise-ms` | manual capture window / sweep time |

### In-app controls

The side panel has, top to bottom: a **λ nm (100–5000)** input box (sets the
wavelength used for the Δλ display and exports — type a value and press
Enter), **scan controls**, a **PD gain (V/A)** dropdown, **Run once** /
**Mode** buttons, and **Export data** / **Align** buttons. Every input box
shows its valid range in the label; dropdowns are native and show all
choices.

**Scan controls** (live — no restart needed): **ampl V (1–30)** (ramp ≈ 1 V
per GHz of span), **offs V (0–15)** (shifts the pattern; also ←/→ keys),
**sweep ms (10–100)** (sweep time at 1×; snapped to the SA201B's step grid),
and the **expand** dropdown (1×–100× sweep expansion). Changing the sweep
time or expansion automatically resizes the scope's capture window (long
sweeps also coarsen the sample interval to keep captures manageable).

All controller writes are applied on a background worker: the UI reacts
instantly, the SA201B follows ~half a second later (its serial link needs a
set + verify round-trip), and a failed write shows a warning. If the write
was lost to a USB dropout, the auto-reconnect re-applies your settings.

**Align** (`t` key) switches the SA201B to a **triangle** scan — the manual's
recommended waveform for initial cavity alignment — and the headline changes
from linewidth to **peak height in volts**: walk the mirror mount to maximize
that number. Alignment sweeps are excluded from the history plot and CSV log.
Click again to return to sawtooth measurement mode.

**Transverse-mode meter.** The stats panel shows
`transverse modes: N% of main (alignment good/fair/poor)` — the height of the
strongest peak found at half-FSR positions relative to the fundamental. Those
peaks are the confocal cavity's odd transverse modes, excited by imperfect
mode matching; they are excluded from the longitudinal-mode count and, above
50%, trigger an alignment warning. Minimizing this number (tip/tilt, iris,
focus position) *is* the alignment procedure — in Align mode the subtitle
shows it live next to the peak height.

**PD gain (V/A) dropdown:** *Auto* (the default) adjusts the SA201B
photodiode amplifier both ways — it steps the gain *down* when the 5 V output
saturates and *up* when the peak falls below ~0.35 V (with a few-second
cooldown). Selecting 10k/100k/1M locks that transimpedance gain manually;
selecting Auto hands control back. The dropdown always shows the current
choice (the stats line shows the actual gain Auto has picked).

**Export:** writes the currently displayed data to `exports\` as two CSVs —
`sweep_*.csv` (time, signal, calibrated frequency offset, Lorentzian fit,
with a `#` metadata header: λ, FSR, calibration, FWHM in MHz and pm, finesse,
gain) and `history_*.csv` (the linewidth-vs-time trend in MHz and pm). Works
in live or single mode.

**Graph 2 x-range.** Above the "Main peak" graph is an **x-range** control
with three modes:

* **Auto** (default) — the window follows the fitted linewidth and any
  neighbouring modes, as it always has.
* **Full 10 GHz** — forces the whole free spectral range (−5000…+5000 MHz),
  so you see every mode in one order at once. Useful for spotting the
  half-FSR higher-order transverse modes that indicate imperfect alignment.
  The label follows `--fsr-ghz` if your cavity differs.
* **Manual** — type a **min** and **max** in MHz (clamped to ±FSR/2, i.e.
  ±5000 MHz for the SA210). Typing in either box switches to Manual
  automatically; bad or inverted entries are rejected with a message.

The choice is remembered between sessions. A drag-zoom on that graph
temporarily overrides the mode; **reset view** returns to it.

**Zoom:** drag with the left mouse button anywhere on a graph to zoom into
that box — this works **while data is streaming**. Auto-scaling for that
graph pauses so your view stays put, and its **reset view** button (above the
graph's top-right corner) lights up; press it to zoom back out and resume
auto-scaling. A wide, flat drag zooms only the x-axis and leaves y
auto-scaling — and vice versa — so you can rescale one dimension at a time.
Small drags (< 8 px) are treated as clicks and ignored. The `v` key resets
all three graphs at once.

**Theme:** the app starts in **dark mode**; the button in the top-right
corner (or the `d` key) switches between dark and light. Launch with
`--theme light` to start light. Snapshots save in whichever theme is active.

**Settings are remembered.** On exit the app writes `settings.json`
(wavelength, theme, ramp amplitude/offset/sweep time/expansion, and the gain
choice) and restores them next launch. Precedence is
*explicit command-line flag → saved value → built-in default*, so
`--wavelength-nm 780` still wins for one session without overwriting your
saved setup until you exit. Delete `settings.json` to return to defaults.

**If the SA201B's USB drops** mid-session the app keeps measuring (the
controller keeps ramping on its own), shows
`! SA201B USB disconnected — retrying...`, and reconnects automatically as
soon as the port reappears — re-applying amplitude, offset, sweep time,
expansion, gain and waveform so the session continues unchanged.

**Keys:** `r` run one sweep · `m` live/single mode · `t` alignment mode ·
`g` cycle PD gain · `a` toggle auto-gain · `e` export data · `d` theme ·
`v` reset all zoomed views · `←/→` DC offset ±0.25 V · `s` snapshot PNG+CSV ·
`p` pause display · `q` quit. (Keys are ignored while typing in any input
box.)

## Interpreting the display

* **Full sweep** — every transmission peak along the 30 V ramp; expect the
  same pattern ~3× (once per FSR). Use this panel while aligning.
* **Main peak** — zoomed, frequency-calibrated view with the Lorentzian fit.
  Multiple peaks inside one FSR = multiple longitudinal modes; their spacings
  are listed in the side panel.
* **History** — measured FWHM vs. time; drift/jitter of the sweep shows up
  here. The CSV log has every value with timestamps.

## Files

| File | Role |
|---|---|
| `linewidth_live.py` | main live application |
| `analysis.py` | peak finding, FSR calibration, Lorentzian fit, uncertainty |
| `sa201b.py` | SA201B USB-serial driver (verified protocol) |
| `pico5000a.py` | PicoScope 5000D block-mode acquisition |
| `capture.py` | the one-sweep data container shared by both |
| `simulator.py` | synthetic source used only by the self-test |
| `test_analysis.py` | pipeline self-test against the simulator |
| `config.py` | FSR, resolution, ports, color themes |

## Tests

`python test_analysis.py` runs the measurement pipeline against a synthetic
multi-mode laser and asserts it recovers the known linewidth, FSR spacing,
mode count, fit uncertainty and unit conversions. It needs **no hardware and
no instrument drivers** (numpy + scipy only), so it runs in CI on every push
— see [`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Requirements

Python 3.12 with `numpy scipy matplotlib pyserial picosdk`
(`pip install -r requirements.txt`) and the Pico Technology **PicoSDK**
(provides `ps5000a.dll`; installed at `C:\Program Files\Pico Technology\SDK`).
Close the PicoScope 7 desktop app before running — only one program can own
the scope at a time.

## License

[MIT](LICENSE)
