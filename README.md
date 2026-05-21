# oscilloscope-readout-4054

Captures waveforms from a **Tektronix DPO4054 Digital Phosphor Oscilloscope** (500 MHz, 4-channel) over USB and saves them to HDF5 files for offline analysis.

## Requirements

- Python 3.10+
- Tektronix DPO4054 connected via USB-B cable

```bash
pip install pyvisa pyvisa-py numpy h5py pyusb pyyaml
```

(`pyyaml` is only needed if you drive captures from a YAML config; see [config_example.yaml](config_example.yaml).)

**Linux only** — grant non-root USB access (run once):

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0699", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/99-tektronix.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Scope setup

On the oscilloscope: `Utility > I/O > USB Network & PC` → set to **USB Device**

## Usage

### GUI (recommended)

```bash
python capture_gui.py
```

Opens an interactive window with:

- **Connect to Scope / Disconnect** — scans USB and connects; coloured indicator shows connection state (red/yellow/green); Disconnect closes the VISA session cleanly
- **Output File** — path for the `.h5` output file, with Browse and New Filename buttons (New Filename stamps the current time)
- **Channels** — checkboxes for CH1–CH4
- **Trigger Window** — pre/post-trigger time in nanoseconds (`0 0` = full record)
- **Capture Options**
  - Number of captures, wait time between captures (seconds), capture label, notes
  - **Acquisition mode** — `SAMPLE` (default), `AVERAGE`, `HIRES`, `ENVELOPE`, or `PEAKDETECT`; selecting `AVERAGE` reveals an averages count field
  - **Read scope measurements** — opt-in; after each capture, queries AMPLITUDE, HIGH, LOW, MEAN, RMS, FREQUENCY, PERIOD, RISETIME, FALLTIME from the scope and stores them as channel attributes in the HDF5 file; adds ~1–2 s overhead per capture for 4 channels
- **Capture** — starts acquisition; waveforms are plotted live as each channel is digitized; a red dashed trigger line is overlaid at t = 0; multiple captures are overlaid using a sequential colormap
- **Stop** — cancels a running capture after the current channel
- **Histograms** — per-channel histograms of the baseline-subtracted pulse integral (V·s) and peak amplitude (V) accumulate across captures; a **Clear Histograms** button resets without interrupting capture
- **Screenshot** — saves a PNG of the scope display to `./data/{prefix}_{timestamp}_shot{N}.png` alongside the waveform file; counter resets when New Filename is clicked

### CLI

`capture_waveforms.py` runs in three modes:

```bash
# 1. Interactive — prompts for each parameter
python capture_waveforms.py

# 2. Headless — all parameters from a YAML file
python capture_waveforms.py --config config_example.yaml

# 3. Generate a template YAML file
python capture_waveforms.py --example-config my_run.yaml
```

The headless mode exposes the same parameters as the GUI (channels, ns-based trigger window, number of captures, wait-between, acquisition mode + averages, scope measurements, ROOT output, label, notes) plus automatic begin/end screenshots. See [config_example.yaml](config_example.yaml) for a fully-commented template.

#### Programmatic use

`run_capture()` and `run_capture_from_yaml()` are importable:

```python
from capture_waveforms import run_capture, run_capture_from_yaml

# From a YAML file
out_path = run_capture_from_yaml("my_run.yaml")

# From a Python dict (any subset of fields; missing keys use DEFAULT_CONFIG)
out_path = run_capture({
    "channels": ["CH1", "CH2"],
    "trigger_window": {"pre_ns": 500, "post_ns": 2000},
    "capture":        {"n_captures": 100, "wait_s": 0.1, "label": "run42"},
    "acquisition":    {"mode": "AVERAGE", "numavg": 16},
    "measurements":   {"enabled": True},
})
```

Pass an already-open `pyvisa` resource as the optional `scope=` argument to reuse a connection across calls; otherwise the function connects and disconnects itself.

#### Interactive mode

In interactive mode the script will:

1. Scan for USB instruments and connect to the scope
2. Prompt for an output `.h5` file (default: timestamped filename)
3. Loop — for each capture:
   - Select channels (`1 2` or `1 2 3 4`, etc.)
   - Enter pre/post trigger window sample counts
   - Enter number of captures and a capture label
   - Enter optional notes
   - Capture and save, then prompt to capture again

Interactive mode uses **samples** for the trigger window; YAML/GUI modes use **nanoseconds**.

## Output files

### HDF5 waveform data

Multiple captures in one session are appended as separate groups in the same file:

```
/  (attrs: created, last_updated)
/<capture_label>/
    attrs:   timestamp, notes
             sample_rate_hz, h_scale_s_div
             trig_type, trig_source, trig_level_v, trig_slope, trig_freq_hz
             acq_mode, acq_numavg
    instrument_setup   dataset  full SET? blob (gzip-compressed), ~10–40 KB
    /CH1/
        time_s   [N]  float64  seconds
        volts    [N]  float64  volts
        attrs:   all WFMPRE preamble fields (XINCR, YMULT, YOFF, YZERO, XZERO, …)
                 CH_COUPLING, CH_SCALE, CH_BANDWIDTH, CH_PROBE
                 meas_amplitude, meas_mean, meas_rms, …  (if measurements enabled)
    /CH2/  ...
```

Reading the output in Python:

```python
import h5py, numpy as np

with h5py.File("waveforms_20260318_142301.h5", "r") as f:
    t = f["run3_signal/CH1/time_s"][:]
    v = f["run3_signal/CH1/volts"][:]
    yunit = f["run3_signal/CH1/volts"].attrs["units"]      # 'V'
    xunit = f["run3_signal/CH1/time_s"].attrs["units"]     # 's'
    amp   = f["run3_signal/CH1"].attrs.get("meas_amplitude")
    setup = f["run3_signal/instrument_setup"][:].tobytes().decode()  # full scope config
```

### TSV capture log

Every successful capture appends one row to a TSV file alongside the HDF5 file (same path, `.tsv` extension). A header row is written automatically on first use.

Columns:

| Column | Description |
|---|---|
| `timestamp` | ISO-8601 timestamp of the save |
| `capture_label` | HDF5 group name for this capture |
| `hdf5_file` | Absolute path to the HDF5 file |
| `channels` | Comma-separated channel list (e.g. `CH1,CH2`) |
| `pre_samples` | Pre-trigger samples, or `full` |
| `post_samples` | Post-trigger samples, or `full` |
| `notes` | Free-text notes |
| `sample_rate_hz` | Sample rate at time of capture |
| `h_scale_s_div` | Horizontal scale (s/div) |
| `trig_type` | Trigger type (e.g. `EDGE`) |
| `trig_source` | Trigger source channel |
| `trig_level_v` | Trigger level in volts |
| `trig_slope` | Trigger slope (`RISE` / `FALL`) |
| `trig_freq_hz` | Measured trigger frequency |
| `acq_mode` | Acquisition mode at capture time |
| `acq_numavg` | Number of averages (AVERAGE mode only) |

### Screenshots

Screenshots are saved as PNG via the `HARDCOPY` SCPI command over USBTMC.

**GUI** — the **Screenshot** button saves on demand to `{prefix}_{timestamp}_shot{N}.png`; the counter resets when New Filename is clicked.

**Headless / YAML** — `output.save_screenshot_begin` (default `true`) and `output.save_screenshot_end` (default `true`) save the scope display automatically before the first capture and after the last one. Files are placed alongside the HDF5 output with the same stem:

```
waveforms_20260520_143022_begin.png
waveforms_20260520_143022_end.png
```

## Notes

- Transfer uses signed 8-bit binary (`RIBINARY` + `DATA:WIDTH 1`) — the DPO4054 has an 8-bit ADC, so 16-bit transfer would just zero-pad each sample and double the `CURVE?` time for no information gain
- All HDF5 datasets are gzip-compressed
- Physical-unit scaling is applied from the WFMPRE preamble: `volts = (raw - YOFF) * YMULT + YZERO`; preamble fields are queried individually (`WFMPRE:YMULT?`, `WFMPRE:YOFF?`, …) because the DPO4054 returns only the WFID string for the bulk `WFMPRE?` query
- Per-channel preamble + display settings are queried once per capture batch and cached; subsequent captures in the batch skip the ~15 SCPI round-trips per channel. The cache is invalidated when the user starts a new Capture run, so changes to scope vertical/horizontal/trigger settings between batches are picked up automatically
- Each capture is a single-sequence acquisition (`ACQUIRE:STOPAFTER SEQUENCE`): the scope is armed with `ACQUIRE:STATE RUN`, completes one full trigger (or NUMAVG triggers in AVERAGE mode), and auto-stops — guaranteeing every readout comes from a fresh trigger and all channels share the same event
- The DPO4054 does not support FastFrame / segmented memory acquisition; every capture incurs the CURVE? transfer time as dead time between triggers
- Programmer reference: Tektronix MSO4000/DPO4000 Series Programmer Manual (077-0248-01)
