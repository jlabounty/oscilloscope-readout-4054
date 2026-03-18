# oscilloscope-readout-4054

Captures waveforms from a **Tektronix DPO4054 Digital Phosphor Oscilloscope** (500 MHz, 4-channel) over USB and saves them to HDF5 files for offline analysis.

## Requirements

- Python 3.10+
- Tektronix DPO4054 connected via USB-B cable

```bash
pip install pyvisa pyvisa-py numpy h5py pyusb
```

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
- **Trigger Window** — pre/post-trigger sample counts (`0 0` = full record)
- **Capture Options** — number of captures, wait time between captures (seconds, default 0), capture label, notes
- **Capture** — starts acquisition; waveforms are plotted live as each channel is digitized; a red dashed trigger line is overlaid at t = 0 (XZERO); multiple captures are overlaid using the viridis colormap; a minimum of 0.1 s is enforced between successive captures regardless of the wait setting
- **Histograms** — displayed alongside the waveforms; per-channel histograms of the baseline-subtracted pulse integral (V·s) and peak amplitude (V) accumulate across captures within a session; baseline is estimated from pre-trigger samples; a **Clear Histograms** button resets without interrupting capture

### CLI

```bash
python capture_waveforms.py
```

The script will:

1. Scan for USB instruments and connect to the scope
2. Prompt for an output `.h5` file (default: timestamped filename)
3. Loop — for each capture:
   - Select channels (`1 2` or `1 2 3 4`, etc.)
   - Enter pre/post trigger window sample counts
   - Enter number of captures and a capture label
   - Enter optional notes
   - Capture and save, then prompt to capture again

## Output files

### HDF5 waveform data

Multiple captures in one session are appended as separate groups in the same file:

```
/  (attrs: created, last_updated)
/<capture_label>/  (attrs: timestamp, notes)
    /CH1/
        time_s   [N]  float64  seconds
        volts    [N]  float64  volts
        attrs:   all WFMPRE preamble fields (XINCR, YMULT, YOFF, …)
    /CH2/  ...
```

Reading the output in Python:

```python
import h5py, numpy as np

with h5py.File("waveforms_20260318_142301.h5", "r") as f:
    t = f["run3_signal/CH1/time_s"][:]
    v = f["run3_signal/CH1/volts"][:]
    yunit = f["run3_signal/CH1/volts"].attrs["units"]   # 'V'
    xunit = f["run3_signal/CH1/time_s"].attrs["units"]  # 's'
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

## Notes

- Transfer uses signed 16-bit binary encoding (`RIBINARY`) for speed — much faster than ASCII at long record lengths
- All HDF5 datasets are gzip-compressed
- Physical-unit scaling is applied from the WFMPRE preamble: `volts = (raw - YOFF) * YMULT + YZERO`; preamble fields are queried individually (`WFMPRE:YMULT?`, `WFMPRE:YOFF?`, …) because the DPO4054 returns only the WFID string for the bulk `WFMPRE?` query
- Memory is frozen with `ACQUIRE:STATE STOP` before reading all channels, ensuring every channel comes from the same trigger event; acquisition is re-armed with `ACQUIRE:STATE RUN` afterwards
- Programmer reference: Tektronix MSO4000/DPO4000 Series Programmer Manual (077-0248-01)
