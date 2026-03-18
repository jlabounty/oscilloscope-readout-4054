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

```bash
python capture_waveforms.py
```

The script will:

1. Scan for USB instruments and connect to the scope
2. Prompt for an output `.h5` file (default: timestamped filename)
3. Loop — for each capture:
   - Select channels (`1 2` or `1 2 3 4`, etc.)
   - Enter a capture label (e.g. `run3_signal`)
   - Enter optional notes
   - Capture and save, then prompt to capture again

## HDF5 output format

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

with h5py.File("waveforms_20250318_142301.h5", "r") as f:
    t = f["run3_signal/CH1/time_s"][:]
    v = f["run3_signal/CH1/volts"][:]
    yunit = f["run3_signal/CH1/volts"].attrs["units"]   # 'V'
    xunit = f["run3_signal/CH1/time_s"].attrs["units"]  # 's'
```

## Notes

- Transfer uses signed 16-bit binary encoding (`RIBINARY`) for speed — much faster than ASCII at long record lengths
- All datasets are gzip-compressed
- Physical-unit scaling is applied from the WFMPRE preamble: `volts = (raw - YOFF) * YMULT + YZERO`
- Programmer reference: Tektronix MSO4000/DPO4000 Series Programmer Manual (077-0248-01)
