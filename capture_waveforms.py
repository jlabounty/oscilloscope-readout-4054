#!/usr/bin/env python3
"""
Tektronix DPO4054 Waveform Capture Script
==========================================
Captures waveforms from up to 4 channels over USB (USBTMC) and saves
them to HDF5 files with full scaling metadata.

Dependencies:
    pip install pyvisa pyvisa-py numpy h5py

On Linux, you may also need:
    pip install pyusb
    # and add udev rule for Tektronix USB access (see README at bottom)
"""

import sys
import time
import re
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pyvisa
except ImportError:
    sys.exit("Missing dependency: pip install pyvisa pyvisa-py")

try:
    import h5py
except ImportError:
    sys.exit("Missing dependency: pip install h5py")


# ── VISA / scope connection ────────────────────────────────────────────────────

TEKTRONIX_USB_ID = "0x0699"   # Tektronix USB vendor ID (for reference)

def find_scope(rm: pyvisa.ResourceManager) -> str:
    """Return the VISA resource string for the first Tektronix scope found."""
    resources = rm.list_resources()
    tek_resources = [r for r in resources if r.upper().startswith("USB")]

    if not tek_resources:
        print("\nNo USB instruments found. All visible resources:")
        for r in resources:
            print(f"  {r}")
        sys.exit(
            "\nCould not find scope. Check the USB cable and that the scope\n"
            "is set to USB Device in Utility > I/O > USB Network & PC."
        )

    if len(tek_resources) == 1:
        return tek_resources[0]

    print("\nMultiple USB instruments found:")
    for i, r in enumerate(tek_resources):
        print(f"  [{i}] {r}")
    idx = int(input("Select instrument index: ").strip())
    return tek_resources[idx]


def connect(resource_str: str) -> pyvisa.Resource:
    """Open and configure the VISA resource."""
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(resource_str)
    scope.timeout = 10_000          # 10 s — generous for slow USB transfers
    scope.read_termination = "\n"
    scope.write_termination = "\n"
    idn = scope.query("*IDN?").strip()
    print(f"\nConnected: {idn}")
    return scope


# ── Waveform acquisition ───────────────────────────────────────────────────────

PREAMBLE_KEYS = [
    "BYT_NR", "BIT_NR", "ENCDG", "BN_FMT", "BYT_OR",
    "NR_PT", "WFID", "PT_FMT", "XINCR", "PT_OFF",
    "XZERO", "XUNIT", "YMULT", "YOFF", "YZERO", "YUNIT"
]

def get_preamble(scope: pyvisa.Resource) -> dict:
    """Fetch and parse the WFMPRE preamble for the currently selected source."""
    raw = scope.query("WFMPRE?").strip()
    preamble = {}
    # Preamble fields are colon-separated key:value pairs, comma-delimited
    # e.g.  BYT_NR 1;BIT_NR 8;ENCDG RIB; ...  (semicolon-separated on DPO4k)
    for field in re.split(r"[;,]", raw):
        field = field.strip()
        if " " in field:
            k, v = field.split(" ", 1)
            preamble[k.upper()] = v.strip('"')
    return preamble


def fetch_channel(scope: pyvisa.Resource, channel: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Capture a single channel waveform.

    Returns
    -------
    time_s   : 1-D array of time values in seconds
    volts    : 1-D array of voltage values in volts
    meta     : dict of preamble metadata
    """
    scope.write(f"DATA:SOURCE {channel}")
    scope.write("DATA:ENCDG RIBINARY")   # signed binary — faster than ASCII
    scope.write("DATA:WIDTH 2")          # 2 bytes per sample → full 16-bit res
    scope.write("DATA:START 1")
    scope.write("DATA:STOP 1E10")        # capture entire record

    meta = get_preamble(scope)

    # Read raw binary curve
    raw_bytes = scope.query_binary_values(
        "CURVE?", datatype="h", is_big_endian=True, container=np.ndarray
    )

    # Scale to physical units
    ymult  = float(meta.get("YMULT",  1.0))
    yoff   = float(meta.get("YOFF",   0.0))
    yzero  = float(meta.get("YZERO",  0.0))
    xincr  = float(meta.get("XINCR",  1e-9))
    xzero  = float(meta.get("XZERO",  0.0))
    pt_off = float(meta.get("PT_OFF", 0.0))
    n_pts  = len(raw_bytes)

    volts  = (raw_bytes.astype(float) - yoff) * ymult + yzero
    time_s = xzero + (np.arange(n_pts) - pt_off) * xincr

    return time_s, volts, meta


# ── HDF5 saving ───────────────────────────────────────────────────────────────

def save_hdf5(
    filepath: Path,
    channels: dict[str, tuple[np.ndarray, np.ndarray, dict]],
    label: str,
    notes: str = "",
) -> None:
    """
    Save one or more channel captures to an HDF5 file.

    File layout
    -----------
    /  (root attrs)  label, timestamp, notes
    /CH1/
         time_s      dataset
         volts       dataset
         attrs       all WFMPRE preamble fields
    /CH2/ ...
    """
    mode = "a" if filepath.exists() else "w"
    with h5py.File(filepath, mode) as f:
        # Root metadata
        if "created" not in f.attrs:
            f.attrs["created"] = datetime.now().isoformat()
        f.attrs["last_updated"] = datetime.now().isoformat()

        grp_name = label if label else datetime.now().strftime("capture_%H%M%S")
        if grp_name in f:
            grp_name += f"_{int(time.time())}"

        cap_grp = f.create_group(grp_name)
        cap_grp.attrs["timestamp"] = datetime.now().isoformat()
        cap_grp.attrs["notes"] = notes

        for ch_name, (time_s, volts, meta) in channels.items():
            ch_grp = cap_grp.create_group(ch_name)
            ds_t = ch_grp.create_dataset("time_s", data=time_s, compression="gzip")
            ds_v = ch_grp.create_dataset("volts",  data=volts,  compression="gzip")
            ds_t.attrs["units"] = "s"
            ds_v.attrs["units"] = meta.get("YUNIT", "V")
            # Store full preamble
            for k, v in meta.items():
                try:
                    ch_grp.attrs[k] = v
                except Exception:
                    pass  # skip unparseable fields

    print(f"  Saved → {filepath}  (group: {grp_name})")


# ── CLI helpers ────────────────────────────────────────────────────────────────

VALID_CHANNELS = {"CH1", "CH2", "CH3", "CH4"}

def prompt_channels() -> list[str]:
    """Ask the user which channels to capture."""
    while True:
        raw = input("\nChannels to capture (e.g. 1 2 3 4  or  1 3): ").strip()
        channels = []
        for tok in raw.split():
            tok = tok.upper().lstrip("C").lstrip("H")
            ch = f"CH{tok}"
            if ch in VALID_CHANNELS:
                channels.append(ch)
            else:
                print(f"  Ignoring unrecognised channel: {tok}")
        if channels:
            return sorted(set(channels))
        print("  Please enter at least one valid channel (1–4).")


def prompt_filepath() -> Path:
    """Ask for an output file path, defaulting to timestamped name."""
    default = f"./data/waveforms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    raw = input(f"\nOutput file [{default}]: ").strip()
    path = Path(raw) if raw else Path(default)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        path = path.with_suffix(".h5")
    return path


def prompt_label() -> str:
    raw = input("Capture label (optional, e.g. 'run3_signal'): ").strip()
    return raw or datetime.now().strftime("capture_%H%M%S")


def prompt_notes() -> str:
    return input("Notes (optional): ").strip()


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(question + suffix).strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Tektronix DPO4054 — Waveform Capture Utility")
    print("=" * 60)

    rm = pyvisa.ResourceManager()

    print("\nScanning for USB instruments...")
    resource_str = find_scope(rm)
    scope = connect(resource_str)

    # Persistent file across session (user can change per capture)
    session_file: Path | None = None

    try:
        while True:
            print("\n" + "─" * 40)

            # Output file
            if session_file is None or prompt_yes_no("Change output file?", default=False):
                session_file = prompt_filepath()

            channels  = prompt_channels()
            label     = prompt_label()
            notes     = prompt_notes()

            print(f"\nCapturing {', '.join(channels)} …")
            captured = {}
            for ch in channels:
                print(f"  {ch} … ", end="", flush=True)
                try:
                    time_s, volts, meta = fetch_channel(scope, ch)
                    captured[ch] = (time_s, volts, meta)
                    n_pts = len(volts)
                    duration_us = (time_s[-1] - time_s[0]) * 1e6
                    print(f"{n_pts:,} pts  |  {duration_us:.1f} µs window")
                except Exception as e:
                    print(f"FAILED ({e})")

            if captured:
                save_hdf5(session_file, captured, label=label, notes=notes)
            else:
                print("  No data captured — nothing saved.")

            if not prompt_yes_no("\nCapture again?", default=True):
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        scope.close()
        print("Scope connection closed. Goodbye.")


if __name__ == "__main__":
    main()


# =============================================================================
# QUICK-START README
# =============================================================================
#
# Installation
# ------------
#   pip install pyvisa pyvisa-py numpy h5py pyusb
#
# Linux udev rule (run once as root so non-root users can access the scope):
#   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0699", MODE="0666"' \
#       | sudo tee /etc/udev/rules.d/99-tektronix.rules
#   sudo udevadm control --reload-rules && sudo udevadm trigger
#
# Scope setup
#   Utility > I/O > USB Network & PC  →  set to "USB Device"
#
# Running
#   python capture_waveforms.py
#
# Reading the HDF5 output in Python
# ----------------------------------
#   import h5py, numpy as np
#
#   with h5py.File("waveforms_20250318_142301.h5", "r") as f:
#       grp = f["run3_signal"]          # capture label you entered
#       t = grp["CH1"]["time_s"][:]
#       v = grp["CH1"]["volts"][:]
#       xunit = grp["CH1"].attrs["XUNIT"]   # 's'
#       yunit = grp["CH1"]["volts"].attrs["units"]  # 'V'
#
# HDF5 file layout
# ----------------
#   /  (attrs: created, last_updated)
#   /<capture_label>/  (attrs: timestamp, notes)
#       /CH1/
#           time_s   [N]  float64  seconds
#           volts    [N]  float64  volts
#           attrs:   all WFMPRE preamble fields (XINCR, YMULT, …)
#       /CH2/ …
#
# Multiple captures in one session are stored as separate groups in the
# same file, so you end up with a clean run log in a single .h5 file.
# =============================================================================