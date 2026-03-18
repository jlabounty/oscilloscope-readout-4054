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

_PREAMBLE_FIELDS = [
    "BYT_NR", "BIT_NR", "ENCDG", "BN_FMT", "BYT_OR",
    "NR_PT",  "WFID",   "PT_FMT", "XUNIT",  "XINCR",
    "PT_OFF", "XZERO",  "YMULT",  "YOFF",   "YZERO",  "YUNIT",
]

def get_preamble(scope: pyvisa.Resource) -> dict:
    """Fetch the WFMPRE preamble for the currently selected source.

    Queries each field individually (e.g. WFMPRE:YMULT?) rather than
    parsing the bulk WFMPRE? response, which on some DPO4054 firmware
    versions returns only the WFID string instead of the full record.
    """
    preamble = {}
    for key in _PREAMBLE_FIELDS:
        try:
            preamble[key] = scope.query(f"WFMPRE:{key}?").strip().strip('"')
        except Exception:
            pass
    return preamble


def fetch_channel(
    scope: pyvisa.Resource,
    channel: str,
    pre_samples: int | None = 1000,
    post_samples: int | None = 1000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Capture a single channel waveform.

    Parameters
    ----------
    pre_samples  : samples to keep before the trigger (None = full record)
    post_samples : samples to keep after the trigger  (None = full record)

    Returns
    -------
    time_s   : 1-D array of time values in seconds
    volts    : 1-D array of voltage values in volts
    meta     : dict of preamble + per-channel display settings
    """
    scope.write(f"DATA:SOURCE {channel}")
    scope.write("DATA:ENCDG RIBINARY")   # signed binary — faster than ASCII
    scope.write("DATA:WIDTH 2")          # 2 bytes per sample → full 16-bit res

    # Fetch preamble first so we know the trigger position (PT_OFF) and
    # record length (NR_PT) before setting the transfer window.
    meta = get_preamble(scope)

    # Per-channel display settings (not in WFMPRE)
    for key, cmd in [
        ("CH_COUPLING",  f"{channel}:COUPling?"),
        ("CH_SCALE",     f"{channel}:SCAle?"),
        ("CH_BANDWIDTH", f"{channel}:BANdwidth?"),
        ("CH_PROBE",     f"{channel}:PRObe?"),
    ]:
        try:
            meta[key] = scope.query(cmd).strip()
        except Exception:
            pass

    pt_off = int(float(meta.get("PT_OFF", 0)))
    nr_pt  = int(meta.get("NR_PT", 0)) or None   # 0 → unknown

    if pre_samples is None and post_samples is None:
        # Full record
        scope.write("DATA:START 1")
        scope.write("DATA:STOP 1E10")
        start_0idx = 0
    else:
        pre  = pre_samples  if pre_samples  is not None else pt_off
        post = post_samples if post_samples is not None else (nr_pt - pt_off if nr_pt else int(1e10))
        start_1idx = max(1, pt_off - pre + 1)
        stop_1idx  = pt_off + post
        if nr_pt:
            stop_1idx = min(nr_pt, stop_1idx)
        scope.write(f"DATA:START {start_1idx}")
        scope.write(f"DATA:STOP {stop_1idx}")
        start_0idx = start_1idx - 1   # convert to 0-based full-record index

    # Read raw binary curve
    raw_bytes = scope.query_binary_values(
        "CURVE?", datatype="h", is_big_endian=True, container=np.ndarray
    )

    # Scale to physical units
    ymult = float(meta.get("YMULT",  1.0))
    yoff  = float(meta.get("YOFF",   0.0))
    yzero = float(meta.get("YZERO",  0.0))
    xincr = float(meta.get("XINCR",  1e-9))
    xzero = float(meta.get("XZERO",  0.0))
    n_pts = len(raw_bytes)

    volts  = (raw_bytes.astype(float) - yoff) * ymult + yzero
    # Offset np.arange by start_0idx so times are correct for windowed captures
    time_s = xzero + (start_0idx + np.arange(n_pts) - pt_off) * xincr

    return time_s, volts, meta


# ── Capture-level scope state ──────────────────────────────────────────────────

# SCPI queries for settings not present in WFMPRE, keyed by the TSV/HDF5 field name
SCOPE_STATE_QUERIES: dict[str, str] = {
    "sample_rate_hz": "HORizontal:SAMPLERate?",
    "h_scale_s_div":  "HORizontal:SCAle?",
    "trig_type":      "TRIGger:MAIn:TYPE?",
    "trig_level_v":   "TRIGger:MAIn:LEVEL?",
    "trig_source":    "TRIGger:MAIn:EDGE:SOUrce?",
    "trig_slope":     "TRIGger:MAIn:EDGE:SLOpe?",
    "trig_freq_hz":   "TRIGger:FREQuency?",
    "acq_mode":       "ACQuire:MODe?",
    "acq_numavg":     "ACQuire:NUMAVg?",
}


def get_scope_state(scope: pyvisa.Resource, channels: list[str]) -> dict:
    """Query capture-level scope settings not present in WFMPRE.

    Queries horizontal settings (sample rate, scale), trigger configuration
    (type, source, level, slope, frequency/rate), and acquire mode.
    Unsupported queries are silently omitted so the function is safe to call
    on any firmware version.

    Returns a flat dict of str → str.
    """
    state: dict[str, str] = {}
    for key, cmd in SCOPE_STATE_QUERIES.items():
        try:
            state[key] = scope.query(cmd).strip()
        except Exception:
            pass
    return state


# ── HDF5 saving ───────────────────────────────────────────────────────────────

def save_hdf5(
    filepath: Path,
    channels: dict[str, tuple[np.ndarray, np.ndarray, dict]],
    label: str,
    notes: str = "",
    scope_state: dict | None = None,
) -> None:
    """
    Save one or more channel captures to an HDF5 file.

    File layout
    -----------
    /  (root attrs)  label, timestamp, notes
    /<capture_label>/  (attrs: timestamp, notes, + all scope_state fields)
        /CH1/
             time_s      dataset
             volts       dataset
             attrs       all WFMPRE preamble fields + CH_* display settings
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

        # Capture-level scope state (horizontal, trigger, acquire settings)
        if scope_state:
            for k, v in scope_state.items():
                try:
                    cap_grp.attrs[k] = v
                except Exception:
                    pass

        for ch_name, (time_s, volts, meta) in channels.items():
            ch_grp = cap_grp.create_group(ch_name)
            ds_t = ch_grp.create_dataset("time_s", data=time_s, compression="gzip")
            ds_v = ch_grp.create_dataset("volts",  data=volts,  compression="gzip")
            ds_t.attrs["units"] = "s"
            ds_v.attrs["units"] = meta.get("YUNIT", "V")
            # Store WFMPRE preamble + per-channel display settings (CH_* keys)
            for k, v in meta.items():
                try:
                    ch_grp.attrs[k] = v
                except Exception:
                    pass  # skip unparseable fields

    print(f"  Saved → {filepath}  (group: {grp_name})")


# ── ROOT output ───────────────────────────────────────────────────────────────

# Numeric preamble fields written as per-sample scalar branches in the TTree.
# Repeated constants compress to near-zero with ROOT's default ZLIB compression.
_ROOT_PREAMBLE_NUMERICS = ["XINCR", "YMULT", "YOFF", "YZERO", "XZERO", "PT_OFF", "NR_PT"]
_ROOT_STATE_NUMERICS    = ["sample_rate_hz", "h_scale_s_div", "trig_level_v", "trig_freq_hz"]


def save_root(
    filepath: Path,
    channels: dict[str, tuple[np.ndarray, np.ndarray, dict]],
    label: str,
    scope_state: dict | None = None,
) -> None:
    """Save waveform data to a ROOT file (via uproot) alongside the HDF5 output.

    File layout
    -----------
    Each channel is stored as a TTree at path ``<label>/<channel>`` inside the
    ROOT file.  Branches per TTree:

      time_s    float64[N]   — time axis in seconds
      volts     float64[N]   — voltage axis in volts
      xincr     float64[N]   — time per sample (s)        ⎫
      ymult     float64[N]   — ADC voltage scale           ⎪
      yoff      float64[N]   — ADC offset (ADC counts)     ⎬ WFMPRE scalars,
      yzero     float64[N]   — voltage zero                ⎪ repeated per sample
      xzero     float64[N]   — trigger time (s)            ⎪ (compress to ~0)
      pt_off    float64[N]   — trigger point index         ⎭
      nr_pt     float64[N]   — full record length (pts)
      sample_rate_hz  float64[N]  ⎫ scope_state fields
      h_scale_s_div   float64[N]  ⎪ (omitted when not
      trig_level_v    float64[N]  ⎪  available)
      trig_freq_hz    float64[N]  ⎭

    Appending
    ---------
    Uses ``uproot.update`` when the ROOT file already exists so that captures
    from successive calls within a session accumulate in one file, mirroring
    the HDF5 append behaviour.

    Raises
    ------
    ImportError if uproot is not installed.
    """
    try:
        import uproot
    except ImportError:
        raise ImportError("uproot is required for ROOT output: pip install uproot")

    root_path = filepath.with_suffix(".root")

    open_fn = uproot.update if root_path.exists() else uproot.recreate
    with open_fn(root_path) as rf:
        s = scope_state or {}
        for ch_name, (time_s, volts, meta) in channels.items():
            n = len(time_s)
            branches: dict[str, np.ndarray] = {
                "time_s": time_s,
                "volts":  volts,
            }
            # Numeric preamble scalars
            for key in _ROOT_PREAMBLE_NUMERICS:
                if key in meta:
                    try:
                        branches[key.lower()] = np.full(n, float(meta[key]), dtype=np.float64)
                    except (ValueError, TypeError):
                        pass
            # Numeric scope-state scalars
            for key in _ROOT_STATE_NUMERICS:
                if key in s:
                    try:
                        branches[key] = np.full(n, float(s[key]), dtype=np.float64)
                    except (ValueError, TypeError):
                        pass
            rf[f"{label}/{ch_name}"] = branches

    print(f"  ROOT   → {root_path}  (tree: {label}/<channel>)")


# ── Capture log ────────────────────────────────────────────────────────────────

TSV_COLUMNS = [
    "timestamp", "capture_label", "hdf5_file",
    "channels", "pre_samples", "post_samples", "notes",
    # Scope state — filled when a live scope connection is available
    "sample_rate_hz", "h_scale_s_div",
    "trig_type", "trig_source", "trig_level_v", "trig_slope", "trig_freq_hz",
    "acq_mode", "acq_numavg",
]

def log_capture_tsv(
    filepath: Path,
    label: str,
    channels: list[str],
    pre_samples: int | None,
    post_samples: int | None,
    notes: str = "",
    scope_state: dict | None = None,
) -> None:
    """Append one row to a TSV log file co-located with the HDF5 output.

    The TSV file has the same path as *filepath* but with a .tsv extension.
    A header row is written automatically the first time the file is created.
    Tab and newline characters in free-text fields are collapsed to spaces so
    each capture occupies exactly one line.

    scope_state keys used (all optional, written as empty string if absent):
        sample_rate_hz, h_scale_s_div, trig_type, trig_source, trig_level_v,
        trig_slope, trig_freq_hz, acq_mode, acq_numavg
    """
    tsv_path = filepath.with_suffix(".tsv")
    write_header = not tsv_path.exists()
    pre_str  = str(pre_samples)  if pre_samples  is not None else "full"
    post_str = str(post_samples) if post_samples is not None else "full"

    def _clean(s: str) -> str:
        return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")

    s = scope_state or {}
    row = "\t".join([
        datetime.now().isoformat(),
        label,
        str(filepath.resolve()),
        ",".join(channels),
        pre_str,
        post_str,
        _clean(notes),
        s.get("sample_rate_hz", ""),
        s.get("h_scale_s_div",  ""),
        s.get("trig_type",      ""),
        s.get("trig_source",    ""),
        s.get("trig_level_v",   ""),
        s.get("trig_slope",     ""),
        s.get("trig_freq_hz",   ""),
        s.get("acq_mode",       ""),
        s.get("acq_numavg",     ""),
    ])
    with open(tsv_path, "a", newline="", encoding="utf-8") as fh:
        if write_header:
            fh.write("\t".join(TSV_COLUMNS) + "\n")
        fh.write(row + "\n")
    print(f"  Log    → {tsv_path}")


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


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds with an appropriate SI prefix."""
    for threshold, scale, unit in [
        (1e-9, 1e12, "ps"),
        (1e-6, 1e9,  "ns"),
        (1e-3, 1e6,  "µs"),
        (1.0,  1e3,  "ms"),
    ]:
        if abs(seconds) < threshold:
            return f"{seconds * scale:.1f} {unit}"
    return f"{seconds:.3f} s"


def prompt_label() -> str:
    raw = input("Capture label (optional, e.g. 'run3_signal'): ").strip()
    return raw or datetime.now().strftime("capture_%H%M%S")


def prompt_notes() -> str:
    return input("Notes (optional): ").strip()


def prompt_n_captures() -> int:
    """Ask how many triggered captures to take (default 1)."""
    while True:
        raw = input("Number of captures [1]: ").strip()
        if not raw:
            return 1
        try:
            n = int(raw)
            if n >= 1:
                return n
        except ValueError:
            pass
        print("  Please enter a positive integer.")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(question + suffix).strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def prompt_window(
    default_pre: int = 1000,
    default_post: int = 1000,
) -> tuple[int | None, int | None]:
    """
    Ask how many samples before/after the trigger to capture.

    Enter two integers (pre post), or press Enter to keep the defaults.
    Enter 0 0 (or 'all') to capture the full record.

    Returns (pre_samples, post_samples); both None means full record.
    """
    while True:
        prompt = (
            f"Trigger window — pre post samples "
            f"[{default_pre} {default_post}] (0 0 = full record): "
        )
        raw = input(prompt).strip()
        if not raw:
            return default_pre, default_post
        if raw.lower() in {"all", "full"}:
            return None, None
        parts = raw.split()
        if len(parts) == 2:
            try:
                pre, post = int(parts[0]), int(parts[1])
                if pre == 0 and post == 0:
                    return None, None
                if pre >= 0 and post >= 0:
                    return pre, post
            except ValueError:
                pass
        print("  Enter two non-negative integers (e.g. '500 2000'), or press Enter for defaults.")


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

            channels   = prompt_channels()
            pre, post  = prompt_window()
            n_captures = prompt_n_captures()
            label      = prompt_label()
            notes      = prompt_notes()

            for i in range(1, n_captures + 1):
                if n_captures > 1:
                    capture_label = f"{label}_{i:03d}"
                    print(f"\nCapture {i}/{n_captures}  ({', '.join(channels)}) …")
                else:
                    capture_label = label
                    print(f"\nCapturing {', '.join(channels)} …")

                scope.write("ACQUIRE:STATE STOP")   # freeze memory — all channels from same trigger
                scope_state = get_scope_state(scope, channels)
                captured = {}
                for ch in channels:
                    print(f"  {ch} … ", end="", flush=True)
                    try:
                        time_s, volts, meta = fetch_channel(scope, ch, pre, post)
                        captured[ch] = (time_s, volts, meta)
                        n_pts = len(volts)
                        print(f"{n_pts:,} pts  |  {_fmt_duration(time_s[-1] - time_s[0])} window")
                    except Exception as e:
                        print(f"FAILED ({e})")

                if captured:
                    save_hdf5(session_file, captured, label=capture_label, notes=notes, scope_state=scope_state)
                    log_capture_tsv(session_file, capture_label, channels, pre, post, notes, scope_state=scope_state)
                else:
                    print("  No data captured — nothing saved.")
                scope.write("ACQUIRE:STATE RUN")     # re-arm for next capture

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