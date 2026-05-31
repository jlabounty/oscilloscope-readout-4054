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

import argparse
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

try:
    import yaml
except ImportError:
    yaml = None   # only required when --config / --example-config is used


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


def fetch_channel_meta(scope: pyvisa.Resource, channel: str) -> dict:
    """Query preamble + per-channel display settings for one channel.

    Also configures DATA:SOURCE, DATA:ENCDG RIBINARY, and DATA:WIDTH 1
    (8-bit transfer, matching the DPO4054's native ADC resolution — using
    16-bit would just zero-pad each sample and double the CURVE? transfer
    time for no information gain).

    The returned dict can be cached and passed to fetch_channel via
    cached_meta=... to skip the ~15 SCPI round-trips this function makes,
    valid for as long as the scope's vertical/horizontal/trigger config
    doesn't change.
    """
    scope.write(f"DATA:SOURCE {channel}")
    scope.write("DATA:ENCDG RIBINARY")
    scope.write("DATA:WIDTH 1")
    meta = get_preamble(scope)
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
    return meta


def fetch_channel(
    scope: pyvisa.Resource,
    channel: str,
    pre_samples: int | None = 1000,
    post_samples: int | None = 1000,
    cached_meta: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Capture a single channel waveform.

    Parameters
    ----------
    pre_samples  : samples to keep before the trigger (None = full record)
    post_samples : samples to keep after the trigger  (None = full record)
    cached_meta  : preamble + channel settings from a prior call to
                   fetch_channel_meta; when supplied, skips the per-channel
                   metadata queries (large speedup for multi-capture sessions)
    Returns
    -------
    time_s   : 1-D array of time values in seconds
    volts    : 1-D array of voltage values in volts
    meta     : dict of preamble + per-channel display settings
    """
    if cached_meta is None:
        meta = fetch_channel_meta(scope, channel)
    else:
        meta = dict(cached_meta)            # copy so callers can mutate freely
        scope.write(f"DATA:SOURCE {channel}")

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

    # Read raw binary curve (signed 8-bit; matches DATA:WIDTH 1 set above)
    raw_bytes = scope.query_binary_values(
        "CURVE?", datatype="b", container=np.ndarray
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
    "setup":          "SET?",           # full instrument configuration blob
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


# ── Instant measurements ──────────────────────────────────────────────────────

_INSTANT_MEAS = [
    "AMPLITUDE", "HIGH", "LOW", "MEAN", "RMS",
    "FREQUENCY", "PERIOD", "RISETIME", "FALLTIME",
]

def get_channel_measurements(scope: pyvisa.Resource, channel: str) -> dict:
    """Query scope-computed immediate measurements for one channel.

    Returns a dict of {meas_<type>: float} for any measurement the scope
    considers valid.  Tektronix uses 9.9E+37 as the "not available" sentinel
    (e.g. frequency on a non-periodic signal); those entries are omitted.
    """
    results: dict[str, float] = {}
    scope.write(f"MEASUREMENT:IMMEd:SOURCE1 {channel}")
    for mtype in _INSTANT_MEAS:
        scope.write(f"MEASUREMENT:IMMEd:TYPE {mtype}")
        try:
            v = float(scope.query("MEASUREMENT:IMMEd:VALUE?").strip())
            if v < 9e37:
                results[f"meas_{mtype.lower()}"] = v
        except Exception:
            pass
    return results


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
            state_copy = dict(scope_state)
            setup_blob = state_copy.pop("setup", None)
            for k, v in state_copy.items():
                try:
                    cap_grp.attrs[k] = v
                except Exception:
                    pass
            if setup_blob:
                # SET? blob is 10–40 KB — too large for an HDF5 attribute; store as
                # a 1-D byte array so it supports gzip (scalars don't).
                blob_bytes = setup_blob.encode() if isinstance(setup_blob, str) else setup_blob
                cap_grp.create_dataset(
                    "instrument_setup",
                    data=np.frombuffer(blob_bytes, dtype="uint8"),
                    compression="gzip",
                    compression_opts=6,
                    shuffle=True,
                )

        for ch_name, (time_s, volts, meta) in channels.items():
            ch_grp = cap_grp.create_group(ch_name)
            ds_t = ch_grp.create_dataset(
                "time_s",
                data=time_s,
                compression="gzip",
                compression_opts=6,
                shuffle=True,
            )
            ds_v = ch_grp.create_dataset(
                "volts",
                data=volts.astype("float32"),
                compression="gzip",
                compression_opts=6,
                shuffle=True,
            )
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


# ── Config-driven capture (YAML) ──────────────────────────────────────────────

VALID_CHANNELS = {"CH1", "CH2", "CH3", "CH4"}

DEFAULT_CONFIG: dict = {
    "output": {
        "prefix":    "waveforms",
        "filename":  None,          # explicit path overrides prefix+timestamp
        "data_dir":  "./data",
        "save_root": False,
        "save_screenshot_begin": True,
        "save_screenshot_end":   True,
    },
    "channels": ["CH1"],
    "trigger_window": {              # nanoseconds (matches the GUI)
        "pre_ns":  1000.0,
        "post_ns": 1000.0,
    },
    "capture": {
        "n_captures": 1,
        "wait_s":     0.0,
        "label":      "",
        "notes":      "",
    },
    "acquisition": {
        "mode":   "SAMPLE",          # SAMPLE | AVERAGE | HIRES | ENVELOPE | PEAKDETECT
        "numavg": 16,                # used only when mode == AVERAGE
    },
    "measurements": {
        "enabled": False,
    },
}

EXAMPLE_CONFIG_YAML = """\
# Tektronix DPO4054 Waveform Capture — Example Configuration
# -----------------------------------------------------------
#   python capture_waveforms.py --config this_file.yaml
#
# All fields mirror the controls in capture_gui.py.  Any field may be
# omitted; omitted fields fall back to the defaults baked into
# DEFAULT_CONFIG in capture_waveforms.py.

output:
  # Final filename is {data_dir}/{prefix}_{YYYYMMDD_HHMMSS}.h5
  # unless `filename` is set explicitly (which overrides prefix+timestamp).
  prefix:    waveforms
  filename:  null
  data_dir:  ./data
  save_root: false           # also write a .root file via uproot
  save_screenshot_begin: true  # save scope PNG before first capture ({prefix}_{ts}_begin.png)
  save_screenshot_end:   true  # save scope PNG after last capture  ({prefix}_{ts}_end.png)

# Channels to capture — any subset of CH1, CH2, CH3, CH4.
channels:
  - CH1

# Time window around the trigger, in nanoseconds.
# Set both to 0 to capture the full record length.
trigger_window:
  pre_ns:  1000.0
  post_ns: 1000.0

capture:
  n_captures: 1              # number of triggered acquisitions
  wait_s:     0.0            # pause between successive captures (seconds)
  label:      ""             # HDF5 group label; "" → capture_HHMMSS
  notes:      ""             # free-text notes saved with each capture

acquisition:
  # SAMPLE | AVERAGE | HIRES | ENVELOPE | PEAKDETECT
  mode:   SAMPLE
  numavg: 16                 # only used when mode == AVERAGE

measurements:
  # Query scope-side measurements (AMPLITUDE, MEAN, RMS, FREQUENCY, …) and
  # store them as channel HDF5 attributes. Adds ~1–2 s per capture.
  enabled: false
"""


def _merge_config(default: dict, user: dict) -> dict:
    """Deep-merge *user* into *default* (user wins). Returns a new dict."""
    out = dict(default)
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_config(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and merge it with DEFAULT_CONFIG."""
    if yaml is None:
        raise ImportError("YAML config requires PyYAML: pip install pyyaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Top level of {path} must be a YAML mapping, got {type(user_cfg).__name__}")
    return _merge_config(DEFAULT_CONFIG, user_cfg)


def write_example_config(path: str | Path) -> Path:
    """Write the example YAML to *path* (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG_YAML)
    return path


def resolve_output_path(cfg: dict) -> Path:
    """Compute the HDF5 output file path from cfg['output']."""
    out = cfg["output"]
    if out.get("filename"):
        path = Path(out["filename"])
    else:
        prefix = out.get("prefix") or "waveforms"
        data_dir = Path(out.get("data_dir") or "./data")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = data_dir / f"{prefix}_{ts}.h5"
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        path = path.with_suffix(".h5")
    return path


def save_screenshot(scope: "pyvisa.Resource", path: Path) -> None:
    """Save a PNG screenshot from the scope's display to *path*.

    Uses HARDCOPY START with FORMAT PNG.  Loops on read_raw() until the PNG
    IEND trailer is seen (the DPO4054 sends the image in ~20 KB chunks).
    """
    _PNG_END = b'IEND\xaeB`\x82'
    scope.write("HARDCOPY:FORMAT PNG")
    scope.write("HARDCOPY:INKSAVER OFF")
    old_timeout = scope.timeout
    scope.timeout = 15_000
    scope.write("HARDCOPY START")
    raw = b''
    try:
        while _PNG_END not in raw[-12:]:
            try:
                raw += scope.read_raw()
                scope.timeout = 3_000
            except Exception:
                break
    finally:
        scope.timeout = old_timeout
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    print(f"  Screenshot → {path}")


def _wait_for_acquisition(scope: "pyvisa.Resource", label: str) -> None:
    """Poll ACQUIRE:STATE? until the scope auto-stops after one sequence."""
    t_start = time.monotonic()
    next_msg = t_start + 1.0
    while True:
        try:
            state = int(scope.query("ACQUIRE:STATE?").strip())
        except Exception:
            state = 0
        if state == 0:
            return
        now = time.monotonic()
        if now >= next_msg:
            print(f"  {label} — waiting for trigger… ({now - t_start:.1f}s)",
                  end="\r", flush=True)
            next_msg = now + 1.0
        time.sleep(0.05)


def run_capture(config: dict, scope: "pyvisa.Resource | None" = None) -> Path:
    """Run a non-interactive capture session driven by a config dict.

    Mirrors the GUI's capture flow (capture_gui.py:_capture_worker):

      1. Apply ACQUIRE:MODE (+ NUMAVG if AVERAGE)
      2. Switch to ACQUIRE:STOPAFTER SEQUENCE so each iteration is a single
         fresh trigger
      3. Cache per-channel preamble + display settings once
      4. For each of n_captures:
           - arm acquisition, wait for it to complete
           - read every selected channel (full record), optionally query
             scope measurements, then slice to ±pre_ns / +post_ns window
           - save to HDF5 (and ROOT, if enabled) and append a TSV log row

    Parameters
    ----------
    config : dict
        Same structure as DEFAULT_CONFIG / the example YAML.  Missing keys
        fall back to defaults.
    scope : pyvisa.Resource, optional
        Open VISA resource.  When None, the function connects to the first
        Tektronix USB scope it finds and closes the session on exit.

    Returns
    -------
    Path
        The HDF5 file written (existing on disk).
    """
    cfg = _merge_config(DEFAULT_CONFIG, config or {})

    channels   = [c.upper() for c in cfg["channels"]]
    pre_ns     = float(cfg["trigger_window"]["pre_ns"])
    post_ns    = float(cfg["trigger_window"]["post_ns"])
    n_caps     = int(cfg["capture"]["n_captures"])
    wait_s     = float(cfg["capture"]["wait_s"])
    label      = cfg["capture"]["label"] or datetime.now().strftime("capture_%H%M%S")
    notes      = cfg["capture"]["notes"] or ""
    save_root_        = bool(cfg["output"]["save_root"])
    screenshot_begin  = bool(cfg["output"]["save_screenshot_begin"])
    screenshot_end    = bool(cfg["output"]["save_screenshot_end"])
    acq_mode   = str(cfg["acquisition"]["mode"]).upper()
    acq_numavg = int(cfg["acquisition"]["numavg"])
    do_measure = bool(cfg["measurements"]["enabled"])

    bad = [c for c in channels if c not in VALID_CHANNELS]
    if bad:
        raise ValueError(f"Unrecognised channel(s): {bad}. Must be one of {sorted(VALID_CHANNELS)}.")
    if not channels:
        raise ValueError("config['channels'] must list at least one channel.")
    if n_caps < 1:
        raise ValueError(f"n_captures must be >= 1, got {n_caps}.")

    full_record = (pre_ns == 0 and post_ns == 0)
    filepath = resolve_output_path(cfg)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output  → {filepath}")
    print(f"Channels: {', '.join(channels)}  |  mode: {acq_mode}"
          f"{' (numavg=' + str(acq_numavg) + ')' if acq_mode == 'AVERAGE' else ''}"
          f"  |  window: {'full record' if full_record else f'-{pre_ns:g} / +{post_ns:g} ns'}"
          f"  |  n_captures: {n_caps}")

    own_scope = scope is None
    if own_scope:
        rm = pyvisa.ResourceManager()
        resource_str = find_scope(rm)
        scope = connect(resource_str)

    try:
        if screenshot_begin:
            try:
                save_screenshot(scope, filepath.with_name(f"{filepath.stem}_begin.png"))
            except Exception as e:
                print(f"  Screenshot (begin) failed: {e}")

        scope.write(f"ACQUIRE:MODE {acq_mode}")
        if acq_mode == "AVERAGE":
            scope.write(f"ACQUIRE:NUMAVG {acq_numavg}")

        try:
            prev_stopafter = scope.query("ACQUIRE:STOPAFTER?").strip()
        except Exception:
            prev_stopafter = "RUNSTOP"
        scope.write("ACQUIRE:STOPAFTER SEQUENCE")

        channel_meta_cache: dict[str, dict] = {}
        for ch in channels:
            try:
                channel_meta_cache[ch] = fetch_channel_meta(scope, ch)
            except Exception:
                channel_meta_cache[ch] = {}

        try:
            for i in range(n_caps):
                if i > 0 and wait_s > 0:
                    time.sleep(wait_s)

                scope.write("ACQUIRE:STATE RUN")
                capture_label = f"{label}_{i+1:03d}" if n_caps > 1 else label
                _wait_for_acquisition(scope, f"Capture {i+1}/{n_caps}")
                print(f"\nCapture {i+1}/{n_caps}: {capture_label}  ({', '.join(channels)})")

                scope_state = get_scope_state(scope, channels)
                captured: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
                for ch in channels:
                    print(f"  {ch} … ", end="", flush=True)
                    try:
                        time_s, volts, meta = fetch_channel(
                            scope, ch, None, None,
                            cached_meta=channel_meta_cache.get(ch),
                        )
                        if do_measure:
                            meta.update(get_channel_measurements(scope, ch))
                        if not full_record:
                            mask   = (time_s >= -pre_ns * 1e-9) & (time_s <= post_ns * 1e-9)
                            time_s = time_s[mask]
                            volts  = volts[mask]
                        captured[ch] = (time_s, volts, meta)
                        span = (time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0
                        print(f"{len(volts):,} pts  |  {_fmt_duration(span)} window")
                    except Exception as e:
                        print(f"FAILED ({e})")

                if captured:
                    save_hdf5(filepath, captured, label=capture_label,
                              notes=notes, scope_state=scope_state)
                    if save_root_:
                        save_root(filepath, captured, label=capture_label,
                                  scope_state=scope_state)
                    pre_log  = None if full_record else pre_ns
                    post_log = None if full_record else post_ns
                    log_capture_tsv(filepath, capture_label, channels,
                                    pre_log, post_log, notes,
                                    scope_state=scope_state)
                else:
                    print("  No data captured — nothing saved.")
        finally:
            scope.write("ACQUIRE:STATE STOP")
            scope.write(f"ACQUIRE:STOPAFTER {prev_stopafter}")
            scope.write("ACQUIRE:STATE RUN")
            if screenshot_end:
                try:
                    save_screenshot(scope, filepath.with_name(f"{filepath.stem}_end.png"))
                except Exception as e:
                    print(f"  Screenshot (end) failed: {e}")
    finally:
        if own_scope:
            try:
                scope.close()
            except Exception:
                pass

    return filepath


def run_capture_from_yaml(path: str | Path,
                          scope: "pyvisa.Resource | None" = None) -> Path:
    """Convenience wrapper: load a YAML file and call run_capture()."""
    return run_capture(load_config(path), scope=scope)


# ── CLI helpers ────────────────────────────────────────────────────────────────

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

def interactive_main():
    """Interactive prompt-driven capture session (no YAML config)."""
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

            # Cache per-channel meta once per batch — skips ~15 SCPI queries
            # per channel per capture.
            channel_meta_cache: dict[str, dict] = {}
            for ch in channels:
                try:
                    channel_meta_cache[ch] = fetch_channel_meta(scope, ch)
                except Exception:
                    channel_meta_cache[ch] = {}

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
                        time_s, volts, meta = fetch_channel(
                            scope, ch, pre, post,
                            cached_meta=channel_meta_cache.get(ch),
                        )
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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Usage
    -----
        python capture_waveforms.py                          # interactive prompts
        python capture_waveforms.py --config run.yaml        # headless, YAML-driven
        python capture_waveforms.py --example-config foo.yaml  # write template & exit
    """
    parser = argparse.ArgumentParser(
        prog="capture_waveforms.py",
        description="Capture waveforms from a Tektronix DPO4054 oscilloscope.",
    )
    parser.add_argument(
        "-c", "--config", metavar="PATH",
        help="YAML config file describing the capture session (headless mode).",
    )
    parser.add_argument(
        "--example-config", metavar="PATH",
        help="Write a default YAML config to PATH and exit.",
    )
    args = parser.parse_args(argv)

    if args.example_config:
        out = write_example_config(args.example_config)
        print(f"Wrote example config → {out}")
        return

    if args.config:
        cfg = load_config(args.config)
        path = run_capture(cfg)
        print(f"\nDone. Output: {path}")
        return

    interactive_main()


if __name__ == "__main__":
    main()


# =============================================================================
# QUICK-START README
# =============================================================================
#
# Installation
# ------------
#   pip install pyvisa pyvisa-py numpy h5py pyusb pyyaml
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
#   python capture_waveforms.py                            # interactive
#   python capture_waveforms.py --config run.yaml          # headless
#   python capture_waveforms.py --example-config run.yaml  # write template
#
# Programmatic use
#   from capture_waveforms import run_capture, run_capture_from_yaml
#   run_capture_from_yaml("run.yaml")
#   run_capture({"channels": ["CH1"], "capture": {"n_captures": 10}})
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