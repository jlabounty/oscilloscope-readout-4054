#!/usr/bin/env python3
"""
Tektronix DPO4054 — Interactive Waveform Capture GUI
=====================================================
Tkinter GUI wrapping capture_waveforms.py. Exposes all capture parameters
and plots waveforms live as each channel is digitized.

Run:
    python capture_gui.py
"""

import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

try:
    import pyvisa
except ImportError:
    import sys; sys.exit("Missing dependency: pip install pyvisa pyvisa-py")

# Import core functions from the existing capture script
from capture_waveforms import find_scope, connect, fetch_channel, save_hdf5, save_root, log_capture_tsv, get_scope_state


# ── Helpers ────────────────────────────────────────────────────────────────────

def si_prefix(value: float) -> tuple[float, str]:
    """Return (scale_factor, prefix) so that value * scale_factor is O(1).
    Copied from plot_waveforms.ipynb."""
    abs_val = abs(value)
    if abs_val == 0:
        return 1.0, ""
    for scale, prefix in [
        (1e9,  "n"),
        (1e6,  "µ"),
        (1e3,  "m"),
        (1.0,  ""),
        (1e-3, "k"),
        (1e-6, "M"),
    ]:
        if abs_val * scale >= 1.0:
            return scale, prefix
    return 1e-6, "M"


def viridis_color(idx: int, total: int) -> tuple:
    """Return an RGBA color from the viridis colormap."""
    cmap = plt.get_cmap("viridis")
    return cmap(idx / max(total - 1, 1))


def validate_inputs(
    file_var: tk.StringVar,
    ch_vars: dict[str, tk.BooleanVar],
    pre_var: tk.StringVar,
    post_var: tk.StringVar,
    n_var: tk.StringVar,
) -> tuple[bool, str]:
    """Return (True, '') on success, or (False, error_message)."""
    selected = [ch for ch, v in ch_vars.items() if v.get()]
    if not selected:
        return False, "Select at least one channel."

    filepath = file_var.get().strip()
    if not filepath:
        return False, "Specify an output file."
    p = Path(filepath)
    if p.suffix.lower() not in {".h5", ".hdf5"}:
        return False, "Output file must have .h5 or .hdf5 extension."

    try:
        pre = int(pre_var.get())
        post = int(post_var.get())
        if pre < 0 or post < 0:
            raise ValueError
    except ValueError:
        return False, "Pre/post samples must be non-negative integers."

    try:
        n = int(n_var.get())
        if n < 1:
            raise ValueError
    except ValueError:
        return False, "Number of captures must be a positive integer."

    return True, ""


def parse_wait(wait_var: tk.StringVar) -> float:
    """Return wait seconds (>=0), or 0.0 on invalid/empty input."""
    try:
        v = float(wait_var.get())
        return max(v, 0.0)
    except ValueError:
        return 0.0


# ── Main application ────────────────────────────────────────────────────────────

CHANNEL_NAMES = ["CH1", "CH2", "CH3", "CH4"]
POLL_MS = 50


class WaveformApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.scope = None
        self.result_queue: queue.Queue = queue.Queue()
        self._capture_running = False
        self._stop_event = threading.Event()
        self._axes_map: dict[str, plt.Axes] = {}
        self._line_map: dict[str, plt.Line2D] = {}
        self._n_captures_total = 1
        self._hist_data: dict[str, dict[str, list]] = {}   # ch → {integral: [], amplitude: []}
        self._hist_axes: dict[str, dict[str, plt.Axes]] = {}  # ch → {integral: ax, amplitude: ax}

        # Tk variables
        self.file_var  = tk.StringVar(value=f"./data/waveforms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
        self.ch_vars   = {ch: tk.BooleanVar(value=(ch == "CH1")) for ch in CHANNEL_NAMES}
        self.pre_var   = tk.StringVar(value="1000")
        self.post_var  = tk.StringVar(value="1000")
        self.n_var     = tk.StringVar(value="1")
        self.wait_var  = tk.StringVar(value="0")
        self.label_var = tk.StringVar(value="")
        self.notes_var = tk.StringVar(value="")
        self.root_var  = tk.BooleanVar(value=False)

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Status bar (top, full width)
        status_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_frame.grid(row=0, column=0, sticky="ew")
        self.status_label = tk.Label(
            status_frame, text="Not connected", anchor="w", padx=6, pady=3
        )
        self.status_label.pack(fill=tk.X)

        # Main area
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=1, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self._build_left_panel(main_frame)
        self._build_right_panel(main_frame)

    def _build_left_panel(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, width=290, padx=8, pady=8)
        left.grid(row=0, column=0, sticky="ns")
        left.grid_propagate(False)

        def section(title: str) -> tk.LabelFrame:
            f = tk.LabelFrame(left, text=title, padx=6, pady=6)
            f.pack(fill=tk.X, pady=(0, 6))
            return f

        def sep() -> None:
            ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=2)

        # -- CONNECTION --
        conn_frame = section("Connection")
        conn_row = tk.Frame(conn_frame)
        conn_row.pack(fill=tk.X)
        self.conn_indicator = tk.Label(conn_row, text="●", fg="red", font=("", 14))
        self.conn_indicator.pack(side=tk.LEFT, padx=(0, 6))
        self.connect_btn = tk.Button(
            conn_row, text="Connect to Scope", command=self._on_connect
        )
        self.connect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        disc_row = tk.Frame(conn_frame)
        disc_row.pack(fill=tk.X, pady=(4, 0))
        self.disconnect_btn = tk.Button(
            disc_row, text="Disconnect", command=self._on_disconnect,
            state=tk.DISABLED,
        )
        self.disconnect_btn.pack(fill=tk.X)

        sep()

        # -- OUTPUT FILE --
        file_frame = section("Output File")
        tk.Entry(file_frame, textvariable=self.file_var).pack(fill=tk.X, pady=(0, 4))
        btn_row = tk.Frame(file_frame)
        btn_row.pack(fill=tk.X)
        tk.Button(btn_row, text="New filename", command=self._on_new_filename).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Browse…", command=self._on_browse).pack(side=tk.RIGHT)
        tk.Checkbutton(
            file_frame, text="Also save .root (uproot)", variable=self.root_var,
        ).pack(anchor="w", pady=(6, 0))

        sep()

        # -- CHANNELS --
        ch_frame = section("Channels")
        grid = tk.Frame(ch_frame)
        grid.pack()
        for i, ch in enumerate(CHANNEL_NAMES):
            tk.Checkbutton(grid, text=ch, variable=self.ch_vars[ch]).grid(
                row=i // 2, column=i % 2, sticky="w", padx=8, pady=2
            )

        sep()

        # -- TRIGGER WINDOW --
        trig_frame = section("Trigger Window")
        for lbl, var in [("Pre-trigger samples", self.pre_var), ("Post-trigger samples", self.post_var)]:
            row = tk.Frame(trig_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=lbl, width=20, anchor="w").pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, width=9).pack(side=tk.RIGHT)
        tk.Label(trig_frame, text="(0  0 = full record)", fg="gray", font=("", 8)).pack(anchor="w")

        sep()

        # -- CAPTURE OPTIONS --
        cap_frame = section("Capture Options")
        rows = [
            ("Number of captures",      self.n_var),
            ("Wait between captures (s)", self.wait_var),
            ("Capture label",            self.label_var),
            ("Notes",                    self.notes_var),
        ]
        for lbl, var in rows:
            row = tk.Frame(cap_frame)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=lbl, anchor="w").pack(fill=tk.X)
            tk.Entry(row, textvariable=var).pack(fill=tk.X)

        sep()

        # -- CAPTURE BUTTON --
        self.capture_btn = tk.Button(
            left,
            text="Capture",
            font=("", 13, "bold"),
            bg="#1a6fbf",
            fg="white",
            activebackground="#155199",
            activeforeground="white",
            pady=8,
            command=self._on_capture,
        )
        self.capture_btn.pack(fill=tk.X, pady=(4, 0))

        # Keep a list of all input widgets for bulk enable/disable
        self._input_widgets = [
            self.connect_btn,
            self.capture_btn,
            *[w for w in left.winfo_children() if isinstance(w, tk.Entry)],
        ]

    def _build_right_panel(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        right.columnconfigure(0, weight=3)
        right.columnconfigure(1, weight=2)
        right.rowconfigure(0, weight=1)

        # ── Waveforms (left) ───────────────────────────────────────────────────
        wave_frame = tk.Frame(right)
        wave_frame.grid(row=0, column=0, sticky="nsew")
        wave_frame.columnconfigure(0, weight=1)
        wave_frame.rowconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=wave_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        wave_toolbar = tk.Frame(wave_frame)
        wave_toolbar.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, wave_toolbar)

        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=14)
        ax.set_axis_off()
        self.canvas.draw()

        # ── Histograms (right) ─────────────────────────────────────────────────
        hist_frame = tk.Frame(right)
        hist_frame.grid(row=0, column=1, sticky="nsew")
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.rowconfigure(0, weight=1)

        self.hist_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=hist_frame)
        self.hist_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        hist_toolbar_frame = tk.Frame(hist_frame)
        hist_toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.hist_canvas, hist_toolbar_frame)
        clear_frame = tk.Frame(hist_frame)
        clear_frame.grid(row=2, column=0, sticky="ew")
        tk.Button(clear_frame, text="Clear Histograms",
                  command=self._clear_histograms).pack(side=tk.RIGHT, padx=4, pady=2)

        ax_h = self.hist_fig.add_subplot(111)
        ax_h.text(0.5, 0.5, "No data yet", ha="center", va="center",
                  transform=ax_h.transAxes, color="gray", fontsize=14)
        ax_h.set_axis_off()
        self.hist_canvas.draw()

    # ── Connect ────────────────────────────────────────────────────────────────

    def _on_connect(self) -> None:
        if self.scope is not None:
            self.scope.close()
            self.scope = None
        self._set_conn_indicator("connecting")
        self._set_status("Scanning for USB instruments…")
        self.connect_btn.config(state=tk.DISABLED)
        rm = pyvisa.ResourceManager()
        t = threading.Thread(target=self._connect_worker, args=(rm,), daemon=True)
        t.start()
        self.root.after(POLL_MS, self._poll_result_queue)

    def _connect_worker(self, rm: pyvisa.ResourceManager) -> None:
        try:
            resource_str = find_scope(rm)
        except SystemExit as e:
            self.result_queue.put(("connect_error", str(e).strip()))
            return
        except Exception as e:
            self.result_queue.put(("connect_error", str(e)))
            return
        try:
            scope = connect(resource_str)
        except SystemExit as e:
            self.result_queue.put(("connect_error", str(e).strip()))
            return
        except Exception as e:
            self.result_queue.put(("connect_error", str(e)))
            return
        idn = scope.query("*IDN?").strip() if scope else "?"
        self.result_queue.put(("connected", scope, idn))

    def _on_disconnect(self) -> None:
        if self.scope is not None:
            try:
                self.scope.close()
            except Exception:
                pass
            self.scope = None
        self._set_conn_indicator("disconnected")
        self._set_status("Disconnected.")
        self.disconnect_btn.config(state=tk.DISABLED)

    # ── Browse ─────────────────────────────────────────────────────────────────

    def _on_new_filename(self) -> None:
        self.file_var.set(f"./data/waveforms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")

    def _on_browse(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*")],
            initialfile=Path(self.file_var.get()).name,
        )
        if path:
            self.file_var.set(path)

    # ── Capture ────────────────────────────────────────────────────────────────

    def _on_capture(self) -> None:
        if self._capture_running:
            return
        if self.scope is None:
            self._set_status("Not connected — click 'Connect to Scope' first.", "red")
            return

        ok, msg = validate_inputs(
            self.file_var, self.ch_vars, self.pre_var, self.post_var, self.n_var
        )
        if not ok:
            self._set_status(msg, "red")
            return

        channels   = [ch for ch, v in self.ch_vars.items() if v.get()]
        pre_val    = int(self.pre_var.get())
        post_val   = int(self.post_var.get())
        pre_arg    = None if (pre_val == 0 and post_val == 0) else pre_val
        post_arg   = None if (pre_val == 0 and post_val == 0) else post_val
        n          = int(self.n_var.get())
        wait_s     = parse_wait(self.wait_var)
        filepath   = Path(self.file_var.get().strip())
        label      = self.label_var.get().strip() or datetime.now().strftime("capture_%H%M%S")
        notes      = self.notes_var.get().strip()
        save_root_ = self.root_var.get()

        filepath.parent.mkdir(parents=True, exist_ok=True)

        self._n_captures_total = n
        self._capture_running  = True
        self._stop_event.clear()
        self._set_controls_enabled(False)
        self._rebuild_figure(channels)

        t = threading.Thread(
            target=self._capture_worker,
            args=(self.scope, channels, pre_arg, post_arg, n, wait_s, filepath, label, notes, save_root_),
            daemon=True,
        )
        t.start()
        self.root.after(POLL_MS, self._poll_result_queue)

    def _capture_worker(
        self,
        scope,
        channels: list[str],
        pre,
        post,
        n: int,
        wait_s: float,
        filepath: Path,
        label: str,
        notes: str,
        save_root_: bool = False,
    ) -> None:
        try:
            for i in range(n):
                if self._stop_event.is_set():
                    self.result_queue.put(("status", "Capture cancelled."))
                    break
                if i > 0 and max(wait_s, 0.1) > 0:
                    deadline = time.monotonic() + max(wait_s, 0.1)
                    while time.monotonic() < deadline:
                        if self._stop_event.is_set():
                            break
                        remaining = deadline - time.monotonic()
                        self.result_queue.put(
                            ("status", f"Waiting… {remaining:.1f} s before capture {i+1}/{n}")
                        )
                        time.sleep(min(0.1, remaining))
                    if self._stop_event.is_set():
                        self.result_queue.put(("status", "Capture cancelled."))
                        break
                capture_label = f"{label}_{i+1:03d}" if n > 1 else label
                scope.write("ACQUIRE:STATE STOP")   # freeze memory — all channels from same trigger
                scope_state = get_scope_state(scope, channels)
                captured = {}
                for ch in channels:
                    if self._stop_event.is_set():
                        break
                    self.result_queue.put(
                        ("status", f"Capture {i+1}/{n} — fetching {ch}…")
                    )
                    time_s, volts, meta = fetch_channel(scope, ch, pre, post)
                    captured[ch] = (time_s, volts, meta)
                    self.result_queue.put(("channel_done", i, ch, time_s, volts, meta))

                if captured:
                    save_hdf5(filepath, captured, label=capture_label, notes=notes, scope_state=scope_state)
                    if save_root_:
                        save_root(filepath, captured, label=capture_label, scope_state=scope_state)
                    log_capture_tsv(filepath, capture_label, channels, pre, post, notes, scope_state=scope_state)
                    self.result_queue.put(("capture_done", i + 1, n, str(filepath)))
                scope.write("ACQUIRE:STATE RUN")     # re-arm for next capture

        except Exception as e:
            self.result_queue.put(("error", str(e)))
            return

        self.result_queue.put(("all_done",))

    # ── Poll queue ─────────────────────────────────────────────────────────────

    def _poll_result_queue(self) -> None:
        done = False
        try:
            while True:
                msg = self.result_queue.get_nowait()
                kind = msg[0]

                if kind == "connected":
                    _, scope, idn = msg
                    self.scope = scope
                    self._set_conn_indicator("connected")
                    self._set_status(f"Connected: {idn}")
                    self.connect_btn.config(state=tk.NORMAL)
                    self.disconnect_btn.config(state=tk.NORMAL)
                    done = True

                elif kind == "connect_error":
                    _, err = msg
                    self._set_conn_indicator("disconnected")
                    self._set_status(f"Connection failed: {err}", "red")
                    self.connect_btn.config(state=tk.NORMAL)
                    self.disconnect_btn.config(state=tk.DISABLED)
                    done = True

                elif kind == "status":
                    self._set_status(msg[1])

                elif kind == "channel_done":
                    _, cap_idx, ch, time_s, volts, meta = msg
                    self._update_plot(ch, time_s, volts, cap_idx, meta)

                elif kind == "capture_done":
                    _, completed, total, path = msg
                    self._set_status(f"Capture {completed}/{total} saved → {path}")

                elif kind == "error":
                    _, err = msg
                    self._set_status(f"Error: {err}", "red")
                    self._capture_running = False
                    self._set_controls_enabled(True)
                    self.disconnect_btn.config(
                        state=tk.NORMAL if self.scope is not None else tk.DISABLED
                    )
                    done = True

                elif kind == "all_done":
                    self._set_status("Done.")
                    self._capture_running = False
                    self._set_controls_enabled(True)
                    self.disconnect_btn.config(
                        state=tk.NORMAL if self.scope is not None else tk.DISABLED
                    )
                    done = True

        except queue.Empty:
            pass

        if not done:
            self.root.after(POLL_MS, self._poll_result_queue)

    # ── Plot ───────────────────────────────────────────────────────────────────

    def _rebuild_figure(self, channels: list[str]) -> None:
        self.fig.clear()
        self._axes_map = {}
        self._line_map = {}
        n = len(channels)
        for i, ch in enumerate(channels):
            ax = self.fig.add_subplot(n, 1, i + 1)
            ax.set_title(ch, loc="left", fontsize=9, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("Voltage (V)")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            (line,) = ax.plot([], [], lw=0.9)
            self._axes_map[ch] = ax
            self._line_map[ch] = line
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

        # Reset histogram state for the new capture session
        self._hist_data = {ch: {"integral": [], "amplitude": []} for ch in channels}
        self._rebuild_hist_figure(channels)

    def _rebuild_hist_figure(self, channels: list[str]) -> None:
        """Set up empty histogram axes (2 columns: integral | amplitude, N rows)."""
        self.hist_fig.clear()
        self._hist_axes = {}
        n = len(channels)
        for i, ch in enumerate(channels):
            ax_int = self.hist_fig.add_subplot(n, 2, 2 * i + 1)
            ax_amp = self.hist_fig.add_subplot(n, 2, 2 * i + 2)
            for ax, title, xlabel in [
                (ax_int, f"{ch} — Integral",  "Integral (V·s)"),
                (ax_amp, f"{ch} — Amplitude", "Amplitude (V)"),
            ]:
                ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel("Counts", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3, linewidth=0.5)
            self._hist_axes[ch] = {"integral": ax_int, "amplitude": ax_amp}
        self.hist_fig.tight_layout(pad=1.5)
        self.hist_canvas.draw_idle()

    def _update_plot(
        self,
        ch: str,
        time_s: np.ndarray,
        volts: np.ndarray,
        capture_idx: int,
        meta: dict,
    ) -> None:
        ax   = self._axes_map.get(ch)
        line = self._line_map.get(ch)
        if ax is None:
            return

        span = time_s[-1] - time_s[0] if len(time_s) > 1 else 1e-6
        t_scale, t_prefix = si_prefix(span)
        t_scaled = time_s * t_scale

        color = viridis_color(capture_idx, self._n_captures_total)

        if capture_idx == 0:
            line.set_xdata(t_scaled)
            line.set_ydata(volts)
            line.set_color(color)

            # Trigger line — drawn once per channel (XZERO is the trigger time in seconds)
            xzero = float(meta.get("XZERO", 0.0))
            ax.axvline(
                x=xzero * t_scale,
                color="red",
                linewidth=0.9,
                linestyle="--",
                alpha=0.7,
                label="trigger",
                zorder=3,
            )
        else:
            ax.plot(t_scaled, volts, lw=0.9, alpha=0.85, color=color)

        ax.relim()
        ax.autoscale_view()
        ax.set_xlabel(f"Time ({t_prefix}s)")
        self.canvas.draw_idle()

        # Accumulate histogram data
        if ch in self._hist_data:
            integral, amplitude = self._compute_pulse_metrics(time_s, volts, meta)
            self._hist_data[ch]["integral"].append(integral)
            self._hist_data[ch]["amplitude"].append(amplitude)
            self._update_histograms(ch)

    def _compute_pulse_metrics(
        self,
        time_s: np.ndarray,
        volts: np.ndarray,
        meta: dict,
    ) -> tuple[float, float]:
        """Return (baseline-subtracted integral in V·s, peak amplitude in V).

        Baseline is the mean of pre-trigger samples (time_s < XZERO).
        Falls back to the first 10 % of samples if no pre-trigger region exists.
        """
        xzero = float(meta.get("XZERO", 0.0))
        pre_mask = time_s < xzero
        if pre_mask.sum() >= 5:
            baseline = volts[pre_mask].mean()
        else:
            n_base = max(1, len(volts) // 10)
            baseline = volts[:n_base].mean()
        bsub = volts - baseline
        integral  = float(np.trapezoid(bsub, time_s))
        amplitude = float(bsub.max())
        return integral, amplitude

    def _update_histograms(self, ch: str) -> None:
        """Redraw the integral and amplitude histograms for one channel."""
        axes = self._hist_axes.get(ch)
        if axes is None:
            return
        data = self._hist_data.get(ch, {})
        for key, ax in axes.items():
            values = data.get(key, [])
            ax.clear()
            xlabel = "Integral (V·s)" if key == "integral" else "Amplitude (V)"
            ax.set_title(
                f"{ch} — {'Integral' if key == 'integral' else 'Amplitude'}  (n={len(values)})",
                loc="left", fontsize=8, fontweight="bold",
            )
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Counts", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if values:
                ax.hist(values, bins=min(50, max(10, len(values))), color="#1a6fbf",
                        edgecolor="white", linewidth=0.4)
                ax.relim()
                ax.autoscale_view()
        self.hist_fig.tight_layout(pad=1.5)
        self.hist_canvas.draw_idle()

    def _clear_histograms(self) -> None:
        """Wipe accumulated histogram data and redraw empty axes."""
        for ch in self._hist_data:
            self._hist_data[ch] = {"integral": [], "amplitude": []}
        channels = list(self._hist_axes.keys())
        if channels:
            self._rebuild_hist_figure(channels)

    # ── Status / indicators ────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str = "black") -> None:
        self.status_label.config(text=msg, fg=color)

    def _set_conn_indicator(self, state: str) -> None:
        colors = {"disconnected": "red", "connecting": "#e6a817", "connected": "#2db83d"}
        self.conn_indicator.config(fg=colors.get(state, "red"))

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED

        def _apply(widget: tk.Widget) -> None:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                _apply(child)

        # Recurse through left panel children (skip the canvas side)
        for child in self.root.winfo_children():
            if child != self.canvas.get_tk_widget():
                _apply(child)

        # Always keep status bar label readable
        self.status_label.config(state=tk.NORMAL)

    # ── Close ──────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop_event.set()
        if self.scope is not None:
            try:
                self.scope.close()
            except Exception:
                pass
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    root.title("DPO4054 Waveform Capture")
    root.minsize(900, 600)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    WaveformApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
