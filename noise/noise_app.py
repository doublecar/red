#!/usr/bin/env python3
"""
Noise Generator — Sleep Aid
5-channel programmable noise mixer.

Channels: Heavy Rain, Brown Noise, Fan, Human Crowd, Waterfall
Output:   Speaker, Save to WAV, or Both
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import butter, sosfilt, lfilter

# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
BLOCK_SIZE  = 2048
DEFAULT_SAVE_DIR = Path.home() / "Documents" / "AK" / "AI" / "Noise"

# Paul Kellett's pink-noise IIR filter (used for Waterfall)
_PK_B = np.array([ 0.049922035, -0.095993537,  0.050612699, -0.004408786])
_PK_A = np.array([ 1.0,         -2.494956002,  2.017265875, -0.522189400])

# ─── Colours / theme ──────────────────────────────────────────────────────────

BG   = "#12121e"   # window background
BG2  = "#1c1c30"   # panel background
FG   = "#dde3f0"   # primary text
FG2  = "#7880a0"   # secondary text
ACC  = "#5b9cf6"   # accent blue
ACC2 = "#3ecf8e"   # green (save button)
RED  = "#e05260"   # stop button
SLD  = "#2a2a45"   # slider trough
FONT = ("Segoe UI", 10)
FONT_B = ("Segoe UI", 10, "bold")
FONT_H = ("Segoe UI", 14, "bold")

# ─── Noise engine ─────────────────────────────────────────────────────────────

def _wn(freq: float, sr: int = SAMPLE_RATE) -> float:
    """Normalised digital frequency."""
    return freq / (sr / 2)


def _norm(x: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(x))
    return x / (m + 1e-12)


class ChannelState:
    """
    Stateful noise generators.
    Each generator maintains filter / oscillator state so successive blocks
    chain together without discontinuities.
    """

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr

        # ── Brown noise: leaky integrator ─────────────────────────────────
        self.brown_y: float = 0.0

        # ── Heavy Rain ────────────────────────────────────────────────────
        # Band-pass 500–9 kHz (rain body) + low-pass 250 Hz (distant rumble)
        self.rain_sos  = butter(3, [_wn(500), _wn(9000)], btype="band", output="sos")
        self.rain_zi   = np.zeros((self.rain_sos.shape[0], 2))
        self.rain_lsos = butter(2,  _wn(250),              btype="low",  output="sos")
        self.rain_lzi  = np.zeros((self.rain_lsos.shape[0], 2))

        # ── Fan ───────────────────────────────────────────────────────────
        # Band-pass 150–4 kHz (airflow) + harmonic tones (motor)
        self.fan_sos    = butter(3, [_wn(150), _wn(4000)], btype="band", output="sos")
        self.fan_zi     = np.zeros((self.fan_sos.shape[0], 2))
        self.fan_phases = np.zeros(7)   # phase accumulators for harmonics 1–7

        # ── Human Crowd ───────────────────────────────────────────────────
        # Band-pass 200–3.5 kHz (speech band) + dense multi-oscillator voices
        self.crowd_sos  = butter(4, [_wn(200), _wn(3500)], btype="band", output="sos")
        self.crowd_zi   = np.zeros((self.crowd_sos.shape[0], 2))
        n_v = 32
        self.v_freqs  = np.random.uniform(90, 500, n_v)
        self.v_phases = np.random.uniform(0, 2 * np.pi, n_v)

        # ── Waterfall ─────────────────────────────────────────────────────
        # Pink noise (Paul Kellett IIR) + high-pass splash component
        self.pk_zi   = np.zeros(len(_PK_B) - 1)
        self.wf_hsos = butter(2, _wn(3500), btype="high", output="sos")
        self.wf_hzi  = np.zeros((self.wf_hsos.shape[0], 2))

    # ── Generators ───────────────────────────────────────────────────────────

    def heavy_rain(self, n: int) -> np.ndarray:
        body,   self.rain_zi  = sosfilt(self.rain_sos,  np.random.normal(0, 1, n), zi=self.rain_zi)
        rumble, self.rain_lzi = sosfilt(self.rain_lsos, np.random.normal(0, 1, n), zi=self.rain_lzi)
        return _norm(body * 0.80 + rumble * 0.20)

    def brown_noise(self, n: int) -> np.ndarray:
        w   = np.random.normal(0, 0.02, n)
        out = np.empty(n)
        y   = self.brown_y
        for i in range(n):
            y      = 0.9985 * y + w[i]
            out[i] = y
        self.brown_y = y
        return _norm(out)

    def fan(self, n: int) -> np.ndarray:
        t    = np.arange(n) / self.sr
        tone = np.zeros(n)
        fund = 60.0   # motor fundamental (Hz)
        for h in range(1, 8):
            tone += (0.5 / h) * np.sin(2 * np.pi * fund * h * t + self.fan_phases[h - 1])
        self.fan_phases += 2 * np.pi * fund * np.arange(1, 8) * n / self.sr
        self.fan_phases %= 2 * np.pi
        broad, self.fan_zi = sosfilt(self.fan_sos, np.random.normal(0, 1, n), zi=self.fan_zi)
        return _norm(tone * 0.15 + broad * 0.85)

    def human_crowd(self, n: int) -> np.ndarray:
        t = np.arange(n) / self.sr
        # Vectorised multi-oscillator: shape (n_voices, n_samples)
        voices = np.sum(
            np.sin(2 * np.pi * self.v_freqs[:, None] * t + self.v_phases[:, None]),
            axis=0,
        ) / len(self.v_freqs)
        self.v_phases += 2 * np.pi * self.v_freqs * n / self.sr
        self.v_phases %= 2 * np.pi
        noise, self.crowd_zi = sosfilt(self.crowd_sos, np.random.normal(0, 1, n), zi=self.crowd_zi)
        return _norm(voices * 0.35 + noise * 0.65)

    def waterfall(self, n: int) -> np.ndarray:
        pink,   self.pk_zi  = lfilter(_PK_B, _PK_A, np.random.normal(0, 1, n), zi=self.pk_zi)
        splash, self.wf_hzi = sosfilt(self.wf_hsos, np.random.normal(0, 1, n), zi=self.wf_hzi)
        return _norm(pink * 0.72 + splash * 0.28)

    def generate(self, channel: str, n: int) -> np.ndarray:
        return getattr(self, channel)(n)


# ─── Main application ─────────────────────────────────────────────────────────

CHANNELS = [
    ("Heavy Rain",   "heavy_rain"),
    ("Brown Noise",  "brown_noise"),
    ("Fan",          "fan"),
    ("Human Crowd",  "human_crowd"),
    ("Waterfall",    "waterfall"),
]


class NoiseApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root  = root
        self.state = ChannelState()
        self._lock = threading.Lock()

        # Per-channel volume (0.0 – 1.0)
        self.volumes: dict[str, tk.DoubleVar] = {
            ch: tk.DoubleVar(value=0.5) for _, ch in CHANNELS
        }
        self.output_mode  = tk.StringVar(value="speaker")
        self.duration_var = tk.StringVar(value="30")
        self.status_var   = tk.StringVar(value="Ready")

        self.is_playing = False
        self.stream: sd.OutputStream | None = None

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self.root
        root.title("Noise Generator — Sleep Aid")
        root.configure(bg=BG)
        root.resizable(False, False)

        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 8))
        tk.Label(hdr, text="NOISE GENERATOR", font=FONT_H,
                 bg=BG, fg=ACC).pack(side="left")
        tk.Label(hdr, text="  sleep aid", font=FONT,
                 bg=BG, fg=FG2).pack(side="left", pady=4)

        _divider(root)

        # ── Channel sliders ───────────────────────────────────────────────
        panel = tk.Frame(root, bg=BG2, padx=20, pady=14)
        panel.pack(fill="x", padx=16, pady=8)

        tk.Label(panel, text="CHANNELS", font=("Segoe UI", 8, "bold"),
                 bg=BG2, fg=FG2).grid(row=0, column=0, columnspan=3,
                                       sticky="w", pady=(0, 8))

        self._val_labels: dict[str, tk.Label] = {}

        for i, (label, ch) in enumerate(CHANNELS, start=1):
            row = i

            # Channel name
            tk.Label(panel, text=label, font=FONT, width=13, anchor="w",
                     bg=BG2, fg=FG).grid(row=row, column=0, sticky="w", pady=5)

            # Slider
            sld = tk.Scale(
                panel,
                variable=self.volumes[ch],
                from_=0.0, to=1.0,
                resolution=0.01,
                orient="horizontal",
                length=260,
                showvalue=False,
                bg=BG2, fg=FG,
                troughcolor=SLD,
                activebackground=ACC,
                highlightthickness=0,
                bd=0,
            )
            sld.grid(row=row, column=1, padx=(8, 6), pady=5)

            # Numeric readout
            lbl = tk.Label(panel, text="0.50", font=FONT, width=4,
                           bg=BG2, fg=ACC, anchor="e")
            lbl.grid(row=row, column=2, pady=5)
            self._val_labels[ch] = lbl

            def _trace(name, *_, v=self.volumes[ch], lbl=lbl):
                lbl.config(text=f"{v.get():.2f}")

            self.volumes[ch].trace_add("write", _trace)

        _divider(root)

        # ── Output mode ───────────────────────────────────────────────────
        out_panel = tk.Frame(root, bg=BG, padx=20)
        out_panel.pack(fill="x", padx=16, pady=4)

        tk.Label(out_panel, text="OUTPUT", font=("Segoe UI", 8, "bold"),
                 bg=BG, fg=FG2).pack(anchor="w")

        rb_frame = tk.Frame(out_panel, bg=BG)
        rb_frame.pack(anchor="w", pady=(4, 0))

        for val, lbl in [("speaker", "Speaker"), ("file", "Save to File"), ("both", "Both")]:
            tk.Radiobutton(
                rb_frame, text=lbl, variable=self.output_mode, value=val,
                font=FONT, bg=BG, fg=FG, selectcolor=SLD,
                activebackground=BG, activeforeground=FG,
                highlightthickness=0,
            ).pack(side="left", padx=(0, 18))

        # ── Save duration ─────────────────────────────────────────────────
        dur_frame = tk.Frame(root, bg=BG, padx=20)
        dur_frame.pack(fill="x", padx=16, pady=(6, 4))

        tk.Label(dur_frame, text="Save duration:", font=FONT,
                 bg=BG, fg=FG).pack(side="left")
        tk.Entry(
            dur_frame, textvariable=self.duration_var, width=6,
            bg=SLD, fg=FG, insertbackground=FG,
            font=FONT, relief="flat", highlightthickness=1,
            highlightcolor=ACC, highlightbackground=SLD,
        ).pack(side="left", padx=6)
        tk.Label(dur_frame, text="minutes", font=FONT,
                 bg=BG, fg=FG2).pack(side="left")

        _divider(root)

        # ── Action buttons ────────────────────────────────────────────────
        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(pady=(8, 4))

        self.play_btn = tk.Button(
            btn_frame, text="▶   PLAY", width=13,
            font=FONT_B, bg=ACC, fg="white",
            activebackground="#3a7de0", activeforeground="white",
            relief="flat", cursor="hand2",
            command=self._toggle_play,
        )
        self.play_btn.pack(side="left", padx=10, ipady=6)

        self.save_btn = tk.Button(
            btn_frame, text="💾   SAVE", width=13,
            font=FONT_B, bg=ACC2, fg="white",
            activebackground="#2daa70", activeforeground="white",
            relief="flat", cursor="hand2",
            command=self._save_dialog,
        )
        self.save_btn.pack(side="left", padx=10, ipady=6)

        # ── Status bar ────────────────────────────────────────────────────
        tk.Label(root, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg=BG, fg=FG2).pack(pady=(6, 14))

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _mix(self, n: int) -> np.ndarray:
        """Generate one mixed block (called from audio callback OR save thread)."""
        vols = {ch: self.volumes[ch].get() for _, ch in CHANNELS}
        total = sum(vols.values()) or 1.0
        mixed = np.zeros(n, dtype=np.float32)
        for _, ch in CHANNELS:
            mixed += vols[ch] * self.state.generate(ch, n)
        mixed /= total
        return mixed

    def _audio_callback(self, outdata: np.ndarray, frames: int, _t, _status) -> None:
        with self._lock:
            block = self._mix(frames)
        outdata[:, 0] = block
        if outdata.shape[1] > 1:
            outdata[:, 1] = block   # mono → stereo copy

    # ── Playback controls ─────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        mode = self.output_mode.get()
        if mode == "file":
            # "File only" — just run the save dialog
            self._save_dialog()
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=2,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            messagebox.showerror("Audio Error", str(exc))
            return

        self.is_playing = True
        self.play_btn.config(text="■   STOP", bg=RED, activebackground="#b03040")
        self._set_status("Playing…")

        if mode == "both":
            threading.Thread(target=self._save_worker,
                             kwargs={"use_dialog": True}, daemon=True).start()

    def _stop_playback(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.play_btn.config(text="▶   PLAY", bg=ACC, activebackground="#3a7de0")
        self._set_status("Stopped")

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_dialog(self) -> None:
        try:
            minutes = float(self.duration_var.get())
            if minutes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Duration",
                                 "Please enter a positive number of minutes.")
            return

        default_name = f"noise_{int(minutes)}min.wav"
        DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        path = filedialog.asksaveasfilename(
            initialdir=str(DEFAULT_SAVE_DIR),
            initialfile=default_name,
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            title="Save Noise File",
        )
        if not path:
            return

        threading.Thread(
            target=self._save_worker,
            kwargs={"path": path, "use_dialog": False},
            daemon=True,
        ).start()

    def _save_worker(self, path: str | None = None, use_dialog: bool = False) -> None:
        """Background thread: generate and write audio to disk."""
        if use_dialog:
            # Called from "both" mode — need to open dialog from main thread
            self.root.after(0, self._save_dialog)
            return

        try:
            minutes = float(self.duration_var.get())
        except ValueError:
            return

        n_total   = int(minutes * 60 * SAMPLE_RATE)
        save_state = ChannelState()                      # independent state for saving
        vols       = {ch: self.volumes[ch].get() for _, ch in CHANNELS}
        total_vol  = sum(vols.values()) or 1.0

        self._set_status(f"Saving {minutes:.0f} min to {Path(path).name}…")
        self.root.after(0, lambda: self.save_btn.config(state="disabled"))

        blocks: list[np.ndarray] = []
        written = 0
        while written < n_total:
            n = min(BLOCK_SIZE, n_total - written)
            mixed = np.zeros(n, dtype=np.float32)
            for _, ch in CHANNELS:
                mixed += vols[ch] * save_state.generate(ch, n)
            mixed /= total_vol
            blocks.append(mixed)
            written += n

        audio = np.concatenate(blocks)
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")

        self._set_status(f"Saved → {Path(path).name}")
        self.root.after(0, lambda: self.save_btn.config(state="normal"))

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.status_var.set(msg))

    def _on_close(self) -> None:
        self._stop_playback()
        self.root.destroy()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _divider(parent: tk.Widget) -> None:
    tk.Frame(parent, bg="#2a2a45", height=1).pack(fill="x", padx=16, pady=4)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = NoiseApp(root)
    root.mainloop()
