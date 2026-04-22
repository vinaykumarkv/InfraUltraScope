#!/usr/bin/env python3
"""
Infrasound & Ultrasound Visualizer + Decoder
=============================================
Captures audio from a high-sample-rate interface, displays real-time
spectrograms across the full spectrum (0–100 kHz+), and transposes
inaudible bands into the audible range (20 Hz–20 kHz) for playback.

Requirements (install via pip):
    pip install numpy scipy sounddevice soundfile matplotlib librosa pyqt5

Hardware:
    - Microphone capable of the target frequency range
      • Infrasound  (<20 Hz): MEMS sensor or dedicated infrasound mic
      • Ultrasound  (>20 kHz): ultrasonic transducer / bat detector mic
    - Audio interface with matching sample rate
      • ≥48 kHz for infrasound work (plenty of low-freq resolution)
      • ≥192 kHz (ideally 384–768 kHz) for ultrasound up to ~150 kHz
"""

import sys
import threading
import queue
import time
import argparse
from pathlib import Path

import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import matplotlib
matplotlib.use("Qt5Agg")          # requires PyQt5
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (override via CLI arguments)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE   = 192_000   # Hz  — set to your interface's capability
DEFAULT_BLOCK_SIZE    = 2048      # samples per callback block
DEFAULT_HISTORY_SEC   = 5.0      # seconds of spectrogram history to show
DEFAULT_NPERSEG       = 2048      # FFT window size
DEFAULT_NOVERLAP      = 1536      # FFT overlap
DEFAULT_FREQ_MIN      = 0        # Hz  — display floor
DEFAULT_FREQ_MAX      = 96_000   # Hz  — display ceiling (Nyquist of 192 k)
DEFAULT_TRANSP_RATIO  = 10.0     # transpose divisor (e.g. 10× → 40 kHz→4 kHz)
DEFAULT_COLORMAP      = "inferno"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def list_devices():
    """Pretty-print available audio devices."""
    print("\n=== Available Audio Devices ===")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        tag = ""
        if d["max_input_channels"] > 0:
            tag += " [IN]"
        if d["max_output_channels"] > 0:
            tag += " [OUT]"
        print(f"  [{i:2d}]{tag:8s}  {d['name']}  "
              f"(max SR: {int(d['default_samplerate'])} Hz)")
    print()


def bandpass_filter(data, lowcut, highcut, fs, order=6):
    """Zero-phase Butterworth band-pass filter."""
    nyq = fs / 2.0
    low  = max(lowcut,  1.0) / nyq
    high = min(highcut, nyq - 1) / nyq
    if low >= high:
        return data
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, data)


def transpose_signal(data, fs, ratio):
    """
    Time-domain frequency transposition by resampling.
    Dividing the sample rate by `ratio` shifts all frequencies
    down by that factor (e.g. ratio=10 → 40 kHz becomes 4 kHz).
    Returns (transposed_data, new_sample_rate).
    """
    new_fs = int(fs / ratio)
    # Resample: treat same samples as if recorded at fs/ratio
    # No actual resampling needed — just re-label the sample rate.
    # For pitch-shift without time-stretch we DO need to resample.
    n_samples_out = int(len(data) * new_fs / fs)
    transposed = signal.resample(data, n_samples_out)
    return transposed, new_fs


# ─────────────────────────────────────────────────────────────────────────────
# CORE TOOL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class InfraUltraScope:
    """
    Real-time infrasound / ultrasound visualizer and frequency transposer.

    Pipeline
    --------
    Microphone → ring buffer → FFT spectrogram display
                             → optional band-pass + transpose → playback / WAV
    """

    def __init__(self, args):
        self.fs          = args.samplerate
        self.block_size  = args.block_size
        self.hist_sec    = args.history
        self.nperseg     = args.nperseg
        self.noverlap    = args.noverlap
        self.freq_min    = args.freq_min
        self.freq_max    = min(args.freq_max, self.fs // 2)
        self.transp_ratio= args.transpose_ratio
        self.device_in   = args.input_device
        self.device_out  = args.output_device
        self.save_raw    = args.save_raw
        self.save_transposed = args.save_transposed
        self.cmap        = args.colormap
        self.play_live   = args.play_live

        # Band selection for transposition (Hz)
        self.band_low    = args.band_low
        self.band_high   = args.band_high

        # Internal buffers
        self.q           = queue.Queue()
        self._history_samples = int(self.fs * self.hist_sec)
        self._ring       = np.zeros(self._history_samples, dtype=np.float32)
        self._ring_lock  = threading.Lock()

        # Recording buffers
        self._raw_chunks       = []
        self._transp_chunks    = []

        # State
        self._running    = False
        self._stream     = None
        self._out_stream = None
        self._transp_buf = np.zeros(0, dtype=np.float32)

        print(f"\n[InfraUltraScope]  fs={self.fs} Hz | "
              f"display {self.freq_min}–{self.freq_max} Hz | "
              f"transpose ×{self.transp_ratio:.1f} "
              f"band {self.band_low}–{self.band_high} Hz")

    # ── Audio callbacks ──────────────────────────────────────────────────────

    def _input_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata[:, 0].copy()
        self.q.put(mono)

    # ── Processing thread ────────────────────────────────────────────────────

    def _process_loop(self):
        """Drain queue → update ring buffer → optionally transpose & play."""
        out_sr = int(self.fs / self.transp_ratio)
        while self._running:
            try:
                chunk = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Update ring buffer (rolling)
            n = len(chunk)
            with self._ring_lock:
                self._ring = np.roll(self._ring, -n)
                self._ring[-n:] = chunk

            # Record raw
            if self.save_raw is not None:
                self._raw_chunks.append(chunk)

            # Transpose selected band
            filtered = bandpass_filter(
                chunk, self.band_low, self.band_high, self.fs)
            transposed, _ = transpose_signal(filtered, self.fs, self.transp_ratio)

            if self.save_transposed is not None:
                self._transp_chunks.append(transposed)

            if self.play_live and self._out_stream is not None:
                # Buffer transposed audio for output
                self._transp_buf = np.concatenate(
                    [self._transp_buf, transposed.astype(np.float32)])

    def _output_callback(self, outdata, frames, time_info, status):
        n = min(frames, len(self._transp_buf))
        outdata[:n, 0] = self._transp_buf[:n]
        if n < frames:
            outdata[n:, 0] = 0.0
        self._transp_buf = self._transp_buf[n:]

    # ── Spectrogram computation ──────────────────────────────────────────────

    def _compute_spectrogram(self):
        with self._ring_lock:
            data = self._ring.copy()

        freqs, times, Sxx = signal.spectrogram(
            data,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling="spectrum",
            mode="magnitude",
        )

        # Frequency slice
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        return freqs[mask], times, Sxx[mask, :]

    # ── GUI ──────────────────────────────────────────────────────────────────

    def _build_gui(self):
        self._fig = plt.figure(
            figsize=(14, 8),
            facecolor="#0a0a0f"
        )
        self._fig.canvas.manager.set_window_title(
            "InfraUltraScope — Infrasound / Ultrasound Visualizer")

        gs = gridspec.GridSpec(
            3, 2,
            figure=self._fig,
            height_ratios=[3, 1, 1],
            hspace=0.45,
            wspace=0.35,
        )

        ax_style = dict(facecolor="#0d0d1a", labelcolor="#aabbdd",
                        tickcolor="#aabbdd")

        # ── Spectrogram ──
        self._ax_spec = self._fig.add_subplot(gs[0, :])
        self._ax_spec.set_facecolor("#0d0d1a")
        self._ax_spec.set_title(
            "FULL-SPECTRUM SPECTROGRAM", color="#66aaff",
            fontsize=11, fontweight="bold", pad=8)
        self._ax_spec.set_xlabel("Time (s)", color="#aabbdd")
        self._ax_spec.set_ylabel("Frequency (Hz)", color="#aabbdd")
        self._ax_spec.tick_params(colors="#aabbdd")
        for spine in self._ax_spec.spines.values():
            spine.set_edgecolor("#334466")

        # Compute initial spectrogram to set up image
        freqs, times, Sxx = self._compute_spectrogram()
        Sxx_db = 20 * np.log10(Sxx + 1e-12)

        self._spec_img = self._ax_spec.imshow(
            Sxx_db,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap=self.cmap,
            vmin=-120, vmax=0,
        )
        cbar = self._fig.colorbar(
            self._spec_img, ax=self._ax_spec, fraction=0.015, pad=0.01)
        cbar.set_label("dB", color="#aabbdd")
        cbar.ax.yaxis.set_tick_params(color="#aabbdd")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aabbdd")

        # Highlight inaudible bands
        self._ax_spec.axhspan(0, 20,    color="#ff3333", alpha=0.08,
                               label="Infrasound (<20 Hz)")
        self._ax_spec.axhspan(20_000, freqs[-1],
                               color="#33aaff", alpha=0.08,
                               label="Ultrasound (>20 kHz)")
        self._ax_spec.axhspan(self.band_low, self.band_high,
                               color="#ffcc00", alpha=0.10,
                               label=f"Transpose band")
        self._ax_spec.legend(
            loc="upper right", fontsize=7,
            facecolor="#111122", edgecolor="#334466",
            labelcolor="#ccddee")

        # ── Waveform ──
        self._ax_wave = self._fig.add_subplot(gs[1, :])
        self._ax_wave.set_facecolor("#0d0d1a")
        self._ax_wave.set_title("WAVEFORM", color="#66aaff",
                                 fontsize=9, fontweight="bold")
        self._ax_wave.set_xlabel("Time (s)", color="#aabbdd")
        self._ax_wave.set_ylabel("Amplitude", color="#aabbdd")
        self._ax_wave.tick_params(colors="#aabbdd")
        for spine in self._ax_wave.spines.values():
            spine.set_edgecolor("#334466")
        t_wave = np.linspace(0, self.hist_sec, self._history_samples)
        self._wave_line, = self._ax_wave.plot(
            t_wave, np.zeros(self._history_samples),
            color="#00ffaa", linewidth=0.6, alpha=0.85)
        self._ax_wave.set_ylim(-1, 1)
        self._ax_wave.set_xlim(0, self.hist_sec)

        # ── Power spectrum ──
        self._ax_psd = self._fig.add_subplot(gs[2, 0])
        self._ax_psd.set_facecolor("#0d0d1a")
        self._ax_psd.set_title("POWER SPECTRUM (current frame)",
                                color="#66aaff", fontsize=9, fontweight="bold")
        self._ax_psd.set_xlabel("Frequency (Hz)", color="#aabbdd")
        self._ax_psd.set_ylabel("dB", color="#aabbdd")
        self._ax_psd.tick_params(colors="#aabbdd")
        for spine in self._ax_psd.spines.values():
            spine.set_edgecolor("#334466")
        self._psd_line, = self._ax_psd.plot(
            freqs, Sxx_db[:, -1],
            color="#ff9900", linewidth=1.0)
        self._ax_psd.set_xlim(self.freq_min, self.freq_max)
        self._ax_psd.set_ylim(-120, 0)

        # ── Peak frequency display ──
        self._ax_info = self._fig.add_subplot(gs[2, 1])
        self._ax_info.set_facecolor("#060610")
        self._ax_info.axis("off")
        self._info_text = self._ax_info.text(
            0.5, 0.5, "Peak: — Hz\nRMS: — dBFS",
            ha="center", va="center",
            fontsize=14, color="#00ffaa",
            fontfamily="monospace",
            transform=self._ax_info.transAxes,
        )

        self._freqs_cache = freqs   # store for update

    def _animate(self, frame):
        try:
            freqs, times, Sxx = self._compute_spectrogram()
        except Exception:
            return

        Sxx_db = 20 * np.log10(Sxx + 1e-12)

        # Update spectrogram image
        self._spec_img.set_data(Sxx_db)
        self._spec_img.set_extent(
            [times[0], times[-1], freqs[0], freqs[-1]])

        # Waveform
        with self._ring_lock:
            wave = self._ring.copy()
        self._wave_line.set_ydata(wave)

        # Power spectrum (last column)
        self._psd_line.set_xdata(freqs)
        self._psd_line.set_ydata(Sxx_db[:, -1])
        self._ax_psd.set_xlim(self.freq_min, self.freq_max)

        # Peak freq + RMS info
        last_col = Sxx[:, -1]
        if last_col.max() > 0:
            peak_hz = freqs[np.argmax(last_col)]
        else:
            peak_hz = 0.0
        rms = np.sqrt(np.mean(wave ** 2))
        rms_db = 20 * np.log10(rms + 1e-9)

        peak_label = (f"{peak_hz:.1f} Hz"
                      if peak_hz < 1000
                      else f"{peak_hz/1000:.2f} kHz")
        band_label = (f"Band: {self.band_low}–{self.band_high} Hz\n"
                      f"→ ÷{self.transp_ratio:.0f} → audible")
        self._info_text.set_text(
            f"Peak: {peak_label}\n"
            f"RMS:  {rms_db:.1f} dBFS\n"
            f"{band_label}"
        )

        return (self._spec_img, self._wave_line,
                self._psd_line, self._info_text)

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self):
        self._running = True

        # Input stream
        kw = dict(
            samplerate=self.fs,
            blocksize=self.block_size,
            dtype="float32",
            channels=1,
            callback=self._input_callback,
        )
        if self.device_in is not None:
            kw["device"] = self.device_in
        self._stream = sd.InputStream(**kw)

        # Output stream (transposed audio playback)
        if self.play_live:
            out_sr = int(self.fs / self.transp_ratio)
            okw = dict(
                samplerate=out_sr,
                blocksize=512,
                dtype="float32",
                channels=1,
                callback=self._output_callback,
            )
            if self.device_out is not None:
                okw["device"] = self.device_out
            self._out_stream = sd.OutputStream(**okw)
            self._out_stream.start()

        self._proc_thread = threading.Thread(
            target=self._process_loop, daemon=True)
        self._proc_thread.start()

        self._stream.start()

        # GUI
        self._build_gui()
        self._anim = FuncAnimation(
            self._fig, self._animate,
            interval=80,        # ~12 fps refresh
            blit=False,
            cache_frame_data=False,
        )

        try:
            plt.show()
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._out_stream:
            self._out_stream.stop()
            self._out_stream.close()

        # Save recordings
        if self.save_raw and self._raw_chunks:
            raw = np.concatenate(self._raw_chunks)
            sf.write(self.save_raw, raw, self.fs, subtype="FLOAT")
            print(f"[saved] raw recording → {self.save_raw}")

        if self.save_transposed and self._transp_chunks:
            tr = np.concatenate(self._transp_chunks)
            out_sr = int(self.fs / self.transp_ratio)
            sf.write(self.save_transposed, tr, out_sr, subtype="FLOAT")
            print(f"[saved] transposed recording → {self.save_transposed}")


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE ANALYSIS  (process a WAV file instead of live input)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(path: str, args):
    """Load a WAV file and produce a static spectrogram + transposed export."""
    print(f"\n[offline] Loading: {path}")
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    mono = data[:, 0]
    print(f"  Sample rate : {fs} Hz")
    print(f"  Duration    : {len(mono)/fs:.2f} s")
    print(f"  Samples     : {len(mono)}")

    freq_max = min(args.freq_max, fs // 2)

    # Spectrogram
    freqs, times, Sxx = signal.spectrogram(
        mono, fs=fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    mask = (freqs >= args.freq_min) & (freqs <= freq_max)
    freqs, Sxx = freqs[mask], Sxx[mask, :]
    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              facecolor="#0a0a0f")
    fig.suptitle(f"Offline Analysis — {Path(path).name}",
                 color="#66aaff", fontsize=13)

    ax1 = axes[0]
    ax1.set_facecolor("#0d0d1a")
    img = ax1.imshow(Sxx_db, aspect="auto", origin="lower",
                     extent=[times[0], times[-1], freqs[0], freqs[-1]],
                     cmap=args.colormap, vmin=-120, vmax=0)
    cbar = fig.colorbar(img, ax=ax1, fraction=0.015)
    cbar.set_label("dB", color="#aabbdd")
    ax1.set_ylabel("Frequency (Hz)", color="#aabbdd")
    ax1.set_title("Spectrogram", color="#aabbdd")
    ax1.tick_params(colors="#aabbdd")
    ax1.axhspan(0, 20, color="#ff3333", alpha=0.1)
    ax1.axhspan(20_000, freq_max, color="#33aaff", alpha=0.1)

    ax2 = axes[1]
    ax2.set_facecolor("#0d0d1a")
    t_wave = np.linspace(0, len(mono)/fs, len(mono))
    ax2.plot(t_wave, mono, color="#00ffaa", linewidth=0.4)
    ax2.set_xlabel("Time (s)", color="#aabbdd")
    ax2.set_ylabel("Amplitude", color="#aabbdd")
    ax2.set_title("Waveform", color="#aabbdd")
    ax2.tick_params(colors="#aabbdd")
    ax2.set_facecolor("#0d0d1a")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#334466")

    plt.tight_layout()

    # Transpose + save
    if args.save_transposed:
        filtered = bandpass_filter(mono, args.band_low, args.band_high, fs)
        transposed, out_sr = transpose_signal(filtered, fs, args.transpose_ratio)
        sf.write(args.save_transposed, transposed, out_sr, subtype="FLOAT")
        print(f"[saved] transposed audio → {args.save_transposed}  "
              f"(SR={out_sr} Hz, ratio={args.transpose_ratio}×)")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Infrasound & Ultrasound Visualizer / Decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live",   action="store_true", default=True,
                      help="Live capture mode (default)")
    mode.add_argument("--file",   metavar="WAV",
                      help="Offline: analyse existing WAV file")
    mode.add_argument("--devices", action="store_true",
                      help="List audio devices and exit")

    p.add_argument("--samplerate",    type=int,   default=DEFAULT_SAMPLE_RATE,
                   help="Input sample rate (Hz)")
    p.add_argument("--block-size",    type=int,   default=DEFAULT_BLOCK_SIZE,
                   dest="block_size")
    p.add_argument("--history",       type=float, default=DEFAULT_HISTORY_SEC,
                   help="Spectrogram history (seconds)")
    p.add_argument("--nperseg",       type=int,   default=DEFAULT_NPERSEG,
                   help="FFT window size")
    p.add_argument("--noverlap",      type=int,   default=DEFAULT_NOVERLAP,
                   help="FFT window overlap")
    p.add_argument("--freq-min",      type=float, default=DEFAULT_FREQ_MIN,
                   dest="freq_min", help="Display min frequency (Hz)")
    p.add_argument("--freq-max",      type=float, default=DEFAULT_FREQ_MAX,
                   dest="freq_max", help="Display max frequency (Hz)")

    p.add_argument("--band-low",      type=float, default=20_000,
                   dest="band_low",
                   help="Transpose band lower edge (Hz)")
    p.add_argument("--band-high",     type=float, default=40_000,
                   dest="band_high",
                   help="Transpose band upper edge (Hz)")
    p.add_argument("--transpose-ratio", type=float, default=DEFAULT_TRANSP_RATIO,
                   dest="transpose_ratio",
                   help="Frequency division ratio for transposition")

    p.add_argument("--play-live",     action="store_true",
                   dest="play_live",
                   help="Play transposed audio in real time (adds latency)")
    p.add_argument("--input-device",  type=int,   default=None,
                   dest="input_device",
                   help="Input device index (see --devices)")
    p.add_argument("--output-device", type=int,   default=None,
                   dest="output_device",
                   help="Output device index (see --devices)")

    p.add_argument("--save-raw",       metavar="FILE.wav", default=None,
                   dest="save_raw",
                   help="Save raw captured audio to WAV")
    p.add_argument("--save-transposed", metavar="FILE.wav", default=None,
                   dest="save_transposed",
                   help="Save transposed audio to WAV")
    p.add_argument("--colormap",      default=DEFAULT_COLORMAP,
                   help="Matplotlib colormap for spectrogram")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.devices:
        list_devices()
        return

    if args.file:
        analyze_file(args.file, args)
        return

    # Live mode
    scope = InfraUltraScope(args)
    try:
        scope.start()
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        scope.stop()


if __name__ == "__main__":
    main()