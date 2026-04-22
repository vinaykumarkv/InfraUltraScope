#!/usr/bin/env python3
"""
InfraUltraScope UI — Infrasound & Ultrasound Visualizer + Audible Decoder
==========================================================================
Full PyQt5 GUI with live spectrogram, adjustable frequency band controls,
and real-time transposition into the audible range.

Install:
    pip install numpy scipy sounddevice soundfile pyqt5 pyqtgraph

Hardware:
    • Infrasound  (<20 Hz)  : MEMS pressure sensor or infrasound mic
    • Ultrasound  (>20 kHz) : ultrasonic transducer / bat-detector mic
    • Interface             : ≥192 kHz SR for ultrasound; 48 kHz fine for infrasound
"""

import sys, threading, queue, time, argparse, struct
from pathlib import Path
from collections import deque

import numpy as np
import scipy.signal as sig
import sounddevice as sd
import soundfile as sf

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# ── colour theme (dark scientific) ──────────────────────────────────────────
BG      = "#0b0c10"
SURFACE = "#12141a"
ACCENT  = "#00c8a0"
ACCENT2 = "#3fa8ff"
WARN    = "#ff8c42"
TEXT    = "#d4dbe8"
MUTED   = "#5a6270"
BORDER  = "#252830"

CMAPS = ["inferno", "viridis", "plasma", "magma", "cividis", "hot"]

# ── audio helpers ────────────────────────────────────────────────────────────

def list_devices():
    devs = sd.query_devices()
    result = []
    for i, d in enumerate(devs):
        tags = []
        if d["max_input_channels"]  > 0: tags.append("IN")
        if d["max_output_channels"] > 0: tags.append("OUT")
        result.append((i, f"[{','.join(tags)}] {d['name']}"))
    return result


def bandpass(data, lo, hi, fs, order=5):
    nyq = fs / 2.0
    lo  = max(lo, 1.0)  / nyq
    hi  = min(hi, fs/2 - 1) / nyq
    if lo >= hi or lo <= 0 or hi >= 1:
        return data
    sos = sig.butter(order, [lo, hi], btype="band", output="sos")
    return sig.sosfiltfilt(sos, data)


def transpose(data, fs, ratio):
    """Pitch-shift by resampling (shifts freq down by `ratio`)."""
    n_out   = max(1, int(len(data) / ratio))
    out_sr  = int(fs / ratio)
    resampled = sig.resample(data, n_out)
    return resampled.astype(np.float32), out_sr


def normalize(data, headroom=0.85):
    peak = np.max(np.abs(data))
    if peak > 1e-9:
        return data / peak * headroom
    return data


# ── spectrogram colour map builder ─────────────────────────────────────────

def make_colormap(name="inferno"):
    import matplotlib.cm as cm
    cmap = cm.get_cmap(name, 256)
    lut  = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return lut


# ═══════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):

    sig_new_block = QtCore.pyqtSignal(np.ndarray)      # raw audio chunk
    sig_status    = QtCore.pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

        # ── state ──────────────────────────────────────────────────────────
        self.fs           = args.samplerate
        self.block_size   = args.block_size
        self.hist_sec     = 6.0
        self.nperseg      = args.nperseg
        self.noverlap     = args.noverlap
        self.in_device    = args.input_device
        self.out_device   = args.output_device

        self.band_lo      = float(args.band_low)
        self.band_hi      = float(args.band_high)
        self.disp_lo      = float(args.freq_min)
        self.disp_hi      = float(min(args.freq_max, self.fs // 2))
        self.trans_ratio  = float(args.transpose_ratio)
        self.play_live    = args.play_live
        self.cmap_name    = "inferno"

        self._q           = queue.Queue(maxsize=200)
        self._hist        = np.zeros(int(self.fs * self.hist_sec), dtype=np.float32)
        self._hist_lock   = threading.Lock()
        self._running     = False
        self._in_stream   = None
        self._out_stream  = None
        self._out_buf     = np.zeros(0, dtype=np.float32)
        self._out_sr      = int(self.fs / self.trans_ratio)
        self._out_lock    = threading.Lock()

        self._raw_chunks  = []
        self._tr_chunks   = []

        self.sig_new_block.connect(self._on_new_block)
        self.sig_status.connect(self._set_status)

        self._build_ui()
        self._apply_theme()

        # refresh timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_plots)
        self._timer.start(80)   # ~12 fps

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("InfraUltraScope  —  Infrasound / Ultrasound Decoder")
        self.resize(1280, 820)

        root  = QtWidgets.QWidget()
        vroot = QtWidgets.QVBoxLayout(root)
        vroot.setContentsMargins(10, 10, 10, 10)
        vroot.setSpacing(8)
        self.setCentralWidget(root)

        # ── toolbar row ────────────────────────────────────────────────────
        tbar = QtWidgets.QHBoxLayout()
        tbar.setSpacing(10)

        self._btn_start = QtWidgets.QPushButton("▶  Start capture")
        self._btn_start.setFixedHeight(36)
        self._btn_start.clicked.connect(self._toggle_capture)
        tbar.addWidget(self._btn_start)

        self._btn_save_raw = QtWidgets.QPushButton("⬇  Save raw WAV")
        self._btn_save_raw.setFixedHeight(36)
        self._btn_save_raw.clicked.connect(self._save_raw)
        self._btn_save_raw.setEnabled(False)
        tbar.addWidget(self._btn_save_raw)

        self._btn_save_tr = QtWidgets.QPushButton("⬇  Save transposed WAV")
        self._btn_save_tr.setFixedHeight(36)
        self._btn_save_tr.clicked.connect(self._save_transposed)
        self._btn_save_tr.setEnabled(False)
        tbar.addWidget(self._btn_save_tr)

        tbar.addStretch()

        tbar.addWidget(QtWidgets.QLabel("Device:"))
        self._cb_device = QtWidgets.QComboBox()
        self._cb_device.setFixedHeight(34)
        self._cb_device.setMinimumWidth(220)
        for idx, name in list_devices():
            self._cb_device.addItem(name, userData=idx)
        if self.in_device is not None:
            self._cb_device.setCurrentIndex(self.in_device)
        tbar.addWidget(self._cb_device)

        tbar.addWidget(QtWidgets.QLabel("SR (Hz):"))
        self._spin_sr = QtWidgets.QSpinBox()
        self._spin_sr.setRange(8_000, 768_000)
        self._spin_sr.setSingleStep(48_000)
        self._spin_sr.setValue(self.fs)
        self._spin_sr.setFixedWidth(100)
        self._spin_sr.setFixedHeight(34)
        self._spin_sr.valueChanged.connect(self._on_sr_changed)
        tbar.addWidget(self._spin_sr)

        tbar.addWidget(QtWidgets.QLabel("Colormap:"))
        self._cb_cmap = QtWidgets.QComboBox()
        self._cb_cmap.setFixedHeight(34)
        for c in CMAPS:
            self._cb_cmap.addItem(c)
        self._cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        tbar.addWidget(self._cb_cmap)

        vroot.addLayout(tbar)

        # ── main splitter ──────────────────────────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        vroot.addWidget(splitter, stretch=1)

        # LEFT — plots
        plots_w = QtWidgets.QWidget()
        plots_v = QtWidgets.QVBoxLayout(plots_w)
        plots_v.setContentsMargins(0, 0, 0, 0)
        plots_v.setSpacing(6)
        splitter.addWidget(plots_w)
        splitter.setStretchFactor(0, 3)

        pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)

        # Spectrogram
        spec_label = QtWidgets.QLabel("FULL-SPECTRUM SPECTROGRAM")
        spec_label.setStyleSheet(f"color:{ACCENT2}; font-size:11px; font-weight:500; letter-spacing:1px;")
        plots_v.addWidget(spec_label)

        self._spec_view = pg.GraphicsLayoutWidget()
        self._spec_view.setFixedHeight(340)
        plots_v.addWidget(self._spec_view)

        self._spec_plot = self._spec_view.addPlot()
        self._spec_plot.setLabel("left",   "Frequency (Hz)", color=TEXT)
        self._spec_plot.setLabel("bottom", "Time (s)",       color=TEXT)
        self._spec_plot.getAxis("left").setTextPen(TEXT)
        self._spec_plot.getAxis("bottom").setTextPen(TEXT)

        self._img_item = pg.ImageItem()
        self._spec_plot.addItem(self._img_item)
        self._lut = make_colormap(self.cmap_name)
        self._img_item.setLookupTable(self._lut)
        self._img_item.setLevels([-120, 0])

        # band overlay regions
        self._region_band = pg.LinearRegionItem(
            values=[self.band_lo, self.band_hi],
            orientation="horizontal",
            brush=pg.mkBrush(255, 200, 50, 35),
            pen=pg.mkPen(255, 200, 50, 200, width=1),
            movable=True,
        )
        self._region_band.sigRegionChanged.connect(self._on_band_region_changed)
        self._spec_plot.addItem(self._region_band)

        self._region_disp = pg.LinearRegionItem(
            values=[self.disp_lo, self.disp_hi],
            orientation="horizontal",
            brush=pg.mkBrush(63, 168, 255, 18),
            pen=pg.mkPen(63, 168, 255, 150, width=1),
            movable=True,
        )
        self._region_disp.sigRegionChanged.connect(self._on_disp_region_changed)
        self._spec_plot.addItem(self._region_disp)

        # Waveform
        wave_label = QtWidgets.QLabel("WAVEFORM")
        wave_label.setStyleSheet(f"color:{ACCENT2}; font-size:11px; font-weight:500; letter-spacing:1px;")
        plots_v.addWidget(wave_label)

        self._wave_view = pg.GraphicsLayoutWidget()
        self._wave_view.setFixedHeight(150)
        plots_v.addWidget(self._wave_view)

        self._wave_plot = self._wave_view.addPlot()
        self._wave_plot.setLabel("left",   "Amplitude",  color=TEXT)
        self._wave_plot.setLabel("bottom", "Time (s)",   color=TEXT)
        self._wave_plot.setYRange(-1, 1)
        self._wave_plot.getAxis("left").setTextPen(TEXT)
        self._wave_plot.getAxis("bottom").setTextPen(TEXT)
        n_hist = int(self.fs * self.hist_sec)
        t_wave = np.linspace(0, self.hist_sec, n_hist)
        self._wave_curve = self._wave_plot.plot(
            t_wave, np.zeros(n_hist),
            pen=pg.mkPen(ACCENT, width=1))

        # PSD
        psd_label = QtWidgets.QLabel("POWER SPECTRUM (latest frame)")
        psd_label.setStyleSheet(f"color:{ACCENT2}; font-size:11px; font-weight:500; letter-spacing:1px;")
        plots_v.addWidget(psd_label)

        self._psd_view = pg.GraphicsLayoutWidget()
        self._psd_view.setFixedHeight(170)
        plots_v.addWidget(self._psd_view)

        self._psd_plot = self._psd_view.addPlot()
        self._psd_plot.setLabel("left",   "dB",          color=TEXT)
        self._psd_plot.setLabel("bottom", "Frequency (Hz)", color=TEXT)
        self._psd_plot.setYRange(-120, 0)
        self._psd_plot.getAxis("left").setTextPen(TEXT)
        self._psd_plot.getAxis("bottom").setTextPen(TEXT)
        self._psd_curve = self._psd_plot.plot(
            pen=pg.mkPen(WARN, width=1.5))
        self._peak_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(ACCENT, width=1.5, style=QtCore.Qt.DashLine))
        self._psd_plot.addItem(self._peak_line)

        # RIGHT — controls panel
        ctrl_scroll = QtWidgets.QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(310)
        ctrl_scroll.setStyleSheet(f"QScrollArea{{border:none; background:{SURFACE};}}")
        splitter.addWidget(ctrl_scroll)

        ctrl_w  = QtWidgets.QWidget()
        ctrl_w.setStyleSheet(f"background:{SURFACE};")
        ctrl_v  = QtWidgets.QVBoxLayout(ctrl_w)
        ctrl_v.setContentsMargins(14, 14, 14, 14)
        ctrl_v.setSpacing(14)
        ctrl_scroll.setWidget(ctrl_w)

        def section(title):
            lbl = QtWidgets.QLabel(title)
            lbl.setStyleSheet(f"color:{ACCENT}; font-size:11px; font-weight:500; "
                               f"letter-spacing:1px; border-bottom:1px solid {BORDER}; "
                               "padding-bottom:4px;")
            ctrl_v.addWidget(lbl)

        # ── Display range ──────────────────────────────────────────────────
        section("DISPLAY FREQUENCY RANGE")
        self._sl_disp_lo, self._lbl_disp_lo = self._freq_slider(
            ctrl_v, "Min display (Hz)", 0, self.fs//2, int(self.disp_lo),
            self._on_disp_lo_slider)
        self._sl_disp_hi, self._lbl_disp_hi = self._freq_slider(
            ctrl_v, "Max display (Hz)", 0, self.fs//2, int(self.disp_hi),
            self._on_disp_hi_slider)

        hint = QtWidgets.QLabel("  ↑ Drag blue region on spectrogram or use sliders")
        hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        hint.setWordWrap(True)
        ctrl_v.addWidget(hint)

        # ── Analysis band ──────────────────────────────────────────────────
        section("ANALYSIS / TRANSPOSE BAND")
        self._sl_band_lo, self._lbl_band_lo = self._freq_slider(
            ctrl_v, "Band low (Hz)", 0, self.fs//2, int(self.band_lo),
            self._on_band_lo_slider)
        self._sl_band_hi, self._lbl_band_hi = self._freq_slider(
            ctrl_v, "Band high (Hz)", 0, self.fs//2, int(self.band_hi),
            self._on_band_hi_slider)

        hint2 = QtWidgets.QLabel("  ↑ Drag yellow region on spectrogram or use sliders")
        hint2.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        hint2.setWordWrap(True)
        ctrl_v.addWidget(hint2)

        # ── Transpose ──────────────────────────────────────────────────────
        section("TRANSPOSITION")

        row_tr = QtWidgets.QHBoxLayout()
        row_tr.addWidget(QtWidgets.QLabel("Ratio (÷):"))
        self._spin_ratio = QtWidgets.QDoubleSpinBox()
        self._spin_ratio.setRange(1.0, 200.0)
        self._spin_ratio.setSingleStep(1.0)
        self._spin_ratio.setDecimals(1)
        self._spin_ratio.setValue(self.trans_ratio)
        self._spin_ratio.setFixedHeight(30)
        self._spin_ratio.valueChanged.connect(self._on_ratio_changed)
        row_tr.addWidget(self._spin_ratio)
        ctrl_v.addLayout(row_tr)

        self._lbl_tr_info = QtWidgets.QLabel()
        self._lbl_tr_info.setStyleSheet(f"color:{ACCENT2}; font-size:12px;")
        self._lbl_tr_info.setWordWrap(True)
        ctrl_v.addWidget(self._lbl_tr_info)
        self._update_tr_label()

        # preset buttons
        preset_row = QtWidgets.QHBoxLayout()
        for label, (lo, hi, ratio) in [
            ("Bat ultrasound\n20–80 kHz ÷10", (20000, 80000, 10)),
            ("Dog whistle\n18–25 kHz ÷8",     (18000, 25000, 8)),
            ("Infrasound\n1–19 Hz ×50",        (1, 19, 0.02)),
        ]:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(52)
            btn.setStyleSheet(f"QPushButton{{background:{BORDER}; color:{TEXT}; border:1px solid {MUTED}; "
                               "border-radius:6px; font-size:11px;}} "
                               f"QPushButton:hover{{background:{SURFACE};}}")
            btn.clicked.connect(lambda _, lo=lo, hi=hi, r=ratio: self._apply_preset(lo, hi, r))
            preset_row.addWidget(btn)
        ctrl_v.addLayout(preset_row)

        # playback
        section("PLAYBACK")
        self._chk_play = QtWidgets.QCheckBox("Live transposed playback")
        self._chk_play.setChecked(self.play_live)
        self._chk_play.stateChanged.connect(self._on_play_toggled)
        ctrl_v.addWidget(self._chk_play)

        vol_row = QtWidgets.QHBoxLayout()
        vol_row.addWidget(QtWidgets.QLabel("Volume:"))
        self._sl_vol = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sl_vol.setRange(0, 100)
        self._sl_vol.setValue(80)
        self._lbl_vol = QtWidgets.QLabel("80%")
        self._sl_vol.valueChanged.connect(lambda v: self._lbl_vol.setText(f"{v}%"))
        vol_row.addWidget(self._sl_vol)
        vol_row.addWidget(self._lbl_vol)
        ctrl_v.addLayout(vol_row)

        # info panel
        section("LIVE METRICS")
        self._lbl_peak = QtWidgets.QLabel("Peak: —")
        self._lbl_rms  = QtWidgets.QLabel("RMS: —")
        self._lbl_sr   = QtWidgets.QLabel(f"SR: {self.fs} Hz")
        for lbl in [self._lbl_peak, self._lbl_rms, self._lbl_sr]:
            lbl.setStyleSheet(f"color:{TEXT}; font-size:13px; font-family:monospace;")
            ctrl_v.addWidget(lbl)

        ctrl_v.addStretch()

        # status bar
        self._status = self.statusBar()
        self._status.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self._status.showMessage("Ready — click Start capture")

    # ── slider helper ──────────────────────────────────────────────────────

    def _freq_slider(self, parent_layout, label_text, lo, hi, val, callback):
        row = QtWidgets.QHBoxLayout()
        lbl_name = QtWidgets.QLabel(label_text)
        lbl_name.setFixedWidth(130)
        lbl_name.setStyleSheet(f"color:{TEXT}; font-size:12px;")
        sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(val)
        lbl_val = QtWidgets.QLabel(self._fmt_hz(val))
        lbl_val.setFixedWidth(72)
        lbl_val.setStyleSheet(f"color:{ACCENT2}; font-size:12px; font-family:monospace;")
        sl.valueChanged.connect(callback)
        sl.valueChanged.connect(lambda v, lv=lbl_val: lv.setText(self._fmt_hz(v)))
        row.addWidget(lbl_name)
        row.addWidget(sl)
        row.addWidget(lbl_val)
        parent_layout.addLayout(row)
        return sl, lbl_val

    @staticmethod
    def _fmt_hz(v):
        return f"{v/1000:.1f} kHz" if v >= 1000 else f"{v} Hz"

    # ── theme ──────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
            QLabel {{ color: {TEXT}; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {SURFACE}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 5px; padding: 3px 8px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QSlider::groove:horizontal {{
                height: 4px; background: {BORDER}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
            QPushButton {{
                background: {SURFACE}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 14px;
            }}
            QPushButton:hover  {{ background: {BORDER}; }}
            QPushButton:pressed {{ background: #1a1d24; }}
            QScrollBar:vertical {{ width: 6px; background: {SURFACE}; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
            QStatusBar {{ background: {SURFACE}; }}
            QCheckBox {{ color: {TEXT}; }}
            QCheckBox::indicator {{ width:14px; height:14px; border:1px solid {MUTED}; border-radius:3px; }}
            QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
        """)

    # ── slider → value sync ────────────────────────────────────────────────

    def _on_disp_lo_slider(self, v):
        self.disp_lo = float(v)
        self._region_disp.blockSignals(True)
        self._region_disp.setRegion([self.disp_lo, self.disp_hi])
        self._region_disp.blockSignals(False)
        self._update_spec_range()

    def _on_disp_hi_slider(self, v):
        self.disp_hi = float(v)
        self._region_disp.blockSignals(True)
        self._region_disp.setRegion([self.disp_lo, self.disp_hi])
        self._region_disp.blockSignals(False)
        self._update_spec_range()

    def _on_band_lo_slider(self, v):
        self.band_lo = float(v)
        self._region_band.blockSignals(True)
        self._region_band.setRegion([self.band_lo, self.band_hi])
        self._region_band.blockSignals(False)
        self._update_tr_label()

    def _on_band_hi_slider(self, v):
        self.band_hi = float(v)
        self._region_band.blockSignals(True)
        self._region_band.setRegion([self.band_lo, self.band_hi])
        self._region_band.blockSignals(False)
        self._update_tr_label()

    # region → slider sync
    def _on_band_region_changed(self):
        lo, hi = self._region_band.getRegion()
        lo = max(0, min(lo, self.fs // 2))
        hi = max(0, min(hi, self.fs // 2))
        self.band_lo, self.band_hi = lo, hi
        self._sl_band_lo.blockSignals(True); self._sl_band_lo.setValue(int(lo)); self._sl_band_lo.blockSignals(False)
        self._sl_band_hi.blockSignals(True); self._sl_band_hi.setValue(int(hi)); self._sl_band_hi.blockSignals(False)
        self._lbl_band_lo.setText(self._fmt_hz(int(lo)))
        self._lbl_band_hi.setText(self._fmt_hz(int(hi)))
        self._update_tr_label()

    def _on_disp_region_changed(self):
        lo, hi = self._region_disp.getRegion()
        lo = max(0, min(lo, self.fs // 2))
        hi = max(0, min(hi, self.fs // 2))
        self.disp_lo, self.disp_hi = lo, hi
        self._sl_disp_lo.blockSignals(True); self._sl_disp_lo.setValue(int(lo)); self._sl_disp_lo.blockSignals(False)
        self._sl_disp_hi.blockSignals(True); self._sl_disp_hi.setValue(int(hi)); self._sl_disp_hi.blockSignals(False)
        self._lbl_disp_lo.setText(self._fmt_hz(int(lo)))
        self._lbl_disp_hi.setText(self._fmt_hz(int(hi)))
        self._update_spec_range()

    def _update_spec_range(self):
        self._spec_plot.setYRange(self.disp_lo, self.disp_hi, padding=0)

    def _on_ratio_changed(self, v):
        self.trans_ratio = v
        self._out_sr = max(1, int(self.fs / v))
        self._update_tr_label()
        # restart output stream if running
        if self._running and self.play_live:
            self._restart_output()

    def _on_sr_changed(self, v):
        self.fs = v
        nyq = v // 2
        self._sl_disp_lo.setRange(0, nyq)
        self._sl_disp_hi.setRange(0, nyq)
        self._sl_band_lo.setRange(0, nyq)
        self._sl_band_hi.setRange(0, nyq)
        self._lbl_sr.setText(f"SR: {v} Hz")

    def _on_cmap_changed(self, name):
        self.cmap_name = name
        self._lut = make_colormap(name)
        self._img_item.setLookupTable(self._lut)

    def _on_play_toggled(self, state):
        self.play_live = bool(state)
        if self._running:
            if self.play_live:
                self._start_output()
            else:
                self._stop_output()

    def _update_tr_label(self):
        lo_out = self.band_lo / self.trans_ratio
        hi_out = self.band_hi / self.trans_ratio
        self._lbl_tr_info.setText(
            f"Band  {self._fmt_hz(int(self.band_lo))} – {self._fmt_hz(int(self.band_hi))}\n"
            f"→ becomes  {self._fmt_hz(int(lo_out))} – {self._fmt_hz(int(hi_out))}\n"
            f"Output SR: {int(self.fs / self.trans_ratio):,} Hz"
        )

    def _apply_preset(self, lo, hi, ratio):
        self.band_lo, self.band_hi, self.trans_ratio = float(lo), float(hi), float(ratio)
        self._sl_band_lo.setValue(int(lo))
        self._sl_band_hi.setValue(int(hi))
        self._spin_ratio.setValue(ratio)
        self._update_tr_label()

    # ── audio ──────────────────────────────────────────────────────────────

    def _input_callback(self, indata, frames, t, status):
        if status:
            self.sig_status.emit(str(status))
        chunk = indata[:, 0].copy()
        if not self._q.full():
            self._q.put_nowait(chunk)

    def _output_callback(self, outdata, frames, t, status):
        vol = self._sl_vol.value() / 100.0
        with self._out_lock:
            n = min(frames, len(self._out_buf))
            outdata[:n, 0] = self._out_buf[:n] * vol
            outdata[n:, 0] = 0.0
            self._out_buf = self._out_buf[n:]

    def _process_loop(self):
        while self._running:
            try:
                chunk = self._q.get(timeout=0.15)
            except queue.Empty:
                continue
            # ring buffer
            n = len(chunk)
            with self._hist_lock:
                self._hist = np.roll(self._hist, -n)
                self._hist[-n:] = chunk
            # raw recording
            self._raw_chunks.append(chunk)
            # transpose
            filtered    = bandpass(chunk, self.band_lo, self.band_hi, self.fs)
            tr, out_sr  = transpose(filtered, self.fs, self.trans_ratio)
            tr_norm     = normalize(tr)
            self._tr_chunks.append(tr_norm)
            if self.play_live:
                with self._out_lock:
                    self._out_buf = np.concatenate([self._out_buf, tr_norm])

    def _start_output(self):
        if self._out_stream is not None:
            return
        out_sr = max(1, int(self.fs / self.trans_ratio))
        kw = dict(samplerate=out_sr, blocksize=512, dtype="float32",
                  channels=1, callback=self._output_callback)
        if self.out_device is not None:
            kw["device"] = self.out_device
        try:
            self._out_stream = sd.OutputStream(**kw)
            self._out_stream.start()
        except Exception as e:
            self.sig_status.emit(f"Output stream error: {e}")

    def _stop_output(self):
        if self._out_stream:
            try:
                self._out_stream.stop(); self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

    def _restart_output(self):
        self._stop_output()
        if self.play_live:
            self._start_output()

    def _toggle_capture(self):
        if not self._running:
            self._start_capture()
        else:
            self._stop_capture()

    def _start_capture(self):
        dev_idx = self._cb_device.currentData()
        self.fs  = self._spin_sr.value()
        n_hist   = int(self.fs * self.hist_sec)
        with self._hist_lock:
            self._hist = np.zeros(n_hist, dtype=np.float32)

        kw = dict(samplerate=self.fs, blocksize=self.block_size,
                  dtype="float32", channels=1, callback=self._input_callback)
        if dev_idx is not None:
            kw["device"] = dev_idx
        try:
            self._in_stream = sd.InputStream(**kw)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Device error", str(e))
            return

        self._running = True
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_thread.start()
        self._in_stream.start()
        if self.play_live:
            self._start_output()

        self._btn_start.setText("■  Stop capture")
        self._btn_save_raw.setEnabled(True)
        self._btn_save_tr.setEnabled(True)
        self._lbl_sr.setText(f"SR: {self.fs} Hz")
        self.sig_status.emit(f"Capturing — device: {self._cb_device.currentText()[:50]}")

    def _stop_capture(self):
        self._running = False
        if self._in_stream:
            try: self._in_stream.stop(); self._in_stream.close()
            except Exception: pass
            self._in_stream = None
        self._stop_output()
        self._btn_start.setText("▶  Start capture")
        self.sig_status.emit("Stopped")

    # ── plot refresh ───────────────────────────────────────────────────────

    def _refresh_plots(self):
        with self._hist_lock:
            data = self._hist.copy()

        if np.all(data == 0):
            return

        # spectrogram
        freqs, times, Sxx = sig.spectrogram(
            data, fs=self.fs,
            nperseg=self.args.nperseg,
            noverlap=self.args.noverlap,
            scaling="spectrum", mode="magnitude")

        Sxx_db = 20 * np.log10(Sxx + 1e-12)

        # display freq mask
        mask = (freqs >= self.disp_lo) & (freqs <= self.disp_hi)
        f_show  = freqs[mask]
        Sxx_show = Sxx_db[mask, :]

        if f_show.size == 0:
            return

        df = f_show[1] - f_show[0] if len(f_show) > 1 else 1.0
        dt = times[1]  - times[0]  if len(times) > 1  else 1.0

        self._img_item.setImage(
            Sxx_show.T,
            autoLevels=False,
            rect=pg.QtCore.QRectF(times[0], f_show[0],
                                   times[-1] - times[0],
                                   f_show[-1] - f_show[0]))
        self._spec_plot.setYRange(self.disp_lo, self.disp_hi, padding=0)
        self._spec_plot.setXRange(times[0], times[-1], padding=0)

        # waveform
        n_hist = len(data)
        t_wave = np.linspace(0, self.hist_sec, n_hist)
        step   = max(1, n_hist // 4000)   # downsample for speed
        self._wave_curve.setData(t_wave[::step], data[::step])

        # PSD — full frequency range of last frame
        last_col = Sxx_db[:, -1]
        self._psd_curve.setData(freqs, last_col)
        self._psd_plot.setXRange(self.disp_lo, self.disp_hi, padding=0)

        # peak marker
        full_mask = (freqs >= self.band_lo) & (freqs <= self.band_hi)
        band_col  = Sxx[:, -1][full_mask]
        if band_col.size > 0 and band_col.max() > 0:
            peak_hz = freqs[full_mask][np.argmax(band_col)]
            self._peak_line.setValue(peak_hz)
            self._lbl_peak.setText(f"Peak: {self._fmt_hz(int(peak_hz))}")

        rms = np.sqrt(np.mean(data ** 2))
        rms_db = 20 * np.log10(rms + 1e-9)
        self._lbl_rms.setText(f"RMS:  {rms_db:.1f} dBFS")

    # ── save ───────────────────────────────────────────────────────────────

    def _save_raw(self):
        if not self._raw_chunks:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save raw audio", "raw_capture.wav", "WAV files (*.wav)")
        if not path:
            return
        data = np.concatenate(self._raw_chunks)
        sf.write(path, data, self.fs, subtype="FLOAT")
        self.sig_status.emit(f"Saved raw → {path}")

    def _save_transposed(self):
        if not self._tr_chunks:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save transposed audio", "transposed_audible.wav",
            "WAV files (*.wav)")
        if not path:
            return
        data  = np.concatenate(self._tr_chunks)
        out_sr = max(1, int(self.fs / self.trans_ratio))
        sf.write(path, data, out_sr, subtype="FLOAT")
        self.sig_status.emit(f"Saved transposed (SR={out_sr} Hz) → {path}")

    @QtCore.pyqtSlot(np.ndarray)
    def _on_new_block(self, chunk):
        pass    # placeholder; processing in thread

    @QtCore.pyqtSlot(str)
    def _set_status(self, msg):
        self._status.showMessage(msg)

    def closeEvent(self, event):
        self._stop_capture()
        super().closeEvent(event)


# ── offline analysis mode ──────────────────────────────────────────────────

def analyze_file(path, args):
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    mono = data[:, 0]
    print(f"  File     : {path}")
    print(f"  SR       : {fs} Hz  |  Duration: {len(mono)/fs:.2f}s")

    freqs, times, Sxx = sig.spectrogram(
        mono, fs=fs, nperseg=args.nperseg, noverlap=args.noverlap,
        scaling="spectrum", mode="magnitude")
    Sxx_db = 20 * np.log10(Sxx + 1e-12)
    freq_max = min(args.freq_max, fs // 2)
    mask = (freqs >= args.freq_min) & (freqs <= freq_max)
    freqs, Sxx_db = freqs[mask], Sxx_db[mask, :]

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(title=f"Offline — {Path(path).name}", show=True,
                                   size=(1200, 700))

    pg.setConfigOptions(antialias=True, background=BG, foreground=TEXT)

    p1 = win.addPlot(title="Spectrogram", row=0, col=0)
    img = pg.ImageItem()
    lut = make_colormap(args.colormap)
    img.setLookupTable(lut)
    img.setLevels([-120, 0])
    img.setImage(Sxx_db.T, autoLevels=False,
                 rect=pg.QtCore.QRectF(times[0], freqs[0],
                                        times[-1]-times[0], freqs[-1]-freqs[0]))
    p1.addItem(img)
    p1.setLabel("left", "Frequency (Hz)"); p1.setLabel("bottom", "Time (s)")

    p2 = win.addPlot(title="Waveform", row=1, col=0)
    t  = np.linspace(0, len(mono)/fs, len(mono))
    step = max(1, len(mono)//8000)
    p2.plot(t[::step], mono[::step], pen=pg.mkPen(ACCENT, width=1))
    p2.setLabel("left","Amplitude"); p2.setLabel("bottom","Time (s)")

    if args.save_transposed:
        filt = bandpass(mono, args.band_low, args.band_high, fs)
        tr, out_sr = transpose(filt, fs, args.transpose_ratio)
        tr = normalize(tr)
        sf.write(args.save_transposed, tr, out_sr, subtype="FLOAT")
        print(f"  Transposed saved → {args.save_transposed}  (SR={out_sr} Hz)")

    app.exec_()


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="InfraUltraScope UI — Infrasound & Ultrasound Visualizer + Decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--file",    metavar="WAV", help="Offline analysis of WAV")
    mode.add_argument("--devices", action="store_true", help="List devices and exit")

    p.add_argument("--samplerate",       type=int,   default=192_000)
    p.add_argument("--block-size",       type=int,   default=2048, dest="block_size")
    p.add_argument("--nperseg",          type=int,   default=2048)
    p.add_argument("--noverlap",         type=int,   default=1536)
    p.add_argument("--freq-min",         type=float, default=0,      dest="freq_min")
    p.add_argument("--freq-max",         type=float, default=96_000, dest="freq_max")
    p.add_argument("--band-low",         type=float, default=20_000, dest="band_low")
    p.add_argument("--band-high",        type=float, default=40_000, dest="band_high")
    p.add_argument("--transpose-ratio",  type=float, default=10.0,   dest="transpose_ratio")
    p.add_argument("--play-live",        action="store_true",         dest="play_live")
    p.add_argument("--input-device",     type=int,   default=None,   dest="input_device")
    p.add_argument("--output-device",    type=int,   default=None,   dest="output_device")
    p.add_argument("--save-transposed",  metavar="FILE.wav", default=None, dest="save_transposed")
    p.add_argument("--colormap",         default="inferno")
    return p


def main():
    args = build_parser().parse_args()

    if args.devices:
        for idx, name in list_devices():
            print(f"  [{idx:2d}]  {name}")
        return

    if args.file:
        analyze_file(args.file, args)
        return

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()