"""
ACAI — gui/app.py
Main application window.
Handles: file selection, session naming, AOI drawing, pose analysis threading + progress.
"""

import os
import sys
import json
import datetime
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QFrame, QSizePolicy,
    QScrollArea, QMessageBox, QListWidget, QListWidgetItem,
    QProgressBar
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QThread
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont
)

# ---- locate project root (one level up from gui/) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ACAI_main

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
import audio_analysis
import gpt_analysis
import drive_upload


# ============================================================
# CONSTANTS
# ============================================================
AOI_SEQUENCE = [
    ("Teaching Slides", QColor(52, 152, 219)),
    ("Students Area",   QColor(46, 204, 113)),
    ("Computer/Laptop", QColor(241, 196, 15)),
    ("Whiteboard",      QColor(231, 76, 60)),
]

DARK_BG      = "#1a1a2e"
PANEL_BG     = "#16213e"
ACCENT       = "#e94560"
ACCENT_HOVER = "#c73652"
TEXT_PRIMARY = "#eaeaea"
TEXT_MUTED   = "#8892a4"
BORDER       = "#2a2a4a"
SUCCESS      = "#2ecc71"
WARNING      = "#f39c12"


# ============================================================
# AUDIO WORKER — runs Whisper transcription in background thread
# ============================================================
class AudioWorker(QThread):
    status   = pyqtSignal(str)    # progress messages
    finished = pyqtSignal(dict)   # output file paths
    error    = pyqtSignal(str)

    def __init__(self, audio_path, output_dir):
        super().__init__()
        self.audio_path = audio_path
        self.output_dir = output_dir

    def run(self):
        try:
            result = audio_analysis.run_transcription(
                self.audio_path,
                self.output_dir,
                progress_callback=lambda msg: self.status.emit(msg)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# GPT WORKER — runs GPT analysis in background thread
# ============================================================
class GPTWorker(QThread):
    status   = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, segments_csv, transcript_txt, audio_path, output_dir, session_name, api_key=None):
        super().__init__()
        self.segments_csv   = segments_csv
        self.transcript_txt = transcript_txt
        self.audio_path     = audio_path
        self.output_dir     = output_dir
        self.session_name   = session_name
        self.api_key        = api_key

    def run(self):
        try:
            result = gpt_analysis.run_gpt_analysis(
                self.segments_csv,
                self.transcript_txt,
                self.audio_path,
                self.output_dir,
                session_name=self.session_name,
                progress_callback=lambda msg: self.status.emit(msg),
                api_key=self.api_key
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# DRIVE WORKER — copies output files to Google Drive sync folder
# ============================================================
class DriveWorker(QThread):
    status   = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, output_dir, drive_root, session_name):
        super().__init__()
        self.output_dir   = output_dir
        self.drive_root   = drive_root
        self.session_name = session_name

    def run(self):
        try:
            result = drive_upload.copy_to_drive(
                output_dir=self.output_dir,
                session_name=self.session_name,
                drive_root=self.drive_root,
                progress_callback=lambda msg: self.status.emit(msg)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# POSE WORKER — runs ACAI_main.main() in a background thread
# ============================================================
class PoseWorker(QThread):
    progress = pyqtSignal(int, int, float)   # frame, total, fps
    finished = pyqtSignal(str)               # output directory path
    error    = pyqtSignal(str)               # error message

    def __init__(self, video_path, aoi_list, output_dir, session_name):
        super().__init__()
        self.video_path   = video_path
        self.aoi_list     = aoi_list
        self.output_dir   = output_dir
        self.session_name = session_name

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(self.video_path))[0]

            # inject globals into ACAI_main
            ACAI_main.VIDEO_NAME               = self.video_path
            ACAI_main.SKIP_AOI_SELECTION       = True
            ACAI_main.areasofinterest_list     = list(self.aoi_list)
            ACAI_main.OUTPUT_VIDEO             = os.path.join(self.output_dir, f"{stem}_out.mp4")
            ACAI_main.OUTPUT_LANDMARK_CSV      = os.path.join(self.output_dir, "reportsourcefile_landmarkcoordinates.csv")
            ACAI_main.OUTPUT_TEACHINGSTYLE_CSV = os.path.join(self.output_dir, "teachingstyle_output.csv")
            ACAI_main.OUTPUT_COG_CSV           = os.path.join(self.output_dir, "reportsourcefile_center_of_gravity.csv")
            ACAI_main.PROGRESS_CALLBACK        = self._on_progress

            ACAI_main.main()

            ACAI_main.PROGRESS_CALLBACK = None
            self.finished.emit(self.output_dir)

        except Exception as e:
            ACAI_main.PROGRESS_CALLBACK = None
            self.error.emit(str(e))
        finally:
            self.msleep(200)   # let Qt clean up before thread exits

    def _on_progress(self, frame, total, fps):
        self.progress.emit(frame, total, fps)


# ============================================================
# AOI CANVAS
# ============================================================
class AOICanvas(QLabel):
    aoi_updated = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._original_frame = None
        self._scale          = 1.0
        self._drawing        = False
        self._drag_start     = None
        self._drag_current   = None
        self._aois           = []

    def load_frame(self, frame_bgr):
        self._original_frame = frame_bgr.copy()
        self._aois = []
        self._redraw()

    def get_aois(self):
        return list(self._aois)

    def undo_last(self):
        if self._aois:
            self._aois.pop()
            self._redraw()
            self.aoi_updated.emit(self._aois)

    def clear_all(self):
        self._aois = []
        self._redraw()
        self.aoi_updated.emit(self._aois)

    def mousePressEvent(self, event):
        if self._original_frame is None:
            return
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._drag_start   = event.pos()
            self._drag_current = event.pos()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._drag_current = event.pos()
            self._redraw()

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            rect_orig = self._display_to_original(self._drag_start, self._drag_current)
            if rect_orig is not None:
                self._aois.append(rect_orig)
                self.aoi_updated.emit(self._aois)
            self._drag_start   = None
            self._drag_current = None
            self._redraw()

    def _redraw(self, available_w=None):
        if self._original_frame is None:
            return
        orig_h, orig_w = self._original_frame.shape[:2]

        # use explicitly passed width from window resize, else fall back to current widget width
        widget_w    = available_w if available_w and available_w > 100 else max(self.width(), 100)
        self._scale = widget_w / orig_w
        display_h   = int(orig_h * self._scale)

        display     = cv2.resize(self._original_frame, (widget_w, display_h))
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        qimg   = QImage(display_rgb.data, widget_w, display_h,
                        display_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        for i, ((x1, y1), (x2, y2)) in enumerate(self._aois):
            color = AOI_SEQUENCE[min(i, len(AOI_SEQUENCE) - 1)][1]
            dx1, dy1 = int(x1 * self._scale), int(y1 * self._scale)
            dx2, dy2 = int(x2 * self._scale), int(y2 * self._scale)
            fill = QColor(color)
            fill.setAlpha(40)
            painter.fillRect(QRect(dx1, dy1, dx2 - dx1, dy2 - dy1), fill)
            painter.setPen(QPen(color, 2))
            painter.drawRect(QRect(dx1, dy1, dx2 - dx1, dy2 - dy1))
            lbl = AOI_SEQUENCE[min(i, len(AOI_SEQUENCE) - 1)][0]
            painter.setFont(QFont("Arial", 13, QFont.Bold))
            painter.setPen(QPen(Qt.white))
            painter.drawText(dx1 + 6, dy1 + 18, lbl)

        if self._drawing and self._drag_start and self._drag_current:
            idx   = len(self._aois)
            color = AOI_SEQUENCE[min(idx, len(AOI_SEQUENCE) - 1)][1]
            painter.setPen(QPen(color, 2, Qt.DashLine))
            painter.drawRect(QRect(self._drag_start, self._drag_current).normalized())

        painter.end()
        self.setPixmap(pixmap)
        self.setFixedHeight(display_h)

    def _display_to_original(self, p1, p2):
        if self._scale == 0:
            return None
        x1 = int(min(p1.x(), p2.x()) / self._scale)
        y1 = int(min(p1.y(), p2.y()) / self._scale)
        x2 = int(max(p1.x(), p2.x()) / self._scale)
        y2 = int(max(p1.y(), p2.y()) / self._scale)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return None
        return (x1, y1), (x2, y2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()


# ============================================================
# SIDE PANEL
# ============================================================
class SidePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(380)
        self.setStyleSheet(f"background-color: {PANEL_BG};")

        # ---- outer layout holds the scroll area ----
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: {PANEL_BG}; }}
            QScrollBar:vertical {{
                background: {PANEL_BG}; width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER}; border-radius: 3px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        # ---- inner content widget ----
        content = QWidget()
        content.setStyleSheet(f"background-color: {PANEL_BG};")
        scroll.setWidget(content)
        outer.addWidget(scroll)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Title
        title = QLabel("ACAI")
        title.setFont(QFont("Arial", 29, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT}; letter-spacing: 6px;")
        subtitle = QLabel("Automated Classroom Analysis & Insights")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px;")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._divider())

        # Session
        layout.addWidget(self._section_label("SESSION"))
        self.session_input = QLineEdit()
        self.session_input.setPlaceholderText("Auto-generated on run...")
        self.session_input.setStyleSheet(self._input_style())
        self.session_input.setFont(QFont("Arial", 12))
        layout.addWidget(self.session_input)
        layout.addWidget(self._divider())

        # OpenAI API Key
        layout.addWidget(self._section_label("OPENAI API KEY"))
        api_key_row = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-...")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setStyleSheet(self._input_style())
        self.api_key_input.setFont(QFont("Arial", 12))
        self.btn_save_key = self._make_button("Save", secondary=True)
        self.btn_save_key.setFixedWidth(70)
        self.btn_save_key.setMinimumHeight(38)
        api_key_row.addWidget(self.api_key_input)
        api_key_row.addWidget(self.btn_save_key)
        layout.addLayout(api_key_row)
        self.api_key_status = QLabel("")
        self.api_key_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: 'Arial';")
        layout.addWidget(self.api_key_status)

        api_file_row = QHBoxLayout()
        self.btn_load_json = self._make_button("Load JSON", secondary=True)
        self.btn_load_txt  = self._make_button("Load TXT", secondary=True)
        api_file_row.addWidget(self.btn_load_json)
        api_file_row.addWidget(self.btn_load_txt)
        layout.addLayout(api_file_row)
        layout.addWidget(self._divider())

        # Google Drive folder
        layout.addWidget(self._section_label("GOOGLE DRIVE FOLDER"))
        self.drive_label = QLabel("No folder selected")
        self.drive_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: 'Arial';")
        self.drive_label.setWordWrap(True)
        drive_row = QHBoxLayout()
        self.btn_drive = self._make_button("Browse", secondary=True)
        self.btn_save_drive = self._make_button("Save", secondary=True)
        self.btn_save_drive.setFixedWidth(70)
        drive_row.addWidget(self.btn_drive)
        drive_row.addWidget(self.btn_save_drive)
        layout.addWidget(self.drive_label)
        layout.addLayout(drive_row)
        self.drive_status = QLabel("")
        self.drive_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: 'Arial';")
        layout.addWidget(self.drive_status)
        layout.addWidget(self._divider())

        # Video
        layout.addWidget(self._section_label("VIDEO FILE"))
        self.video_label = QLabel("No file selected")
        self.video_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-family: 'Arial';")
        self.video_label.setWordWrap(True)
        self.btn_video = self._make_button("Browse Video", secondary=True)
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_video)
        layout.addWidget(self._divider())

        # Audio
        layout.addWidget(self._section_label("AUDIO FILE"))
        self.audio_label = QLabel("No file selected")
        self.audio_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-family: 'Arial';")
        self.audio_label.setWordWrap(True)
        self.btn_audio = self._make_button("Browse Audio", secondary=True)
        layout.addWidget(self.audio_label)
        layout.addWidget(self.btn_audio)
        layout.addWidget(self._divider())

        # AOI list
        layout.addWidget(self._section_label("AREAS OF INTEREST"))
        self.aoi_list_widget = QListWidget()
        self.aoi_list_widget.setStyleSheet(f"""
            QListWidget {{
                background: {DARK_BG}; border: 1px solid {BORDER};
                border-radius: 4px; color: {TEXT_PRIMARY};
                font-family: 'Arial'; font-size: 12px; padding: 4px;
            }}
            QListWidget::item {{ padding: 4px 6px; }}
        """)
        self.aoi_list_widget.setMinimumHeight(80)
        self.aoi_list_widget.setMaximumHeight(160)
        self.aoi_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.aoi_list_widget)

        aoi_btns = QHBoxLayout()
        self.btn_undo  = self._make_button("Undo", secondary=True)
        self.btn_clear = self._make_button("Clear All", secondary=True)
        aoi_btns.addWidget(self.btn_undo)
        aoi_btns.addWidget(self.btn_clear)
        layout.addLayout(aoi_btns)

        hint = QLabel("① Slides  ② Students  ③ Computer  ④+ Whiteboard")
        hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: 'Arial';")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch()
        layout.addWidget(self._divider())

        # Progress
        layout.addWidget(self._section_label("PROGRESS"))

        self.stage_label = QLabel("—")
        self.stage_label.setFont(QFont("Arial", 12))
        self.stage_label.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {DARK_BG}; border-radius: 3px; border: none;
            }}
            QProgressBar::chunk {{
                background: {ACCENT}; border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        self.fps_label = QLabel("")
        self.fps_label.setFont(QFont("Arial", 10))
        self.fps_label.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self.fps_label)

        layout.addWidget(self._divider())

        # Run button
        self.btn_run = self._make_button("RUN ANALYSIS", secondary=False)
        self.btn_run.setFixedHeight(44)
        self.btn_run.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-family: 'Arial';")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 10, QFont.Bold))
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 3px;")
        return lbl

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    def _make_button(self, text, secondary=False):
        btn = QPushButton(text)
        btn.setFont(QFont("Arial", 12))
        btn.setCursor(Qt.PointingHandCursor)
        if secondary:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: 1px solid {BORDER};
                    border-radius: 4px; color: {TEXT_PRIMARY}; padding: 6px 10px;
                }}
                QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
                QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {BORDER}; }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {ACCENT}; border: none; border-radius: 4px;
                    color: white; padding: 8px 10px; font-weight: bold;
                }}
                QPushButton:hover {{ background: {ACCENT_HOVER}; }}
                QPushButton:disabled {{ background: #444; color: {TEXT_MUTED}; }}
            """)
        return btn

    def _input_style(self):
        return f"""
            QLineEdit {{
                background: {DARK_BG}; border: 1px solid {BORDER};
                border-radius: 4px; color: {TEXT_PRIMARY}; padding: 8px 10px; min-height: 20px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """

    def update_aoi_list(self, aois):
        self.aoi_list_widget.clear()
        for i, rect in enumerate(aois):
            label = AOI_SEQUENCE[min(i, len(AOI_SEQUENCE) - 1)][0]
            color = AOI_SEQUENCE[min(i, len(AOI_SEQUENCE) - 1)][1]
            (x1, y1), (x2, y2) = rect
            item = QListWidgetItem(f"  {i+1}. {label}  [{x1},{y1} → {x2},{y2}]")
            item.setForeground(color)
            self.aoi_list_widget.addItem(item)

    def set_running(self, running: bool):
        """Lock/unlock UI during analysis."""
        self.btn_run.setEnabled(not running)
        self.btn_video.setEnabled(not running)
        self.btn_audio.setEnabled(not running)
        self.btn_undo.setEnabled(not running)
        self.btn_clear.setEnabled(not running)
        self.session_input.setEnabled(not running)
        self.api_key_input.setEnabled(not running)
        self.btn_save_key.setEnabled(not running)
        self.btn_load_json.setEnabled(not running)
        self.btn_load_txt.setEnabled(not running)
        self.btn_drive.setEnabled(not running)
        self.btn_save_drive.setEnabled(not running)


# ============================================================
# MAIN WINDOW
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACAI — Classroom Analysis")
        self.setMinimumSize(800, 500)
        self.setStyleSheet(f"background-color: {DARK_BG};")
        self.showMaximized()

        self.video_path    = None
        self.audio_path    = None
        self._worker       = None
        self._audio_worker = None
        self._gpt_worker   = None
        self._drive_worker = None
        self._output_dir   = None
        self._drive_root   = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.panel = SidePanel()
        root.addWidget(self.panel)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {BORDER};")
        root.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: {DARK_BG}; }}
            QScrollBar:vertical {{
                background: {PANEL_BG}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; }}
        """)

        canvas_container = QWidget()
        canvas_container.setStyleSheet(f"background: {DARK_BG};")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(24, 24, 24, 24)

        self.canvas_instruction = QLabel("Select a video file to begin drawing Areas of Interest")
        self.canvas_instruction.setAlignment(Qt.AlignCenter)
        self.canvas_instruction.setFont(QFont("Arial", 13))
        self.canvas_instruction.setStyleSheet(f"color: {TEXT_MUTED};")
        canvas_layout.addWidget(self.canvas_instruction)

        self.canvas = AOICanvas()
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addStretch()

        scroll.setWidget(canvas_container)
        root.addWidget(scroll, stretch=1)

        # signals
        self.panel.btn_video.clicked.connect(self._browse_video)
        self.panel.btn_audio.clicked.connect(self._browse_audio)
        self.panel.btn_undo.clicked.connect(self.canvas.undo_last)
        self.panel.btn_clear.clicked.connect(self.canvas.clear_all)
        self.panel.btn_run.clicked.connect(self._run_analysis)
        self.canvas.aoi_updated.connect(self._on_aoi_updated)
        self.panel.btn_save_key.clicked.connect(self._save_api_key)
        self.panel.btn_load_json.clicked.connect(self._load_key_from_json)
        self.panel.btn_load_txt.clicked.connect(self._load_key_from_txt)
        self.panel.api_key_input.textChanged.connect(self._check_ready)
        self.panel.btn_drive.clicked.connect(self._browse_drive_folder)
        self.panel.btn_save_drive.clicked.connect(self._save_drive_path)

        # load saved config on startup
        self._load_config()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, 'panel') or not hasattr(self, 'canvas'):
            return
        available_w = self.width() - self.panel.width() - 60
        self.canvas._redraw(available_w=max(available_w, 100))

    def _load_config(self):
        """Load saved config and pre-fill API key and Drive folder fields."""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    cfg = json.load(f)
                key = cfg.get("openai_api_key", "")
                if key:
                    self.panel.api_key_input.setText(key)
                    self.panel.api_key_status.setText("Key loaded from config.")
                    self.panel.api_key_status.setStyleSheet(
                        f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
                    )
                drive = cfg.get("drive_root", "")
                if drive:
                    self._drive_root = drive
                    self.panel.drive_label.setText(drive)
                    self.panel.drive_label.setStyleSheet(
                        f"color: {TEXT_PRIMARY}; font-size: 10px; font-family: 'Arial';"
                    )
                    self.panel.drive_status.setText("Drive folder loaded from config.")
                    self.panel.drive_status.setStyleSheet(
                        f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
                    )
            except Exception:
                pass

    def _save_api_key(self):
        """Save API key to config.json."""
        key = self.panel.api_key_input.text().strip()
        if not key:
            self.panel.api_key_status.setText("No key entered.")
            return
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    cfg = json.load(f)
            except Exception:
                pass
        cfg["openai_api_key"] = key
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        self.panel.api_key_status.setText("Key saved ✓")
        self.panel.api_key_status.setStyleSheet(
            f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
        )

    def _browse_drive_folder(self):
        """Browse for the local Google Drive ACAI_Reports folder."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Google Drive ACAI_Reports Folder", ""
        )
        if not path:
            return
        self._drive_root = path
        self.panel.drive_label.setText(path)
        self.panel.drive_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 10px; font-family: 'Arial';"
        )
        self.panel.drive_status.setText("Folder selected — click Save to persist.")
        self.panel.drive_status.setStyleSheet(
            f"color: {WARNING}; font-size: 10px; font-family: 'Arial';"
        )

    def _save_drive_path(self):
        """Save Drive folder path to config.json."""
        if not self._drive_root:
            self.panel.drive_status.setText("No folder selected.")
            return
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    cfg = json.load(f)
            except Exception:
                pass
        cfg["drive_root"] = self._drive_root
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        self.panel.drive_status.setText("Drive folder saved ✓")
        self.panel.drive_status.setStyleSheet(
            f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
        )

    def _get_api_key(self) -> str:
        """Return current API key from input field."""
        return self.panel.api_key_input.text().strip()

    def _load_key_from_json(self):
        """Browse for a JSON file and extract the API key."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Config File", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            # accept any of these common key names
            key = (cfg.get("openai_api_key") or
                   cfg.get("api_key") or
                   cfg.get("OPENAI_API_KEY") or
                   cfg.get("key") or "")
            if not key:
                self.panel.api_key_status.setText("No API key found in JSON.")
                self.panel.api_key_status.setStyleSheet(
                    f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
                )
                return
            self.panel.api_key_input.setText(key.strip())
            self.panel.api_key_status.setText(f"Key loaded from {os.path.basename(path)}")
            self.panel.api_key_status.setStyleSheet(
                f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
            )
        except Exception as e:
            self.panel.api_key_status.setText(f"Error reading JSON: {e}")
            self.panel.api_key_status.setStyleSheet(
                f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
            )

    def _load_key_from_txt(self):
        """Browse for a TXT file and use its contents as the API key."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TXT Key File", "",
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                key = f.read().strip()
            if not key:
                self.panel.api_key_status.setText("TXT file is empty.")
                self.panel.api_key_status.setStyleSheet(
                    f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
                )
                return
            self.panel.api_key_input.setText(key)
            self.panel.api_key_status.setText(f"Key loaded from {os.path.basename(path)}")
            self.panel.api_key_status.setStyleSheet(
                f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
            )
        except Exception as e:
            self.panel.api_key_status.setText(f"Error reading TXT: {e}")
            self.panel.api_key_status.setStyleSheet(
                f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
            )

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4);;All Files (*)"
        )
        if not path:
            return
        self.video_path = path
        self.panel.video_label.setText(os.path.basename(path))
        self.panel.video_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-family: 'Arial';"
        )
        self._load_first_frame(path)
        self._auto_session_name()
        self._check_ready()

    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.mp3 *.m4a *.wav *.flac);;All Files (*)"
        )
        if not path:
            return
        self.audio_path = path
        self.panel.audio_label.setText(os.path.basename(path))
        self.panel.audio_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-family: 'Arial';"
        )
        self._check_ready()

    def _load_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, "Error", "Could not read video file.")
            return
        self.canvas.load_frame(frame)
        self.canvas_instruction.setText(
            "Draw AOIs in order: ① Slides  ② Students  ③ Computer  ④+ Whiteboard(s)  —  then click Run"
        )
        self.canvas_instruction.setStyleSheet(f"color: {TEXT_PRIMARY};")

    def _on_aoi_updated(self, aois):
        self.panel.update_aoi_list(aois)
        self._check_ready()

    def _auto_session_name(self):
        if self.video_path and not self.panel.session_input.text().strip():
            stem = os.path.splitext(os.path.basename(self.video_path))[0]
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.panel.session_input.setText(f"{stem}_{ts}")

    def _check_ready(self):
        aois        = self.canvas.get_aois()
        video_ok    = self.video_path is not None
        audio_ok    = self.audio_path is not None
        aoi_ok      = len(aois) >= 3
        api_key_ok  = bool(self.panel.api_key_input.text().strip())

        if video_ok and audio_ok and aoi_ok and api_key_ok:
            self.panel.btn_run.setEnabled(True)
            self.panel.status_label.setText("Ready to run.")
            self.panel.status_label.setStyleSheet(
                f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
            )
        else:
            self.panel.btn_run.setEnabled(False)
            missing = []
            if not video_ok:   missing.append("video")
            if not audio_ok:   missing.append("audio")
            if not aoi_ok:     missing.append(f"AOIs ({len(aois)}/3 minimum)")
            if not api_key_ok: missing.append("API key")
            self.panel.status_label.setText("Waiting: " + ", ".join(missing))
            self.panel.status_label.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 10px; font-family: 'Arial';"
            )

    def _run_analysis(self):
        session = self.panel.session_input.text().strip()
        if not session:
            self._auto_session_name()
            session = self.panel.session_input.text().strip()

        aois = self.canvas.get_aois()

        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Run")
        msg.setText(
            f"Session:  {session}\n"
            f"Video:    {os.path.basename(self.video_path)}\n"
            f"Audio:    {os.path.basename(self.audio_path)}\n"
            f"AOIs:     {len(aois)} defined\n\n"
            f"Start analysis?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setStyleSheet(f"background: {PANEL_BG}; color: {TEXT_PRIMARY};")
        if msg.exec_() != QMessageBox.Yes:
            return

        output_dir = os.path.join(PROJECT_ROOT, "outputs", session)

        self.panel.set_running(True)
        self.panel.progress_bar.setValue(0)
        self.panel.stage_label.setText("Pose analysis running...")
        self.panel.stage_label.setStyleSheet(f"color: {WARNING};")
        self.panel.fps_label.setText("")
        self.panel.status_label.setText("Running — do not close the window.")
        self.panel.status_label.setStyleSheet(
            f"color: {WARNING}; font-size: 10px; font-family: 'Arial';"
        )

        self._worker = PoseWorker(self.video_path, aois, output_dir, session)
        self._worker.progress.connect(self._on_pose_progress)
        self._worker.finished.connect(self._on_pose_finished)
        self._worker.error.connect(self._on_pose_error)
        self._worker.start()

    def _on_pose_progress(self, frame, total, fps):
        pct = int((frame / total) * 100) if total > 0 else 0
        self.panel.progress_bar.setValue(pct)
        self.panel.stage_label.setText(f"Pose analysis — frame {frame}/{total}")
        if fps > 0:
            self.panel.fps_label.setText(f"{fps:.2f} fps")

    def _on_pose_finished(self, output_dir):
        self._output_dir = output_dir
        self.panel.progress_bar.setValue(100)
        self.panel.stage_label.setText("Pose analysis complete ✓  —  starting transcription...")
        self.panel.stage_label.setStyleSheet(f"color: {SUCCESS};")
        self.panel.fps_label.setText("")
        self._worker.quit()
        self._worker.wait()    # block until thread fully stops
        self._worker = None
        self._start_audio()

    def _start_audio(self):
        self.panel.stage_label.setText("Transcribing audio...")
        self.panel.stage_label.setStyleSheet(f"color: {WARNING};")
        self.panel.progress_bar.setValue(0)
        self.panel.status_label.setText("Whisper running — do not close the window.")
        self.panel.status_label.setStyleSheet(
            f"color: {WARNING}; font-size: 10px; font-family: 'Arial';"
        )
        self._audio_worker = AudioWorker(self.audio_path, self._output_dir)
        self._audio_worker.status.connect(lambda msg: self.panel.stage_label.setText(msg))
        self._audio_worker.finished.connect(self._on_audio_finished)
        self._audio_worker.error.connect(self._on_audio_error)
        self._audio_worker.start()

    def _on_audio_finished(self, result):
        self.panel.progress_bar.setValue(100)
        self.panel.stage_label.setText("Transcription complete ✓  —  starting GPT analysis...")
        self.panel.stage_label.setStyleSheet(f"color: {SUCCESS};")
        self._audio_worker.quit()
        self._audio_worker.wait()
        self._audio_worker = None
        self._start_gpt(result)

    def _on_audio_error(self, message):
        self.panel.stage_label.setText("Transcription error")
        self.panel.stage_label.setStyleSheet(f"color: {ACCENT};")
        self.panel.status_label.setText(f"Error: {message}")
        self.panel.status_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
        )
        self.panel.set_running(False)
        if self._audio_worker:
            self._audio_worker.quit()
            self._audio_worker.wait()
        self._audio_worker = None
        QMessageBox.critical(self, "Transcription Error", f"An error occurred:\n\n{message}")

    def _start_gpt(self, audio_result):
        session = self.panel.session_input.text().strip()
        self.panel.stage_label.setText("GPT analysis running...")
        self.panel.stage_label.setStyleSheet(f"color: {WARNING};")
        self.panel.progress_bar.setValue(0)
        self.panel.status_label.setText("GPT API calls in progress — do not close the window.")
        self.panel.status_label.setStyleSheet(
            f"color: {WARNING}; font-size: 10px; font-family: 'Arial';"
        )
        self._gpt_worker = GPTWorker(
            segments_csv=audio_result["segments_csv"],
            transcript_txt=audio_result["transcript_txt"],
            audio_path=self.audio_path,
            output_dir=self._output_dir,
            session_name=session,
            api_key=self._get_api_key()
        )
        self._gpt_worker.status.connect(self._on_gpt_status)
        self._gpt_worker.finished.connect(self._on_gpt_finished)
        self._gpt_worker.error.connect(self._on_gpt_error)
        self._gpt_worker.start()

    def _on_gpt_status(self, message):
        self.panel.stage_label.setText(message)

    def _on_gpt_finished(self, result):
        self.panel.progress_bar.setValue(100)
        self.panel.stage_label.setText("GPT analysis complete ✓  —  uploading to Drive...")
        self.panel.stage_label.setStyleSheet(f"color: {SUCCESS};")
        self._gpt_worker.quit()
        self._gpt_worker.wait()
        self._gpt_worker = None
        self._start_drive_upload()

    def _start_drive_upload(self):
        session = self.panel.session_input.text().strip()

        # if no Drive folder configured, skip upload and finish
        if not self._drive_root:
            self.panel.stage_label.setText("All analysis complete ✓  (no Drive folder set)")
            self.panel.stage_label.setStyleSheet(f"color: {SUCCESS};")
            self.panel.status_label.setText(f"Saved locally to: {self._output_dir}")
            self.panel.status_label.setStyleSheet(
                f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
            )
            self.panel.set_running(False)
            QMessageBox.information(
                self, "Analysis Complete",
                f"All analysis finished.\n\nOutputs saved locally to:\n{self._output_dir}\n\n"
                f"No Google Drive folder configured — files not uploaded."
            )
            return

        self.panel.stage_label.setText("Uploading to Google Drive...")
        self.panel.stage_label.setStyleSheet(f"color: {WARNING};")
        self.panel.progress_bar.setValue(0)
        self.panel.status_label.setText("Copying files to Drive sync folder...")
        self.panel.status_label.setStyleSheet(
            f"color: {WARNING}; font-size: 10px; font-family: 'Arial';"
        )

        self._drive_worker = DriveWorker(self._output_dir, self._drive_root, session)
        self._drive_worker.status.connect(lambda msg: self.panel.stage_label.setText(msg))
        self._drive_worker.finished.connect(self._on_drive_finished)
        self._drive_worker.error.connect(self._on_drive_error)
        self._drive_worker.start()

    def _on_drive_finished(self, result):
        self.panel.progress_bar.setValue(100)
        self.panel.stage_label.setText("All complete ✓  — files uploaded to Drive")
        self.panel.stage_label.setStyleSheet(f"color: {SUCCESS};")
        self.panel.status_label.setText(f"Drive: {result['drive_folder_path']}")
        self.panel.status_label.setStyleSheet(
            f"color: {SUCCESS}; font-size: 10px; font-family: 'Arial';"
        )
        self.panel.set_running(False)
        self._drive_worker.quit()
        self._drive_worker.wait()
        self._drive_worker = None

        skipped_msg = ""
        if result.get("skipped_files"):
            skipped_msg = f"\n\nSkipped (not found):\n" + "\n".join(f"  • {f}" for f in result["skipped_files"])

        QMessageBox.information(
            self, "Analysis Complete",
            f"All analysis finished.\n\n"
            f"Local output:\n{self._output_dir}\n\n"
            f"Drive folder:\n{result['drive_folder_path']}\n\n"
            f"Uploaded: {len(result['copied_files'])} files"
            f"{skipped_msg}"
        )

    def _on_drive_error(self, message):
        self.panel.stage_label.setText("Drive upload failed")
        self.panel.stage_label.setStyleSheet(f"color: {ACCENT};")
        self.panel.status_label.setText(f"Drive error: {message}")
        self.panel.status_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
        )
        self.panel.set_running(False)
        if self._drive_worker:
            self._drive_worker.quit()
            self._drive_worker.wait()
        self._drive_worker = None
        QMessageBox.warning(
            self, "Drive Upload Failed",
            f"Analysis completed but Drive upload failed:\n\n{message}\n\n"
            f"Your files are saved locally at:\n{self._output_dir}"
        )

    def _on_gpt_error(self, message):
        self.panel.stage_label.setText("GPT analysis error")
        self.panel.stage_label.setStyleSheet(f"color: {ACCENT};")
        self.panel.status_label.setText(f"Error: {message}")
        self.panel.status_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
        )
        self.panel.set_running(False)
        if self._gpt_worker:
            self._gpt_worker.quit()
            self._gpt_worker.wait()
        self._gpt_worker = None
        QMessageBox.critical(self, "GPT Analysis Error", f"An error occurred:\n\n{message}")

    def _on_pose_error(self, message):
        self.panel.stage_label.setText("Error")
        self.panel.stage_label.setStyleSheet(f"color: {ACCENT};")
        self.panel.status_label.setText(f"Error: {message}")
        self.panel.status_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-family: 'Arial';"
        )
        self.panel.set_running(False)
        self._worker = None

        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n\n{message}")