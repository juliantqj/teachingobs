"""
ACAI — upload.py
Entry Point 2: Upload a completed session to Google Drive (OAuth2).

Usage: python upload.py
"""

import os
import sys
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QLineEdit,
    QFrame, QProgressBar, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import drive_upload

CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config.json")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
CLIENT_PATH  = os.path.join(PROJECT_ROOT, "credentials", "client_secret.json")
TOKEN_PATH   = os.path.join(PROJECT_ROOT, "credentials", "token.json")

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
# UPLOAD WORKER
# ============================================================
class UploadWorker(QThread):
    status   = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, session_dir, drive_folder_id,
                 client_secret_path, include_landmarks=False):
        super().__init__()
        self.session_dir        = session_dir
        self.drive_folder_id    = drive_folder_id
        self.client_secret_path = client_secret_path
        self.include_landmarks  = include_landmarks

    def run(self):
        try:
            result = drive_upload.upload_session(
                session_dir=self.session_dir,
                drive_folder_id=self.drive_folder_id,
                client_secret_path=self.client_secret_path,
                include_landmarks=self.include_landmarks,
                progress_callback=lambda msg: self.status.emit(msg)
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# MAIN WINDOW
# ============================================================
class UploadWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACAI — Upload to Google Drive")
        self.setFixedSize(540, 700)
        self.setStyleSheet(f"background-color: {DARK_BG};")

        self._worker              = None
        self._creds_path_override = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(14)

        # Title
        title = QLabel("ACAI")
        title.setFont(QFont("Arial", 26, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT}; letter-spacing: 6px;")
        subtitle = QLabel("Upload Session to Google Drive")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._divider())

        # ---- Google account status ----
        layout.addWidget(self._section_label("GOOGLE ACCOUNT"))
        account_row = QHBoxLayout()
        self.account_label = QLabel("")
        self.account_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        self.account_label.setWordWrap(True)
        self.btn_sign_out = self._make_button("Sign Out", secondary=True)
        self.btn_sign_out.setFixedWidth(90)
        account_row.addWidget(self.account_label, stretch=1)
        account_row.addWidget(self.btn_sign_out)
        layout.addLayout(account_row)
        self._refresh_account_status()
        layout.addWidget(self._divider())

        # ---- Client secret ----
        layout.addWidget(self._section_label("OAUTH CREDENTIALS (client_secret.json)"))
        creds_row = QHBoxLayout()
        creds_exists = os.path.exists(CLIENT_PATH)
        self.creds_label = QLabel(
            "credentials/client_secret.json found ✓" if creds_exists
            else "Not found — browse to select"
        )
        self.creds_label.setStyleSheet(
            f"color: {SUCCESS if creds_exists else ACCENT}; font-size: 10px;"
        )
        self.creds_label.setWordWrap(True)
        self.btn_browse_creds = self._make_button("Browse", secondary=True)
        creds_row.addWidget(self.creds_label, stretch=1)
        creds_row.addWidget(self.btn_browse_creds)
        layout.addLayout(creds_row)
        layout.addWidget(self._divider())

        # ---- Session selection ----
        layout.addWidget(self._section_label("SESSION TO UPLOAD"))
        self.session_combo = QComboBox()
        self.session_combo.setStyleSheet(self._combo_style())
        self.session_combo.setFont(QFont("Arial", 11))
        self.session_combo.setFixedHeight(38)
        self._populate_sessions()
        layout.addWidget(self.session_combo)

        browse_row = QHBoxLayout()
        self.manual_path_label = QLabel("Or browse manually:")
        self.manual_path_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        self.btn_browse_session = self._make_button("Browse Folder", secondary=True)
        browse_row.addWidget(self.manual_path_label, stretch=1)
        browse_row.addWidget(self.btn_browse_session)
        layout.addLayout(browse_row)

        self.selected_path_label = QLabel("")
        self.selected_path_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        self.selected_path_label.setWordWrap(True)
        layout.addWidget(self.selected_path_label)
        layout.addWidget(self._divider())

        # ---- Drive folder ID ----
        layout.addWidget(self._section_label("GOOGLE DRIVE FOLDER ID"))
        self.folder_id_input = QLineEdit()
        self.folder_id_input.setPlaceholderText("Paste folder ID from Drive URL...")
        self.folder_id_input.setStyleSheet(self._input_style())
        self.folder_id_input.setFont(QFont("Arial", 11))
        layout.addWidget(self.folder_id_input)

        save_row = QHBoxLayout()
        self.btn_save_id = self._make_button("Save ID", secondary=True)
        self.folder_id_status = QLabel("")
        self.folder_id_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        save_row.addWidget(self.btn_save_id)
        save_row.addWidget(self.folder_id_status)
        save_row.addStretch()
        layout.addLayout(save_row)
        layout.addWidget(self._divider())

        # ---- Options ----
        layout.addWidget(self._section_label("OPTIONS"))
        self.chk_landmarks = QCheckBox("Include landmark coordinates CSV (large file)")
        self.chk_landmarks.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.chk_landmarks.setFont(QFont("Arial", 11))
        layout.addWidget(self.chk_landmarks)
        layout.addWidget(self._divider())

        # ---- Progress ----
        self.stage_label = QLabel("—")
        self.stage_label.setFont(QFont("Arial", 11))
        self.stage_label.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {DARK_BG}; border-radius: 3px; border: none;
            }}
            QProgressBar::chunk {{
                background: {ACCENT}; border-radius: 3px;
            }}
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # ---- Upload button ----
        self.btn_upload = self._make_button("UPLOAD TO DRIVE", secondary=False)
        self.btn_upload.setFixedHeight(44)
        self.btn_upload.setFont(QFont("Arial", 13, QFont.Bold))
        layout.addWidget(self.btn_upload)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # signals
        self.btn_sign_out.clicked.connect(self._sign_out)
        self.btn_browse_creds.clicked.connect(self._browse_creds)
        self.btn_browse_session.clicked.connect(self._browse_session)
        self.btn_save_id.clicked.connect(self._save_folder_id)
        self.btn_upload.clicked.connect(self._start_upload)

        self._load_config()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 9, QFont.Bold))
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 3px;")
        return lbl

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        return line

    def _make_button(self, text, secondary=False):
        btn = QPushButton(text)
        btn.setFont(QFont("Arial", 11))
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
                border-radius: 4px; color: {TEXT_PRIMARY};
                padding: 8px 10px; min-height: 20px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """

    def _combo_style(self):
        return f"""
            QComboBox {{
                background: {DARK_BG}; border: 1px solid {BORDER};
                border-radius: 4px; color: {TEXT_PRIMARY}; padding: 6px 10px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {PANEL_BG}; color: {TEXT_PRIMARY};
                selection-background-color: {ACCENT};
            }}
        """

    def _refresh_account_status(self):
        if os.path.exists(TOKEN_PATH):
            self.account_label.setText("Signed in ✓  (token saved locally)")
            self.account_label.setStyleSheet(f"color: {SUCCESS}; font-size: 10px;")
            self.btn_sign_out.setEnabled(True)
        else:
            self.account_label.setText(
                "Not signed in — browser will open on first upload"
            )
            self.account_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
            self.btn_sign_out.setEnabled(False)

    def _populate_sessions(self):
        self.session_combo.clear()
        self.session_combo.addItem("— select a session —", userData=None)
        if os.path.exists(OUTPUTS_DIR):
            sessions = sorted(
                [d for d in os.listdir(OUTPUTS_DIR)
                 if os.path.isdir(os.path.join(OUTPUTS_DIR, d))],
                reverse=True
            )
            for s in sessions:
                self.session_combo.addItem(
                    s, userData=os.path.join(OUTPUTS_DIR, s)
                )

    def _load_config(self):
        cfg = drive_upload.load_drive_config(CONFIG_PATH)
        folder_id = cfg.get("drive_folder_id", "")
        if folder_id:
            self.folder_id_input.setText(folder_id)
            self.folder_id_status.setText("Loaded from config.")
            self.folder_id_status.setStyleSheet(f"color: {SUCCESS}; font-size: 10px;")

    def _get_session_dir(self):
        manual = self.selected_path_label.text().strip()
        if manual and os.path.isdir(manual):
            return manual
        return self.session_combo.currentData()

    def _get_creds_path(self):
        return self._creds_path_override or CLIENT_PATH

    def _set_ui_enabled(self, enabled: bool):
        for w in [self.btn_upload, self.btn_browse_session, self.btn_browse_creds,
                  self.btn_save_id, self.btn_sign_out, self.session_combo,
                  self.folder_id_input, self.chk_landmarks]:
            w.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _sign_out(self):
        cleared = drive_upload.clear_token()
        if cleared:
            QMessageBox.information(
                self, "Signed Out",
                "Token removed. You will be prompted to sign in again on next upload."
            )
        self._refresh_account_status()

    def _browse_creds(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select client_secret.json", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._creds_path_override = path
        self.creds_label.setText(os.path.basename(path))
        self.creds_label.setStyleSheet(f"color: {SUCCESS}; font-size: 10px;")

    def _browse_session(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Session Folder", OUTPUTS_DIR
        )
        if not path:
            return
        self.selected_path_label.setText(path)
        self.selected_path_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 10px;"
        )
        self.session_combo.setCurrentIndex(0)

    def _save_folder_id(self):
        folder_id = self.folder_id_input.text().strip()
        if not folder_id:
            self.folder_id_status.setText("No ID entered.")
            return
        drive_upload.save_drive_config(CONFIG_PATH, folder_id)
        self.folder_id_status.setText("Saved ✓")
        self.folder_id_status.setStyleSheet(f"color: {SUCCESS}; font-size: 10px;")

    def _start_upload(self):
        session_dir = self._get_session_dir()
        creds_path  = self._get_creds_path()
        folder_id   = self.folder_id_input.text().strip()

        if not session_dir:
            QMessageBox.warning(self, "Missing Session",
                                "Please select a session from the dropdown or browse for a folder.")
            return
        if not os.path.exists(creds_path):
            QMessageBox.warning(self, "Missing Credentials",
                                f"client_secret.json not found:\n{creds_path}\n\n"
                                "See drive_upload.py header for setup instructions.")
            return
        if not folder_id:
            QMessageBox.warning(self, "Missing Folder ID",
                                "Please enter your Google Drive folder ID.")
            return

        self._set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.stage_label.setText("Authenticating with Google Drive...")
        self.stage_label.setStyleSheet(f"color: {WARNING};")
        self.status_label.setText("Uploading — do not close the window.")
        self.status_label.setStyleSheet(f"color: {WARNING}; font-size: 10px;")

        self._worker = UploadWorker(
            session_dir=session_dir,
            drive_folder_id=folder_id,
            client_secret_path=creds_path,
            include_landmarks=self.chk_landmarks.isChecked()
        )
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_status(self, message):
        self.stage_label.setText(message)
        self._refresh_account_status()

    def _on_finished(self, result):
        self.progress_bar.setVisible(False)
        self.stage_label.setText("Upload complete ✓")
        self.stage_label.setStyleSheet(f"color: {SUCCESS};")
        n = len(result.get("files", {}))
        self.status_label.setText(
            f"{n} files uploaded — session: {result['session_name']}"
        )
        self.status_label.setStyleSheet(f"color: {SUCCESS}; font-size: 10px;")
        self._set_ui_enabled(True)
        self._refresh_account_status()
        self._worker = None
        QMessageBox.information(
            self, "Upload Complete",
            f"Session '{result['session_name']}' uploaded successfully.\n"
            f"{n} files in Drive folder."
        )

    def _on_error(self, message):
        self.progress_bar.setVisible(False)
        self.stage_label.setText("Upload failed")
        self.stage_label.setStyleSheet(f"color: {ACCENT};")
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet(f"color: {ACCENT}; font-size: 10px;")
        self._set_ui_enabled(True)
        self._worker = None
        QMessageBox.critical(self, "Upload Error", f"An error occurred:\n\n{message}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("ACAI Upload")
    window = UploadWindow()
    window.show()
    sys.exit(app.exec_())
