"""
ACAI Dashboard — data.py
Data loading layer.

Phase 1: reads from local outputs/ folder.
Phase 2: swap load_session_files() to call Google Drive API.
         Everything else in the codebase stays identical.
"""

import json
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# Phase 1: local path. Phase 2: change this one line only.
# ============================================================
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


# ============================================================
# SESSION DISCOVERY
# ============================================================
def list_sessions() -> list:
    """
    Return reverse-chronological list of valid session folder names.
    A valid session must contain at least teachingstyle_output.csv.
    """
    if not OUTPUTS_DIR.exists():
        return []
    sessions = []
    for d in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "teachingstyle_output.csv").exists():
            sessions.append(d.name)
    return sessions


def parse_session_name(session_name: str) -> dict:
    """
    Parse {filename}_{date}_{time} into components.
    e.g. luis_s01_20260331_143022 ->
         { filename: luis_s01, date: 20260331, time: 143022 }
    Falls back gracefully if format doesn't match.
    """
    parts = session_name.rsplit("_", 2)
    if len(parts) == 3:
        return {"filename": parts[0], "date": parts[1], "time": parts[2]}
    return {"filename": session_name, "date": "", "time": ""}


# ============================================================
# SESSION FILE LOADING  <-- swap this function for Phase 2
# ============================================================
def load_session_files(session_name: str) -> dict:
    """
    Load all output files for a session into a standardised dict.

    Returns:
        {
            teaching_style : DataFrame
            cog            : DataFrame
            blooms         : DataFrame
            acoustic       : DataFrame
            segments       : DataFrame
            report         : dict (parsed JSON)
        }

    Phase 2: replace the file reads below with Drive API calls.
    The return structure must stay identical — nothing else changes.
    """
    session_dir = OUTPUTS_DIR / session_name
    result = {}

    def _read_csv(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    result["teaching_style"] = _read_csv(session_dir / "teachingstyle_output.csv")
    result["cog"]            = _read_csv(session_dir / "reportsourcefile_center_of_gravity.csv")
    result["blooms"]         = _read_csv(session_dir / "blooms_classification.csv")
    result["acoustic"]       = _read_csv(session_dir / "acoustic_prosody.csv")

    # audio segments CSV has variable name prefix
    segments_df = pd.DataFrame()
    for f in session_dir.iterdir():
        if f.name.endswith("_segments.csv"):
            segments_df = pd.read_csv(f)
            break
    result["segments"] = segments_df

    report_path = session_dir / "gpt_summary_report.json"
    result["report"] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    # session metadata (fps, duration, frame dimensions)
    meta_path = session_dir / "session_meta.json"
    result["session_meta"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    # first video frame for heatmap background
    frame_path = session_dir / "first_frame.jpg"
    result["first_frame_path"] = str(frame_path) if frame_path.exists() else None

    return result


# ============================================================
# DERIVED METRICS  (pure functions — no I/O)
# ============================================================
def get_teaching_style_summary(df: pd.DataFrame) -> dict:
    if df.empty or "teaching_style" not in df.columns:
        return {}
    return (df["teaching_style"].value_counts(normalize=True) * 100).round(1).to_dict()


def get_aoi_time_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    mapping = {
        "slidesarea":     "Slides",
        "studentsarea":   "Students",
        "computerarea":   "Computer",
        "whiteboardarea": "Whiteboard",
    }
    total = len(df)
    return {
        label: round(df[col].sum() / total * 100, 1)
        for col, label in mapping.items()
        if col in df.columns
    }


def get_blooms_summary(df: pd.DataFrame) -> dict:
    if df.empty or "blooms_level" not in df.columns:
        return {}
    return (df["blooms_level"].value_counts(normalize=True) * 100).round(1).to_dict()


def get_acoustic_summary(report: dict) -> dict:
    return report.get("acoustic_prosody", {})


def get_linguistic_summary(report: dict) -> dict:
    return report.get("linguistic_prosody", {})


def get_content_summary(report: dict) -> dict:
    return report.get("content_summary", {})