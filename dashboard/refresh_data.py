"""
ACAI Dashboard — refresh_data.py
Downloads session files from a publicly shared Google Drive folder.

No service account or API key required.
The ACAI_Reports folder must be shared as "Anyone with the link can view".

Setup:
  1. Open ACAI_Reports in Google Drive
  2. Share → Anyone with the link → Viewer
  3. Copy the folder ID from the URL:
     drive.google.com/drive/folders/XXXXXXXXXXXXXXXX
  4. Paste it as DRIVE_FOLDER_ID below
"""

import os
import subprocess
import json
import urllib.request
import re
from pathlib import Path

# ============================================================
# CONFIG — set your folder ID here
# ============================================================
DRIVE_FOLDER_ID = "1LL1u1QejrUlSNIjwrMTkwkeRPYvQW5Tf"   # replace with real ID
OUTPUTS_DIR     = Path(__file__).parent.parent / "outputs"
SKIP_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ============================================================
# DRIVE FOLDER LISTING (no API key — uses public HTML scrape)
# ============================================================
def _list_public_folder(folder_id: str) -> list:
    """
    List files in a public Google Drive folder by scraping the folder page.
    Returns list of {name, id} dicts.
    Works as long as the folder is shared as 'Anyone with the link'.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="ignore")

    # extract file metadata embedded in the page as JSON
    # Drive embeds file info in a JS variable — extract file IDs and names
    pattern = r'\["([a-zA-Z0-9_-]{25,})"(?:.*?)"([^"]+\.[a-zA-Z0-9]+)"'
    matches = re.findall(pattern, html)

    seen = set()
    files = []
    for file_id, name in matches:
        if file_id not in seen and len(file_id) > 25:
            seen.add(file_id)
            ext = Path(name).suffix.lower()
            if ext and ext not in SKIP_EXTENSIONS:
                files.append({"id": file_id, "name": name})
    return files


def _list_public_subfolders(folder_id: str) -> list:
    """
    List session subfolders inside the root ACAI_Reports folder.
    Returns list of {name, id} dicts representing session folders.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="ignore")

    # folder IDs in Drive are the same format as file IDs
    # subfolder links appear as /drive/folders/{id} in the HTML
    pattern = r'/drive/folders/([a-zA-Z0-9_-]{25,})'
    folder_ids = list(dict.fromkeys(re.findall(pattern, html)))
    folder_ids = [fid for fid in folder_ids if fid != folder_id]

    # get folder names from the data-id attributes or title tags
    name_pattern = r'data-id="([a-zA-Z0-9_-]{25,})"[^>]*>([^<]+)<'
    name_map = {m[0]: m[1].strip() for m in re.findall(name_pattern, html)}

    subfolders = []
    for fid in folder_ids:
        name = name_map.get(fid, fid)
        subfolders.append({"id": fid, "name": name})
    return subfolders


# ============================================================
# DOWNLOAD
# ============================================================
def _wget_download(file_id: str, dest_path: Path, progress_callback=None):
    """Download a single public Drive file using wget."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

    if progress_callback:
        progress_callback(f"Downloading: {dest_path.name}")

    result = subprocess.run(
        ["wget", "-q", "--no-check-certificate", "-O", str(dest_path), url],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0 or dest_path.stat().st_size < 100:
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Download failed for {dest_path.name}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def sync_from_drive(progress_callback=None) -> dict:
    """
    List session subfolders on Drive, compare with local outputs/,
    download missing files (excluding video files).

    Returns summary dict.
    """
    if not is_configured():
        return {
            "sessions_checked": 0,
            "files_downloaded": 0,
            "files_skipped":    0,
            "new_sessions":     [],
            "errors":           ["DRIVE_FOLDER_ID not set in refresh_data.py"],
        }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Scanning Google Drive folder...")

    sessions_checked = 0
    files_downloaded = 0
    files_skipped    = 0
    new_sessions     = []
    errors           = []

    try:
        subfolders = _list_public_subfolders(DRIVE_FOLDER_ID)
    except Exception as e:
        return {
            "sessions_checked": 0,
            "files_downloaded": 0,
            "files_skipped":    0,
            "new_sessions":     [],
            "errors":           [f"Could not read Drive folder: {e}"],
        }

    if progress_callback:
        progress_callback(f"Found {len(subfolders)} session(s) on Drive.")

    local_sessions = {
        d.name for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and (d / "teachingstyle_output.csv").exists()
    } if OUTPUTS_DIR.exists() else set()

    for folder in subfolders:
        session_name      = folder["name"]
        session_folder_id = folder["id"]
        sessions_checked += 1
        is_new            = session_name not in local_sessions
        session_dir       = OUTPUTS_DIR / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(f"Checking: {session_name}")

        try:
            drive_files = _list_public_folder(session_folder_id)
        except Exception as e:
            errors.append(f"{session_name}: could not list files — {e}")
            continue

        for f in drive_files:
            local_path = session_dir / f["name"]
            if local_path.exists():
                files_skipped += 1
                continue
            try:
                _wget_download(f["id"], local_path, progress_callback)
                files_downloaded += 1
            except Exception as e:
                errors.append(f"{session_name}/{f['name']}: {e}")

        if is_new:
            new_sessions.append(session_name)

    if progress_callback:
        summary = (
            f"Complete — {files_downloaded} file(s) downloaded, "
            f"{files_skipped} already up to date."
        )
        if errors:
            summary += f" {len(errors)} error(s)."
        progress_callback(summary)

    return {
        "sessions_checked": sessions_checked,
        "files_downloaded": files_downloaded,
        "files_skipped":    files_skipped,
        "new_sessions":     new_sessions,
        "errors":           errors,
    }


def is_configured() -> bool:
    return DRIVE_FOLDER_ID != "YOUR_ACAI_REPORTS_FOLDER_ID"
