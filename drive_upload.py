"""
ACAI — drive_upload.py
Layer 5: Copy session output files to local Google Drive sync folder.

No API, no OAuth, no credentials needed.
Requires Google Drive for Desktop to be installed and syncing.

Setup (one-time):
  1. Install Google Drive for Desktop
  2. In Google Drive (browser), find the shared ACAI_Reports folder
  3. Right-click → "Add shortcut to Drive" → place in My Drive
  4. In Drive for Desktop preferences, make sure the folder is set to sync
  5. Update DRIVE_ROOT below to match your local Google Drive path
"""

import shutil
from pathlib import Path


# ============================================================
# CONFIG — update this path to match your machine
# ============================================================
DRIVE_ROOT = r"G:\.shortcut-targets-by-id\1LL1u1QejrUlSNIjwrMTkwkeRPYvQW5Tf\teaching_dashboard"


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def copy_to_drive(output_dir: str, session_name: str = "",
                  drive_root: str = None,
                  progress_callback=None) -> dict:
    """
    Copy all files from a local session output folder into the
    Google Drive sync folder. Google Drive for Desktop handles
    the actual upload in the background.

    Parameters
    ----------
    output_dir       : local session output folder path
    session_name     : subfolder name to create in ACAI_Reports
                       (defaults to the output_dir folder name)
    drive_root       : override for DRIVE_ROOT constant
    progress_callback: optional callable(str) for status messages

    Returns
    -------
    {
        "drive_folder_path": local path of the created Drive folder,
        "copied_files":      list of filenames copied,
        "skipped_files":     list of filenames that failed,
    }
    """
    output_dir  = Path(output_dir)
    root        = Path(drive_root) if drive_root else Path(DRIVE_ROOT)
    folder_name = session_name if session_name else output_dir.name
    dest        = root / folder_name

    if not root.exists():
        raise FileNotFoundError(
            f"Google Drive folder not found: {root}\n\n"
            "Please check:\n"
            "  1. Google Drive for Desktop is installed and running\n"
            "  2. The ACAI_Reports folder has been added as a shortcut to My Drive\n"
            "  3. The DRIVE_ROOT path in drive_upload.py matches your local Drive path"
        )

    dest.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Copying files to Drive: {folder_name}")

    copied  = []
    skipped = []

    for file in sorted(output_dir.iterdir()):
        if not file.is_file():
            continue
        try:
            shutil.copy2(file, dest / file.name)
            copied.append(file.name)
            if progress_callback:
                progress_callback(f"Copied: {file.name}")
        except Exception as e:
            skipped.append(file.name)
            if progress_callback:
                progress_callback(f"Failed to copy {file.name}: {e}")

    if progress_callback:
        progress_callback(
            f"Done — {len(copied)} files copied to Drive. "
            f"Google Drive will sync in the background."
        )

    return {
        "drive_folder_path": str(dest),
        "copied_files":      copied,
        "skipped_files":     skipped,
    }


# ============================================================
# STANDALONE TEST --> ONLY RUN THIS IF YOU WANT TO TEST
# ============================================================
if __name__ == "__main__":
    result = copy_to_drive(
        output_dir=r"C:\Users\USER\Documents\GitHub\teachingobs\outputs\test_video_20260331_170126",
        session_name="test_session",
        progress_callback=print
    )
    print(result)