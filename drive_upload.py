"""
ACAI — drive_upload.py
Uploads a session output folder to Google Drive using OAuth2.

═══════════════════════════════════════════════════════════
ONE-TIME SETUP (5 minutes, done once per project)
═══════════════════════════════════════════════════════════
1. Go to https://console.cloud.google.com/
2. Create a new project (e.g. "ACAI")
3. Go to APIs & Services → Library → search "Google Drive API" → Enable
4. Go to APIs & Services → OAuth consent screen
   - User Type: External → Create
   - App name: ACAI, enter your email → Save and Continue (skip scopes/test users)
5. Go to APIs & Services → Credentials
   - Create Credentials → OAuth client ID
   - Application type: Desktop app → Name: ACAI → Create
   - Download the JSON → save as credentials/client_secret.json
6. Create a folder in Google Drive you want uploads to go to
   - Share it with anyone who will be uploading (Editor access)
   - Copy the folder ID from the URL:
     https://drive.google.com/drive/folders/THIS_IS_THE_FOLDER_ID
   - Paste it into the upload UI → Save ID

First run: browser opens for Google sign-in → click Allow → token saved to
credentials/token.json. All future runs are silent (token auto-refreshes).

To hand off to a collaborator:
  - Share the project folder (they get credentials/client_secret.json)
  - They run upload.py — browser opens for their Google sign-in
  - Their token.json is saved on their machine
  - Uploads go to the same shared Drive folder (same folder ID in config.json)

Install dependencies:
  pip install google-api-python-client google-auth google-auth-oauthlib
═══════════════════════════════════════════════════════════
"""

import os
import json
from pathlib import Path

SCOPES       = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "credentials", "token.json")
CLIENT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "credentials", "client_secret.json")

UPLOAD_FILES = [
    "teachingstyle_output.csv",
    "reportsourcefile_center_of_gravity.csv",
    "blooms_classification.csv",
    "gpt_summary_report.json",
]

OPTIONAL_FILES = [
    "reportsourcefile_landmarkcoordinates.csv",
]


def _get_drive_service(client_secret_path: str = None):
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Google API packages not installed.\n"
            "Run: pip install google-api-python-client google-auth google-auth-oauthlib"
        )

    creds_path  = client_secret_path or CLIENT_PATH
    token_path  = TOKEN_PATH
    os.makedirs(os.path.dirname(token_path), exist_ok=True)

    creds = None

    # load existing token if available
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # if no valid token, run OAuth flow (opens browser once)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"client_secret.json not found: {creds_path}\n"
                    "See drive_upload.py header for setup instructions."
                )
            flow  = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # save token for future runs
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def upload_session(session_dir: str, drive_folder_id: str,
                   client_secret_path: str = None,
                   include_landmarks: bool = False,
                   progress_callback=None) -> dict:
    """
    Upload session output files to a Google Drive folder.
    Creates a subfolder named after the session inside drive_folder_id.
    Returns dict of {filename: drive_file_id}.
    """
    from googleapiclient.http import MediaFileUpload

    session_dir  = Path(session_dir)
    session_name = session_dir.name

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    if progress_callback:
        progress_callback("Authenticating with Google Drive...")

    service = _get_drive_service(client_secret_path)

    # create session subfolder on Drive
    if progress_callback:
        progress_callback(f"Creating Drive folder: {session_name}")

    folder_meta = {
        "name":     session_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents":  [drive_folder_id],
    }
    folder = service.files().create(body=folder_meta, fields="id").execute()
    session_folder_id = folder["id"]

    # determine files to upload
    to_upload = list(UPLOAD_FILES)
    if include_landmarks:
        to_upload += OPTIONAL_FILES

    # also pick up any *_segments.csv and transcript files dynamically
    for f in session_dir.iterdir():
        if f.name not in to_upload and f.suffix in (".csv", ".json", ".txt", ".docx"):
            to_upload.append(f.name)

    uploaded = {}
    for filename in to_upload:
        filepath = session_dir / filename
        if not filepath.exists():
            if progress_callback:
                progress_callback(f"Skipping (not found): {filename}")
            continue

        if progress_callback:
            progress_callback(f"Uploading: {filename}")

        mime      = _guess_mime(filepath.suffix)
        file_meta = {"name": filename, "parents": [session_folder_id]}
        media     = MediaFileUpload(str(filepath), mimetype=mime, resumable=True)
        result    = service.files().create(
            body=file_meta, media_body=media, fields="id"
        ).execute()
        uploaded[filename] = result["id"]

    if progress_callback:
        progress_callback(f"Upload complete — {len(uploaded)} files uploaded.")

    return {
        "session_folder_id": session_folder_id,
        "files":             uploaded,
        "session_name":      session_name,
    }


def _guess_mime(suffix: str) -> str:
    return {
        ".csv":  "text/csv",
        ".json": "application/json",
        ".txt":  "text/plain",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".mp4":  "video/mp4",
    }.get(suffix.lower(), "application/octet-stream")


def load_drive_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.load(f)


def save_drive_config(config_path: str, folder_id: str):
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    cfg["drive_folder_id"] = folder_id
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def clear_token():
    """Delete saved token to force re-authentication on next run."""
    if os.path.exists(TOKEN_PATH):
        os.remove(TOKEN_PATH)
        return True
    return False
