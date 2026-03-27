from tqdm import tqdm
import math
import cv2
import numpy as np
import pandas as pd
import statistics
import datetime
import os
from enum import IntEnum

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "pose_landmarker_full.task"   # download this .task model file
VIDEO_NAME = "luis_trim_lesson.mp4"
BLUR = 56                                  # percentage of image height to blur from bottom
NUM_POSES = 1
MIN_POSE_DET_CONF = 0.2
MIN_POSE_PRES_CONF = 0.2
MIN_TRACK_CONF = 0.2

OUTPUT_VIDEO = f"{VIDEO_NAME}_out.mp4"
OUTPUT_LANDMARK_CSV = "reportsourcefile_landmarkcoordinates.csv"
OUTPUT_TEACHINGSTYLE_CSV = "teachingstyle_output.csv"
OUTPUT_COG_CSV = "reportsourcefile_center_of_gravity.csv"

# AOI transparency settings
AOI_SELECTION_BASE_WEIGHT = 0.9   # 70% original image
AOI_SELECTION_FILL_WEIGHT = 0.1   # 30% red overlay
AOI_DISPLAY_BASE_WEIGHT = 0.9     # same for display during processing
AOI_DISPLAY_FILL_WEIGHT = 0.1


# ============================================================
# LANDMARK ENUM
# ============================================================
class PoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (3, 7), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]


# ============================================================
# GLOBAL STATE
# ============================================================
teaching_style_dict = {
    "frame": [],
    "teaching_style": [],
    "slidesarea": [],
    "studentsarea": [],
    "pointingslides": [],
    "lookingstudents": [],
    "computerarea": [],
    "whiteboardarea": [],
    "facingstudents": []
}

landmark_coordinate_dict = {"frame": []}
for x in range(33):
    landmark_coordinate_dict[f"lm{x}_x"] = []
    landmark_coordinate_dict[f"lm{x}_y"] = []
    landmark_coordinate_dict[f"lm{x}_z"] = []
    landmark_coordinate_dict[f"lm{x}_visibility"] = []

areasofinterest_list = []
list_of_COG = []

frame_width = 0
frame_height = 0


# ============================================================
# HELPERS
# ============================================================
def ensure_rect(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))


def point_in_rect(point, rect):
    if point is None or rect is None:
        return False
    x, y = point[:2]
    (x1, y1), (x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def get_aoi(index):
    if 0 <= index < len(areasofinterest_list):
        return areasofinterest_list[index]
    return None


def safe_visibility(lm):
    return getattr(lm, "visibility", None)


def clamp_int(v):
    return int(round(v))


def append_default_logs(frame_idx, label="No Pose Detected"):
    teaching_style_dict["frame"].append(frame_idx)
    teaching_style_dict["teaching_style"].append(label)
    teaching_style_dict["slidesarea"].append(0)
    teaching_style_dict["studentsarea"].append(0)
    teaching_style_dict["pointingslides"].append(0)
    teaching_style_dict["lookingstudents"].append(0)
    teaching_style_dict["computerarea"].append(0)
    teaching_style_dict["whiteboardarea"].append(0)
    teaching_style_dict["facingstudents"].append(0)


def append_landmark_row(frame_idx, landmarks_norm, width, height):
    landmark_coordinate_dict["frame"].append(frame_idx)

    if landmarks_norm is None:
        for i in range(33):
            landmark_coordinate_dict[f"lm{i}_x"].append(None)
            landmark_coordinate_dict[f"lm{i}_y"].append(None)
            landmark_coordinate_dict[f"lm{i}_z"].append(None)
            landmark_coordinate_dict[f"lm{i}_visibility"].append(None)
        return

    for i, lm in enumerate(landmarks_norm):
        landmark_coordinate_dict[f"lm{i}_x"].append(clamp_int(lm.x * width))
        landmark_coordinate_dict[f"lm{i}_y"].append(clamp_int(lm.y * height))
        landmark_coordinate_dict[f"lm{i}_z"].append(lm.z * width)
        landmark_coordinate_dict[f"lm{i}_visibility"].append(safe_visibility(lm))


def draw_pose(frame, pixel_landmarks, visibility=None):
    for a, b in POSE_CONNECTIONS:
        if a >= len(pixel_landmarks) or b >= len(pixel_landmarks):
            continue

        pa = pixel_landmarks[a]
        pb = pixel_landmarks[b]
        if pa is None or pb is None:
            continue

        cv2.line(frame, (pa[0], pa[1]), (pb[0], pb[1]), (0, 255, 0), 2)

    for i, pt in enumerate(pixel_landmarks):
        if pt is None:
            continue

        vis_ok = True
        if visibility is not None and i < len(visibility):
            vis_val = visibility[i]
            if vis_val is not None and vis_val < 0.2:
                vis_ok = False

        radius = 4 if vis_ok else 2
        color = (0, 255, 255) if vis_ok else (100, 100, 100)
        cv2.circle(frame, (pt[0], pt[1]), radius, color, -1)


def normalize_to_pixels(norm_landmarks, width, height):
    pixels = []
    vis = []
    for lm in norm_landmarks:
        pixels.append((
            clamp_int(lm.x * width),
            clamp_int(lm.y * height),
            lm.z * width
        ))
        vis.append(safe_visibility(lm))
    return pixels, vis


def export_csvs():
    pd.DataFrame.from_dict(landmark_coordinate_dict).to_csv(
        OUTPUT_LANDMARK_CSV, index=False
    )
    pd.DataFrame.from_dict(teaching_style_dict).to_csv(
        OUTPUT_TEACHINGSTYLE_CSV, index=False
    )


# ============================================================
# AOI SELECTION
# ============================================================
def areasofinterest(video_name):
    rectangles = []
    drawing = False
    temp_points = []
    current_mouse = [0, 0]

    vid = cv2.VideoCapture(video_name)
    if not vid.isOpened():
        raise RuntimeError(f"Could not open video for AOI selection: {video_name}")

    ret, first_frame = vid.read()
    vid.release()

    if not ret:
        raise RuntimeError("Could not read first frame for AOI selection.")

    # --- Resize for display, keep scale factors to map coords back ---
    orig_h, orig_w = first_frame.shape[:2]
    display_w = 1280
    scale = display_w / orig_w
    display_h = int(orig_h * scale)
    display_frame = cv2.resize(first_frame, (display_w, display_h))

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, temp_points, rectangles

        current_mouse[0] = x
        current_mouse[1] = y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            temp_points = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            temp_points.append((x, y))

            if len(temp_points) == 2:
                rect = ensure_rect(temp_points[0], temp_points[1])
                rectangles.append(rect)
                print("Added AOI (display coords):", rect)

    cv2.namedWindow("Define Areas of Interest")
    cv2.setMouseCallback("Define Areas of Interest", draw_rectangle)

    while True:
        frame = display_frame.copy()

        if len(rectangles) == 0:
            prompt = "Mark out teaching slides"
        elif len(rectangles) == 1:
            prompt = "Mark out area where students are sitting"
        elif len(rectangles) == 2:
            prompt = "Mark out area where computer/laptop is"
        else:
            prompt = "Mark out whiteboard area(s), then press q"

        cv2.putText(frame, prompt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        labels = ["Teaching Slides", "Students", "Computer", "Whiteboard"]
        for i, rect in enumerate(rectangles):
            label = labels[i] if i < 3 else "Whiteboard"
            overlay = frame.copy()
            cv2.rectangle(frame, rect[0], rect[1], (0, 0, 255), -1)
            cx = int((rect[0][0] + rect[1][0]) / 2)
            cy = int((rect[0][1] + rect[1][1]) / 2)
            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            frame = cv2.addWeighted(overlay, AOI_SELECTION_BASE_WEIGHT, frame, AOI_SELECTION_FILL_WEIGHT, 0)

        # Live drag preview
        if drawing and len(temp_points) == 1:
            preview_rect = ensure_rect(temp_points[0], (current_mouse[0], current_mouse[1]))
            cv2.rectangle(frame, preview_rect[0], preview_rect[1], (0, 255, 255), 2)

        cv2.imshow("Define Areas of Interest", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    # --- Scale AOI coords back to original resolution ---
    scaled = []
    for (x1, y1), (x2, y2) in rectangles:
        scaled.append((
            (int(x1 / scale), int(y1 / scale)),
            (int(x2 / scale), int(y2 / scale))
        ))
    return scaled


# ============================================================
# DETECTION
# ============================================================
def detect_pose(frame_bgr, landmarker, timestamp_ms):
    height, width = frame_bgr.shape[:2]

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    students_aoi = get_aoi(1)
    if students_aoi is not None:
        cv2.rectangle(image_rgb, students_aoi[0], students_aoi[1], (0, 0, 0), -1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks:
        return frame_bgr, None, None, result

    norm_landmarks = result.pose_landmarks[0]
    pixel_landmarks, visibility = normalize_to_pixels(norm_landmarks, width, height)

    output = frame_bgr.copy()
    draw_pose(output, pixel_landmarks, visibility)
    return output, pixel_landmarks, norm_landmarks, result


# ============================================================
# CLASSIFICATION
# ============================================================
def classify_pose(landmarks, output_image, history, current_frame_count):
    global frame_width, frame_height

    label = "Passive Teaching"
    facingstudentslog = False

    history_Rwrist_x = [lm[PoseLandmark.RIGHT_WRIST][0] for lm in history]
    history_Lwrist_x = [lm[PoseLandmark.LEFT_WRIST][0] for lm in history]

    if len(history) >= 30:
        try:
            rw = int(statistics.stdev(history_Rwrist_x))
            lw = int(statistics.stdev(history_Lwrist_x))
            if rw > 10 and lw > 10:
                label = "Active Gesturing"
        except statistics.StatisticsError:
            pass

    left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
    left_wrist = landmarks[PoseLandmark.LEFT_WRIST]
    right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
    right_wrist = landmarks[PoseLandmark.RIGHT_WRIST]

    slides_aoi = get_aoi(0)
    students_aoi = get_aoi(1)

    left_line = []
    right_line = []

    if slides_aoi is not None and left_wrist[1] <= slides_aoi[1][1]:
        try:
            if left_wrist[0] >= left_shoulder[0] and left_wrist[1] <= left_shoulder[1]:
                angle = math.degrees(
                    math.atan((left_shoulder[1] - left_wrist[1]) / ((left_wrist[0] - left_shoulder[0]) + 1e-6))
                )
                ticks = max(0, int((frame_width - left_wrist[0]) / 200))
                for x in range(ticks):
                    px = 100
                    extendedx = (left_wrist[0] - left_shoulder[0]) + (px * x)
                    extendedy = extendedx * math.tan(angle * math.pi / 180)
                    newx = left_wrist[0] + (px * x)
                    newy = left_shoulder[1] - extendedy
                    left_line.append((int(newx), int(newy)))
                for pt in left_line:
                    cv2.circle(output_image, pt, 5, (0, 255, 255), 5)
        except Exception:
            pass

    if slides_aoi is not None and right_wrist[1] <= slides_aoi[1][1]:
        try:
            if right_wrist[0] <= right_shoulder[0] and right_wrist[1] <= right_shoulder[1]:
                angle = math.degrees(
                    math.atan((right_wrist[1] - right_shoulder[1]) / ((right_wrist[0] - right_shoulder[0]) + 1e-6))
                )
                ticks = max(0, int((frame_width - right_wrist[0]) / 200))
                for x in range(ticks):
                    px = 100
                    extendedx = (right_shoulder[0] - right_wrist[0]) + (px * x)
                    extendedy = extendedx * math.tan(angle * math.pi / 180)
                    newx = right_wrist[0] - (px * x)
                    newy = right_shoulder[1] - extendedy
                    right_line.append((int(newx), int(newy)))
                for pt in right_line:
                    cv2.circle(output_image, pt, 5, (0, 255, 255), 5)
        except Exception:
            pass

    checkifpointing = False
    if slides_aoi is not None:
        for pt in left_line:
            if point_in_rect(pt, slides_aoi):
                checkifpointing = True
                break

        if not checkifpointing:
            for pt in right_line:
                if point_in_rect(pt, slides_aoi):
                    cv2.putText(output_image, "Lecturer pointing at slides", (10, 250),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    checkifpointing = True
                    break

    teaching_style_dict["pointingslides"].append(1 if checkifpointing else 0)

    if landmarks[PoseLandmark.RIGHT_SHOULDER][0] > landmarks[PoseLandmark.LEFT_SHOULDER][0]:
        cv2.putText(output_image, "Lecturer not facing forwards", (10, 500),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        if not facingstudentslog:
            teaching_style_dict["facingstudents"].append(0)
            facingstudentslog = True

        shoulders_mid = (
            int((landmarks[PoseLandmark.LEFT_SHOULDER][0] + landmarks[PoseLandmark.RIGHT_SHOULDER][0]) / 2),
            int((landmarks[PoseLandmark.LEFT_SHOULDER][1] + landmarks[PoseLandmark.RIGHT_SHOULDER][1]) / 2)
        )

        near_whiteboard = False
        if len(areasofinterest_list) > 3:
            for wb in areasofinterest_list[3:]:
                if point_in_rect(shoulders_mid, wb):
                    near_whiteboard = True
                    break

        if near_whiteboard:
            teaching_style_dict["whiteboardarea"].append(1)
            cv2.putText(output_image, "Facing away & Near Whiteboard", (10, 750),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        else:
            teaching_style_dict["whiteboardarea"].append(0)

        teaching_style_dict["lookingstudents"].append(0)

    else:
        lineofsight_students = False

        if landmarks[PoseLandmark.LEFT_EAR][0] > landmarks[PoseLandmark.NOSE][0]:
            try:
                y_certain = (
                    (landmarks[PoseLandmark.NOSE][1] - landmarks[PoseLandmark.LEFT_EAR][1]) /
                    ((landmarks[PoseLandmark.NOSE][0] - landmarks[PoseLandmark.LEFT_EAR][0]) + 1e-6)
                ) * (0 - landmarks[PoseLandmark.LEFT_EAR][0]) + landmarks[PoseLandmark.LEFT_EAR][1]
                y_certain = int(y_certain)
            except Exception:
                y_certain = frame_height

            if y_certain < frame_height:
                cv2.line(output_image,
                         (landmarks[PoseLandmark.LEFT_EAR][0], landmarks[PoseLandmark.LEFT_EAR][1]),
                         (0, y_certain), (255, 255, 255), 1)
                if students_aoi is not None and y_certain > students_aoi[0][1]:
                    lineofsight_students = True

        if landmarks[PoseLandmark.RIGHT_EAR][0] < landmarks[PoseLandmark.NOSE][0]:
            try:
                y_certain = (
                    (landmarks[PoseLandmark.NOSE][1] - landmarks[PoseLandmark.RIGHT_EAR][1]) /
                    ((landmarks[PoseLandmark.NOSE][0] - landmarks[PoseLandmark.RIGHT_EAR][0]) + 1e-6)
                ) * (frame_width - landmarks[PoseLandmark.RIGHT_EAR][0]) + landmarks[PoseLandmark.RIGHT_EAR][1]
                y_certain = int(y_certain)
            except Exception:
                y_certain = frame_height

            if y_certain < frame_height:
                cv2.line(output_image,
                         (landmarks[PoseLandmark.RIGHT_EAR][0], landmarks[PoseLandmark.RIGHT_EAR][1]),
                         (frame_width, y_certain), (255, 255, 255), 1)
                if students_aoi is not None and y_certain > students_aoi[0][1]:
                    lineofsight_students = True

        teaching_style_dict["lookingstudents"].append(1 if lineofsight_students else 0)

        head_mid = (
            int((landmarks[PoseLandmark.LEFT_EYE][0] + landmarks[PoseLandmark.RIGHT_EYE][0]) / 2),
            int((landmarks[PoseLandmark.LEFT_EYE][1] + landmarks[PoseLandmark.RIGHT_EYE][1]) / 2)
        )

        try:
            angle_r = math.degrees(
                math.atan((head_mid[0] - landmarks[PoseLandmark.RIGHT_SHOULDER][0]) /
                          ((landmarks[PoseLandmark.RIGHT_SHOULDER][1] - head_mid[1]) + 1e-6))
            )
        except Exception:
            angle_r = 1

        try:
            angle_l = math.degrees(
                math.atan((landmarks[PoseLandmark.LEFT_SHOULDER][0] - head_mid[0]) /
                          ((landmarks[PoseLandmark.LEFT_SHOULDER][1] - head_mid[1]) + 1e-6))
            )
        except Exception:
            angle_l = 1

        r_to_frame = (frame_height - head_mid[1]) * math.tan(angle_r * math.pi / 180)
        l_to_frame = (frame_height - head_mid[1]) * math.tan(angle_l * math.pi / 180)

        xr = int(head_mid[0] - r_to_frame)
        xl = int(head_mid[0] + l_to_frame)

        cv2.line(output_image, head_mid, (xr, frame_height), (0, 255, 0), 1)
        cv2.line(output_image, head_mid, (xl, frame_height), (0, 255, 0), 1)

        if students_aoi is not None:
            if students_aoi[0][0] <= xr or xl < students_aoi[1][0]:
                cv2.putText(output_image, "Body facing students", (10, 1350),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                if not facingstudentslog:
                    teaching_style_dict["facingstudents"].append(1)
                    facingstudentslog = True

        if not facingstudentslog:
            teaching_style_dict["facingstudents"].append(0)

        teaching_style_dict["whiteboardarea"].append(0)

    color = (0, 0, 255) if label == "Passive Teaching" else (0, 255, 0)
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    teaching_style_dict["frame"].append(current_frame_count)
    teaching_style_dict["teaching_style"].append(label)

    return output_image


# ============================================================
# TRACK COG
# ============================================================
def track_cog(landmarks, output_image):
    left_heel = landmarks[PoseLandmark.LEFT_HEEL]
    right_heel = landmarks[PoseLandmark.RIGHT_HEEL]

    head_mid = (
        int((landmarks[PoseLandmark.LEFT_EYE][0] + landmarks[PoseLandmark.RIGHT_EYE][0]) / 2),
        int((landmarks[PoseLandmark.LEFT_EYE][1] + landmarks[PoseLandmark.RIGHT_EYE][1]) / 2)
    )
    cv2.circle(output_image, head_mid, 5, (0, 255, 255), 5)

    center_of_gravity = None
    if left_heel is not None and right_heel is not None:
        center_of_gravity = (
            (left_heel[0] + right_heel[0]) / 2,
            (left_heel[1] + right_heel[1]) / 2,
            (left_heel[2] + right_heel[2]) / 2
        )
        cv2.circle(output_image, (int(center_of_gravity[0]), int(center_of_gravity[1])), 5, (255, 0, 0), 5)
        list_of_COG.append((int(center_of_gravity[0]), int(center_of_gravity[1])))
    else:
        list_of_COG.append(0)

    slides_aoi = get_aoi(0)
    students_aoi = get_aoi(1)
    computer_aoi = get_aoi(2)

    slides_bool = slides_aoi is not None and point_in_rect(head_mid, slides_aoi)
    students_bool = students_aoi is not None and center_of_gravity is not None and point_in_rect(center_of_gravity, students_aoi)

    if slides_bool and students_bool:
        cv2.putText(output_image, "Interacting with slides", (10, 90), cv2.QT_FONT_NORMAL, 1, (1, 194, 252), 2)
        cv2.putText(output_image, "Interacting with students", (10, 110), cv2.QT_FONT_NORMAL, 1, (0, 254, 255), 2)
        teaching_style_dict["slidesarea"].append(1)
        teaching_style_dict["studentsarea"].append(1)
    elif slides_bool:
        cv2.putText(output_image, "Interacting with slides", (10, 90), cv2.QT_FONT_NORMAL, 1, (1, 194, 252), 2)
        teaching_style_dict["slidesarea"].append(1)
        teaching_style_dict["studentsarea"].append(0)
    elif students_bool:
        cv2.putText(output_image, "Interacting with students", (10, 90), cv2.QT_FONT_NORMAL, 1, (0, 254, 255), 2)
        teaching_style_dict["slidesarea"].append(0)
        teaching_style_dict["studentsarea"].append(1)
    else:
        teaching_style_dict["slidesarea"].append(0)
        teaching_style_dict["studentsarea"].append(0)

    shoulders_mid = (
        int((landmarks[PoseLandmark.LEFT_SHOULDER][0] + landmarks[PoseLandmark.RIGHT_SHOULDER][0]) / 2),
        int((landmarks[PoseLandmark.LEFT_SHOULDER][1] + landmarks[PoseLandmark.RIGHT_SHOULDER][1]) / 2)
    )
    cv2.circle(output_image, shoulders_mid, 5, (102, 5, 255), 2)

    if computer_aoi is not None and point_in_rect(shoulders_mid, computer_aoi):
        cv2.putText(output_image, "Lecturer near computer", (10, 900), cv2.QT_FONT_NORMAL, 1, (102, 5, 255), 2)
        teaching_style_dict["computerarea"].append(1)
    else:
        teaching_style_dict["computerarea"].append(0)

    return output_image


# ============================================================
# DRAW AOIS / BLUR
# ============================================================
def draw_aois(frame):
    labels = ["Teaching Slides", "Students", "Computer", "Whiteboard"]
    for i, rect in enumerate(areasofinterest_list):
        label = labels[i] if i < 3 else "Whiteboard"
        overlay = frame.copy()
        cv2.rectangle(frame, rect[0], rect[1], (0, 0, 255), -1)
        cx = int((rect[0][0] + rect[1][0]) / 2)
        cy = int((rect[0][1] + rect[1][1]) / 2)
        cv2.putText(frame, label, (cx, cy), cv2.QT_FONT_NORMAL, 0.8, (255, 255, 255), 1)
        frame = cv2.addWeighted(
            overlay,
            AOI_DISPLAY_BASE_WEIGHT,
            frame,
            AOI_DISPLAY_FILL_WEIGHT,
            0
        )
    return frame


def blur_lower_region(frame):
    blur_height_value = int(frame_height - ((BLUR / 100) * frame_height))
    x1, y1 = 0, blur_height_value
    x2, y2 = frame_width, frame_height

    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        blur = cv2.GaussianBlur(roi, (55, 55), 0)
        frame[y1:y2, x1:x2] = blur
    return frame


# ============================================================
# MAIN
# ============================================================
def main():
    global areasofinterest_list, frame_width, frame_height

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Download a Pose Landmarker .task model and place it next to this script."
        )

    print("Select AOIs:")
    print("1 = slides, 2 = students, 3 = computer, 4+ = whiteboards, then press q")
    areasofinterest_list = areasofinterest(VIDEO_NAME)
    print("AOIs:", areasofinterest_list)

    video = cv2.VideoCapture(VIDEO_NAME)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_NAME}")

    video_fps = video.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video Frame Rate:", video_fps)
    print("Total frames:", total_frame_count)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, video_fps, (frame_width, frame_height))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_POSE_DET_CONF,
        min_pose_presence_confidence=MIN_POSE_PRES_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
        output_segmentation_masks=False,
    )

    landmarks_history = []
    running_frame_count = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while video.isOpened():
            start_time = datetime.datetime.now()
            ok, frame = video.read()
            if not ok:
                break

            running_frame_count += 1
            print(f"{running_frame_count}/{total_frame_count} total frames analyzed.")

            timestamp_ms = int((running_frame_count / max(video_fps, 1e-6)) * 1000)

            frame_out, landmarks_px, landmarks_norm, _ = detect_pose(frame, landmarker, timestamp_ms)

            if landmarks_px is not None:
                landmarks_history.append(landmarks_px)
                if len(landmarks_history) > 60:
                    landmarks_history.pop(0)

                frame_out = classify_pose(
                    landmarks_px,
                    frame_out,
                    landmarks_history,
                    running_frame_count
                )

                frame_out = track_cog(landmarks_px, frame_out)

                append_landmark_row(running_frame_count, landmarks_norm, frame_width, frame_height)
            else:
                append_default_logs(running_frame_count)
                append_landmark_row(running_frame_count, None, frame_width, frame_height)

            frame_out = draw_aois(frame_out)
            frame_out = blur_lower_region(frame_out)

            end_time = datetime.datetime.now()
            delta = (end_time - start_time).total_seconds()
            if delta > 0:
                cv2.putText(frame_out, f"FPS: {1.0 / delta:.2f}", (10, frame_height - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            cv2.imshow("Pose Detection", frame_out)
            out.write(frame_out)

            export_csvs()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video.release()
    out.release()
    cv2.destroyAllWindows()

    cog_x, cog_y, frames = [], [], []
    for i, value in enumerate(list_of_COG, start=1):
        frames.append(i)
        if value == 0:
            cog_x.append(None)
            cog_y.append(None)
        else:
            cog_x.append(value[0])
            cog_y.append(value[1])

    pd.DataFrame({"frame": frames, "x": cog_x, "y": cog_y}).to_csv(OUTPUT_COG_CSV, index=False)
    export_csvs()

    print("Done.")
    print("Saved:")
    print(" -", OUTPUT_VIDEO)
    print(" -", OUTPUT_LANDMARK_CSV)
    print(" -", OUTPUT_TEACHINGSTYLE_CSV)
    print(" -", OUTPUT_COG_CSV)


if __name__ == "__main__":
    main()
    
    
    
    