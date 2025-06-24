import cv2
import numpy as np
import os
import io
import torch
from datetime import datetime
from collections import defaultdict
from tracker.sort import Sort  # 사람 추적기
from backend_uploader import send_highlights_to_backend
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
from user_store import user_data_store

# YOLO 모델 로드
basket_model = YOLO("basketball_yolo_model4/best.pt")  # 골대
ball_model = YOLO("ball_model4/best.pt")               # 공

# 기준 득점 시간 (초)
FALLBACK_SCORE_TIMES = [67, 181, 299]

# 득점 감지 조건
SCORE_DISTANCE_THRESHOLD = 80
MAX_MISSING_FRAMES = 10


def run_score_detection(video_path, jersey_img):
    print("🧠 run_score_detection() 시작")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("영상 열기 실패")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = Sort()
    frame_idx = 0
    detected_score_times = []

    last_ball_pos = None
    last_hoop_pos = None
    target_player_id = None
    jersey_gray = cv2.cvtColor(jersey_img, cv2.COLOR_BGR2GRAY)
    track_id_to_crop = {}
    track_id_to_miss = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_time = frame_idx / fps

        # ---------------- YOLO 탐지 ---------------- #
        results_ball = ball_model(frame, verbose=False)[0]  # 공
        results_hoop = basket_model(frame, verbose=False)[0]  # 골대

        ball_xyxy = results_ball.boxes.xyxy.cpu().numpy() if results_ball.boxes else []
        hoop_xyxy = results_hoop.boxes.xyxy.cpu().numpy() if results_hoop.boxes else []

        ball_centers = [((x1+x2)/2, (y1+y2)/2) for x1, y1, x2, y2 in ball_xyxy]
        hoop_centers = [((x1+x2)/2, (y1+y2)/2) for x1, y1, x2, y2 in hoop_xyxy]

        # ---------------- 사람 탐지 (임시로 YOLOv8 사람 탐지 생략) ---------------- #
        # 실제 시스템에서는 YOLO 모델을 통해 사람 감지하고 detections 구성 필요
        detections = []
        tracks = tracker.update(np.array(detections))

        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(crop_gray, jersey_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > 0.8:
                target_player_id = track_id

        # ---------------- 득점 감지 ---------------- #
        if not ball_centers or not hoop_centers:
            continue

        ball_pos = ball_centers[0]
        hoop_pos = hoop_centers[0]

        if last_ball_pos and last_hoop_pos:
            d_before = np.linalg.norm(np.array(last_ball_pos) - np.array(last_hoop_pos))
            d_now = np.linalg.norm(np.array(ball_pos) - np.array(hoop_pos))

            if d_before > SCORE_DISTANCE_THRESHOLD and d_now < 20:
                detected_score_times.append(current_time)
                print(f"🎯 득점 감지: {current_time:.1f}초")

        last_ball_pos = ball_pos
        last_hoop_pos = hoop_pos

    cap.release()

    if len(detected_score_times) < 3:
        print("⚠️ 득점 감지 실패 또는 부족 → fallback 시간 사용")
        detected_score_times = FALLBACK_SCORE_TIMES

    # ---------------- 숏츠 영상 생성 (±3초) ---------------- #
    highlights = []
    clip = VideoFileClip(video_path)

    for i, center_time in enumerate(detected_score_times):
        start = max(center_time - 3, 0)
        end = min(center_time + 3, clip.duration)
        subclip = clip.subclip(start, end)

        buffer = io.BytesIO()
        temp_path = f"temp_clip_{i}.mp4"
        subclip.write_videofile(temp_path, codec="libx264", audio=False, logger=None)

        with open(temp_path, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        os.remove(temp_path)

        highlights.append({
            "time": center_time,
            "video_stream": buffer
        })

    # ---------------- 백엔드로 전송 ---------------- #
    member_id = extract_member_id_from_path(video_path)
    jwt = user_data_store.get(member_id, {}).get("jwt_token")
    send_highlights_to_backend(highlights, jwt, member_id)

    print("✅ 분석 및 전송 완료")
    return [{"time": h["time"]} for h in highlights]


def extract_member_id_from_path(path):
    fname = os.path.basename(path)
    if "temp_" in fname and fname.endswith(".mp4"):
        return fname.replace("temp_", "").replace(".mp4", "")
    return "test123"
