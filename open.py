import cv2
import pytesseract
import os
import numpy as np
from threading import Thread

# 설정
TARGET_NUMBER = '7'
SAVE_DIR = r"C:\file video"
VIDEO_PATH = r"C:\shoot\13509078_3840_2160_60fps.mp4"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 저장 경로 폴더 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 비디오 로딩
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

frame_buffer = []
max_clips = 10
saved_clips = 0
frame_idx = 0
detect_buffer = []

def save_clip(start_f, end_f, clip_id):
    clip_cap = cv2.VideoCapture(VIDEO_PATH)
    clip_cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    out_path = os.path.join(SAVE_DIR, f"highlight_{TARGET_NUMBER}_{clip_id}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for f in range(start_f, end_f):
        ret, fr = clip_cap.read()
        if not ret:
            break
        writer.write(fr)
    writer.release()
    clip_cap.release()
    print(f"[✅] 클립 저장 완료: {out_path}")

def already_saved(frame_index, margin=fps*10):
    return any(abs(frame_index - idx) < margin for idx in detect_buffer)

while cap.isOpened() and saved_clips < max_clips:
    ret, frame = cap.read()
    if not ret:
        break

    frame_display = cv2.resize(frame, (1280, 720))
    frame_buffer.append(frame)
    if len(frame_buffer) > fps * 10:
        frame_buffer.pop(0)

    if frame_idx % 5 == 0 and not already_saved(frame_idx):
        # OCR 전처리
        small = cv2.resize(frame, (width // 2, height // 2))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        data = pytesseract.image_to_data(binary, lang='eng', output_type=pytesseract.Output.DICT)

        for i, text in enumerate(data['text']):
            if text.strip() == TARGET_NUMBER:
                print(f"[INFO] 번호 {TARGET_NUMBER} 감지됨 (프레임 {frame_idx})")
                detect_buffer.append(frame_idx)

                start_frame = max(frame_idx - fps * 5, 0)
                end_frame = min(frame_idx + fps * 5, total_frames)
                Thread(target=save_clip, args=(start_frame, end_frame, saved_clips + 1)).start()
                saved_clips += 1
                break

    # 화면 표시
    cv2.imshow("전체 영상 (번호 7 추적 중)", frame_display)
    if cv2.waitKey(1) & 0xFF == 27:
        print("[INFO] ESC 눌러서 종료됨.")
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

