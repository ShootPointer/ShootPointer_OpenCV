import cv2
import os
from ultralytics import YOLO

# ====== 사용자 정의 경로 ======
video_path = r"C:\Users\이동현\Desktop\shootpoint\basketball.mp4"
model_path = r"C:\Users\이동현\Desktop\shootpoint\runs\detect\basketball_yolo_model4\weights\best.pt"
output_dir = r"C:\Users\이동현\Desktop\shootpoint\ball_detected_frames"
conf_threshold = 0.15  # 신뢰도 낮춰서 공까지 잡게 하기

# ====== 클래스 설정 ======
BALL_CLASS_ID = 1  # 공 클래스 번호 (makesense에서 지정한 번호로 맞춰야 함)

# ====== 디렉토리 준비 ======
os.makedirs(output_dir, exist_ok=True)

# ====== 모델 로드 ======
model = YOLO(model_path)

# ====== 영상 열기 ======
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 영상 열기에 실패했습니다.")
    exit()

frame_idx = 0
saved_count = 0

print("📦 공 탐지 중...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 추론
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]

    # 공 클래스가 포함되어 있는지 확인
    ball_detected = any(int(cls) == BALL_CLASS_ID for cls in results.boxes.cls)

    if ball_detected:
        save_path = os.path.join(output_dir, f"ball_frame_{frame_idx:05d}.jpg")
        cv2.imwrite(save_path, frame)
        saved_count += 1

    frame_idx += 1

cap.release()
print(f"✅ 공이 감지된 프레임 {saved_count}장 저장 완료: {output_dir}")
