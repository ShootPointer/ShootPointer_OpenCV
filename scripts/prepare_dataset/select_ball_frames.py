import os
import shutil
from ultralytics import YOLO
from PIL import Image

# 1. 경로 설정
model_path = r"C:\Users\이동현\Desktop\shootpoint\runs\detect\ball_model\weights\best.pt"  # 공 인식 모델
frames_dir = r"C:\Users\이동현\Desktop\shootpoint\frames"  # 원본 프레임 이미지 폴더
output_dir = r"C:\Users\이동현\Desktop\shootpoint\frames_selected_ball"  # 추출된 이미지 저장 폴더

# 2. 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 3. 모델 로드
model = YOLO(model_path)

# 4. confidence 기준
CONFIDENCE_THRESHOLD = 0.4
MAX_IMAGES = 300  # 추출할 이미지 최대 수

# 5. 이미지 분석 및 선별
count = 0
for filename in sorted(os.listdir(frames_dir)):
    if filename.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(frames_dir, filename)

        # 추론
        results = model(image_path, verbose=False)[0]
        boxes = results.boxes

        # 공이 감지되었고 confidence가 충분히 높을 때만
        if boxes and any(float(box.conf) >= CONFIDENCE_THRESHOLD for box in boxes):
            shutil.copy(image_path, os.path.join(output_dir, filename))
            count += 1

        if count >= MAX_IMAGES:
            break

print(f"✅ 총 {count}장의 공 이미지가 '{output_dir}'에 저장되었습니다.")
