import cv2
import os
import shutil

# 경로 설정
input_folder = r"C:\Users\이동현\Desktop\shootpoint\frames_ball_all"
output_folder = r"C:\Users\이동현\Desktop\shootpoint\frames_selected_ball"

# 출력 폴더 초기화
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# 조건 설정
min_area = 100  # 최소 검출 면적
max_frames = 300  # 최대 추출할 프레임 수
ball_color_range = {
    "lower": (5, 50, 50),     # HSV 기준, 오렌지/갈색 계열
    "upper": (20, 255, 255)
}

# 추출된 수
selected_count = 0

# 파일명 기준 정렬
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))])

for file_name in image_files:
    img_path = os.path.join(input_folder, file_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ball_color_range["lower"], ball_color_range["upper"])

    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_ball = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            has_ball = True
            break

    if has_ball:
        shutil.copy(img_path, os.path.join(output_folder, file_name))
        selected_count += 1

    if selected_count >= max_frames:
        break

print(f"[완료] 공이 잘 나온 이미지 {selected_count}장 추출 완료 ✅")
