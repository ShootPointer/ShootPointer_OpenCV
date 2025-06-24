import os
import shutil
import random

# 2천장 이미지 폴더 경로
source_folder = r"C:\Users\이동현\Desktop\shootpoint\ball_detected_frames"

# 샘플링한 이미지 저장할 폴더 경로
target_folder = r"C:\Users\이동현\Desktop\shootpoint\ball_frames_sampled"

# 샘플 개수
num_to_sample = 300

# 저장 폴더 없으면 생성
os.makedirs(target_folder, exist_ok=True)

# 이미지 확장자 필터링
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# 모든 이미지 불러오기
all_images = [f for f in os.listdir(source_folder) if os.path.splitext(f)[1].lower() in image_extensions]

# 총 이미지보다 샘플 수가 많으면 제한
num_to_sample = min(num_to_sample, len(all_images))

# 무작위 샘플 추출
sampled_images = random.sample(all_images, num_to_sample)

# 복사
for img in sampled_images:
    shutil.copy2(os.path.join(source_folder, img), os.path.join(target_folder, img))

print(f"{num_to_sample}장 샘플링 완료 → {target_folder}")
