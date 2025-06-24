import os
import random
import shutil

# 설정
base_path = r"C:\Users\이동현\Desktop\yolo_ball_dataset"
train_img_dir = os.path.join(base_path, "images", "train")
val_img_dir = os.path.join(base_path, "images", "val")
train_label_dir = os.path.join(base_path, "labels", "train")
val_label_dir = os.path.join(base_path, "labels", "val")

val_sample_count = 30  # 복사할 검증 이미지 수

# val 디렉토리 생성
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# train 이미지 리스트 가져오기
img_files = [f for f in os.listdir(train_img_dir) if f.endswith((".jpg", ".png"))]

# 랜덤 샘플링
val_samples = random.sample(img_files, min(val_sample_count, len(img_files)))

for img_name in val_samples:
    label_name = os.path.splitext(img_name)[0] + ".txt"

    # 이미지 복사
    shutil.move(os.path.join(train_img_dir, img_name), os.path.join(val_img_dir, img_name))

    # 라벨 복사
    src_label = os.path.join(train_label_dir, label_name)
    dst_label = os.path.join(val_label_dir, label_name)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)

print(f"[✔] 총 {len(val_samples)}개의 이미지와 라벨을 검증용(val) 폴더로 이동했습니다.")
