import os
import shutil
import random

# 경로 설정
image_src = r"C:\Users\이동현\Desktop\shootpoint\frames"
label_src = r"C:\Users\이동현\Desktop\labels_my-project-name_2025-06-23-02-05-48"

base_dir = r"C:\Users\이동현\Desktop\yolo_dataset"
images_train = os.path.join(base_dir, "images", "train")
images_val = os.path.join(base_dir, "images", "val")
labels_train = os.path.join(base_dir, "labels", "train")
labels_val = os.path.join(base_dir, "labels", "val")

# 폴더 생성
for d in [images_train, images_val, labels_train, labels_val]:
    os.makedirs(d, exist_ok=True)

# 라벨이 있는 이미지만 필터링
valid_images = []
for fname in os.listdir(image_src):
    if fname.endswith(".jpg"):
        label_name = fname.replace(".jpg", ".txt")
        if os.path.exists(os.path.join(label_src, label_name)):
            valid_images.append(fname)

print(f"✅ 라벨 있는 이미지 수: {len(valid_images)}")

# 데이터 분할
random.shuffle(valid_images)
split_idx = int(len(valid_images) * 0.8)
train_files = valid_images[:split_idx]
val_files = valid_images[split_idx:]

def copy_pair(files, img_dst, lbl_dst):
    for img_name in files:
        label_name = img_name.replace(".jpg", ".txt")
        shutil.copy(os.path.join(image_src, img_name), os.path.join(img_dst, img_name))
        shutil.copy(os.path.join(label_src, label_name), os.path.join(lbl_dst, label_name))

# 복사 실행
copy_pair(train_files, images_train, labels_train)
copy_pair(val_files, images_val, labels_val)

print("✅ 라벨 있는 데이터만 정리 완료 (train/val)")
