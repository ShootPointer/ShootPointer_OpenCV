import os
import random
import shutil

# 경로 설정
base_dir = r"C:\Users\이동현\Desktop\yolo_dataset"  # ← 본인 폴더 경로 맞게 수정
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# 결과 폴더
output_dirs = {
    "train_images": os.path.join(base_dir, "images", "train"),
    "val_images": os.path.join(base_dir, "images", "val"),
    "train_labels": os.path.join(base_dir, "labels", "train"),
    "val_labels": os.path.join(base_dir, "labels", "val")
}

# 결과 폴더 생성
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

# 이미지 파일 목록 불러오기
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
random.shuffle(image_files)

# 8:2로 분할
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def copy_files(file_list, img_dst, lbl_dst):
    for filename in file_list:
        img_src = os.path.join(images_dir, filename)
        lbl_src = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))

        shutil.copy(img_src, os.path.join(img_dst, filename))
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(lbl_dst, os.path.basename(lbl_src)))
        else:
            print(f"⚠️ 라벨 없음: {lbl_src}")

# 파일 복사
copy_files(train_files, output_dirs["train_images"], output_dirs["train_labels"])
copy_files(val_files, output_dirs["val_images"], output_dirs["val_labels"])

print("✅ 데이터셋 분할 완료! 🎉")
