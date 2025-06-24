import cv2
import os
import argparse

def extract_frames(video_path, save_dir, interval_sec=5):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 영상 열기에 실패했습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("❌ FPS가 0입니다.")
        return

    frame_interval = int(fps * interval_sec)
    print(f"🕒 {interval_sec}초마다 프레임 저장 (interval = {frame_interval} 프레임마다)")

    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            save_path = os.path.join(save_dir, f"frame_{saved_count:04d}.jpg")
            if cv2.imwrite(save_path, frame):
                print(f"✅ Saved: {save_path}")
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"📦 총 저장된 프레임 수: {saved_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="🎬 입력 영상 경로")
    parser.add_argument("--output", type=str, required=True, help="💾 프레임 저장 경로")
    parser.add_argument("--interval", type=int, default=5, help="⏱️ 저장 간격 (초)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.interval)
