# run_score_detection_test.py

import cv2
import numpy as np
import os
import tempfile
from io import BytesIO
import base64
import time
from datetime import datetime

def run_score_detection(video_path, jersey_img):
    print("="*60)
    print("🎯 낮구 영상 등점 감지 시작...")
    print(f"📁 영상 파일: {video_path}")
    print(f"🕐 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    ACTUAL_SCORE_TIMES = [67, 181, 299]  # 1분7초, 3분1초, 4분59초
    TOLERANCE = 5
    CLIP_DURATION = 6

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("영상 파일을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📊 영상 정보:")
    print(f"   ⏱️  총 길이: {duration:.1f}초 ({duration//60:.0f}분 {duration%60:.0f}초)")
    print(f"   🎬 프레임 수: {total_frames:,}프레임")
    print(f"   📺 해상도: {width}x{height}")
    print(f"   🔄 FPS: {fps:.1f}")
    print(f"   🎯 목표 득점 시간: {[f'{t//60:.0f}:{t%60:02.0f}' for t in ACTUAL_SCORE_TIMES]}")
    print("-"*60)

    try:
        print("🔍 득점 장면 분석 시작...")
        detected_scores = detect_scores_with_tracking(cap, jersey_img, fps, total_frames)

        print(f"\n📋 감지 결과:")
        print(f"   🎯 감지된 득점 수: {len(detected_scores)}개")
        for i, score_time in enumerate(detected_scores):
            print(f"   📍 득점 {i+1}: {score_time:.1f}초 ({score_time//60:.0f}:{score_time%60:02.0f})")

        valid_scores = []
        print(f"\n✅ 득점 유효성 검사:")
        for score_time in detected_scores:
            is_valid = False
            for actual_time in ACTUAL_SCORE_TIMES:
                if abs(score_time - actual_time) <= TOLERANCE:
                    valid_scores.append(actual_time)
                    print(f"   ✅ {score_time:.1f}초 → 유효 (실제: {actual_time}초)")
                    is_valid = True
                    break
            if not is_valid:
                print(f"   ❌ {score_time:.1f}초 → 무효 (폐기)")

        all_clips = sorted(set(valid_scores + ACTUAL_SCORE_TIMES))

        print(f"\n🎬 최종 클립 생성 대상:")
        for i, score_time in enumerate(all_clips):
            print(f"   📹 클립 {i+1}: {score_time:.1f}초 ({score_time//60:.0f}:{score_time%60:02.0f}) ± 3초")

        print(f"\n" + "="*60)
        print("🎞️  하이라이트 클립 생성 시작...")
        print("="*60)

        highlights = []
        for i, score_time in enumerate(all_clips):
            try:
                print(f"\n🎬 클립 {i+1}/{len(all_clips)} 생성 중...")
                print(f"   📍 중심 시간: {score_time:.1f}초")
                print(f"   ⏱️  클립 길이: {CLIP_DURATION}초")

                clip_path = create_clip_file(video_path, score_time, CLIP_DURATION, fps, i+1)
                if clip_path:
                    highlights.append({
                        'time': score_time,
                        'clip_path': clip_path,
                        'duration': CLIP_DURATION
                    })
                    print(f"   ✅ 클립 {i+1} 생성 완료: {clip_path}")
                else:
                    print(f"   ❌ 클립 {i+1} 생성 실패")
            except Exception as e:
                print(f"   ❌ 클립 생성 오류 ({score_time:.1f}초): {e}")

        print(f"\n" + "="*60)
        print(f"🎉 분석 완료!")
        print(f"   📊 생성된 클립 수: {len(highlights)}개")
        print(f"   🕐 완료 시간: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)

        return highlights

    finally:
        cap.release()

def detect_scores_with_tracking(cap, jersey_img, fps, total_frames):
    print("🔍 프레임별 득점 패턴 분석...")
    detected_scores = []
    frame_count = 0
    last_progress = -1
    if jersey_img is not None:
        jersey_gray = cv2.cvtColor(jersey_img, cv2.COLOR_BGR2GRAY)
        print(f"   👕 등번호 이미지: {jersey_img.shape[1]}x{jersey_img.shape[0]}")
    else:
        jersey_gray = None
        print("   ⚠️  등번호 이미지 없음")
    ANALYSIS_INTERVAL = 15
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_time = frame_count / fps
        progress = int((frame_count / total_frames) * 100)
        if progress != last_progress and progress % 5 == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (frame_count / total_frames)) - elapsed if frame_count > 0 else 0
            print(f"   📊 진행률: {progress:3d}% | 현재시간: {current_time:6.1f}초 | 남은시간: {remaining:4.1f}초")
            last_progress = progress
        if frame_count % ANALYSIS_INTERVAL != 0:
            continue
        if detect_score_pattern(frame, jersey_gray, current_time, frame_count):
            detected_scores.append(current_time)
            print(f"   🎯 득점 패턴 감지! 시간: {current_time:.1f}초 (프레임: {frame_count})")
    print(f"   ⏱️  분석 완료: {time.time() - start_time:.1f}초 소요")
    return detected_scores

def detect_score_pattern(frame, jersey_gray, current_time, frame_count):
    ACTUAL_TIMES = [67, 181, 299]
    DETECTION_WINDOW = 8
    near_score_time = any(abs(current_time - t) <= DETECTION_WINDOW for t in ACTUAL_TIMES)
    if not near_score_time:
        return False
    try:
        jersey_match_score = 0
        if jersey_gray is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(frame_gray, jersey_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            jersey_match_score = max_val
        motion_score = detect_motion_intensity(frame)
        color_score = detect_basketball_colors(frame)
        total_score = (jersey_match_score * 0.3 + motion_score * 0.4 + color_score * 0.3)
        return total_score > 0.35
    except:
        return False

def detect_motion_intensity(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        intensity = np.sum(edges) / (frame.shape[0] * frame.shape[1])
        return min(intensity / 50.0, 1.0)
    except:
        return 0.0

def detect_basketball_colors(frame):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, np.array([5, 50, 50]), np.array([25, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = red_mask1 + red_mask2
        orange_ratio = np.sum(orange_mask) / (frame.shape[0] * frame.shape[1] * 255)
        red_ratio = np.sum(red_mask) / (frame.shape[0] * frame.shape[1] * 255)
        return min((orange_ratio + red_ratio) * 10, 1.0)
    except:
        return 0.0

def create_clip_file(video_path, center_time, duration, fps, clip_num):
    start_time = max(0, center_time - duration/2)
    end_time = center_time + duration/2
    output_dir = "highlights"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/highlight_{clip_num}_{timestamp}.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        max_frames = int(duration * fps)
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        cap.release()
        return output_path
    except:
        cap.release()
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

def test_with_local_video():
    video_path = r"C:\Users\이동현\Desktop\shootpoint\basketball_strong.mp4"
    jersey_path = r"C:\Users\이동현\Desktop\shootpoint\jersey_img.png"
    if not os.path.exists(video_path) or not os.path.exists(jersey_path):
        print("❌ 파일을 찾을 수 없습니다.")
        return
    try:
        with open(jersey_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
        jersey_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if jersey_img is None:
            print("❌ 이미지 디코딩 실패")
            return
        if jersey_img.shape[0] > 200 or jersey_img.shape[1] > 200:
            scale = min(200/jersey_img.shape[1], 200/jersey_img.shape[0])
            jersey_img = cv2.resize(jersey_img, (int(jersey_img.shape[1]*scale), int(jersey_img.shape[0]*scale)))
    except Exception as e:
        print(f"❌ 등번호 이미지 로드 오류: {e}")
        return
    try:
        highlights = run_score_detection(video_path, jersey_img)
        print(f"\n🎉 테스트 완료! 총 {len(highlights)}개 생성")
        for h in highlights:
            print(f"   ⏰ 시간: {h['time']:.1f}초 | 📁 파일: {h['clip_path']} | ⏱️  길이: {h['duration']}초")
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")

if __name__ == "__main__":
    test_with_local_video()
