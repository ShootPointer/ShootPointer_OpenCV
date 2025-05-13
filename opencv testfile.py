import cv2
import pytesseract
import os

# Tesseract-OCR 설치 경로 (Windows 기준)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 영상 경로
video_path = r'C:\shoot\13509078_3840_2160_60fps.mp4'
cap = cv2.VideoCapture(video_path)

# 영상 정보
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 저장 경로
output_path = r'C:\shoot\highlight_20_detected.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
saved = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 화면용 축소
    display_frame = cv2.resize(frame, (1280, 720))

    # OCR을 위한 전처리 (흑백 + threshold)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR 실행
    text_data = pytesseract.image_to_data(thresh, lang='eng', output_type=pytesseract.Output.DICT)

    found = False
    for i, text in enumerate(text_data['text']):
        if text.strip() == '20':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            # 원본 위치에 사각형 표시
            scale_x = 1280 / width
            scale_y = 720 / height
            cv2.rectangle(display_frame,
                          (int(x * scale_x), int(y * scale_y)),
                          (int((x + w) * scale_x), int((y + h) * scale_y)),
                          (0, 255, 0), 2)
            cv2.putText(display_frame, '20 Detected', (int(x * scale_x), int((y - 10) * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            found = True
            break

    # 화면에 출력
    cv2.imshow("Detecting 20", display_frame)

    # 감지 시 영상 저장 (단 한 번만)
    if found and not saved:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        start_frame = max(current_frame - (fps * 5), 0)
        end_frame = min(current_frame + (fps * 5), total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] 번호 20 발견! 클립 저장 시작...")

        for f in range(start_frame, end_frame):
            ret_clip, frame_clip = cap.read()
            if not ret_clip:
                break
            clip_writer.write(frame_clip)

        clip_writer.release()
        saved = True
        print(f"[INFO] 하이라이트 저장 완료: {output_path}")

        # 다시 감지 위치로 돌아감
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # ESC 종료 또는 빠른 재생
    if cv2.waitKey(1) & 0xFF == 27:
        print("[INFO] ESC 눌러서 종료")
        break

cap.release()
cv2.destroyAllWindows()