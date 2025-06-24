import cv2

video_path = r"C:\Users\이동현\Desktop\shootpoint\basketball_converted.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 영상 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    cv2.imshow('🎬 영상 미리보기', frame)

    # 키보드에서 'q' 누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
