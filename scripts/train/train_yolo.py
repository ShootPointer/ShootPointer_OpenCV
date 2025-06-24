from ultralytics import YOLO

# ① 모델 로드 (가볍게 시작: yolov8n)
model = YOLO('yolov8n.pt')  # 처음엔 small 모델로 실험

# ② 학습 시작
model.train(
    data='C:/Users/이동현/Desktop/yolo_dataset/data.yaml',
    epochs=30,         # 학습 횟수 (적게는 10, 많게는 50~100)
    imgsz=640,         # 입력 이미지 사이즈
    batch=8,           # GPU 성능에 따라 8~16
    name='basketball_yolo_model'  # 저장될 모델 폴더 이름
)
