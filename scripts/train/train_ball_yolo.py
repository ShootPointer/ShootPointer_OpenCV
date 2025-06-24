from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')  # 혹은 yolov8s.pt 등 다른 모델
    model.train(
        data='C:/Users/이동현/Desktop/yolo_ball_dataset/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,  # GPU 사용
        name='ball_model'
    )

if __name__ == '__main__':
    main()
