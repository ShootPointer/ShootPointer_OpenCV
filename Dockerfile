FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 안정적인 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 먼저 복사하고 패키지 설치 (캐시 효율성)
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 프로젝트 파일 전체 복사
COPY . /app/

# YOLO 모델 디렉토리가 제대로 복사되었는지 확인하고 디버깅
RUN ls -la /app/
RUN ls -la /app/basketball_yolo_model4/ || echo "basketball_yolo_model4 directory not found"

# 포트 노출
EXPOSE 8888

# 앱 실행
CMD ["python", "app.py"]