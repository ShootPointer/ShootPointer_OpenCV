FROM python:3.11-slim

# FFmpeg + Tesseract OCR + 빌드 도구 + 폰트
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스
COPY app ./app

# ✅ 런타임 디렉토리(저장/임시) 사전 생성 (권한 이슈 방지)
#   - SAVE_ROOT=/data/highlights
#   - TEMP_ROOT=/app/.tmp/upload_data  (환경변수에서 사용)
RUN mkdir -p /data/highlights \
    && mkdir -p /app/.tmp/upload_data \
    && mkdir -p /app/.tmp/upload_data/chunks

EXPOSE 8000

# uvicorn 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
