FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# compose에서 build.context가 ./ai_worker 이므로, 아래 COPY들은 그 기준 경로임
COPY services   ./ai_worker/services
COPY utils      ./ai_worker/utils
COPY configs    ./ai_worker/configs
COPY ai_modules ./ai_worker/ai_modules

# 패키지 인식 (소스에 이미 __init__.py 있으면 생략 가능)
# RUN touch /app/ai_worker/__init__.py \
#          /app/ai_worker/services/__init__.py \
#          /app/ai_worker/utils/__init__.py \
#          /app/ai_worker/configs/__init__.py \
#          /app/ai_worker/ai_modules/__init__.py

# PYTHONPATH는 /app 만 설정 (불필요한 다중 경로 제거)
ENV PYTHONPATH=/app

# 모듈 방식 실행
CMD ["python", "-m", "ai_worker.services.ai_worker_simulator"]
