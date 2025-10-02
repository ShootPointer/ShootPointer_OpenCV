# Basket Highlight AI

농구 **풀경기 영상**에서 특정 **등번호 선수**의 플레이를 찾아, **득점 장면 전/후 ±5초**로 잘라 **숏츠 하이라이트 ZIP**을 만들어주는 API 서버입니다.  
Docker 이미지에 **OpenCV + Tesseract + FFmpeg**가 포함되어 있어, 팀원이 어디서든 동일 환경으로 실행할 수 있습니다.

## ✨ Features
- 파일 업로드 → 임시 저장 → 처리 후 자동 정리
- 등번호 감지(OpenCV + Tesseract) → 선수 구간 추정
- 오디오 무음 분석(ffmpeg `silencedetect`) 기반 자동 후보 타임스탬프
- 지정된 타임스탬프들을 ±5초로 클립 생성 → ZIP 스트리밍 응답
- Swagger UI 제공 (`/docs`)

## 🚀 Quick Start (Docker)
```bash
git clone https://github.com/ShootPointer/ShootPointer_OpenCV.git
cd ShootPointer_OpenCV
cp .env.example .env   # Windows: copy .env.example .env

docker compose build --no-cache
docker compose up
