# ai_worker/services/spring_reporter.py

import json
import logging
import time
import os
from typing import Dict, Any, List
from datetime import datetime
import requests
import asyncio

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. 설정 및 상수
# ─────────────────────────────────────────────────────────────

# 스프링 서버 API 엔드포인트
SPRING_HIGHLIGHT_UPLOAD_URL = "https://tkv00.ddns.net/api/highlight/upload-result"
MAX_RETRIES = 5  # 5회 재시도 약속

# 하이라이트 결과물이 저장되는 베이스 디렉토리 (컨테이너 기준)
# docker-compose 에서 OUTPUT_DIR=/data/highlights/processed 로 주입됨
OUTPUT_BASE_DIR = os.getenv("OUTPUT_DIR", "/data/highlights/processed")

# (옵션) 호스트 기준 prefix. 혹시라도 풀 경로가 /home/videos/... 로 들어오는 경우 대비
HIGHLIGHT_STRIP_PREFIX = os.getenv("HIGHLIGHT_STRIP_PREFIX", "/home/videos")

# ─────────────────────────────────────────────────────────────
# 2. 내부 유틸리티 함수
# ─────────────────────────────────────────────────────────────

def _extract_score_from_label(label: str) -> Dict[str, int]:
    """
    registry.py의 라벨을 기반으로 twoPointCount/threePointCount를 추출합니다.
    """
    if label == "3PT":
        return {"twoPointCount": 0, "threePointCount": 1}
    elif label == "2PT":
        return {"twoPointCount": 1, "threePointCount": 0}
    # "FT" (자유투) 및 기타는 0, 0으로 처리
    return {"twoPointCount": 0, "threePointCount": 0}


def _to_relative_highlight_path(full_path: str) -> str:
    """
    내부 파일 경로를 'highlight/...' 뒤에 붙일 상대 경로로 변환한다.

    우선 순서:
    1) 컨테이너 내부 OUTPUT_BASE_DIR 기준 상대 경로
       예) /data/highlights/processed/{jobId}/file.mp4
           → {jobId}/file.mp4
    2) (옵션) 호스트 기준 HIGHLIGHT_STRIP_PREFIX=/home/videos 기준 상대 경로
       예) /home/videos/highlight/{jobId}/file.mp4
           → highlight/{jobId}/file.mp4
    3) 둘 다 안 맞으면 파일명만 사용 (폴백)
    """
    try:
        full_abs = os.path.abspath(full_path)
    except Exception:
        full_abs = full_path

    # 1) OUTPUT_BASE_DIR 기준
    try:
        base_abs = os.path.abspath(OUTPUT_BASE_DIR)
        if full_abs.startswith(base_abs):
            rel = os.path.relpath(full_abs, base_abs)  # 예) {jobId}/file.mp4
            return rel.replace("\\", "/").lstrip("/")
    except Exception:
        pass

    # 2) /home/videos 기준 (혹시라도 그런 경로가 넘어온 경우)
    try:
        host_base_abs = os.path.abspath(HIGHLIGHT_STRIP_PREFIX)
        if full_abs.startswith(host_base_abs):
            rel = os.path.relpath(full_abs, host_base_abs)  # 예) highlight/{jobId}/file.mp4
            return rel.replace("\\", "/").lstrip("/")
    except Exception:
        pass

    # 3) 폴백: 파일명만
    return os.path.basename(full_abs).replace("\\", "/")


def _build_payload(
    job_id: str,
    member_id: str,  # 현재는 payload 에 사용하지 않고, 헤더(X-Member-Id)로만 사용
    highlight_id: str,
    output_files_with_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    스프링 API 명세에 맞는 JSON Payload를 구성합니다.

    {
      "jobId": "...",
      "highlightIdentifier": "...",
      "highlightUrls": [
        {
          "highlightUrl": "highlight/{jobId}/파일명.mp4",
          "twoPointCount": 0,
          "threePointCount": 1
        }
      ],
      "createdAt": "2025-11-16T17:31:41.544Z"
    }
    """
    # createdAt: ISO 8601, 밀리초, Z(UTC) 형식
    current_time_iso = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    highlight_urls_list: List[Dict[str, Any]] = []

    for item in output_files_with_segments:
        # 실제 파일 경로(컨테이너 기준 or 호스트 기준)
        output_path = item["output_path"]

        # OUTPUT_BASE_DIR 또는 /home/videos 기준으로 상대 경로 변환
        #   예)
        #     /data/highlights/processed/{jobId}/{jobId}_segment_01.mp4
        #       -> {jobId}/{jobId}_segment_01.mp4
        #     /home/videos/highlight/{jobId}/{jobId}_segment_01.mp4
        #       -> highlight/{jobId}/{jobId}_segment_01.mp4
        relative_path = _to_relative_highlight_path(output_path)

        # 최종적으로 백엔드에 전달되는 값:
        #   highlight/{relative_path}
        # 예) highlight/a1d1f806c8324d02/a1d1f806c8324d02_segment_01.mp4
        relative_url = f"https://tkv00.ddns.net/highlight/{relative_path.lstrip('/')}"


        segment_label = item["segment"].get("label", "UNKNOWN")
        score = _extract_score_from_label(segment_label)

        highlight_urls_list.append(
            {
                "highlightUrl": relative_url,                 # ✅ 스펙: highlightUrl
                "twoPointCount": score["twoPointCount"],
                "threePointCount": score["threePointCount"],
            }
        )

    return {
        "jobId": job_id,                                   # ✅ 추가: jobId 루트에 포함
        "highlightIdentifier": highlight_id,               # ✅ 그대로 사용
        "highlightUrls": highlight_urls_list,
        "createdAt": current_time_iso,                     # ✅ 루트에 createdAt 추가
    }

# ─────────────────────────────────────────────────────────────
# 3. HTTP 전송 및 재시도 (동기 함수)
# ─────────────────────────────────────────────────────────────

def _send_highlight_sync(
    job_id: str,
    member_id: str,
    highlight_id: str,
    output_files_with_segments: List[Dict[str, Any]],
) -> bool:
    """실제 동기 HTTP POST 요청을 실행합니다."""

    payload = _build_payload(job_id, member_id, highlight_id, output_files_with_segments)

    headers = {
        "Content-Type": "application/json",
        "X-Member-Id": member_id,  # 명세의 필수 헤더
    }

    for attempt in range(MAX_RETRIES):
        logger.info(f"[{highlight_id}] HTTP POST Attempt: {attempt + 1} / {MAX_RETRIES}")

        try:
            response = requests.post(
                SPRING_HIGHLIGHT_UPLOAD_URL,
                data=json.dumps(payload),
                headers=headers,
                timeout=15,
            )

            # 응답 코드 및 바디 확인 로직 (이전 코드와 동일, 성공 시 True 반환)
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if response_json.get("success") is True:
                        logger.info("✅ HTTP POST Success: Status 200 & success: true")
                        return True

                    # success:false 인 경우, message / error 도 같이 로그
                    message = (
                        response_json.get("message")
                        or response_json.get("error")
                        or str(response_json)
                    )
                    logger.warning(
                        "❌ API Internal Failure (success: false). "
                        f"message={message}. Retrying..."
                    )
                except json.JSONDecodeError:
                    logger.error(
                        "⚠️ 응답 JSON 디코딩 실패. Raw response text: %r. Retrying...",
                        response.text,
                    )
            else:
                # 200 이 아닌 경우, body 도 같이 출력
                logger.error(
                    f"❌ HTTP Status Code Error: {response.status_code}. "
                    f"Response body: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ HTTP Request Error: {e.__class__.__name__}: {e}. Retrying...")

        # 재시도 대기 (지수 백오프)
        if attempt < MAX_RETRIES - 1:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
        else:
            logger.critical("❌ Maximum HTTP POST retries exceeded. Final report failed.")
            return False

    return False

# ─────────────────────────────────────────────────────────────
# 4. 메인 비동기 진입점
# ─────────────────────────────────────────────────────────────

async def send_highlight_result_with_retry(
    job_id: str,
    member_id: str,
    highlight_id: str,
    output_files_with_segments: List[Dict[str, Any]],
) -> bool:
    """
    비동기 환경에서 동기 HTTP 전송 함수를 안전하게 호출합니다.
    """
    return await asyncio.to_thread(
        _send_highlight_sync,
        job_id,
        member_id,
        highlight_id,
        output_files_with_segments,
    )
