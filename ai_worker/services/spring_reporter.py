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
MAX_RETRIES = 5 # 5회 재시도 약속

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


def _build_payload(member_id: str, highlight_id: str, output_files_with_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    API 명세에 맞는 JSON Payload를 구성합니다.
    """
    
    # createdDate: ISO 8601, 밀리초, Z (UTC)를 포함하는 형식
    current_time_iso = datetime.now().isoformat(timespec='milliseconds') + 'Z' 

    highlight_urls_list = []
    for item in output_files_with_segments:
        # URL 형식: 'highlight/파일명.mp4'
        relative_path = os.path.basename(item["output_path"]) 
        relative_url = f"highlight/{relative_path}" 
        
        segment_label = item["segment"].get("label", "UNKNOWN")
        score = _extract_score_from_label(segment_label)
        
        highlight_urls_list.append({
            "url": relative_url,
            "twoPointCount": score["twoPointCount"],
            "threePointCount": score["threePointCount"],
            "createdDate": current_time_iso 
        })

    return {
        "highlightIdentifier": highlight_id,
        "highlightUrls": highlight_urls_list
    }

# ─────────────────────────────────────────────────────────────
# 3. HTTP 전송 및 재시도 (동기 함수)
# ─────────────────────────────────────────────────────────────

def _send_highlight_sync(
    member_id: str, 
    highlight_id: str, 
    output_files_with_segments: List[Dict[str, Any]]
) -> bool:
    """실제 동기 HTTP POST 요청을 실행합니다."""
    
    payload = _build_payload(member_id, highlight_id, output_files_with_segments)
    
    headers = {
        "Content-Type": "application/json",
        "X-Member-Id": member_id # 명세의 필수 헤더
    }
    
    for attempt in range(MAX_RETRIES):
        logger.info(f"[{highlight_id}] HTTP POST Attempt: {attempt + 1} / {MAX_RETRIES}")
        
        try:
            response = requests.post(
                SPRING_HIGHLIGHT_UPLOAD_URL, 
                data=json.dumps(payload), 
                headers=headers,
                timeout=15 
            )

            # 응답 코드 및 바디 확인 로직 (이전 코드와 동일, 성공 시 True 반환)
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if response_json.get("success") is True:
                        logger.info("✅ HTTP POST Success: Status 200 & success: true")
                        return True
                    logger.warning(f"❌ API Internal Failure (success: false). Retrying...")
                except json.JSONDecodeError:
                    logger.error(f"⚠️ 응답 JSON 디코딩 실패. Retrying...")
            else:
                logger.error(f"❌ HTTP Status Code Error: {response.status_code}. Retrying...")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ HTTP Request Error: {e.__class__.__name__}. Retrying...")
        
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
    member_id: str, 
    highlight_id: str, 
    output_files_with_segments: List[Dict[str, Any]]
) -> bool:
    """
    비동기 환경에서 동기 HTTP 전송 함수를 안전하게 호출합니다.
    """
    return await asyncio.to_thread(_send_highlight_sync, member_id, highlight_id, output_files_with_segments)