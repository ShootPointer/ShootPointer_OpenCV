# backend_uploader.py

import requests
from datetime import datetime
from io import BytesIO
import uuid
import json

def send_highlights_to_backend(highlight_clips, member_id, jwt_token):
    """
    3개의 하이라이트 영상을 백엔드로 전송하는 함수

    :param highlight_clips: List[Tuple[BytesIO, datetime]]
        - BytesIO: 메모리에 저장된 mp4 영상
        - datetime: 생성 시간
    :param member_id: str
    :param jwt_token: str
    :return: requests.Response
    """
    if len(highlight_clips) != 3:
        raise ValueError("반드시 하이라이트 클립이 3개여야 합니다.")

    # ✅ 하이라이트 키 (예: 123_20250624_221500)
    highlight_key = f"{member_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ✅ multipart 파일 준비 (key는 모두 'highlights')
    files = []
    for idx, (clip_io, created_at) in enumerate(highlight_clips):
        clip_io.seek(0)
        filename = f"highlight_{idx+1}.mp4"
        files.append((
            'highlights',
            (filename, clip_io, 'video/mp4')
        ))

    # ✅ uploadHighlightDto JSON 생성
    created_at_list = [dt.strftime("%Y-%m-%dT%H:%M:%S") for _, dt in highlight_clips]
    upload_highlight_dto = {
        "highlightKey": highlight_key,
        "createdAt": created_at_list
    }

    # ✅ form-data 필드 추가
    data = {
        "uploadHighlightDto": json.dumps(upload_highlight_dto)
    }

    # ✅ 헤더 설정
    headers = {
        "X-Member-Id": member_id,
        "Authorization": jwt_token  # 예: "Bearer eyJ..."
    }

    # ✅ 백엔드 요청
    try:
        response = requests.post(
            url="http://tkv00.ddns.net:9000/api/highlight/upload-result",
            headers=headers,
            data=data,
            files=files,
            timeout=30
        )

        print("📤 백엔드 응답 상태:", response.status_code)
        print("📨 응답 본문:", response.text)
        return response

    except requests.exceptions.RequestException as e:
        print("❌ 백엔드 전송 실패:", str(e))
        return None
