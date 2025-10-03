# app/routers/player.py
from __future__ import annotations

import logging
from pathlib import Path
import shutil
import uuid
from typing import List, Tuple

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.config import settings
from app.services.jersey import detect_player_segments
from app.services.pipeline import generate_highlight_clips, build_zip_in_memory

router = APIRouter(prefix="/player", tags=["player"])
logger = logging.getLogger(__name__)

# 업로드 임시 폴더
TMP_DIR = Path("/tmp/uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _save_upload(video: UploadFile) -> Path:
    suffix = Path(video.filename or "").suffix or ".mp4"
    tmp = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    logger.info(f"[player] saved upload -> {tmp.name} ({tmp.stat().st_size} bytes)")
    return tmp


def _in_any_segment(t: float, segments: List[Tuple[float, float]]) -> bool:
    for s, e in segments:
        if s <= t <= e:
            return True
    return False


def _parse_timestamps(raw: str) -> List[float]:
    """
    '30, 75 ,120' 같은 문자열을 float 리스트로 변환.
    공백/중복/빈 항목 제거, 음수 제거, 정렬/중복제거.
    """
    out: List[float] = []
    for part in (raw or "").replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            val = float(p)
            if val >= 0:
                out.append(val)
        except Exception:
            # 잘못된 입력은 조용히 무시
            pass
    # 정렬 + 중복 제거
    return sorted(set(out))


# ─────────────────────────────────────────────────────────────
# API: 세그먼트(플레이 구간) 검출
# ─────────────────────────────────────────────────────────────
@router.post("/segments")
async def player_segments(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    jersey_number: int = Form(..., description="찾을 등번호 (예: 23)"),
):
    """
    업로드된 영상에서 등번호 기반 플레이 구간을 검출해 [start,end] 초 리스트 반환.
    (현재는 간단 파이프라인; 모델 고도화 시 정확도 향상)
    """
    tmp_in = _save_upload(video)
    try:
        segs = detect_player_segments(tmp_in, jersey_number)
        logger.info(f"[player/segments] jersey={jersey_number} -> segments={len(segs)}")
        return {"segments": segs}
    except Exception as e:
        logger.exception(f"[player/segments] failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        try:
            tmp_in.unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# API: 하이라이트 ZIP 생성 (타임스탬프 필터링 + 컷팅)
# ─────────────────────────────────────────────────────────────
@router.post("/highlight")
async def player_highlight_zip(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    jersey_number: int = Form(..., description="찾을 등번호 (예: 23)"),
    timestamps: str = Form(..., description="쉼표구분 초 리스트, 예: 30,75,120"),
    pre: float = Form(default=settings.DEFAULT_PRE),
    post: float = Form(default=settings.DEFAULT_POST),
    max_clips: int = Form(default=settings.MAX_CLIPS),
):
    """
    1) 등번호 기반 플레이 구간 검출
    2) 입력 timestamps 중 해당 구간에 속한 것만 필터
    3) 각 이벤트 주변 [t-pre, t+post] 컷 생성
    4) ZIP으로 묶어 스트리밍 다운로드
    """
    tmp_in = _save_upload(video)
    clip_paths: List[str] = []

    try:
        # 1) 선수 구간 검출
        segments = detect_player_segments(tmp_in, jersey_number)

        # 2) 타임스탬프 파싱 + 선수 구간 필터
        events = _parse_timestamps(timestamps)
        events = [t for t in events if _in_any_segment(t, segments)]
        if not events:
            msg = "선수 구간과 겹치는 이벤트가 없습니다. (timestamps/등번호 확인)"
            logger.warning(f"[player/highlight] {msg}")
            raise RuntimeError(msg)

        # 사용자가 너무 많이 넣어도 상한 적용
        if max_clips and max_clips > 0:
            events = events[:max_clips]

        logger.info(
            f"[player/highlight] jersey={jersey_number} events={events} pre={pre} post={post} max={max_clips}"
        )

        # 3) 클립 생성
        results = generate_highlight_clips(
            tmp_in, jersey_number, events, pre=pre, post=post, max_clips=max_clips
        )
        clip_paths = [p for _, p in results]

        # 4) ZIP 메모리 스트림
        buf = build_zip_in_memory(clip_paths)

    except Exception as e:
        logger.exception(f"[player/highlight] failed: {e}")
        try:
            tmp_in.unlink(missing_ok=True)
        except Exception:
            pass
        # 개별 에러를 프론트/백에서 보기 쉽게 400으로 반환
        return JSONResponse(status_code=400, content={"error": str(e)})

    # 임시파일 정리
    try:
        tmp_in.unlink(missing_ok=True)
    except Exception:
        pass
    for p in clip_paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="player_highlights.zip"'},
    )
