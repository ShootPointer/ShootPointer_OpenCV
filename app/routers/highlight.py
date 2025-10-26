# app/routers/highlight.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
import shutil
import uuid
import asyncio
from typing import List, Optional

from app.core.config import settings
from app.services.ffmpeg import detect_loud_midpoints
from app.services.pipeline import (
    generate_highlight_clips,
    build_zip_in_memory,
    save_clip_to_repo,  # ✅ UUID 파일명으로 SAVE_ROOT로 이동 + 공개 URL 생성
)

# Redis 진행률/결과 발행 유틸 (async)
from app.services.pubsub import (
    publish_progress,
    publish_result,
    publish_error,
)

router = APIRouter()

TMP_DIR = Path("/tmp/uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(video: UploadFile) -> Path:
    suffix = Path(video.filename).suffix or ".mp4"
    tmp = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return tmp


def _cleanup_paths(paths: List[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except:
            pass


@router.post("/clip")
async def clip_single(
    background: BackgroundTasks,
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    center_time: float = Form(..., description="잘라낼 중심 타임(초)"),
    pre: float = Form(default=settings.DEFAULT_PRE, description="앞쪽 초"),
    post: float = Form(default=settings.DEFAULT_POST, description="뒤쪽 초"),
):
    """
    단일 시점 기준 ± 구간을 잘라 mp4 파일로 반환.
    (저장/Redis 발행 없음 — 기존 동작 유지)
    """
    tmp_in = _save_upload(video)
    try:
        results = generate_highlight_clips(
            tmp_in, jersey_number=-1, timestamps=[center_time], pre=pre, post=post, max_clips=1
        )
        if not results:
            raise RuntimeError("clip generation failed")
        _, out_path = results[0]
    except Exception as e:
        try:
            tmp_in.unlink(missing_ok=True)
        except:
            pass
        return JSONResponse(status_code=400, content={"error": str(e)})

    def cleanup():
        try:
            tmp_in.unlink(missing_ok=True)
        except:
            pass
        _cleanup_paths([out_path])

    background.add_task(cleanup)
    return FileResponse(out_path, filename=out_path.name, media_type="video/mp4", background=background)


@router.post("/highlight")
async def highlight_zip(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    jersey_number: int = Form(..., description="찾을 등번호 (예: 23)"),
    timestamps: str = Form(..., description="쉼표구분 초 리스트, 예: 30,75,120"),
    pre: float = Form(default=settings.DEFAULT_PRE),
    post: float = Form(default=settings.DEFAULT_POST),
    max_clips: int = Form(default=settings.MAX_CLIPS),
    # ✅ 선택: 주면 저장 + Redis 발행 진행 (안 주면 기존처럼 ZIP만 반환)
    memberId: Optional[str] = Form(default=None),
    jobId: Optional[str] = Form(default=None),
):
    """
    수동/외부에서 받은 타임스탬프 목록으로 클립 생성 후 ZIP 반환(MVP).
    - memberId/jobId가 주어지면, 각 클립을 SAVE_ROOT/{memberId}/{jobId}/{uuid}.mp4 로 저장하고
      진행률(PROCESSING)과 최종 결과(COMPLETE)를 Redis에 발행 + KV 저장.
    - 미제공 시엔 기존처럼 ZIP만 반환.
    """
    tmp_in = _save_upload(video)

    try:
        # 입력 파싱
        events: List[float] = []
        for s in timestamps.split(","):
            s = s.strip()
            if s:
                events.append(float(s))

        # 진행률 시작(옵션)
        do_publish = bool(memberId and jobId)
        if do_publish:
            await publish_progress(memberId, jobId, 0.01, "downloading/upload saved")

        # 클립 생성 (임시 디렉터리에 생성)
        results = generate_highlight_clips(
            tmp_in, jersey_number, events, pre=pre, post=post, max_clips=max_clips
        )
        clip_paths = [p for _, p in results]

        # 저장 + 공개 URL 생성 (옵션)
        public_urls: List[str] = []
        if do_publish:
            total = len(clip_paths)
            if total == 0:
                raise RuntimeError("no clips generated")

            for i, tmp_clip in enumerate(clip_paths):
                # SAVE_ROOT/{memberId}/{jobId}/{uuid}.mp4 로 이동/복사
                _, url = save_clip_to_repo(tmp_clip, memberId, jobId, index=i)
                public_urls.append(url)

                # 대략적 진행률 (생성/저장 기준)
                progress = (i + 1) / total
                await publish_progress(memberId, jobId, progress, f"saved {i+1}/{total}")

            # 최종 COMPLETE (KV + PUB/SUB)
            await publish_result(memberId, jobId, public_urls, final=True)

        # ZIP 빌드 (반환은 기존대로)
        buf = build_zip_in_memory(clip_paths)

    except Exception as e:
        # 에러 시 Redis에도 알림(옵션)
        if memberId and jobId:
            await publish_error(memberId, jobId, str(e))
        try:
            tmp_in.unlink(missing_ok=True)
        except:
            pass
        return JSONResponse(status_code=400, content={"error": str(e)})

    # 임시 정리
    try:
        tmp_in.unlink(missing_ok=True)
    except:
        pass
    _cleanup_paths(clip_paths)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="highlights.zip"'},
    )


@router.post("/highlight/auto")
async def highlight_auto_zip(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    jersey_number: int = Form(..., description="찾을 등번호 (예: 23). (현재 버전은 필터 미적용, 자리만 유지)"),
    pre: float = Form(default=settings.DEFAULT_POST),
    post: float = Form(default=settings.DEFAULT_POST),
    max_clips: int = Form(default=settings.MAX_CLIPS),
    silence_threshold_db: float = Form(default=settings.SILENCE_THRESHOLD_DB),
    silence_min_dur: float = Form(default=settings.SILENCE_MIN_DUR),
    topk: int = Form(default=settings.AUTO_TOPK),
    # ✅ 선택: 저장 + Redis 발행
    memberId: Optional[str] = Form(default=None),
    jobId: Optional[str] = Form(default=None),
):
    """
    (간단 자동) 오디오 무음 감지 기반 후보 시점(top-k)을 뽑아 ZIP 반환.
    - memberId/jobId가 주어지면 저장 + Redis 발행까지 수행.
    """
    tmp_in = _save_upload(video)

    try:
        # 후보 타임스탬프 추출
        candidates = detect_loud_midpoints(
            tmp_in,
            silence_thresh_db=float(silence_threshold_db),
            silence_min_dur=float(silence_min_dur),
            topk=int(topk),
        )
        if not candidates:
            raise RuntimeError(
                "no candidate timestamps found (try lowering silence_threshold_db or topk)"
            )

        do_publish = bool(memberId and jobId)
        if do_publish:
            await publish_progress(memberId, jobId, 0.01, "downloading/upload saved")

        # 클립 생성
        results = generate_highlight_clips(
            tmp_in, jersey_number, candidates, pre=pre, post=post, max_clips=max_clips
        )
        clip_paths = [p for _, p in results]

        # 저장 + 공개 URL 생성 (옵션)
        public_urls: List[str] = []
        if do_publish:
            total = len(clip_paths)
            for i, tmp_clip in enumerate(clip_paths):
                _, url = save_clip_to_repo(tmp_clip, memberId, jobId, index=i)
                public_urls.append(url)
                await publish_progress(memberId, jobId, (i + 1) / total, f"saved {i+1}/{total}")
            await publish_result(memberId, jobId, public_urls, final=True)

        # ZIP 빌드
        buf = build_zip_in_memory(clip_paths)

    except Exception as e:
        if memberId and jobId:
            await publish_error(memberId, jobId, str(e))
        try:
            tmp_in.unlink(missing_ok=True)
        except:
            pass
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        tmp_in.unlink(missing_ok=True)
    except:
        pass
    _cleanup_paths(clip_paths)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="highlights_auto.zip"'},
    )
