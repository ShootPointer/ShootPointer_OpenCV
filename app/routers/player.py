# app/routers/player.py
from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple

import httpx
from fastapi import APIRouter, UploadFile, File, Form, Body, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.core.config import settings
from app.services.jersey import (
    detect_player_segments,
    ocr_jersey_image_bytes,  # ← 등번호 사진 OCR
)
from app.services.pipeline import generate_highlight_clips
from app.services.streaming import build_zip_spooled, stream_zip_and_cleanup
from app.services.downloader import download_to_temp  # URL 다운로드(비영구, 임시파일)

# (선택) 콜백 업로드 공용 헬퍼
from app.services.callback import post_zip_to_callback

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
    try:
        size = tmp.stat().st_size
    except Exception:
        size = -1
    logger.info(f"[player] saved upload -> {tmp.name} ({size} bytes)")
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
            # 잘못된 입력은 무시
            pass
    return sorted(set(out))


def _check_backend_token(token: str | None):
    """백엔드-서버 간 공유 시크릿 검증(옵션). 설정되어 있으면 필수."""
    if settings.BACKEND_SECRET:
        if not token or token != settings.BACKEND_SECRET:
            raise HTTPException(status_code=403, detail="forbidden")

# ─────────────────────────────────────────────────────────────
# API: 등번호 사진 → 숫자 추출(OCR)
# ─────────────────────────────────────────────────────────────
@router.post("/parse-jersey")
async def parse_jersey_number_from_image(
    image: UploadFile = File(..., description="등번호가 보이는 이미지")
):
    """등번호 '이미지'에서 숫자(OCR)만 추출."""
    try:
        img_bytes = await image.read()
        digits, conf = ocr_jersey_image_bytes(img_bytes)
        if not digits:
            return JSONResponse(
                status_code=400,
                content={"error": "숫자를 읽지 못했습니다. 더 선명한 이미지로 시도하세요."},
            )
        return {"number": digits, "confidence": conf}
    except Exception as e:
        logger.exception(f"[parse-jersey] failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

# ─────────────────────────────────────────────────────────────
# API: 세그먼트(플레이 구간) 검출
#  - jersey_image가 있으면 OCR로 번호 추출 후 사용
# ─────────────────────────────────────────────────────────────
@router.post("/segments")
async def player_segments(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    jersey_number: int | None = Form(None, description="찾을 등번호 (예: 23). 이미지가 있으면 생략 가능"),
    jersey_image: UploadFile | None = File(None, description="등번호가 보이는 이미지(선택)"),
):
    """
    업로드된 영상에서 등번호 기반 플레이 구간을 검출해 [start,end] 초 리스트 반환.
    jersey_image가 제공되면 OCR로 숫자를 추출해 jersey_number로 사용.
    """
    tmp_in = _save_upload(video)
    try:
        # 이미지 우선(OCR)
        if jersey_image is not None:
            img_bytes = await jersey_image.read()
            digits, conf = ocr_jersey_image_bytes(img_bytes)
            if not digits:
                raise RuntimeError("등번호 이미지를 읽지 못했습니다. 더 선명하게 촬영해주세요.")
            jersey_number = int(digits)
            logger.info(f"[segments] jersey_image OCR -> {jersey_number} (conf={conf:.2f})")

        if jersey_number is None:
            raise RuntimeError("jersey_number 또는 jersey_image 중 하나는 반드시 제공해야 합니다.")

        segs = detect_player_segments(tmp_in, jersey_number)
        logger.info(f"[player/segments] jersey={jersey_number} -> segments={len(segs)}")
        return {"segments": segs, "jersey_number": jersey_number}
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
#  - 업로드(멀티파트) → 처리 → ZIP을 "스트리밍"으로 바로 응답
#  - 디스크에 영구 저장 없음(임시파일은 응답 종료 후 자동 삭제)
#  - jersey_image가 있으면 OCR로 번호 추출 후 사용
# ─────────────────────────────────────────────────────────────
@router.post("/highlight")
async def player_highlight_zip(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    timestamps: str = Form(..., description="쉼표구분 초 리스트, 예: 30,75,120"),
    jersey_number: int | None = Form(None, description="찾을 등번호 (예: 23). 이미지가 있으면 생략 가능"),
    jersey_image: UploadFile | None = File(None, description="등번호가 보이는 이미지(선택)"),
    pre: float = Form(default=settings.DEFAULT_PRE),
    post: float = Form(default=settings.DEFAULT_POST),
    max_clips: int = Form(default=settings.MAX_CLIPS),
):
    """
    1) jersey_image가 있으면 OCR로 번호 추출 → jersey_number로 사용
    2) 등번호 기반 구간 검출 → timestamps 필터 → 컷팅 → ZIP 스트리밍
    """
    tmp_in = _save_upload(video)
    clip_paths: List[Path] = []
    tmp_to_cleanup: List[Path] = [tmp_in]

    try:
        # 이미지 우선(OCR)
        if jersey_image is not None:
            img_bytes = await jersey_image.read()
            digits, conf = ocr_jersey_image_bytes(img_bytes)
            if not digits:
                raise RuntimeError("등번호 이미지를 읽지 못했습니다. 더 선명하게 촬영해주세요.")
            jersey_number = int(digits)
            logger.info(f"[highlight] jersey_image OCR -> {jersey_number} (conf={conf:.2f})")

        if jersey_number is None:
            raise RuntimeError("jersey_number 또는 jersey_image 중 하나는 반드시 제공해야 합니다.")

        segments = detect_player_segments(tmp_in, jersey_number)

        events = _parse_timestamps(timestamps)
        events = [t for t in events if _in_any_segment(t, segments)] if segments else events
        if not events:
            raise RuntimeError("선수 구간과 겹치는 이벤트가 없습니다. (timestamps/등번호/이미지 확인)")

        if max_clips and max_clips > 0:
            events = events[:max_clips]

        logger.info(
            f"[player/highlight] jersey={jersey_number} events={events} pre={pre} post={post} max={max_clips}"
        )

        # 클립 생성
        results = generate_highlight_clips(
            tmp_in, jersey_number, events, pre=pre, post=post, max_clips=max_clips
        )
        clip_paths = [Path(p) for _, p in results]

        # ZIP → 스풀 파일 → 스트리밍 응답 (+ 백그라운드에서 임시파일 정리)
        spooled = build_zip_spooled(clip_paths, arc_prefix=f"#{jersey_number}_")
        return stream_zip_and_cleanup(spooled, clip_paths, tmp_to_cleanup)

    except Exception as e:
        logger.exception(f"[player/highlight] failed: {e}")
        # 실패 시 임시파일 수동 정리
        for p in clip_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            Path(tmp_in).unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse(status_code=400, content={"error": str(e)})

# ─────────────────────────────────────────────────────────────
# (선택) API: URL 입력 버전 (S3 presigned URL 등) — 서버 간 대용량에 적합
#  - 원본을 영구 저장하지 않고, 임시파일만 사용 후 스트리밍 응답
# ─────────────────────────────────────────────────────────────
class HighlightByUrlReq(BaseModel):
    video_url: str
    jersey_number: int
    timestamps: List[float]
    pre: float | None = None
    post: float | None = None
    max_clips: int | None = None

@router.post("/highlight-by-url")
async def highlight_by_url(payload: HighlightByUrlReq = Body(...)):
    clip_paths: List[Path] = []
    tmp_to_cleanup: List[Path] = []

    try:
        # URL 원본 → 임시파일 (영구 저장 없음)
        suffix = ".mp4"
        tmp_in = await download_to_temp(payload.video_url, suffix=suffix)
        tmp_to_cleanup.append(tmp_in)

        pre = payload.pre if payload.pre is not None else settings.DEFAULT_PRE
        post = payload.post if payload.post is not None else settings.DEFAULT_POST
        max_clips = payload.max_clips if payload.max_clips is not None else settings.MAX_CLIPS

        segments = detect_player_segments(tmp_in, payload.jersey_number)

        events = sorted(set([t for t in (payload.timestamps or []) if t >= 0]))
        events = [t for t in events if _in_any_segment(t, segments)] if segments else events
        if not events:
            raise RuntimeError("선수 구간과 겹치는 타임스탬프가 없습니다.")

        if max_clips and max_clips > 0:
            events = events[:max_clips]

        results = generate_highlight_clips(
            tmp_in, payload.jersey_number, events, pre=pre, post=post, max_clips=max_clips
        )
        clip_paths = [Path(p) for _, p in results]

        spooled = build_zip_spooled(clip_paths, arc_prefix=f"#{payload.jersey_number}_")
        return stream_zip_and_cleanup(spooled, clip_paths, tmp_to_cleanup)

    except Exception as e:
        logger.exception(f"[player/highlight-by-url] failed: {e}")
        for p in clip_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        for t in tmp_to_cleanup:
            try:
                if Path(t).is_dir():
                    shutil.rmtree(t, ignore_errors=True)
                else:
                    Path(t).unlink(missing_ok=True)
            except Exception:
                pass
        return JSONResponse(status_code=400, content={"error": str(e)})

# ─────────────────────────────────────────────────────────────
# 옵션 B: ZIP을 백엔드 콜백 URL로 멀티파트 POST 전송(서버에 저장 없이)
#  - 백엔드 수신 엔드포인트는 multipart/form-data 의 file 필드명으로 ZIP 수신
# ─────────────────────────────────────────────────────────────
class HighlightCallbackReq(BaseModel):
    callback_url: str               # 백엔드 수신 URL
    jersey_number: int
    timestamps: List[float]
    pre: float | None = None
    post: float | None = None
    max_clips: int | None = None

@router.post("/highlight-callback")
async def player_highlight_callback(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    payload: str = Form(..., description='JSON 문자열: {"callback_url": "...", "jersey_number": 5, "timestamps": [..], "pre":..,"post":..,"max_clips":..}')
):
    """
    영상과 파라미터(JSON)를 받아 클립 ZIP을 생성하고,
    ZIP 파일을 백엔드가 제공한 callback_url 로 멀티파트(필드명: file) POST 전송.
    로컬/컨테이너에는 ZIP 파일을 저장하지 않음.
    """
    # 0) JSON 파싱 → pydantic 검증
    try:
        data = HighlightCallbackReq(**json.loads(payload))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"invalid payload: {e}"})

    tmp_in = _save_upload(video)
    clip_paths: List[Path] = []
    tmp_to_cleanup: List[Path] = [tmp_in]

    try:
        # 1) 선수 구간 검출
        segments = detect_player_segments(tmp_in, data.jersey_number)

        # 2) 타임스탬프 정리 + 선수 구간 필터
        events = sorted(set([float(t) for t in (data.timestamps or []) if t is not None and t >= 0.0]))
        events = [t for t in events if _in_any_segment(t, segments)] if segments else events
        if not events:
            raise RuntimeError("선수 구간과 겹치는 타임스탬프가 없습니다.")

        pre = data.pre if data.pre is not None else settings.DEFAULT_PRE
        post = data.post if data.post is not None else settings.DEFAULT_POST
        maxc = data.max_clips if data.max_clips is not None else settings.MAX_CLIPS

        logger.info(f"[player/highlight-callback] jersey={data.jersey_number} events={events} pre={pre} post={post} max={maxc}")

        # 3) 클립 생성
        results = generate_highlight_clips(
            tmp_in, data.jersey_number, events, pre=pre, post=post, max_clips=maxc
        )
        clip_paths = [Path(p) for _, p in results]

        # 4) ZIP 스풀 파일 생성
        spooled = build_zip_spooled(clip_paths, arc_prefix=f"#{data.jersey_number}_")

        # 5) 백엔드 콜백으로 멀티파트 POST
        timeout = httpx.Timeout(60.0, connect=30.0, read=60.0, write=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {"file": ("player_highlights.zip", spooled, "application/zip")}
            resp = await client.post(data.callback_url, files=files)
            resp.raise_for_status()

        # 6) 임시 정리
        for p in clip_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for t in tmp_to_cleanup:
            try:
                Path(t).unlink(missing_ok=True)
            except Exception:
                pass

        return {"status": "delivered", "callback_url": data.callback_url, "clips": len(clip_paths)}

    except Exception as e:
        logger.exception(f"[player/highlight-callback] failed: {e}")
        # 실패 시 정리
        for p in clip_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for t in tmp_to_cleanup:
            try:
                Path(t).unlink(missing_ok=True)
            except Exception:
                pass
        return JSONResponse(status_code=400, content={"error": str(e)})

# ─────────────────────────────────────────────────────────────
# 백엔드 연동: 업로드 or URL로 받으면 → ZIP 생성 → 콜백 POST
#  - ingest-upload  : multipart 업로드 방식
#  - ingest-url     : presigned URL 등 링크 방식
#  - ingest-async   : 202 즉시 응답, 백그라운드에서 콜백 POST (간단 버전)
# ─────────────────────────────────────────────────────────────
@router.post("/ingest-upload")
async def ingest_upload_for_backend(
    video: UploadFile = File(..., description="백엔드에서 전달한 원본 영상"),
    jersey_number: int = Form(...),
    timestamps: str = Form(..., description="예: 30,75,120"),
    callback_url: str = Form(..., description="백엔드 콜백 URL"),
    pre: float = Form(default=settings.DEFAULT_PRE),
    post: float = Form(default=settings.DEFAULT_POST),
    max_clips: int = Form(default=settings.MAX_CLIPS),
    token: str | None = Form(default=None, description="백엔드-서버 공유 시크릿"),
):
    _check_backend_token(token)
    tmp_in = _save_upload(video)
    tmp_to_cleanup: List[Path] = [tmp_in]
    clips: List[Path] = []
    try:
        # 등번호 구간
        segs = detect_player_segments(tmp_in, jersey_number)
        # 타임스탬프 정리 + 필터
        events = _parse_timestamps(timestamps)
        events = [t for t in events if _in_any_segment(t, segs)] if segs else events
        if not events:
            raise RuntimeError("선수 구간과 겹치는 타임스탬프가 없습니다.")
        if max_clips and max_clips > 0:
            events = events[:max_clips]

        # 클립 생성
        res = generate_highlight_clips(tmp_in, jersey_number, events, pre=pre, post=post, max_clips=max_clips)
        clips = [Path(p) for _, p in res]

        # ZIP 스풀
        spooled = build_zip_spooled(clips, arc_prefix=f"#{jersey_number}_")

        # 콜백 POST
        status = await post_zip_to_callback(
            callback_url,
            spooled,
            meta={"jersey_number": jersey_number, "events": events},
            bearer_or_token=None,
        )
        logger.info(f"[ingest-upload] delivered -> {callback_url} ({status})")
        return {"status": "delivered", "clips": len(clips)}
    except Exception as e:
        logger.exception(f"[ingest-upload] failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        for p in clips:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for t in tmp_to_cleanup:
            try:
                Path(t).unlink(missing_ok=True)
            except Exception:
                pass


class IngestUrlReq(BaseModel):
    video_url: str
    jersey_number: int
    timestamps: List[float]
    callback_url: str
    pre: float | None = None
    post: float | None = None
    max_clips: int | None = None
    token: str | None = None

@router.post("/ingest-url")
async def ingest_url_for_backend(body: IngestUrlReq):
    _check_backend_token(body.token)

    tmp_to_cleanup: List[Path] = []
    clips: List[Path] = []
    try:
        # URL → 임시파일
        tmp_in = await download_to_temp(body.video_url, suffix=".mp4")
        tmp_to_cleanup.append(tmp_in)

        pre = body.pre if body.pre is not None else settings.DEFAULT_PRE
        post = body.post if body.post is not None else settings.DEFAULT_POST
        maxc = body.max_clips if body.max_clips is not None else settings.MAX_CLIPS

        # 등번호 구간
        segs = detect_player_segments(tmp_in, body.jersey_number)
        # 타임스탬프 정리 + 필터
        events = sorted(set([float(t) for t in (body.timestamps or []) if t >= 0.0]))
        events = [t for t in events if _in_any_segment(t, segs)] if segs else events
        if not events:
            raise RuntimeError("선수 구간과 겹치는 타임스탬프가 없습니다.")
        if maxc and maxc > 0:
            events = events[:maxc]

        # 클립 생성
        res = generate_highlight_clips(tmp_in, body.jersey_number, events, pre=pre, post=post, max_clips=maxc)
        clips = [Path(p) for _, p in res]

        # ZIP 스풀
        spooled = build_zip_spooled(clips, arc_prefix=f"#{body.jersey_number}_")

        # 콜백 POST
        status = await post_zip_to_callback(
            body.callback_url,
            spooled,
            meta={"jersey_number": body.jersey_number, "events": events},
            bearer_or_token=None,
        )
        logger.info(f"[ingest-url] delivered -> {body.callback_url} ({status})")
        return {"status": "delivered", "clips": len(clips)}
    except Exception as e:
        logger.exception(f"[ingest-url] failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        for p in clips:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for t in tmp_to_cleanup:
            try:
                if Path(t).is_dir():
                    shutil.rmtree(t, ignore_errors=True)
                else:
                    Path(t).unlink(missing_ok=True)
            except Exception:
                pass

# 202 즉시 응답 + 백그라운드에서 콜백 POST (간단 버전)
JOBS: dict[str, dict] = {}

class IngestAsyncReq(BaseModel):
    video_url: str
    jersey_number: int
    timestamps: List[float]
    callback_url: str
    pre: float | None = None
    post: float | None = None
    max_clips: int | None = None
    token: str | None = None

@router.post("/ingest-async")
async def ingest_async_for_backend(body: IngestAsyncReq, background: BackgroundTasks):
    _check_backend_token(body.token)
    import uuid as _uuid
    job_id = _uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued"}

    def _task():
        tmp_to_cleanup: List[Path] = []
        clips: List[Path] = []
        try:
            JOBS[job_id]["status"] = "downloading"

            # 동기 다운로드(백그라운드 스레드용)
            tmp_path = TMP_DIR / f"{uuid.uuid4().hex}.mp4"
            timeout = httpx.Timeout(120, connect=30)
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream("GET", body.video_url) as resp:
                    resp.raise_for_status()
                    with tmp_path.open("wb") as f:
                        for chunk in resp.iter_bytes(1024 * 1024):
                            if chunk:
                                f.write(chunk)
            tmp_to_cleanup.append(tmp_path)

            pre = body.pre if body.pre is not None else settings.DEFAULT_PRE
            post = body.post if body.post is not None else settings.DEFAULT_POST
            maxc = body.max_clips if body.max_clips is not None else settings.MAX_CLIPS

            JOBS[job_id]["status"] = "detecting"
            segs = detect_player_segments(tmp_path, body.jersey_number)

            events = sorted(set([float(t) for t in (body.timestamps or []) if t >= 0.0]))
            events = [t for t in events if _in_any_segment(t, segs)] if segs else events
            if not events:
                raise RuntimeError("선수 구간과 겹치는 타임스탬프가 없습니다.")
            if maxc and maxc > 0:
                events = events[:maxc]

            JOBS[job_id]["status"] = "cutting"
            res = generate_highlight_clips(tmp_path, body.jersey_number, events, pre=pre, post=post, max_clips=maxc)
            clips[:] = [Path(p) for _, p in res]

            spooled = build_zip_spooled(clips, arc_prefix=f"#{body.jersey_number}_")

            JOBS[job_id]["status"] = "uploading"
            # 동기 업로드
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=30.0, read=60.0, write=60.0)) as client:
                files = {"file": ("player_highlights.zip", spooled, "application/zip")}
                r = client.post(body.callback_url, files=files)
                r.raise_for_status()

            JOBS[job_id]["status"] = "done"
        except Exception as e:
            logger.exception(f"[ingest-async] failed: {e}")
            JOBS[job_id]["status"] = f"error: {e}"
        finally:
            for p in clips:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            for t in tmp_to_cleanup:
                try:
                    if Path(t).is_dir():
                        shutil.rmtree(t, ignore_errors=True)
                    else:
                        Path(t).unlink(missing_ok=True)
                except Exception:
                    pass

    background.add_task(_task)
    return JSONResponse(status_code=202, content={"job_id": job_id})

@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    return JOBS.get(job_id, {"status": "unknown"})
