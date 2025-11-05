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

# ──────────────────────────────────────────────────────────────
# ↓↓↓ 새로 추가되는 일괄 처리(batch)용 유틸/의존성 ↓↓↓
# 자동 코트 보정(Homography) + 2/3점/1점 판정
from app.services.bh_geometry import (
    CourtSpec, FIBA, NBA,
    warp_pixel_to_world, is_free_throw, classify_2pt3pt, compute_homography_auto,
)
# 공/득점 이벤트/등번호 OCR 훅 (필요 시 YOLO/PP-OCR로 교체 가능)
from app.services.bh_detect import (
    detect_ball_hsv, is_score_event, ocr_digits, shooter_roi_near_ball,
)
# FFmpeg 컷/오버레이/합치기
from app.services.bh_edit import (
    ffprobe_duration, cut_and_overlay, concat_videos,
)
import cv2
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# ✅ 새로 추가된: 여러 영상 일괄 처리 + 2/3점 라벨 + ±컷 + 합본
# (기존 라우트는 건드리지 않음)
@router.post("/highlight/batch")
async def highlight_batch(
    videos: List[UploadFile] = File(..., description="여러 개 영상 (프론트 멀티 업로드)"),
    jerseys: str = Form(..., description="쉼표 구분 등번호, 예: '24,24,10' (영상 순서와 매칭)"),
    format: str = Form("FIBA"),
    pre: float = Form(default=settings.DEFAULT_PRE),
    post: float = Form(default=settings.DEFAULT_POST),
    # 선택: 저장/발행까지 하고 싶으면 넘김 (기존 save_clip_to_repo + pub/sub 재사용)
    memberId: Optional[str] = Form(default=None),
    jobId: Optional[str] = Form(default=None),
):
    """
    여러 영상을 한 번에 처리:
    - 각 영상에서 대상 등번호의 득점만 감지
    - 2/3점/1점 라벨 후 ±pre/post로 컷 & 오버레이
    - 모든 클립 합본(세로 1080x1920) 생성
    """
    # 업로드 저장
    local_paths: List[Path] = []
    for f in videos:
        local_paths.append(_save_upload(f))

    # 등번호 파싱/검증
    jersey_list = [int(x.strip()) for x in jerseys.split(",") if x.strip()]
    if len(jersey_list) != len(local_paths):
        _cleanup_paths(local_paths)
        return JSONResponse(status_code=400, content={"error": "영상 개수와 등번호 개수가 다릅니다."})

    spec: CourtSpec = FIBA if format.upper() == "FIBA" else NBA
    out_dir = TMP_DIR
    all_clips: List[Path] = []

    # (옵션) 진행률 시작
    do_publish = bool(memberId and jobId)
    if do_publish:
        await publish_progress(memberId, jobId, 0.01, "batch: upload saved")

    try:
        # 영상별 처리
        for idx, (src_path, jersey_target) in enumerate(zip(local_paths, jersey_list)):
            clips = await _process_one_video_batch(
                src_path, jersey_target, spec, pre, post, out_dir
            )
            all_clips.extend(clips)
            if do_publish:
                await publish_progress(
                    memberId, jobId,
                    (idx + 1) / max(1, len(local_paths)),
                    f"processed {idx+1}/{len(local_paths)}"
                )

        # 합본
        merged = None
        if all_clips:
            merged = out_dir / "merged_shorts.mp4"
            concat_videos([str(p) for p in all_clips], str(merged))

        # (옵션) 저장 + 공개 URL 발행
        public_urls: List[str] = []
        if do_publish and all_clips:
            for i, tmp_clip in enumerate(all_clips):
                _, url = save_clip_to_repo(tmp_clip, memberId, jobId, index=i)
                public_urls.append(url)
                await publish_progress(
                    memberId, jobId,
                    0.8 + 0.2 * (i + 1) / len(all_clips),
                    f"saved {i+1}/{len(all_clips)}"
                )
            await publish_result(memberId, jobId, public_urls, final=True)

        # 응답(JSON 경로)
        resp = {
            "merged_video": str(merged) if merged else None,
            "clips": [str(p) for p in all_clips],
        }
        return JSONResponse(content=resp)

    except Exception as e:
        if do_publish:
            await publish_error(memberId, jobId, str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # 입력 원본 정리 (생성된 클립은 남겨둠: 프론트가 읽거나 save_clip_to_repo가 이동)
        _cleanup_paths(local_paths)


async def _process_one_video_batch(
    src_path: Path, jersey_target: int, spec: CourtSpec,
    pre_s: float, post_s: float, out_dir: Path
) -> List[Path]:
    """
    단일 영상에서:
    - 림/코트 라인 기반 자동 보정
    - 득점 이벤트(공이 림을 관통) 검출
    - 등번호 OCR 필터
    - 2/3점/1점 라벨 후 컷/오버레이
    """
    clips: List[Path] = []
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return clips

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = ffprobe_duration(str(src_path))

    ok, key = cap.read()
    if not ok:
        cap.release()
        return clips

    # 코트 자동 보정 & 림 픽셀
    Hmat, hoop_px = compute_homography_auto(key, spec)
    if Hmat is None or hoop_px is None:
        # 보정 실패해도 파이프라인은 계속(라벨은 UNK 가능)
        hoop_px = None

    scale = max(1, int(max(W, Hh) / 960))
    frame_idx = 0
    ball_hist: List[tuple] = []
    recent_scores: List[float] = []
    dedup_gap = 4.0
    base = src_path.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1

        small = frame if scale == 1 else cv2.resize(frame, (W // scale, Hh // scale))
        ball = detect_ball_hsv(small)
        if ball:
            x, y, r = ball
            ball_hist.append((x * scale, y * scale, max(1, int(r * scale))))
        elif ball_hist:
            ball_hist.append(ball_hist[-1])  # 간단한 관성

        # 득점 이벤트
        if hoop_px and is_score_event(ball_hist[-12:], hoop_px):
            # 중복 방지
            if any(abs(t - s) < dedup_gap for s in recent_scores):
                continue
            recent_scores.append(t)

            # 등번호 필터
            jersey_ok = False
            if ball_hist:
                bx, by, br = ball_hist[-1]
                roi = shooter_roi_near_ball(frame, (bx, by, br))
                if roi is not None:
                    digits = ocr_digits(roi)  # 필요 시 PP-OCR/CRNN으로 교체
                    jersey_ok = (str(jersey_target) in digits)
            if not jersey_ok:
                continue

            # 2/3점/1점 라벨
            label = "UNK"
            if Hmat is not None:
                bx, by, br = ball_hist[-1]
                foot_px = (bx, by + int(1.2 * br))  # 슈터 발 근사
                world = warp_pixel_to_world(Hmat, foot_px)
                if world is not None:
                    if is_free_throw(world, spec):
                        label = "1PT"
                    else:
                        label = classify_2pt3pt(world, spec)

            # 컷 & 오버레이
            s = max(0.0, t - pre_s)
            e = min(duration, t + post_s)
            out_path = out_dir / f"{base}_{int(t * 1000)}.mp4"
            text = f"#{jersey_target} · {label} · t={int(t//60):02d}:{int(t%60):02d}"
            cut_and_overlay(str(src_path), s, e, text, str(out_path))
            clips.append(out_path)

    cap.release()
    return clips
