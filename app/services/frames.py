# app/services/frames.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Literal, Optional
import subprocess
import uuid
import re
import logging

import cv2

from app.core.config import settings

logger = logging.getLogger(__name__)

FRAMES_DIR = Path("/tmp/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FFmpeg 실행 유틸 (하드웨어 가속 주입 + 타임아웃 + 로깅)
# ─────────────────────────────────────────────────────────────
def _inject_hwaccel(cmd: list[str]) -> list[str]:
    """
    settings.FFMPEG_HWACCEL 이 설정되어 있으면 적절한 -hwaccel 옵션 삽입.
    입력 cmd는 ["ffmpeg", ...] 형태라고 가정.
    """
    if not settings.FFMPEG_HWACCEL:
        return cmd
    out = cmd[:1] + ["-hwaccel", settings.FFMPEG_HWACCEL]
    if settings.FFMPEG_HWACCEL_DEVICE:
        out += ["-hwaccel_device", settings.FFMPEG_HWACCEL_DEVICE]
    out += cmd[1:]
    return out


def _run(cmd: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """
    공통 서브프로세스 실행 함수.
    - stdout/stderr 캡처
    - settings.FFMPEG_TIMEOUT_SEC 기본 적용
    - 예외는 상위에서 처리(라우터가 JSON으로 변환)
    """
    to = timeout if timeout is not None else settings.FFMPEG_TIMEOUT_SEC
    cmd = _inject_hwaccel(cmd)
    try:
        preview = " ".join(cmd[:10])
        logger.debug(f"[frames.ffproc] exec (timeout={to}s): {preview} ...")
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=to,
        )
        logger.debug(f"[frames.ffproc] returncode={p.returncode}")
        return p
    except subprocess.TimeoutExpired:
        logger.warning(f"[frames.ffproc] timeout after {to}s")
        raise
    except Exception as e:
        logger.exception(f"[frames.ffproc] failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# ffmpeg 기반 썸네일 + pts_time 파싱 (권장)
# ─────────────────────────────────────────────────────────────
def sample_frames_with_timestamps_ffmpeg(
    video_path: Path,
    fps: float = 0.5,
    max_frames: int = 40,
    scale_width: int = 640,
) -> List[Dict]:
    """
    ffmpeg의 showinfo 로그에서 pts_time(초)를 파싱해 썸네일과 함께 반환.
    - fps → scale → showinfo 순으로 필터 구성
    - -vframes 로 '필터 이후' 프레임 수 상한 보장 → 필요한 수만큼 뽑고 즉시 종료
    - pts_time 정수/소수 모두 파싱
    반환: [{ "t": float, "image_path": str }, ...]
    """
    assert video_path.exists(), f"input not found: {video_path}"
    assert fps > 0.0, "fps must be > 0"
    assert max_frames >= 1, "max_frames must be >= 1"
    assert scale_width >= 160, "scale_width must be >= 160"

    sess = FRAMES_DIR / uuid.uuid4().hex
    sess.mkdir(parents=True, exist_ok=True)
    out_pattern = sess / "frame_%05d.jpg"

    vf = f"fps={fps},scale='min({scale_width},iw)':-2,showinfo"
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "3",
        "-vframes", str(max_frames),  # 🔑 필터 이후 상한
        str(out_pattern),
    ]
    proc = _run(cmd)

    # pts_time: 정수/소수 모두 대응
    times: list[float] = []
    for line in (proc.stderr or "").splitlines():
        m = re.search(r"pts_time:([0-9]+(?:\.[0-9]+)?)", line)
        if m:
            try:
                times.append(float(m.group(1)))
            except Exception:
                pass

    images = sorted(sess.glob("frame_*.jpg"))
    # 길이 맞춰서 자르기
    n = min(len(images), len(times), max_frames)
    if n == 0:
        raise RuntimeError("no frames extracted (ffmpeg). try a different fps or check the input.")

    frames = [{"t": float(round(times[i], 3)), "image_path": str(images[i])} for i in range(n)]
    logger.info(f"[frames.ffmpeg] session={sess.name} frames={n} (fps={fps}, width<={scale_width})")
    return frames


# ─────────────────────────────────────────────────────────────
# OpenCV 기반 썸네일 + t≈frame_index/native_fps (대안)
# ─────────────────────────────────────────────────────────────
def sample_frames_with_timestamps_opencv(
    video_path: Path,
    fps: float = 0.5,
    max_frames: int = 40,
    scale_width: int = 640,
) -> List[Dict]:
    """
    OpenCV VideoCapture로 일정 간격 프레임을 읽고,
    타임스탬프는 frame_index / native_fps 로 근사 계산(정밀도는 ffmpeg 방법보다 낮을 수 있음).
    """
    assert video_path.exists(), f"input not found: {video_path}"
    assert fps > 0.0, "fps must be > 0"
    assert max_frames >= 1, "max_frames must be >= 1"
    assert scale_width >= 160, "scale_width must be >= 160"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("failed to open video with OpenCV")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 목표 간격(프레임 단위)
    step = 1
    if native_fps > 0:
        step = max(1, int(round(native_fps / max(fps, 1e-6))))

    sess = FRAMES_DIR / uuid.uuid4().hex
    sess.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    frame_idx = 0
    saved = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step == 0:
                H, W = frame.shape[:2]
                if W > scale_width:
                    r = scale_width / float(W)
                    new_w = int(W * r)
                    new_h = int(H * r)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # 근사 타임스탬프
                t = (frame_idx / native_fps) if native_fps > 0 else 0.0

                out_path = sess / f"frame_{frame_idx:05d}.jpg"
                ok_write = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_write:
                    results.append({"t": float(round(t, 3)), "image_path": str(out_path)})
                    saved += 1
                    if saved >= max_frames:
                        break

            frame_idx += 1
            if total_frames and frame_idx >= total_frames:
                break
    finally:
        cap.release()

    if not results:
        raise RuntimeError("no frames extracted (opencv).")

    logger.info(f"[frames.opencv] session={sess.name} frames={len(results)} (fps={fps}, width<={scale_width})")
    return results


# ─────────────────────────────────────────────────────────────
# 통합 프런트 함수
# ─────────────────────────────────────────────────────────────
def sample_frames_with_timestamps(
    video_path: Path,
    fps: float = 0.5,
    max_frames: int = 40,
    scale_width: int = 640,
    engine: Literal["ffmpeg", "opencv"] = "ffmpeg",
) -> List[Dict]:
    """
    engine으로 'ffmpeg' 또는 'opencv' 선택.
    - ffmpeg 엔진: showinfo 의 pts_time을 사용(정밀)
    - opencv 엔진: frame_index/native_fps 근사(속도/환경 호환성 ↑)
    """
    if engine == "opencv":
        return sample_frames_with_timestamps_opencv(video_path, fps, max_frames, scale_width)
    return sample_frames_with_timestamps_ffmpeg(video_path, fps, max_frames, scale_width)
