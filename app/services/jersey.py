# app/services/jersey.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import subprocess
import re
import tempfile
import shutil
import logging

import cv2
import numpy as np
import pytesseract

from app.core.config import settings
from app.services.ffmpeg import get_duration

logger = logging.getLogger(__name__)

# ---------------------------
# FFmpeg helpers
# ---------------------------
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """공통 서브프로세스 실행(타임아웃/로깅 포함)"""
    to = settings.FFMPEG_TIMEOUT_SEC
    logger.debug(f"[jersey.ffproc] exec (timeout={to}s): {' '.join(cmd[:6])} ...")
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=to,
        )
        logger.debug(f"[jersey.ffproc] returncode={p.returncode}")
        return p
    except subprocess.TimeoutExpired:
        logger.warning(f"[jersey.ffproc] timeout after {to}s")
        raise
    except Exception as e:
        logger.exception(f"[jersey.ffproc] failed: {e}")
        raise


def _sample_frames_to_dir(
    video_path: Path, fps: float
) -> tuple[Path, list[tuple[float, Path]]]:
    """
    ffmpeg로 일정 fps 간격 프레임을 추출하고 showinfo 로그에서 pts_time(초)을 파싱.
    반환: (임시폴더경로, [(timestamp_sec, frame_path), ...])
    """
    assert video_path.exists(), f"input not found: {video_path}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="jersey_"))
    out_pattern = tmp_dir / "f_%05d.jpg"
    # 960px로 스케일 제한(속도/인식 안정성), showinfo로 타임스탬프 얻기
    vf = f"fps={fps},scale='min(960,iw)':-2,showinfo"

    proc = _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-q:v",
            "3",
            str(out_pattern),
        ]
    )

    times: list[float] = []
    for line in (proc.stderr or "").splitlines():
        m = re.search(r"pts_time:([0-9]+\.[0-9]+)", line)
        if m:
            try:
                times.append(float(m.group(1)))
            except Exception:
                pass

    images = sorted(tmp_dir.glob("f_*.jpg"))
    pairs: list[tuple[float, Path]] = []
    for i, img in enumerate(images):
        if i < len(times):
            pairs.append((times[i], img))

    logger.info(f"[jersey.sample] frames={len(pairs)} (fps={fps})")
    return tmp_dir, pairs


# ---------------------------
# OCR helpers
# ---------------------------
def _ocr_digits(img: np.ndarray) -> tuple[str, float]:
    """
    숫자만 인식(whitelist). 간단 신뢰도 추정치(길이에 기반).
    pytesseract timeout 설정 적용.
    """
    cfg = (
        f"-l eng --oem {settings.JERSEY_TESSERACT_OEM} "
        f"--psm {settings.JERSEY_TESSERACT_PSM} "
        f"-c tessedit_char_whitelist=0123456789"
    )
    try:
        text = pytesseract.image_to_string(
            img, config=cfg, timeout=settings.OCR_TIMEOUT_SEC
        )
    except RuntimeError as e:
        # pytesseract는 timeout 시 RuntimeError("Timeout ...")를 던질 수 있음
        logger.warning(f"[jersey.ocr] timeout after {settings.OCR_TIMEOUT_SEC}s: {e}")
        return "", 0.0
    except Exception as e:
        logger.exception(f"[jersey.ocr] failed: {e}")
        return "", 0.0

    text = re.sub(r"\D+", "", text or "")  # 숫자만
    if not text:
        return "", 0.0
    conf = min(1.0, max(0.0, len(text) / 2.0))  # 2자리면 1.0 근사
    return text, conf


def _digit_roi_candidates(gray: np.ndarray) -> list[np.ndarray]:
    """
    숫자 후보 ROI 추출: 블러 → 적응형 이진화 → 모폴로지 → 컨투어 필터.
    상단 HUD(스코어보드) 오검을 줄이기 위해 화면 상단 일부는 제외.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois: list[np.ndarray] = []
    H, W = gray.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 400:  # 너무 작은 박스 제외
            continue
        ar = w / (h + 1e-6)
        if ar < 0.3 or ar > 6.0:  # 비정상 종횡비 제외
            continue
        if y < H * 0.1:  # 화면 상단 10%는 HUD일 확률↑ → 제외(필요시 0.15~0.2로 조절)
            continue
        x0 = max(0, x - 2)
        y0 = max(0, y - 2)
        x1 = min(W, x + w + 2)
        y1 = min(H, y + h + 2)
        roi = gray[y0:y1, x0:x1]
        rois.append(roi)
    return rois


# ---------------------------
# time utils
# ---------------------------
def _merge_times(times: list[float], max_gap: float) -> list[tuple[float, float]]:
    """검출 시점들을 인접 간격(max_gap) 기준으로 [start,end] 구간으로 묶기."""
    if not times:
        return []
    times = sorted(times)
    segs: list[tuple[float, float]] = []
    s = e = times[0]
    for t in times[1:]:
        if t - e <= max_gap:
            e = t
        else:
            segs.append((s, e))
            s = e = t
    segs.append((s, e))
    return segs


def _expand_and_filter(
    segs: list[tuple[float, float]],
    pad: float,
    min_dur: float,
    total: float,
) -> list[tuple[float, float]]:
    """구간 앞뒤 pad 확장 후, 너무 짧은 구간 제거 + [0,total] 범위 클램프."""
    out: list[tuple[float, float]] = []
    for s, e in segs:
        s2 = max(0.0, s - pad)
        e2 = min(total, e + pad)
        if e2 - s2 >= min_dur:
            out.append((s2, e2))
    return out


# ---------------------------
# main API
# ---------------------------
def detect_player_segments(
    video_path: Path, jersey_number: int
) -> List[Tuple[float, float]]:
    """
    간단한 등번호 감지 파이프라인:
      1) fps로 프레임 샘플링 (ffmpeg)
      2) 숫자 후보 ROI 추출(OpenCV) → Tesseract OCR(숫자만)
      3) jersey_number와 일치하는 프레임의 시점들을 병합해 구간 반환
    """
    assert video_path.exists(), f"input not found: {video_path}"

    fps = settings.JERSEY_SAMPLE_FPS
    min_seg = settings.JERSEY_MIN_SEG_DUR
    merge_gap = settings.JERSEY_MERGE_GAP
    num_conf = settings.JERSEY_NUM_CONF

    tmp_dir = None
    try:
        tmp_dir, frames = _sample_frames_to_dir(video_path, fps=fps)
        total = get_duration(video_path)
        target = str(jersey_number)

        hit_times: list[float] = []
        for t, img_path in frames:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 후보 ROI들에 대해 OCR
            rois = _digit_roi_candidates(gray)
            found = False
            for roi in rois:
                # 작은 패치일수록 OCR 유리 → 2배 확대
                roi_big = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                text, conf = _ocr_digits(roi_big)
                if conf >= num_conf and target in text:
                    found = True
                    break
            if found:
                hit_times.append(t)

        # 시점 → 구간 병합, 살짝 확장, 짧은 구간 제거
        segs = _merge_times(hit_times, max_gap=merge_gap)
        segs = _expand_and_filter(segs, pad=0.5, min_dur=min_seg, total=total)
        logger.info(f"[jersey.segs] jersey={jersey_number} hits={len(hit_times)} segments={len(segs)}")
        return segs

    finally:
        # 임시 프레임 폴더 정리
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
def ocr_jersey_image_bytes(image_bytes: bytes) -> tuple[str, float]:
    """
    업로드된 등번호 '이미지'에서 숫자만 추출.
    반환: (digits, conf) — conf는 간단 추정치(0.0~1.0)
    """
    if not image_bytes:
        return "", 0.0
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "", 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 작은 입력일수록 확대하면 OCR이 잘 되는 편
    h, w = gray.shape[:2]
    if min(h, w) < 120:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    digits, conf = _ocr_digits(gray)
    return digits, conf