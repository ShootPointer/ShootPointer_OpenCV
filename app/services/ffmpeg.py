# app/services/ffmpeg.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json
import re
import subprocess
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 내부 실행 유틸 (HW 가속 주입 + 타임아웃/로깅)
# ─────────────────────────────────────────────────────────────
def _inject_hwaccel(cmd: list[str]) -> list[str]:
    """
    settings.FF_HWACCEL / FF_HWACCEL_DEVICE 설정 시 -hwaccel 옵션 삽입.
    ["ffmpeg" | "ffprobe", ...] 형태의 커맨드만 대응.
    """
    hw = getattr(settings, "FF_HWACCEL", "") or ""
    dev = getattr(settings, "FF_HWACCEL_DEVICE", "") or ""
    if not hw or not cmd:
        return cmd

    tool = cmd[0]
    rest = cmd[1:]
    injected = [tool, "-hwaccel", hw]
    if dev:
        injected += ["-hwaccel_device", dev]
    injected += rest
    return injected


def _run(cmd: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """
    공통 서브프로세스 실행.
    - stdout/stderr 캡처
    - settings.FFMPEG_TIMEOUT_SEC 기본 적용
    """
    to = timeout if timeout is not None else settings.FFMPEG_TIMEOUT_SEC
    cmd = _inject_hwaccel(cmd)
    try:
        logger.debug(f"[ffproc] exec (timeout={to}s): {' '.join(cmd[:10])} ...")
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=to,
        )
        logger.debug(f"[ffproc] returncode={p.returncode}")
        return p
    except subprocess.TimeoutExpired:
        logger.warning(f"[ffproc] timeout after {to}s")
        raise
    except Exception as e:
        logger.exception(f"[ffproc] failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# 메타데이터: 총 길이(초)
# ─────────────────────────────────────────────────────────────
def get_duration(video_path: Path) -> float:
    """ffprobe로 영상 총 길이(초) 얻기"""
    assert video_path.exists(), f"not found: {video_path}"
    cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    p = _run(cmd)
    dur = 0.0
    if p.returncode != 0:
        logger.error(f"[duration] ffprobe error: {p.stderr[:400]}")
    try:
        data = json.loads(p.stdout or "{}")
        dur = float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        # stdout 누락 시 stderr에서 보조 파싱
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr or "")
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            dur = float(h * 3600 + mi * 60 + s)
    dur = max(0.0, dur)
    logger.info(f"[duration] {video_path.name}: {dur:.3f}s")
    return dur


# ─────────────────────────────────────────────────────────────
# 컷팅: 빠른 자르기(코덱 copy, 키프레임 단위) + 선택적 폴백
# ─────────────────────────────────────────────────────────────
def _cut_clip_reencode(src: Path, dst: Path, start: float, end: float) -> None:
    """
    느리지만 정확한 컷(재인코드). copy 실패 시 폴백용.
    - 프리셋/스레드 등은 환경에 맞게 .env로 조절 가능
    """
    duration = max(0.0, end - start)
    threads = getattr(settings, "FFMPEG_THREADS", None)
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", getattr(settings, "FFMPEG_REENC_VCODEC", "libx264"),
        "-preset", getattr(settings, "FFMPEG_REENC_PRESET", "veryfast"),
        "-crf", str(getattr(settings, "FFMPEG_REENC_CRF", 23)),
        "-c:a", getattr(settings, "FFMPEG_REENC_ACODEC", "aac"),
        "-movflags", "+faststart",
    ]
    if threads:
        cmd += ["-threads", str(threads)]
    cmd += [str(dst)]
    p = _run(cmd)
    if p.returncode != 0 or not dst.exists() or (dst.stat().st_size <= 0):
        logger.error(f"[cut:reenc] failed ({start:.3f}~{end:.3f}) -> {dst.name} : {p.stderr[:400]}")
        raise RuntimeError(f"ffmpeg reencode cut failed: {p.stderr[:200]}")
    logger.info(f"[cut:reenc] ok ({start:.3f}~{end:.3f}) -> {dst.name}")


def cut_clip_fast(src: Path, dst: Path, start: float, end: float) -> None:
    """
    빠른 컷팅(키프레임 단위) - 코덱 copy.
    start~end 구간을 잘라 dst로 저장.
    - 실패 시(키프레임 제약, 컨테이너 이슈 등) .env 의 FFMPEG_CUT_FALLBACK_REENCODE=1 인 경우 재인코드 폴백 수행
    """
    assert src.exists(), f"not found: {src}"
    duration = max(0.0, end - start)
    threads = getattr(settings, "FFMPEG_THREADS", None)

    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c", "copy",
    ]
    if threads:
        cmd += ["-threads", str(threads)]
    cmd += [str(dst)]

    p = _run(cmd)
    ok = (p.returncode == 0) and dst.exists() and (dst.stat().st_size > 0)

    if not ok and getattr(settings, "FFMPEG_CUT_FALLBACK_REENCODE", False):
        logger.warning(f"[cut] copy failed → reencode fallback: {p.stderr[:200]}")
        _cut_clip_reencode(src, dst, start, end)
        return

    if not ok:
        logger.error(f"[cut] failed ({start:.3f}~{end:.3f}) -> {dst.name} : {p.stderr[:400]}")
        raise RuntimeError(f"ffmpeg cut failed: {p.stderr[:200]}")

    logger.info(f"[cut] ok ({start:.3f}~{end:.3f}) -> {dst.name}")


# ─────────────────────────────────────────────────────────────
# 오디오: 무음 감지(silencedetect) → 비무음 구간 중앙 후보 추출
# ─────────────────────────────────────────────────────────────
def detect_silences(
    src: Path,
    silence_threshold_db: float = -28.0,
    silence_min_dur: float = 0.30,
) -> List[Tuple[float, float]]:
    """
    ffmpeg silencedetect로 (start,end) 무음 구간 리스트 반환.
    - pts 출력은 정수/소수 모두 대응
    """
    assert src.exists(), f"not found: {src}"
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(src),
        "-vn",
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={silence_min_dur}",
        "-f", "null", "-",
    ]
    p = _run(cmd)
    lines = (p.stderr or "").splitlines()

    silences: List[Tuple[float, float]] = []
    cur_start: Optional[float] = None
    # 정수/소수 모두 매칭: ([0-9]+(?:\.[0-9]+)?)
    for line in lines:
        m1 = re.search(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)", line)
        m2 = re.search(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*silence_duration", line)
        if m1:
            try:
                cur_start = float(m1.group(1))
            except Exception:
                cur_start = None
        if m2 and cur_start is not None:
            try:
                end = float(m2.group(1))
                silences.append((cur_start, end))
            except Exception:
                pass
            cur_start = None

    logger.info(f"[silence] found={len(silences)} (thr={silence_threshold_db}dB, d>={silence_min_dur}s)")
    return silences


def non_silent_centers(total: float, silences: List[Tuple[float, float]], topk: int = 5) -> List[float]:
    """
    무음 구간을 제외한 '소리 나는' 구간들의 중앙값을 후보로 추출하고,
    길이가 긴 구간 기준 상위 topk 반환.
    """
    if total <= 0:
        return []

    silent = sorted(silences)
    # 비무음 구간 만들기
    segments: List[Tuple[float, float]] = []
    cur = 0.0
    for s, e in silent:
        if s > cur:
            segments.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        segments.append((cur, total))

    # 길이 기준 상위 topk 선택
    longest = sorted(segments, key=lambda x: (x[1] - x[0]), reverse=True)[:topk]
    centers = [round((a + b) / 2, 3) for (a, b) in longest if (b - a) > 0.5]
    centers = sorted(set(centers))
    logger.info(f"[nonsilent] segments={len(segments)} -> centers(top{topk})={centers}")
    return centers


# ─────────────────────────────────────────────────────────────
# 호환용 래퍼: 예전 코드가 쓰던 이름 유지
# ─────────────────────────────────────────────────────────────
def detect_loud_midpoints(
    src: Path,
    topk: int = 5,
    silence_threshold_db: float = -28.0,
    silence_min_dur: float = 0.30,
) -> List[float]:
    """
    [Backward-compat] 과거 이름 유지용.
    내부적으로 detect_silences + non_silent_centers를 사용.
    """
    total = get_duration(src)
    sil = detect_silences(src, silence_threshold_db=silence_threshold_db, silence_min_dur=silence_min_dur)
    return non_silent_centers(total, sil, topk=topk)
