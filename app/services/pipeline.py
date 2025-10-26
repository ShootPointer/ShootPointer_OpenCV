# app/services/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import io
import os
import uuid
import zipfile
import logging

from app.core.config import settings
from app.services.ffmpeg import (
    get_duration,
    cut_clip_fast,
    detect_silences,
    non_silent_centers,
)

logger = logging.getLogger(__name__)

TMP_CLIPS = Path("/tmp/clips")
TMP_CLIPS.mkdir(parents=True, exist_ok=True)


def _safe_window(total: float, t: float, pre: float, post: float) -> Tuple[float, float]:
    """
    영상 길이를 벗어나지 않도록 [start, end] 창을 보정한다.
    최소 길이 0.5초를 보장.
    """
    s = max(0.0, t - pre)
    e = min(total, t + post)
    if e <= s:
        e = min(total, s + 0.5)
    return s, e


def _norm_timestamps(ts: List[float], maxn: int) -> List[float]:
    """
    타임스탬프를 float으로 정규화하고, 음수 제거 → 정렬 → 중복 제거 → 최대개수 제한.
    """
    out: List[float] = []
    for x in ts:
        try:
            v = float(x)
            if v >= 0:
                out.append(v)
        except Exception:
            continue
    out = sorted(set(out))
    if maxn and maxn > 0:
        out = out[:maxn]
    return out


def generate_highlight_clips(
    src: Path,
    jersey_number: int | None,
    timestamps: List[float],
    pre: float | None = None,
    post: float | None = None,
    max_clips: int | None = None,
) -> List[Tuple[float, Path]]:
    """
    각 타임스탬프마다 [pre, post] 만큼 잘라 개별 mp4 생성.
    반환: [(t, clip_path), ...]
    - src: 입력 영상
    - jersey_number: 파일명 접두사 등에 활용(없어도 동작)
    - timestamps: 초 단위
    - pre/post: None이면 settings.DEFAULT_PRE/POST 사용
    - max_clips: 최대 생성 개수 제한(None이면 settings.MAX_CLIPS)
    """
    assert src.exists(), f"not found: {src}"
    total = get_duration(src)
    pre = settings.DEFAULT_PRE if pre is None else float(pre)
    post = settings.DEFAULT_POST if post is None else float(post)
    maxn = settings.MAX_CLIPS if max_clips is None else int(max_clips)

    ts = _norm_timestamps([float(x) for x in timestamps], maxn=maxn)
    if not ts:
        raise RuntimeError("no valid timestamps")

    out: List[Tuple[float, Path]] = []

    # 파일명 접두사(등번호가 있으면 '#12_', 없으면 'clip_')
    prefix = f"#{jersey_number}_" if jersey_number is not None and jersey_number >= 0 else "clip_"

    for t in ts:
        s, e = _safe_window(total, t, pre, post)
        # 사람이 보기 좋은 임시파일명: 접두사 + 중심초(정수) + 랜덤 id (임시 디렉터리)
        name = f"{prefix}{int(round(t))}_{uuid.uuid4().hex[:8]}.mp4"
        dst = TMP_CLIPS / name

        # 컷팅(빠른 copy 모드) — 실패 시 예외 발생
        cut_clip_fast(src, dst, s, e)
        out.append((t, dst))

    logger.info(f"[pipeline] generated {len(out)} clips (pre={pre}, post={post}, src={src.name})")
    return out


def build_zip_in_memory(paths: List[Path]) -> io.BytesIO:
    """
    파일들을 메모리 ZIP으로 만들어 StreamingResponse에 바로 쓰기.
    - 큰 ZIP은 메모리 점유가 커질 수 있으니, 대용량은 streaming.build_zip_spooled 사용 권장
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            zf.write(p, arcname=p.name)
    buf.seek(0)
    logger.info(f"[pipeline] build_zip_in_memory -> {len(paths)} files")
    return buf


def auto_candidates(src: Path, topk: int | None = None) -> List[float]:
    """
    오디오 기반 자동 후보 타임스탬프.
    - 무음 구간을 찾고, 비무음 구간의 중앙값을 길이 기준 상위 top-k 반환
    """
    assert src.exists(), f"not found: {src}"
    total = get_duration(src)
    sil = detect_silences(
        src,
        silence_threshold_db=settings.SILENCE_THRESHOLD_DB,
        silence_min_dur=settings.SILENCE_MIN_DUR,
    )
    k = settings.AUTO_TOPK if topk is None else int(topk)
    centers = non_silent_centers(total, sil, topk=k)
    logger.info(f"[pipeline.auto] candidates={centers}")
    return centers


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_clip_to_repo(src_path: Path, member_id: str, job_id: str, index: int) -> tuple[Path, str]:
    """
    src_path(임시 클립)를 서비스 저장소(SAVE_ROOT)로 이동/복사하고, 공개 URL을 만들어 반환.
    - 최종 파일명은 index 대신 UUID로 부여(요구사항 반영).
    return: (dst_path, public_url)
    """
    root = Path(settings.SAVE_ROOT)
    dst_dir = root / member_id / job_id
    ensure_dir(dst_dir)

    # 🔁 변경 포인트: 최종 파일명은 uuid 기반으로 생성
    #   - 하이픈 없는 32자 hex 사용 (경로/호환성 안전)
    uuid_name = f"{uuid.uuid4().hex}{src_path.suffix}"
    dst_path = dst_dir / uuid_name

    # 이동(같은 볼륨이면 rename으로 빠르게), 다르면 copy 후 원본 삭제
    try:
        src_path.rename(dst_path)
    except Exception:
        dst_path.write_bytes(src_path.read_bytes())
        try:
            src_path.unlink(missing_ok=True)
        except Exception:
            pass

    public_url = f"{settings.STATIC_BASE_URL.rstrip('/')}/{member_id}/{job_id}/{dst_path.name}"
    return dst_path, public_url
