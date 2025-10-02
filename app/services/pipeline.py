# app/services/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import io
import uuid
import zipfile

from app.core.config import settings
from app.services.ffmpeg import get_duration, cut_clip_fast, detect_silences, non_silent_centers

TMP_CLIPS = Path("/tmp/clips")
TMP_CLIPS.mkdir(parents=True, exist_ok=True)

def _safe_window(total: float, t: float, pre: float, post: float) -> Tuple[float, float]:
    s = max(0.0, t - pre)
    e = min(total, t + post)
    if e <= s:  # 최소 길이 보장
        e = min(total, s + 0.5)
    return s, e

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
    """
    assert src.exists()
    total = get_duration(src)
    pre = settings.DEFAULT_PRE if pre is None else pre
    post = settings.DEFAULT_POST if post is None else post
    maxn = settings.MAX_CLIPS if max_clips is None else max_clips

    ts = sorted([float(x) for x in timestamps])[:maxn]
    out: List[Tuple[float, Path]] = []
    for t in ts:
        s, e = _safe_window(total, t, pre, post)
        dst = TMP_CLIPS / f"{uuid.uuid4().hex}_{int(t)}.mp4"
        cut_clip_fast(src, dst, s, e)
        out.append((t, dst))
    return out

def build_zip_in_memory(paths: List[Path]) -> io.BytesIO:
    """
    파일들을 메모리 ZIP으로 만들어 StreamingResponse에 바로 쓰기.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf

def auto_candidates(src: Path, topk: int | None = None) -> List[float]:
    """
    오디오 기반 자동 후보 타임스탬프.
    """
    total = get_duration(src)
    sil = detect_silences(
        src,
        silence_threshold_db=settings.SILENCE_THRESHOLD_DB,
        silence_min_dur=settings.SILENCE_MIN_DUR,
    )
    k = settings.AUTO_TOPK if topk is None else topk
    return non_silent_centers(total, sil, topk=k)
