# app/services/bh_edit.py
from __future__ import annotations
from typing import List, Optional
import os
import subprocess
import tempfile
import pathlib

# ─────────────────────────────────────────────────────────────
# ffprobe: duration
# ─────────────────────────────────────────────────────────────
def ffprobe_duration(path: str) -> float:
    """
    Return video duration in seconds. 0.0 if probe fails.
    """
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        out = (proc.stdout or "").strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# drawtext-safe helpers
# ─────────────────────────────────────────────────────────────
def _escape_drawtext_text(s: str) -> str:
    """
    FFmpeg drawtext용 안전 이스케이프:
    :, ',', '\', ''' 는 각각 \:, \,, \\, \'
    """
    return (
        s.replace("\\", "\\\\")
         .replace(":", r"\:")
         .replace(",", r"\,")
         .replace("'", r"\'")
    )

def _detect_fontfile() -> Optional[str]:
    """
    컨테이너 내 흔한 폰트 경로 자동 탐색.
    없으면 None 반환(= drawtext가 시스템 기본 폰트 사용)
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    env_font = os.getenv("DRAW_FONTFILE", "").strip()
    if env_font and pathlib.Path(env_font).exists():
        return env_font
    for p in candidates:
        if pathlib.Path(p).exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────
# cut + overlay
# ─────────────────────────────────────────────────────────────
def cut_and_overlay(src: str, start: float, end: float, text: str, out_path: str) -> None:
    """
    [start, end] 구간을 1080x1920 캔버스에 맞춰 패딩하고 drawtext 오버레이 후 저장.
    - drawtext 특수문자 이스케이프 적용
    - 폰트 자동 탐색(DRAW_FONTFILE 환경변수로 명시 가능)
    - 오류시 stderr 마지막 줄로 이유를 띄워줌
    """
    start = max(0.0, float(start))
    dur   = max(0.1, float(end) - start)

    safe_text = _escape_drawtext_text(text)
    fontfile  = _detect_fontfile()

    draw = f"drawtext=text='{safe_text}':x=40:y=50:fontsize=48:fontcolor=white:borderw=2"
    if fontfile:
        draw = f"drawtext=fontfile='{fontfile}':text='{safe_text}':x=40:y=50:fontsize=48:fontcolor=white:borderw=2"

    vf = ",".join([
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease",
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
        draw,
    ])

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{dur:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # drawtext 파싱 문제 등 디버깅에 도움되도록 마지막 에러 라인 반환
        last = (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "unknown error"
        raise RuntimeError(f"ffmpeg failed: {last}")


# ─────────────────────────────────────────────────────────────
# concat
# ─────────────────────────────────────────────────────────────
def concat_videos(paths: List[str], dst: str) -> None:
    """
    동일 인코딩/해상도로 만들어진 mp4들을 빠르게 병합.
    - 1개면 그대로 복사
    - concat copy가 실패하면 libx264 재인코드 폴백
    """
    paths = [str(p) for p in paths if p]
    if not paths:
        raise ValueError("concat_videos: empty paths")

    if len(paths) == 1:
        proc = subprocess.run(["ffmpeg", "-y", "-i", paths[0], "-c", "copy", dst],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            # 드물게 컨테이너/코덱 태그 차이로 copy 실패 → 재인코드
            subprocess.run(
                ["ffmpeg", "-y", "-i", paths[0], "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst],
                check=True
            )
        return

    # concat list 파일 생성
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for p in paths:
            # ffmpeg concat demuxer는 절대경로 권장
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_path = f.name

    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", dst],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode != 0:
            # 스트림 파라미터가 조금이라도 다르면 copy 실패할 수 있음 → 재인코드 폴백
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst],
                check=True
            )
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass
