# app/services/frames.py
from __future__ import annotations
from pathlib import Path
import subprocess
import uuid
import re
from typing import List, Dict, Tuple, Literal

import cv2

FRAMES_DIR = Path("/tmp/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# ── ffmpeg 기반 (기존) ───────────────────────────────────────────────────────
def sample_frames_with_timestamps_ffmpeg(
    video_path: Path,
    fps: float = 0.5,
    max_frames: int = 40,
    scale_width: int = 640,
) -> List[Dict]:
    """
    ffmpeg의 showinfo 로그에서 pts_time(초)를 파싱해 썸네일과 함께 반환.
    """
    assert video_path.exists(), f"input not found: {video_path}"
    sess = FRAMES_DIR / uuid.uuid4().hex
    sess.mkdir(parents=True, exist_ok=True)
    out_pattern = sess / "frame_%05d.jpg"

    vf = f"fps={fps},scale='min({scale_width},iw)':-2,showinfo"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "3",
        str(out_pattern),
    ]
    proc = _run(cmd)

    times: list[float] = []
    for line in proc.stderr.splitlines():
        m = re.search(r"pts_time:([0-9]+\.[0-9]+)", line)
        if m:
            try:
                times.append(float(m.group(1)))
            except Exception:
                pass

    images = sorted(sess.glob("frame_*.jpg"))
    n = min(len(images), len(times), max_frames)
    return [{"t": times[i], "image_path": str(images[i])} for i in range(n)]

# ── OpenCV 기반 (신규) ───────────────────────────────────────────────────────
def sample_frames_with_timestamps_opencv(
    video_path: Path,
    fps: float = 0.5,
    max_frames: int = 40,
    scale_width: int = 640,
) -> List[Dict]:
    """
    OpenCV VideoCapture로 일정 간격 프레임을 읽고,
    타임스탬프는 frame_index / native_fps 로 근사 계산.
    """
    assert video_path.exists(), f"input not found: {video_path}"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("failed to open video with OpenCV")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 목표 간격(프레임 단위)
    step = int(max(1, round((native_fps / max(fps, 1e-6))))) if native_fps > 0 else 1

    sess = FRAMES_DIR / uuid.uuid4().hex
    sess.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            H, W = frame.shape[:2]
            if W > scale_width:
                r = scale_width / float(W)
                frame = cv2.resize(frame, (int(W*r), int(H*r)), interpolation=cv2.INTER_AREA)

            # 타임스탬프 근사
            t = (frame_idx / native_fps) if native_fps > 0 else 0.0

            out_path = sess / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            results.append({"t": float(round(t, 3)), "image_path": str(out_path)})
            saved += 1
            if saved >= max_frames:
                break

        frame_idx += 1
        if total_frames and frame_idx >= total_frames:
            break

    cap.release()
    return results

# ── 통합 프런트 함수 ─────────────────────────────────────────────────────────
def sample_frames_with_timestamps(
    video_path: Path,
    fps: float = 0.5,
    max_frames: int = 40,
    scale_width: int = 640,
    engine: Literal["ffmpeg", "opencv"] = "ffmpeg",
) -> List[Dict]:
    """
    engine으로 'ffmpeg' 또는 'opencv' 선택.
    """
    if engine == "opencv":
        return sample_frames_with_timestamps_opencv(video_path, fps, max_frames, scale_width)
    return sample_frames_with_timestamps_ffmpeg(video_path, fps, max_frames, scale_width)
