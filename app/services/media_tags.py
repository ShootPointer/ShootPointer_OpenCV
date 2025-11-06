# app/services/media_tags.py
from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import subprocess
import tempfile
import os

def set_mp4_metadata(src: str, dst: str, tags: Dict[str, str]) -> None:
    """
    ffmpeg로 MP4 메타데이터 키/값을 씌움(스트림 copy).
    예: tags={"clip_type":"2PT","video_id":"1","segment_index":"1","start_sec":"5","end_sec":"15"}
    """
    args: List[str] = ["ffmpeg", "-y", "-i", src, "-movflags", "use_metadata_tags"]
    for k, v in tags.items():
        args += ["-metadata", f"{k}={v}"]
    args += ["-c", "copy", dst]
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg metadata failed:\n{p.stdout}")

def annotate_inplace(src: str, tags: Dict[str, str]) -> str:
    """
    (편의) 임시 파일에 태그 넣고 원본 교체. 교체된 경로를 반환.
    """
    src_p = Path(src)
    with tempfile.TemporaryDirectory() as td:
        tmp_out = Path(td) / src_p.name
        set_mp4_metadata(str(src_p), str(tmp_out), tags)
        # 원본 교체
        src_p.write_bytes(tmp_out.read_bytes())
    return str(src_p)
