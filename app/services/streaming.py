# app/services/streaming.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union, List
from tempfile import SpooledTemporaryFile
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import shutil


PathLike = Union[str, Path]


def build_zip_spooled(clip_paths: Iterable[PathLike], arc_prefix: str = "") -> SpooledTemporaryFile:
    """
    클립 파일들을 ZIP으로 묶어 SpooledTemporaryFile로 반환.
    - max_size를 넘기면 자동으로 디스크 스풀로 전환되어 메모리 과점유를 방지.
    - arc_prefix: ZIP 내부 파일명 앞에 붙일 접두사(예: '#12_')
    """
    spooled = SpooledTemporaryFile(max_size=100 * 1024 * 1024, mode="w+b")  # ~100MB까지 메모리, 이후 디스크 스풀
    with ZipFile(spooled, mode="w", compression=ZIP_DEFLATED) as zf:
        for p in clip_paths:
            p = Path(p)
            if not p.exists():
                continue
            arcname = f"{arc_prefix}{p.name}"
            zf.write(p, arcname=arcname)
    spooled.seek(0)  # 읽기 시작점으로 이동
    return spooled


def get_spooled_size(spooled_file: SpooledTemporaryFile) -> int:
    """
    SpooledTemporaryFile의 총 크기(bytes)를 반환.
    (업로드 전에 Content-Length 헤더 계산용)
    """
    cur = spooled_file.tell()
    spooled_file.seek(0, 2)  # 끝으로
    size = spooled_file.tell()
    spooled_file.seek(cur, 0)  # 원래 위치로 복귀
    return size


def _iter_file(fobj, chunk_size: int = 64 * 1024):
    """StreamingResponse에 쓸 청크 제너레이터."""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            break
        yield chunk


def stream_zip_and_cleanup(
    spooled_file: SpooledTemporaryFile,
    clip_paths: Iterable[PathLike],
    tmp_paths: Iterable[PathLike],
) -> StreamingResponse:
    """
    ZIP 스트림을 HTTP 응답으로 내보내고, 응답 종료 시 임시 리소스 모두 삭제.
    - 옵션 B(콜백 업로드)에서는 쓰지 않아도 되지만, 다운로드 엔드포인트 유지용으로 제공.
    """

    def _cleanup():
        # 클립 파일 제거
        try:
            for p in clip_paths:
                Path(p).unlink(missing_ok=True)
        except Exception:
            pass
        # 임시 경로 정리(폴더/파일 모두 대응)
        try:
            for t in tmp_paths:
                t = Path(t)
                if t.is_dir():
                    shutil.rmtree(t, ignore_errors=True)
                else:
                    t.unlink(missing_ok=True)
        except Exception:
            pass
        # 스풀 닫기
        try:
            spooled_file.close()
        except Exception:
            pass

    headers = {"Content-Disposition": 'attachment; filename="player_highlights.zip"'}
    return StreamingResponse(
        _iter_file(spooled_file),
        media_type="application/zip",
        headers=headers,
        background=BackgroundTask(_cleanup),
    )
