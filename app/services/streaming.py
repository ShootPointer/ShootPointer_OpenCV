# app/services/streaming.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union, List, Optional
from tempfile import SpooledTemporaryFile
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import shutil
import logging

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def build_zip_spooled(clip_paths: Iterable[PathLike], arc_prefix: str = "") -> SpooledTemporaryFile:
    """
    클립 파일들을 ZIP으로 묶어 SpooledTemporaryFile로 반환.
    - max_size를 넘기면 자동으로 디스크 스풀로 전환되어 메모리 과점유 방지.
    - arc_prefix: ZIP 내부 파일명 앞에 붙일 접두사(예: '#12_')
    """
    spooled = SpooledTemporaryFile(max_size=100 * 1024 * 1024, mode="w+b")  # ~100MB까지 메모리, 이후 디스크 스풀
    count = 0
    with ZipFile(spooled, mode="w", compression=ZIP_DEFLATED) as zf:
        for p in clip_paths:
            p = Path(p)
            if not p.exists():
                continue
            arcname = f"{arc_prefix}{p.name}"
            zf.write(p, arcname=arcname)
            count += 1
    spooled.seek(0)  # 읽기 시작점으로 이동
    logger.info(f"[streaming] build_zip_spooled -> {count} files (prefix='{arc_prefix}')")
    return spooled


def get_spooled_size(spooled_file: SpooledTemporaryFile) -> int:
    """
    SpooledTemporaryFile의 총 크기(bytes)를 반환.
    (업로드/다운로드 전에 Content-Length 헤더 계산용)
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
    *,
    download_name: str = "player_highlights.zip",
    chunk_size: int = 64 * 1024,
    add_content_length: bool = False,
) -> StreamingResponse:
    """
    ZIP 스트림을 HTTP 응답으로 내보내고, 응답 종료 시 임시 리소스 모두 삭제.
    - download_name: Content-Disposition 파일명
    - chunk_size: 스트리밍 청크 크기
    - add_content_length: True면 Content-Length 헤더 추가(일부 프록시/클라이언트 호환)
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
        logger.info("[streaming] cleanup done")

    headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
    if add_content_length:
        try:
            headers["Content-Length"] = str(get_spooled_size(spooled_file))
        except Exception:
            # 길이 계산 실패는 무시하고 진행
            pass

    logger.info(f"[streaming] start response (file={download_name})")
    return StreamingResponse(
        _iter_file(spooled_file, chunk_size=chunk_size),
        media_type="application/zip",
        headers=headers,
        background=BackgroundTask(_cleanup),
    )
