# app/routers/presigned_upload.py

from __future__ import annotations

import logging
import base64
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, BackgroundTasks

from app.core.config import settings
from app.core.crypto import AESGCMCrypto, DecryptedToken, get_crypto_service
from app.services.file_manager import merge_chunks_and_cleanup  # 파일 I/O 및 진행률 알림 위임 (chunk_saved_progress 제거됨)

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Depends 타입 별칭
# ─────────────────────────────────────────────────────────────
CryptoDep = Annotated[AESGCMCrypto, Depends(get_crypto_service)]

# ─────────────────────────────────────────────────────────────
# 유틸: Base64(any) 디코더
# ─────────────────────────────────────────────────────────────
def _b64_any_decode(s: str) -> bytes:
    """
    URL-safe / 표준 Base64 모두 수용. 필요 시 패딩 보정 후 디코드.
    """
    s = s.strip()
    if not s:
        return b""
    # data URL prefix 제거 대응
    if "," in s and s.lstrip().lower().startswith("data:"):
        s = s.split(",", 1)[1].strip()

    # Base64 패딩 보정 (4의 배수)
    pad = "=" * ((4 - (len(s) % 4)) % 4)

    # URL-safe 먼저 시도 후, 실패 시 표준 Base64 시도
    try:
        return base64.urlsafe_b64decode(s + pad)
    except Exception:
        return base64.b64decode(s + pad)


# ─────────────────────────────────────────────────────────────
# 내부 래퍼: file_manager 시그니처(4/5인자) 모두 호환
# ─────────────────────────────────────────────────────────────
async def _merge_task_wrapper(
    job_id: str,
    file_name: str,
    total_parts: int,
    chunk_dir: Path,
    member_id: str,
):
    """
    file_manager.merge_chunks_and_cleanup 이
    - (job_id, file_name, total_parts, chunk_dir, member_id) 5인자를 받으면 그대로 호출
    - 아니라면 4인자 시그니처로 폴백
    """
    try:
        # 먼저 5인자 시도 (향후 파일매니저가 member_id를 지원할 때)
        return await merge_chunks_and_cleanup(job_id, file_name, total_parts, chunk_dir, member_id)  # type: ignore
    except TypeError:
        # 기존 4인자 시그니처와의 호환 유지
        return await merge_chunks_and_cleanup(job_id, file_name, total_parts, chunk_dir)  # type: ignore


# ─────────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────────
@router.post("/chunk")
async def upload_presigned_chunk(
    crypto: CryptoDep,
    # file은 Base64 문자열로 받음
    file: str = Form(..., description="Base64 인코딩된 청크 문자열"),
    chunkIndex: int = Form(..., ge=1, description="현재 청크 번호 (1부터 시작)"),
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
    fileName: str = Form(..., description="클라이언트가 업로드하는 파일명"),
):
    """
    클라이언트로부터 Base64 인코딩된 청크를 받아 디코딩 및 저장
    """
    job_id: str | None = None
    token_file_name: str | None = None

    try:
        # 0) 유효성 검사 추가 (FastAPI가 처리하지 못하는 비즈니스 로직)
        if chunkIndex > totalParts:
            raise HTTPException(status_code=400, detail="chunkIndex cannot be greater than totalParts.")

        # 1) 토큰 복호화/검증
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            token_file_name = token_data.fileName
        except ValueError as e:
            logger.error(f"Token validation failed (ValueError) / ERR-A: {e}")
            # [오류 코드 적용] ERR-A: 토큰 복호화 실패/만료/변조 시 401
            raise HTTPException(status_code=401, detail=f"ERR-A: Invalid or expired access token: {str(e)}")

        # 2) fileName 일치 검증
        if fileName != token_file_name:
            logger.error(
                f"Filename mismatch for Job {job_id} / ERR-B: Token expects '{token_file_name}', received '{fileName}'"
            )
            # [오류 코드 적용] ERR-B: 파일명 불일치 시 400
            raise HTTPException(
                status_code=400,
                detail=f"ERR-B: Upload file name mismatch with token (Expected: '{token_file_name}', Received: '{fileName}')",
            )

        # 3) Base64 데이터 디코딩
        base64_str = file
        try:
            if not base64_str.strip():
                raise ValueError("Empty base64 payload")

            chunk_binary_data = _b64_any_decode(base64_str)
        except Exception as e:
            logger.error(f"Base64 decoding failed for chunk {chunkIndex} (Job {job_id}) / ERR-C: {e}")
            # [오류 코드 적용] ERR-C: Base64 디코딩 실패 또는 페이로드 비어있음 시 493
            raise HTTPException(status_code=493, detail="ERR-C: Invalid or empty Base64 chunk payload.")

        if not chunk_binary_data:
            logger.error(f"Decoded chunk is empty for chunk {chunkIndex} (Job {job_id}) / ERR-C")
            # [오류 코드 적용] ERR-C: 디코딩 후 바이너리 데이터가 비어있음 시 493
            raise HTTPException(status_code=493, detail="ERR-C: Decoded chunk binary data is empty.")

        # 4) 청크 저장 경로 설정 및 저장 (file_manager의 책임)
        chunk_dir = Path(settings.TEMP_ROOT) / job_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_filename = f"{job_id}_{token_file_name}.{chunkIndex:04d}"
        chunk_path = chunk_dir / chunk_filename

        with chunk_path.open("wb") as buffer:
            buffer.write(chunk_binary_data)

        logger.info(f"Chunk {chunkIndex}/{totalParts} saved for Job {job_id}")

        # 5) 파일 저장 성공 후, 진행률 통보 로직 제거 (병합 시에만 보고하도록 변경됨)
        # chunk_saved_progress(job_id, chunkIndex, totalParts) # <--- 이 라인이 제거되었습니다.

        return {"message": "Chunk uploaded successfully", "jobId": job_id, "chunkIndex": chunkIndex}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected Error during chunk upload: {e}", exc_info=True)
        # 처리되지 않은 모든 예외는 500
        raise HTTPException(status_code=500, detail="Internal server error during chunk upload")


@router.post("/complete")
async def complete_presigned_upload(
    crypto: CryptoDep,
    background_tasks: BackgroundTasks,
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
):
    """
    청크 완료 확인 (무결성 검증) → 프론트엔드에 즉시 응답 → 백그라운드에서 병합 및 AI 트리거
    """
    job_id: str | None = None
    chunk_dir: Path | None = None
    file_name: str | None = None
    member_id: str | None = None

    try:
        # 1) 토큰 복호화/검증
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            file_name = token_data.fileName
            member_id = token_data.memberId  # ← A-1: memberId 추출
        except ValueError as e:
            logger.error(f"Token validation failed in /complete / ERR-A: {e}")
            # [오류 코드 적용] ERR-A: 토큰 복호화 실패/만료/변조 시 401
            raise HTTPException(status_code=401, detail=f"ERR-A: Invalid or expired access token: {str(e)}")

        # 2) 임시 청크 폴더 확인
        chunk_dir = Path(settings.TEMP_ROOT) / job_id
        if not chunk_dir.exists():
            raise HTTPException(status_code=404, detail="Job ID not found or no chunks uploaded.")

        # 3) 개수 검증 (무결성 검증)
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        actual_parts = len(chunk_files)

        if actual_parts != totalParts:
            logger.error(
                f"Integrity check failed for Job {job_id} / ERR-D: Expected {totalParts} parts, found {actual_parts}."
            )
            # [오류 코드 적용] ERR-D: 청크 개수 불일치 시 494
            raise HTTPException(
                status_code=494,
                detail=f"ERR-D: Incomplete upload: Expected {totalParts} chunks, but only {actual_parts} were received.",
            )

        logger.info(f"Integrity check SUCCESS for Job {job_id}. Found {actual_parts}/{totalParts} chunks.")

        # 4) 백그라운드 작업 추가 (병합, 정리, AI 트리거)
        #    - A-1: memberId를 래퍼에 함께 전달 (file_manager가 4인자여도 안전)
        background_tasks.add_task(
            _merge_task_wrapper,
            job_id,
            file_name,
            totalParts,
            chunk_dir,
            member_id or "",  # Not None 보장
        )
        logger.info(f"Background merge/cleanup task added for Job {job_id} (memberId={member_id}).")

        # 5) 클라이언트에게 즉시 성공 응답
        return {"message": "Upload integrity verified, processing started in background.", "jobId": job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during completion integrity check: {e}", exc_info=True)
        # 처리되지 않은 모든 예외는 500
        raise HTTPException(status_code=500, detail="Internal server error during integrity check")
