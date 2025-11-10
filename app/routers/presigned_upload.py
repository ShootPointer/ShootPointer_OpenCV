from __future__ import annotations

import logging
import shutil
import base64
from pathlib import Path
from typing import Annotated, Optional # Optional (또는 | None) import

# 🚨 수정: File, UploadFile 제거 (Form만 사용)
from fastapi import APIRouter, Depends, Form, HTTPException, BackgroundTasks

from app.core.config import settings
from app.core.crypto import AESGCMCrypto, DecryptedToken, get_crypto_service

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────
# 비동기 AI 데모 트리거 (placeholder)
# ─────────────────────────────────────────────────────────────
def trigger_ai_demo(job_id: str, final_path: Path) -> None:
    """
    원본 영상 저장 완료 후 다음 단계로 넘어가는 placeholder 함수입니다.
    """
    try:
        # TODO: 실제 큐/잡 트리거 로직으로 교체
        logger.info(f"AI Demo Triggered (Placeholder) for Job ID: {job_id}. Source: {final_path}")
        logger.info(f"AI processing successfully initiated for {job_id}")
    except Exception as e:
        logger.error(f"Failed to trigger AI demo for {job_id}: {e}")

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
    # data URL prefix 제거 대응 (ex: "data:application/octet-stream;base64,AAAA...")
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
# 엔드포인트
# ─────────────────────────────────────────────────────────────
@router.post("/chunk")
async def upload_presigned_chunk(
    # ⬇⬇ 기본값 없는 파라미터(Depends 주입)는 맨 앞에 둔다 (파이썬 규칙 충족)
    crypto: CryptoDep,
    background_tasks: BackgroundTasks,

    # ⬇⬇ 기본값 있는 파라미터들 (Form)
    # 🚨 수정: file: UploadFile 대신 base64Chunk: str로 Base64 문자열을 Form으로 받음
    base64Chunk: str = Form(..., description="Base64 인코딩된 청크 문자열"),
    
    chunkIndex: int = Form(..., ge=1, description="현재 청크 번호 (1부터 시작)"),
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
    fileName: str = Form(..., description="클라이언트가 업로드하는 파일명"),
):
    """
    클라이언트로부터 Base64 인코딩된 청크를 받아 디코딩 및 저장
    """
    # Optional 타입 힌트를 위한 변수 선언
    job_id: str | None = None
    token_file_name: str | None = None
    
    try:
        # 0) 유효성 검사 추가 (chunkIndex와 totalParts는 FastAPI/Pydantic이 이미 검사함)
        if chunkIndex > totalParts:
            raise HTTPException(status_code=400, detail="chunkIndex cannot be greater than totalParts.")

        # 1) 토큰 복호화/검증
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            token_file_name = token_data.fileName
        except ValueError as e:
            logger.error(f"Token validation failed (ValueError): {e}")
            raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

        # 2) fileName 일치 검증
        if fileName != token_file_name:
            logger.error(
                f"Filename mismatch for Job {job_id}: Token expects '{token_file_name}', received '{fileName}'"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Filename mismatch: Token expects '{token_file_name}', received '{fileName}'",
            )

        # 3) Base64 데이터 디코딩
        base64_str = base64Chunk # 🚨 수정: Form으로 받은 문자열 변수를 사용
        try:
            if not base64_str.strip():
                raise ValueError("empty base64 payload")
                
            # 강력한 유틸리티 함수를 사용하여 디코딩
            chunk_binary_data = _b64_any_decode(base64_str)
        except Exception as e:
            logger.error(f"Base64 decoding failed for chunk {chunkIndex} (Job {job_id}): {e}")
            raise HTTPException(status_code=400, detail="Invalid Base64 data received in chunk.")

        if not chunk_binary_data:
            logger.error(f"Decoded chunk is empty for chunk {chunkIndex} (Job {job_id})")
            raise HTTPException(status_code=400, detail="Decoded chunk is empty.")

        # 4) 청크 저장 경로 설정
        chunk_dir = Path(settings.TEMP_ROOT) / job_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_filename = f"{job_id}_{token_file_name}.{chunkIndex:04d}"
        chunk_path = chunk_dir / chunk_filename

        # 5) 디코딩된 바이너리 데이터를 파일에 쓰기
        with chunk_path.open("wb") as buffer:
            buffer.write(chunk_binary_data)

        logger.info(
            f"Chunk {chunkIndex}/{totalParts} saved for Job {job_id} "
            f"(size={len(chunk_binary_data)}B, path={chunk_path})"
        )
        return {"message": "Chunk uploaded successfully", "jobId": job_id, "chunkIndex": chunkIndex}

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        logger.error(f"Unexpected Error during chunk upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during upload")

@router.post("/complete")
async def complete_presigned_upload(
    # ⬇⬇ 기본값 없는 파라미터 먼저
    crypto: CryptoDep,
    background_tasks: BackgroundTasks,

    # ⬇⬇ 기본값 있는 Form 파라미터들
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
):
    """
    청크 완료 확인 → 병합 → AI 처리 트리거 (Placeholder 유지)
    """
    job_id: str | None = None
    chunk_dir: Path | None = None # Path 타입 힌트
    try:
        # 1) 토큰 복호화/검증
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            file_name = token_data.fileName
        except ValueError as e:
            logger.error(f"Token validation failed in /complete: {e}")
            raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

        # 2) 임시 청크 폴더 확인
        chunk_dir = Path(settings.TEMP_ROOT) / job_id
        if not chunk_dir.exists():
            raise HTTPException(status_code=404, detail="Job ID not found or no chunks uploaded.")

        # 3) 개수 검증
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        actual_parts = len(chunk_files)
        if actual_parts != totalParts:
            logger.error(
                f"Integrity check failed for Job {job_id}: Expected {totalParts} parts, found {actual_parts}."
            )
            shutil.rmtree(chunk_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Incomplete upload: Expected {totalParts} chunks, but only {actual_parts} were received.",
            )

        # 4) 병합 (최종 저장 루트: SAVE_ROOT)
        final_path = Path(settings.SAVE_ROOT) / f"{job_id}_{file_name}"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        with final_path.open("wb") as outfile:
            for chunk_file in chunk_files:
                with chunk_file.open("rb") as infile:
                    shutil.copyfileobj(infile, outfile)

        logger.info(f"File merged successfully: {final_path}")

        # 5) 임시 청크 폴더 정리
        shutil.rmtree(chunk_dir, ignore_errors=True)

        # 6) AI 데모 비동기 시작 (Placeholder 유지)
        background_tasks.add_task(trigger_ai_demo, job_id, final_path)
        logger.info("AI processing successfully initiated (via background task).")

        return {"message": "Upload complete, file merged, and AI processing initiated", "jobId": job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during completion: {e}")
        # chunk_dir이 정의되었고 존재하는 경우에만 정리 시도
        if 'chunk_dir' in locals() and chunk_dir and chunk_dir.exists():
             shutil.rmtree(chunk_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Internal server error during file merge")