from __future__ import annotations

import logging
import shutil
import base64
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, BackgroundTasks

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
        # TODO: Redis 큐에 job_id와 final_path를 push하는 실제 로직으로 대체해야 합니다.
        logger.info(f"AI Demo Triggered (Placeholder) for Job ID: {job_id}. Source: {final_path}")
        logger.info(f"AI processing successfully initiated for {job_id}")
    except Exception as e:
        logger.error(f"Failed to trigger AI demo for {job_id}: {e}")

# ─────────────────────────────────────────────────────────────
# Depends 타입 별칭
# ─────────────────────────────────────────────────────────────

CryptoDep = Annotated[AESGCMCrypto, Depends(get_crypto_service)]

# ─────────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────────

@router.post("/chunk")
async def upload_presigned_chunk(
    # ⬇⬇ 기본값 없는 파라미터(Depends 주입)는 맨 앞에 둔다 (파이썬 규칙 충족)
    crypto: CryptoDep,
    background_tasks: BackgroundTasks,

    # ⬇⬇ 기본값 있는 파라미터들 (Form/File)
    file: UploadFile = File(..., description="Base64 인코딩된 청크 데이터"),
    chunkIndex: int = Form(..., ge=1, description="현재 청크 번호 (1부터 시작)"),
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
    fileName: str = Form(..., description="클라이언트가 업로드하는 파일명"),
):
    """
    클라이언트로부터 Base64 인코딩된 청크를 받아 디코딩 및 저장
    """
    job_id = None
    token_file_name = None
    try:
        # 1) 토큰 복호화/검증
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            token_file_name = token_data.fileName
        except ValueError as e:
            # 🚨 JSON serializable 오류를 방지하기 위해 ValueError 발생 시 HTTPException으로 변환 (핵심 수정)
            logger.error(f"Token validation failed (ValueError): {e}")
            raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")


        # 2) fileName 일치 검증
        if fileName != token_file_name:
            logger.error(f"Filename mismatch for Job {job_id}: Token expects '{token_file_name}', received '{fileName}'")
            raise HTTPException(
                status_code=400,
                detail=f"Filename mismatch: Token expects '{token_file_name}', received '{fileName}'",
            )
        
        # 3) Base64 데이터 읽기 및 디코딩
        base64_data: bytes = await file.read() 
        
        try:
            # Base64 문자열을 실제 바이너리 데이터로 디코딩
            chunk_binary_data = base64.b64decode(base64_data.strip())
        except (ValueError, Exception) as e:
            # Base64 디코딩 실패 시 400 Bad Request 및 로그 기록
            logger.error(f"Base64 decoding failed for chunk {chunkIndex} (Job {job_id}): {e}")
            raise HTTPException(status_code=400, detail="Invalid Base64 data received in chunk.")


        # 4) 청크 저장 경로 설정
        chunk_dir = settings.TEMP_ROOT / job_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_filename = f"{job_id}_{token_file_name}.{chunkIndex:04d}"
        chunk_path = chunk_dir / chunk_filename

        # 5) 디코딩된 바이너리 데이터를 파일에 쓰기
        with chunk_path.open("wb") as buffer:
            buffer.write(chunk_binary_data)
        
        logger.info(f"Chunk {chunkIndex}/{totalParts} saved for Job ID: {job_id}")
        return {"message": "Chunk uploaded successfully", "jobId": job_id, "chunkIndex": chunkIndex}

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 그 외 모든 예상치 못한 오류에 대해 500 응답
        logger.error(f"Unexpected Error during chunk upload: {e}", exc_info=True)
        # JSON 직렬화 오류를 피하기 위해 HTTPException으로 변환하여 반환
        raise HTTPException(status_code=500, detail="Internal server error during upload")

@router.post("/complete")
async def complete_presigned_upload(
    # ⬇⬇ 기본값 없는 파라미터 먼저
    crypto: CryptoDep,
    background_tasks: BackgroundTasks,
    # redis_service: RedisDep, # Redis 의존성 제거됨

    # ⬇⬇ 기본값 있는 Form 파라미터들
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
):
    """
    청크 완료 확인 → 병합 → AI 처리 트리거 (Placeholder 유지)
    """
    job_id = None
    chunk_dir = None
    try:
        # 1) 토큰 복호화/검증
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            file_name = token_data.fileName
        except ValueError as e:
            # 🚨 JSON serializable 오류를 방지하기 위해 ValueError 발생 시 HTTPException으로 변환 (핵심 수정)
            logger.error(f"Token validation failed in /complete: {e}")
            raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")


        chunk_dir = settings.TEMP_ROOT / job_id
        if not chunk_dir.exists():
            raise HTTPException(status_code=404, detail="Job ID not found or no chunks uploaded.")

        # 2) 개수 검증
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        actual_parts = len(chunk_files)
        if actual_parts != totalParts:
            logger.error(
                f"Integrity check failed for Job {job_id}: Expected {totalParts} parts, found {actual_parts}."
            )
            shutil.rmtree(chunk_dir) # 실패 시 임시 파일 정리
            raise HTTPException(
                status_code=400,
                detail=f"Incomplete upload: Expected {totalParts} chunks, but only {actual_parts} were received.",
            )

        # 3) 병합
        final_path = settings.ORIGINAL_VIDEO_ROOT / f"{job_id}_{file_name}"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        with final_path.open("wb") as outfile:
            for chunk_file in chunk_files:
                with chunk_file.open("rb") as infile:
                    shutil.copyfileobj(infile, outfile)

        logger.info(f"File merged successfully: {final_path}")

        # 4) 임시 청크 폴더 정리
        shutil.rmtree(chunk_dir)

        # 5) AI 데모 비동기 시작 (Placeholder 유지)
        background_tasks.add_task(trigger_ai_demo, job_id, final_path)
        logger.info(f"AI processing successfully initiated (via background task).")

        return {"message": "Upload complete, file merged, and AI processing initiated", "jobId": job_id}

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        logger.error(f"Error during completion: {e}")
        if 'chunk_dir' in locals() and chunk_dir and chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
        # JSON 직렬화 오류를 피하기 위해 HTTPException으로 변환하여 반환
        raise HTTPException(status_code=500, detail="Internal server error during file merge")