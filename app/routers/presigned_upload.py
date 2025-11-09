# app/routers/presigned_upload.py
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.crypto import AESGCMCrypto, DecryptedToken, get_crypto_service

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Pydantic 폼 데이터 스키마
# ─────────────────────────────────────────────────────────────

class ChunkUploadForm(BaseModel):
    chunkIndex: int = Field(..., ge=1, description="현재 청크 번호 (1부터 시작)")
    totalParts: int = Field(..., ge=1, description="전체 청크 개수")
    presignedToken: str = Field(..., description="AES-GCM 복호화 가능한 토큰")
    fileName: str = Field(..., description="클라이언트가 업로드하는 파일명")

class CompleteUploadForm(BaseModel):
    totalParts: int = Field(..., ge=1, description="전체 청크 개수")
    presignedToken: str = Field(..., description="AES-GCM 복호화 가능한 토큰")

# ─────────────────────────────────────────────────────────────
# 비동기 AI 데모 트리거 (placeholder)
# ─────────────────────────────────────────────────────────────

def trigger_ai_demo(job_id: str, final_path: Path) -> None:
    try:
        logger.info(f"AI Demo Triggered for Job ID: {job_id}. Source: {final_path}")
        # 실제 처리 로직/큐 연동은 프로젝트 환경에 맞춰 구현
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
    file: UploadFile = File(..., description="청크 바이너리 데이터"),
    chunkIndex: int = Form(..., ge=1, description="현재 청크 번호 (1부터 시작)"),
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
    fileName: str = Form(..., description="클라이언트가 업로드하는 파일명"),
):
    """
    청크 파일을 임시 디렉토리에 저장
    """
    try:
        # 1) 토큰 복호화/검증
        token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
        job_id = token_data.jobId
        token_file_name = token_data.fileName

        # 2) fileName 일치 검증
        if fileName != token_file_name:
            raise HTTPException(
                status_code=400,
                detail=f"Filename mismatch: Token expects '{token_file_name}', received '{fileName}'",
            )

        # 3) 청크 저장 경로
        chunk_dir = Path(settings.TEMP_ROOT) / job_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_filename = f"{job_id}_{token_file_name}.{chunkIndex:04d}"
        chunk_path = chunk_dir / chunk_filename

        # 4) 저장
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Chunk {chunkIndex}/{totalParts} saved for Job ID: {job_id}")
        return {"message": "Chunk uploaded successfully", "jobId": job_id, "chunkIndex": chunkIndex}

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Error during chunk upload: {e}")
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
    청크 완료 확인 → 병합 → AI 처리 트리거
    """
    try:
        # 1) 토큰 복호화/검증
        token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
        job_id = token_data.jobId
        file_name = token_data.fileName

        chunk_dir = Path(settings.TEMP_ROOT) / job_id
        if not chunk_dir.exists():
            raise HTTPException(status_code=404, detail="Job ID not found or no chunks uploaded.")

        # 2) 개수 검증
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        actual_parts = len(chunk_files)
        if actual_parts != totalParts:
            logger.error(
                f"Integrity check failed for Job {job_id}: Expected {totalParts} parts, found {actual_parts}."
            )
            shutil.rmtree(chunk_dir)
            raise HTTPException(
                status_code=400,
                detail=f"Incomplete upload: Expected {totalParts} chunks, but only {actual_parts} were received.",
            )

        # 3) 병합
        final_path = Path(settings.SAVE_ROOT) / f"{job_id}_{file_name}"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        with final_path.open("wb") as outfile:
            for chunk_file in chunk_files:
                with chunk_file.open("rb") as infile:
                    shutil.copyfileobj(infile, outfile)

        logger.info(f"File merged successfully: {final_path}")

        # 4) 임시 폴더 정리
        shutil.rmtree(chunk_dir)

        # 5) AI 데모 비동기 시작
        background_tasks.add_task(trigger_ai_demo, job_id, final_path)

        return {"message": "Upload complete, file merged, and AI processing initiated", "jobId": job_id}

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Error during completion: {e}")
        if 'chunk_dir' in locals() and chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Internal server error during file merge")
