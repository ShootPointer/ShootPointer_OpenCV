from __future__ import annotations

import logging
import base64
import re
from pathlib import Path
from typing import Annotated, Dict, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, BackgroundTasks

from app.core.config import settings
from app.core.crypto import AESGCMCrypto, DecryptedToken, get_crypto_service
from app.services.file_manager import (
    merge_chunks_and_cleanup,
    report_progress_to_spring,  # ✅ 진행률 보고 함수 사용
)
from app.schemas.redis import UploadStatus  # ✅ 상태 Enum (UPLOADING 등)

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Depends 타입 별칭
# ─────────────────────────────────────────────────────────────
CryptoDep = Annotated[AESGCMCrypto, Depends(get_crypto_service)]

# ─────────────────────────────────────────────────────────────
# 업로드 컨텍스트: jobId별 최초 fileName을 락으로 고정
# ─────────────────────────────────────────────────────────────
UPLOAD_CTX: Dict[str, str] = {}

# ─────────────────────────────────────────────────────────────
# 유틸: Base64(any) 디코더
# ─────────────────────────────────────────────────────────────
def _b64_any_decode(s: str) -> bytes:
    """
    URL-safe / 표준 Base64 모두 수용. 필요 시 패딩 보정 후 디코드.
    """
    s = (s or "").strip()
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
# 파일명 정화 + 확장자 화이트리스트
# ─────────────────────────────────────────────────────────────
_ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".m4v"}

def _sanitize_filename(name: str, max_len: int = 120) -> str:
    """
    - 경로 구분자 제거
    - 허용 문자만 유지 (영문/숫자/._- 공백)
    - 길이 제한
    - 확장자 화이트리스트 검사
    """
    if not name:
        raise HTTPException(status_code=400, detail="invalid_filename: empty")

    # 경로 구분자 제거
    name = name.replace("\\", "/").split("/")[-1]

    # 허용 문자만 남김 (한글 허용 필요 시 [^A-Za-z0-9._\\- ㄱ-힣] 등으로 확장)
    name = re.sub(r"[^A-Za-z0-9._\- ]", "_", name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="invalid_filename: sanitized_to_empty")

    # 너무 긴 이름 컷
    if len(name) > max_len:
        base, dot, ext = name.rpartition(".")
        if dot:
            base = base[: max(1, max_len - len(ext) - 1)]
            name = f"{base}.{ext}"
        else:
            name = name[:max_len]

    # 확장자 체크
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="invalid_extension: only video files are allowed")

    return name

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
# 엔드포인트: 청크 업로드
# ─────────────────────────────────────────────────────────────
@router.post("/chunk")
async def upload_presigned_chunk(
    crypto: CryptoDep,
    # 파일은 Base64 문자열로 받음
    file: str = Form(..., description="Base64 인코딩된 청크 문자열"),
    chunkIndex: int = Form(..., ge=1, description="현재 청크 번호 (1부터 시작)"),
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
    fileName: str = Form(..., description="클라이언트가 업로드하는 파일명"),
):
    """
    클라이언트로부터 Base64 인코딩된 청크를 받아 디코딩 및 저장.
    - 토큰은 expires/memberId/jobId 인증만 수행 (fileName 비교 제거)
    - fileName은 폼에서 받고 sanitize + 확장자 체크 + jobId 단위로 락
    """
    job_id: Optional[str] = None

    try:
        # 0) 유효성 검사
        if chunkIndex > totalParts:
            raise HTTPException(status_code=400, detail="chunkIndex cannot be greater than totalParts.")

        # 1) 토큰 복호화/검증 → jobId, memberId 확인 (fileName 비교는 하지 않음)
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            member_id = token_data.memberId  # 로그/추적용
        except ValueError as e:
            logger.error(f"Token validation failed (ValueError) / ERR-A: {e}")
            # ERR-A: 토큰 복호화 실패/만료/변조 시 401
            raise HTTPException(status_code=401, detail=f"ERR-A: Invalid or expired access token: {str(e)}")

        # 2) 파일명 sanitize + 확장자 체크, 그리고 최초 파일명 락
        safe_name = _sanitize_filename(fileName)
        prev_name = UPLOAD_CTX.get(job_id)
        if prev_name is None:
            UPLOAD_CTX[job_id] = safe_name
        elif prev_name != safe_name:
            logger.error(
                f"[upload_chunk] filename changed within the same job. "
                f"jobId={job_id}, prev={prev_name}, got={safe_name}"
            )
            # ERR-B2: 동일 jobId에서 파일명 변경 시 거부
            raise HTTPException(status_code=400, detail="ERR-B2: filename changed within the same upload job.")

        # 3) Base64 디코딩
        try:
            if not file.strip():
                raise ValueError("Empty base64 payload")
            chunk_binary_data = _b64_any_decode(file)
        except Exception as e:
            logger.error(f"Base64 decoding failed for chunk {chunkIndex} (Job {job_id}) / ERR-C: {e}")
            # ERR-C: Base64 디코딩 실패 또는 페이로드 비어있음 시 493
            raise HTTPException(status_code=493, detail="ERR-C: Invalid or empty Base64 chunk payload.")

        if not chunk_binary_data:
            logger.error(f"Decoded chunk is empty for chunk {chunkIndex} (Job {job_id}) / ERR-C")
            # ERR-C: 디코딩 후 바이너리 데이터가 비어있음 시 493
            raise HTTPException(status_code=493, detail="ERR-C: Decoded chunk binary data is empty.")

        # 4) 저장 경로 구성 (경로 탈출 방지)
        base_dir = Path(settings.TEMP_ROOT).resolve()
        chunk_dir = (base_dir / job_id).resolve()
        if base_dir not in chunk_dir.parents and chunk_dir != base_dir:
            raise HTTPException(status_code=400, detail="invalid_path")

        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_filename = f"{job_id}_{safe_name}.{chunkIndex:04d}"
        chunk_path = (chunk_dir / chunk_filename).resolve()
        if chunk_dir not in chunk_path.parents and chunk_path != chunk_dir:
            raise HTTPException(status_code=400, detail="invalid_path")

        with chunk_path.open("wb") as buffer:
            buffer.write(chunk_binary_data)

        logger.info(f"Chunk {chunkIndex}/{totalParts} saved for Job {job_id} (memberId={member_id})")

        # ✅ 청크 업로드 진행률 보고 (0 ~ 90%)
        try:
            # 전체 업로드에서 청크 업로드 단계는 0~90% 구간 사용
            upload_ratio = chunkIndex / totalParts  # 0.0 ~ 1.0
            progress_int = int(round(upload_ratio * 90))  # 0 ~ 90 정수

            # 0%로 남지 않도록 1%부터 시작, 최대 90%까지만
            if progress_int < 1:
                progress_int = 1
            if progress_int > 90:
                progress_int = 90

            await report_progress_to_spring(
                job_id,
                UploadStatus.UPLOADING.value,  # "UPLOADING"
                float(progress_int),
                member_id
            )
        except Exception as e:
            # 진행률 보고 실패해도 업로드 자체는 계속 되도록
            logger.error(
                f"Failed to report upload progress for chunk {chunkIndex}/{totalParts} of Job {job_id}: {e}"
            )

        # 기존 응답은 그대로 유지
        return {"message": "Chunk uploaded successfully", "jobId": job_id, "chunkIndex": chunkIndex}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected Error during chunk upload: {e}", exc_info=True)
        # 처리되지 않은 모든 예외는 500
        raise HTTPException(status_code=500, detail="Internal server error during chunk upload")

# ─────────────────────────────────────────────────────────────
# 엔드포인트: 업로드 완료 → 병합/정리 백그라운드 실행
# ─────────────────────────────────────────────────────────────
@router.post("/complete")
async def complete_presigned_upload(
    crypto: CryptoDep,
    background_tasks: BackgroundTasks,
    totalParts: int = Form(..., ge=1, description="전체 청크 개수"),
    presignedToken: str = Form(..., description="AES-GCM 복호화 가능한 토큰"),
    # 토큰에서 fileName을 제거했으므로, 필요 시 프론트가 함께 보내줄 수 있도록 옵션화
    fileName: Optional[str] = Form(
        None,
        description="(선택) 업로드 파일명 – 미보내면 서버가 첫 청크에서 잠근 이름을 사용",
    ),
):
    """
    청크 완료 확인 (무결성 검증) → 프론트엔드에 즉시 응답 → 백그라운드에서 병합 및 AI 트리거.
    - fileName은 클라이언트가 보낼 수도 있고, 안 보내면 UPLOAD_CTX[jobId] 사용
    """
    job_id: Optional[str] = None
    chunk_dir: Optional[Path] = None
    safe_name: Optional[str] = None
    member_id: Optional[str] = None

    try:
        # 1) 토큰 복호화/검증
        try:
            token_data: DecryptedToken = crypto.decrypt_token(presignedToken)
            job_id = token_data.jobId
            member_id = token_data.memberId
        except ValueError as e:
            logger.error(f"Token validation failed in /complete / ERR-A: {e}")
            # ERR-A: 토큰 복호화 실패/만료/변조 시 401
            raise HTTPException(status_code=401, detail=f"ERR-A: Invalid or expired access token: {str(e)}")

        # 2) 파일명 결정 (폼 > 컨텍스트)
        if fileName and fileName.strip():
            safe_name = _sanitize_filename(fileName)
            # 컨텍스트가 이미 있으면 동일한지 확인(일관성 유지)
            prev_name = UPLOAD_CTX.get(job_id)
            if prev_name and prev_name != safe_name:
                # 사용자가 마지막에 다른 이름을 보내왔다면, 보안상 거부(업로드 중 이름 변경 방지)
                raise HTTPException(status_code=400, detail="ERR-B2: filename changed within the same upload job.")
            UPLOAD_CTX[job_id] = safe_name
        else:
            safe_name = UPLOAD_CTX.get(job_id)
            if not safe_name:
                raise HTTPException(status_code=400, detail="missing_filename")

        # 3) 임시 청크 폴더 확인 + 경로 탈출 방지
        base_dir = Path(settings.TEMP_ROOT).resolve()
        chunk_dir = (base_dir / job_id).resolve()
        if not chunk_dir.exists():
            raise HTTPException(status_code=404, detail="Job ID not found or no chunks uploaded.")
        if base_dir not in chunk_dir.parents and chunk_dir != base_dir:
            raise HTTPException(status_code=400, detail="invalid_path")

        # 4) 개수 검증 (무결성 검증)
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{safe_name}.*"))
        actual_parts = len(chunk_files)
        if actual_parts != totalParts:
            logger.error(
                f"Integrity check failed for Job {job_id} / ERR-D: "
                f"Expected {totalParts} parts, found {actual_parts}."
            )
            # ERR-D: 청크 개수 불일치 시 494
            raise HTTPException(
                status_code=494,
                detail=(
                    "ERR-D: Incomplete upload: "
                    f"Expected {totalParts} chunks, but only {actual_parts} were received."
                ),
            )

        logger.info(f"Integrity check SUCCESS for Job {job_id}. Found {actual_parts}/{totalParts} chunks.")

        # 5) 백그라운드 작업 추가 (병합, 정리, AI 트리거)
        background_tasks.add_task(
            _merge_task_wrapper,
            job_id,
            safe_name,
            totalParts,
            chunk_dir,
            member_id or "",  # Not None 보장
        )
        logger.info(
            f"Background merge/cleanup task added for Job {job_id} "
            f"(memberId={member_id}, fileName={safe_name})."
        )

        # 6) 클라이언트에게 즉시 성공 응답
        return {"message": "Upload integrity verified, processing started in background.", "jobId": job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during completion integrity check: {e}", exc_info=True)
        # 처리되지 않은 모든 예외는 500
        raise HTTPException(status_code=500, detail="Internal server error during integrity check")
