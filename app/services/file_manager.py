import logging
import shutil
from pathlib import Path

from app.core.config import settings
from app.services.redis_pubsub import publish_progress # Redis Pub/Sub 서비스로 변경
from app.schemas.redis import UploadStatus

logger = logging.getLogger(__name__)

def chunk_saved_progress(job_id: str, chunk_index: int, total_parts: int) -> None:
    """
    단일 청크가 성공적으로 저장되었음을 Redis에 통보합니다. (Chunk 엔드포인트에서 호출됨)
    """
    # pubsub_service를 통해 Redis에 진행률 통보
    publish_progress(
        job_id, 
        UploadStatus.UPLOADING, 
        f"Chunk {chunk_index}/{total_parts} received and saved.",
        current=chunk_index, 
        total=total_parts
    )

def merge_chunks_and_cleanup(job_id: str, file_name: str, total_parts: int, chunk_dir: Path):
    """
    [Background Task] 파일 병합, 임시 폴더 정리, AI 작업 큐 트리거를 실행합니다.
    
    Args:
        job_id: The unique job identifier.
        file_name: The original name of the file.
        total_parts: The total number of chunks expected.
        chunk_dir: The Path object pointing to the temporary chunk directory.
    """
    logger.info(f"Starting background merge and cleanup for Job ID: {job_id}")
    
    # 최종 저장 경로: SAVE_ROOT/[jobId]/[fileName]
    final_path = Path(settings.SAVE_ROOT) / job_id / file_name
    
    try:
        # 1. 병합 사전 준비 및 진행률 통보 시작
        final_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        
        publish_progress(
            job_id, 
            UploadStatus.PROCESSING, 
            "Start merging chunks...",
            current=0, total=total_parts
        )
        
        # 2. 병합 작업 실행
        with final_path.open("wb") as outfile:
            for i, chunk_file in enumerate(chunk_files):
                with chunk_file.open("rb") as infile:
                    shutil.copyfileobj(infile, outfile)
                
                progress = i + 1
                # 병합 중에도 진행률을 주기적으로 업데이트
                if total_parts < 20 or progress % max(1, total_parts // 20) == 0 or progress == total_parts:
                     publish_progress(
                        job_id, 
                        UploadStatus.PROCESSING, 
                        f"Merging chunk {progress}/{total_parts}",
                        current=progress, total=total_parts
                    )
        
        logger.info(f"File merged successfully: {final_path}")

        # 3. 임시 청크 폴더 삭제 (정리)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        logger.info(f"Successfully cleaned up temporary chunk directory: {chunk_dir}")
        
        # 4. 완료 통보 (Pub/Sub)
        publish_progress(
            job_id, 
            UploadStatus.UPLOAD_COMPLETE, 
            "Original video file saved and cleanup finished.",
            current=total_parts, total=total_parts
        )
        
        # 5. AI 작업 지시 (Queue)
        # TODO: Redis L PUSH (AI Queue) 로직을 redis_pubsub.py의 새로운 함수를 통해 구현하여 연결해야 함
        logger.info(f"Job {job_id} is ready for AI processing. (AI Queue Trigger Pending)")

    except Exception as e:
        logger.error(f"Critical error during background merge/cleanup for Job {job_id}: {e}", exc_info=True)
        # 실패 시에도 임시 폴더 삭제 시도
        shutil.rmtree(chunk_dir, ignore_errors=True)
        publish_progress(
            job_id, 
            UploadStatus.ERROR, 
            f"Merge or cleanup failed: {e}",
            current=0, total=total_parts
        )