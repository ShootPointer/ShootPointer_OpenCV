# ai_worker/services/ai_worker_simulator.py

import asyncio
import json
import logging
import os
import time
import subprocess
import cv2
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from redis.asyncio import Redis, ConnectionError as RedisConnectionError

# ─────────────────────────────────────────────────────────────
# 1. 모듈 및 설정 임포트/정의
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 패키지 기준 절대 임포트 (ai_worker. 접두사 고정)
try:
    from ai_worker.utils.bh_edit import get_video_metadata
    from ai_worker.configs.registry import PLANS, DURATION_TOLERANCE_SEC, SIZE_TOLERANCE_BYTES
    from ai_worker.ai_modules.bh_geometry import compute_homography_auto, NBA
    from ai_worker.ai_modules.bh_detect import detect_ball_hsv
    # 외부 모듈: 스프링 서버로 결과 전송 (재시도 포함)
    from ai_worker.services.spring_reporter import send_highlight_result_with_retry, MAX_RETRIES
except ImportError as e:
    raise ImportError(
        "Failed to import ai_worker submodules. "
        "Ensure Dockerfile COPY paths place sources under /app/ai_worker and PYTHONPATH includes /app. "
        f"Original error: {e}"
    ) from e


class Settings(BaseModel):
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://:rlaehdus00123@host.docker.internal:6379/0")
    AI_QUEUE_NAME: str = os.getenv("REDIS_AI_JOB_QUEUE", "opencv-ai-job-queue")
    SPRING_API_URL: str = os.getenv("SPRING_API_URL", "http://host.docker.internal:8080/api/v1/jobs")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/data/highlights/processed")
    # FastAPI 쪽 REDIS_UPLOAD_PROGRESS_CHANNEL 과 같은 prefix 를 사용
    PROGRESS_CHANNEL_PREFIX: str = os.getenv("PROGRESS_CHANNEL_PREFIX", "opencv-progress-highlight")


settings = Settings()
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

_redis_client: Optional[Redis] = None


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception:
            raise RedisConnectionError("Redis connection failed.")
    return _redis_client


class AITaskPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    jobId: str = Field(..., description="작업 고유 ID")
    memberId: str = Field(..., description="요청 사용자 ID")
    originalFilePath: str = Field(..., description="원본 파일의 서버 내부 경로")
    # FastAPI → Redis: highlightKey  → (alias 매핑) → highlightIdentifier(필수)
    highlightIdentifier: str = Field(..., alias="highlightKey", description="하이라이트 식별 키(필수)")
    maxClips: Optional[int] = Field(default=None)


# ─────────────────────────────────────────────────────────────
# 2. 공통: 진행률/완료 보고 유틸 (Spring ProgressData 스펙 맞추기)
# ─────────────────────────────────────────────────────────────

def _get_status_key(job_id: str) -> str:
    """FastAPI 쪽과 맞추기 위해 동일한 스냅샷 키 사용."""
    return f"job:{job_id}:status"


async def _publish_progress(job_id: str, payload: Dict[str, Any]) -> None:
    """
    - Redis SET (스냅샷)
    - Redis PUBLISH (Pub/Sub)
    둘 다 수행. FastAPI 쪽과 형태를 맞춘다.
    """
    redis = get_redis_client()
    status_key = _get_status_key(job_id)
    try:
        # 스냅샷
        await redis.set(status_key, json.dumps(payload), ex=3600)
    except Exception as e:
        logger.error(f"[{job_id}] Failed to SET progress snapshot: {e}")

    try:
        # Pub/Sub
        channel_name = f"{settings.PROGRESS_CHANNEL_PREFIX}:{job_id}"
        await redis.publish(channel_name, json.dumps(payload))
        logger.info(
            f"[{job_id}] Progress PUBLISHED to {channel_name}: "
            f"type={payload.get('type')}, success={payload.get('success')}, "
            f"progress={payload.get('progress')}, message={payload.get('message')}"
        )
    except Exception as e:
        logger.error(f"[{job_id}] Failed to PUBLISH progress via Pub/Sub: {e}")


async def _report_processing_stage(
    job_id: str,
    member_id: str,
    stage: str,
    progress: float,
    current_clip: int,
    total_clips: int,
    highlight_id: Optional[str] = None,
) -> None:
    """
    ProgressType.PROCESSING 단계 보고.

    👉 최종 Redis/PubSub JSON 형식 (AI 처리 중):

    {
      "status": 200,
      "success": true,
      "timeStamp": 1731990002000,
      "type": "PROCESSING",
      "progress": 32.5,
      "jobId": "job1",
      "memberId": "xxxx"
    }

    stage / current_clip / total_clips / highlight_id 는
    - 로그에만 사용하고
    - Redis payload 에는 포함하지 않는다.
    """
    payload: Dict[str, Any] = {
        "status": 200,
        "success": True,
        "timeStamp": int(time.time() * 1000),
        "type": "PROCESSING",
        "memberId": str(member_id),
        "jobId": str(job_id),
        "progress": float(round(progress, 2)),
    }

    # 디버깅용 상세 로그 (Redis 로는 안 나감)
    logger.info(
        f"[{job_id}] PROCESSING stage={stage}, "
        f"clip={current_clip}/{total_clips}, "
        f"highlightIdentifier={highlight_id}, "
        f"progress={payload['progress']}"
    )

    await _publish_progress(job_id, payload)


async def _report_completion(
    job_id: str,
    member_id: str,
    success: bool,
    total_clips: int,
    highlight_id: Optional[str] = None,
    output_paths: Optional[List[str]] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    ProgressType.COMPLETE 단계 보고.

    👉 최종 Redis/PubSub JSON 형식 (AI 처리 완료):

    (성공)
    {
      "status": 200,
      "success": true,
      "timeStamp": 1731990003000,
      "type": "COMPLETE",
      "jobId": "job1",
      "memberId": "xxxx"
    }

    (에러)
    {
      "status": 200,
      "success": false,
      "timeStamp": 1731990003000,
      "type": "COMPLETE",
      "jobId": "job1",
      "memberId": "xxxx",
      "message": "에러 설명"
    }

    - COMPLETE 단계에서는 progress / stage / currentClip / totalClips 등은
      전송하지 않는다.
    """
    progress_value = 100.0 if success else -1.0  # 로그용

    payload: Dict[str, Any] = {
        "status": 200,
        "success": bool(success),
        "timeStamp": int(time.time() * 1000),
        "type": "COMPLETE",
        "memberId": str(member_id),
        "jobId": str(job_id),
    }

    # 에러 시에는 message 포함
    if not success and error_message:
        payload["message"] = str(error_message)

    await _publish_progress(job_id, payload)

    # 로그는 기존 느낌 유지
    status_str = "COMPLETED" if success else "FAILED"
    logger.info(
        f"[{job_id}] Final Status: {status_str} "
        f"(success={success}, totalClips={total_clips}, progress={progress_value})"
        f"{' | highlightIdentifier='+highlight_id if highlight_id else ''}"
    )
    if output_paths:
        logger.info(f"[{job_id}] Output files: {output_paths}")
    logger.info(f"[{job_id}] (Simulated) Final report target: {settings.SPRING_API_URL}/complete")


# ─────────────────────────────────────────────────────────────
# 3. 핵심 로직: AI 시뮬레이션, FFmpeg (기능은 그대로)
# ─────────────────────────────────────────────────────────────

def _find_matching_plan(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    input_size = metadata.get("size_bytes", 0)
    input_duration = metadata.get("duration_sec", 0.0)
    input_width = metadata.get("width", 0)
    input_height = metadata.get("height", 0)

    for plan in PLANS:
        size_diff = abs(input_size - plan.get("size_bytes", 0))
        is_size_match = size_diff <= SIZE_TOLERANCE_BYTES

        duration_diff = abs(input_duration - plan.get("duration_sec", 0.0))
        is_duration_match = duration_diff <= DURATION_TOLERANCE_SEC

        is_resolution_match = (
            input_width == plan.get("width", 0)
            and input_height == plan.get("height", 0)
        )

        if is_size_match and is_duration_match and is_resolution_match:
            segments = [
                {"start": s, "duration": d, "label": l}
                for s, d, l in plan["segments"]
            ]
            return {"name": f"Demo Plan {plan['id']}", "segments": segments}
    return None


async def _run_ai_pipeline_simulation(
    job_id: str,
    original_path: str,
    metadata: Dict[str, Any],
    member_id: str,
    highlight_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    기존 AI 파이프라인 시뮬레이션 로직은 그대로 두고,
    분석 구간(15% → 70%) 동안 progress 를 촘촘하게 보내도록만 수정.
    """
    start_time = time.time()
    try:
        logger.info(f"[{job_id}] [SIMU 1/5] Starting video decoding and AI pipeline initialization...")
        cap = cv2.VideoCapture(original_path)
        if not cap.isOpened():
            logger.error(
                f"[{job_id}] ERROR: Could not open video file for AI analysis simulation. "
                f"Path: {original_path}"
            )
            return []
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            logger.error(f"[{job_id}] ERROR: Failed to read first frame from {original_path}")
            return []
        await asyncio.sleep(0.5)

        compute_homography_auto(frame=frame, spec=NBA)
        logger.info(f"[{job_id}] [SIMU 2/5] Court Geometry Analysis finished (simulated).")
        await asyncio.sleep(0.3)

        detect_ball_hsv(frame=frame)
        logger.info(f"[{job_id}] [SIMU 3/5] Object Detector run finished (simulated).")
        await asyncio.sleep(0.2)

        matching_plan = _find_matching_plan(metadata)
        if not matching_plan:
            analysis_time = (time.time() - start_time) + (metadata.get("duration_sec", 10.0) * 0.2)
            logger.warning(
                f"[{job_id}] [SIMU 4/5] AI Analysis Completed in {analysis_time:.2f}s, "
                f"but no highlights found. Metadata: {metadata}"
            )
            return []

        logger.info(f"[{job_id}] [SIMU 4/5] AI Analysis Success! Found matching plan: {matching_plan['name']}")
        logger.info(f"[{job_id}] Starting Frame-by-Frame Inference (Time Consuming Process)...")

        # --- 여기서부터가 실제 '긴 분석 구간' 시뮬레이션 ---
        analysis_time_sim = metadata.get("duration_sec", 10.0) * 0.7

        # highlight_id 가 있는 경우에만 세분화된 진행률 전송
        if highlight_id:
            steps = 50  # B안: 적당한 단계 수
            for step in range(steps):
                await asyncio.sleep(analysis_time_sim / steps)
                smooth_progress = 15 + ((step + 1) / steps * (70 - 15))  # 15% → 70%
                await _report_processing_stage(
                    job_id=job_id,
                    member_id=member_id,
                    stage="ANALYZING",
                    progress=smooth_progress,
                    current_clip=0,
                    total_clips=0,
                    highlight_id=highlight_id,
                )
        else:
            # 안전장치: 혹시 highlight_id 가 비어있으면 기존처럼 한 번에 sleep
            await asyncio.sleep(analysis_time_sim)

        logger.info(
            f"[{job_id}] [SIMU 5/5] Inference finished in {analysis_time_sim:.2f}s. "
            f"{len(matching_plan['segments'])} segments identified."
        )
        return matching_plan["segments"]

    except Exception as e:
        logger.critical(f"[{job_id}] AI PIPELINE CRASHED: {type(e).__name__}: {e}")
        return []


def _run_ffmpeg_cut(job_id: str, src_path: str, segment: Dict[str, Any], output_path: str) -> None:
    """
    하이라이트 클립을 생성할 때,
    - 원본 코덱/해상도를 그대로 복사하던 기존 방식(-c copy)을 버리고
    - 모바일/웹 플레이어 호환성이 높은 표준 H.264 + AAC + yuv420p 로 재인코딩.
      (해상도는 가로 1280 고정, 세로는 비율 유지 & 짝수)
    """
    start = max(0.0, float(segment["start"]))
    duration = max(0.1, float(segment["duration"]))

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        src_path,
        "-t",
        f"{duration:.3f}",
        # ▼ 재인코딩 설정 (모바일/Expo-AV/ExoPlayer 친화)
        # 해상도: 가로 1280, 세로는 비율 유지 + 짝수(-2)
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
        # 비디오 코덱: H.264
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        # 오디오 코덱: AAC
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        # MP4 스트리밍/모바일 재생 친화 옵션
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"[{job_id}] Segment cut & re-encoded successfully: {output_path} ({segment['label']})")
    except subprocess.CalledProcessError as e:
        logger.error(
            f"[{job_id}] FFmpeg cutting/re-encoding failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR: {e.stderr.decode(errors='ignore')}"
        )
        raise RuntimeError(f"FFmpeg cut failed for segment: {segment['label']}")


# ─────────────────────────────────────────────────────────────
# 4. Task 처리 로직 (기능 유지, 진행률만 새 스펙으로)
# ─────────────────────────────────────────────────────────────

async def process_task(task: AITaskPayload):
    job_id = task.jobId
    member_id = task.memberId
    original_path = task.originalFilePath
    highlight_id = (task.highlightIdentifier or "").strip()

    if not highlight_id:
        # 하이라이트 키 없으면 즉시 실패 (COMPLETE + success:false + message)
        await _report_completion(
            job_id=job_id,
            member_id=member_id,
            success=False,
            total_clips=0,
            highlight_id=None,
            output_paths=[],
            error_message="highlightKey/highlightIdentifier is missing in payload.",
        )
        logger.error(f"[{job_id}] Task FAILED: highlightKey/highlightIdentifier is missing in payload.")
        return

    # 파일 경로 + 세그먼트 정보를 함께 담는 리스트
    output_files_with_segments: List[Dict[str, Any]] = []

    try:
        logger.info(
            f"[{job_id}] Task Received and Starting for member {member_id}. "
            f"File: {original_path} | highlightIdentifier={highlight_id}"
        )

        # 1단계: 작업 수신 (1%)
        await _report_processing_stage(
            job_id=job_id,
            member_id=member_id,
            stage="JOB_RECEIVED_INIT",
            progress=1.0,
            current_clip=0,
            total_clips=0,
            highlight_id=highlight_id,
        )

        logger.info(f"[{job_id}] Simulating initial video file load and decoding setup (1.0s delay)...")
        await asyncio.sleep(1.0)

        # 2단계: 비디오 로드 완료 (5%)
        await _report_processing_stage(
            job_id=job_id,
            member_id=member_id,
            stage="VIDEO_LOAD_INIT",
            progress=5.0,
            current_clip=0,
            total_clips=0,
            highlight_id=highlight_id,
        )

        if not os.path.exists(original_path):
            try:
                base_dir = os.path.dirname(original_path)
                logger.error(f"[{job_id}] Task FAILED: Original file not found at {original_path}")
                logger.error(f"[{job_id}] DEBUG: base_dir exists? {os.path.exists(base_dir)}")
                logger.error(f"[{job_id}] DEBUG: /data/highlights exists? {os.path.exists('/data/highlights')}")
                if os.path.exists('/data/highlights'):
                    logger.error(f"[{job_id}] DEBUG: /data/highlights entries: {os.listdir('/data/highlights')}")
            except Exception as debug_e:
                logger.error(f"[{job_id}] DEBUG: error while listing /data/highlights: {debug_e}")

            raise FileNotFoundError(f"Original file not found: {original_path}")

        # 3단계: 메타데이터 추출 (15%)
        metadata = get_video_metadata(original_path)
        logger.info(
            f"[{job_id}] Metadata: Duration={metadata.get('duration_sec'):.2f}s, "
            f"Size={metadata.get('size_bytes')}, "
            f"Resolution={metadata.get('width')}x{metadata.get('height')}"
        )
        await _report_processing_stage(
            job_id=job_id,
            member_id=member_id,
            stage="METADATA_EXTRACTED",
            progress=15.0,
            current_clip=0,
            total_clips=0,
            highlight_id=highlight_id,
        )

        # 4단계: 분석 구간 (15 → 70% 구간은 내부에서 촘촘하게 보고)
        highlight_segments = await _run_ai_pipeline_simulation(
            job_id=job_id,
            original_path=original_path,
            metadata=metadata,
            member_id=member_id,
            highlight_id=highlight_id,
        )

        # 분석 최종 마무리 포인트 70% (기존 호환용)
        await _report_processing_stage(
            job_id=job_id,
            member_id=member_id,
            stage="ANALYSIS_FINISHED",
            progress=70.0,
            current_clip=0,
            total_clips=0,
            highlight_id=highlight_id,
        )

        # 5단계: 하이라이트 컷팅 (70 → 99%)
        if not highlight_segments:
            logger.warning(f"[{job_id}] AI analysis completed successfully, but found no highlights to cut.")
            total_segments = 0
        else:
            logger.info(f"[{job_id}] Starting video cutting for {len(highlight_segments)} segments.")

            # ✅ jobId별 하위 폴더 생성: /data/highlights/processed/{jobId}
            job_output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
            os.makedirs(job_output_dir, exist_ok=True)

            total_segments = len(highlight_segments)
            for i, segment in enumerate(highlight_segments):
                # 파일 이름은 기존 패턴 유지 (jobId_segment_xx.mp4)
                output_filename = f"{job_id}_segment_{i+1:02d}.mp4"
                output_path = os.path.join(job_output_dir, output_filename)

                _run_ffmpeg_cut(job_id, original_path, segment, output_path)

                # 파일 경로와 세그먼트 정보를 묶어 저장 (spring_reporter 로 전달)
                output_files_with_segments.append(
                    {
                        "output_path": output_path,
                        "segment": segment,
                    }
                )

                # 🔹 컷팅 구간 progress: 70% → 99% 사이를 세그먼트 개수 기준으로 분배
                cut_progress = 70 + ((i + 1) / total_segments * (99 - 70))
                await _report_processing_stage(
                    job_id=job_id,
                    member_id=member_id,
                    stage="CUTTING",
                    progress=cut_progress,
                    current_clip=i + 1,
                    total_clips=total_segments,
                    highlight_id=highlight_id,
                )

        # 6단계: 완료 보고 (HTTP 전송 + COMPLETE 이벤트)
        final_output_paths = [d["output_path"] for d in output_files_with_segments]

        if not output_files_with_segments:
            # 하이라이트가 없으면 HTTP 전송 없이 COMPLETE 만 보고 (성공)
            await _report_completion(
                job_id=job_id,
                member_id=member_id,
                success=True,
                total_clips=0,
                highlight_id=highlight_id,
                output_paths=[],
            )
            final_message = "Task processing COMPLETE. No highlights found."
        else:
            # 하이라이트가 있으면 HTTP 전송 시도 (재시도 포함)
            http_success = await send_highlight_result_with_retry(
                job_id,                # ✅ jobId 바디에 넣기 위해 추가
                member_id,
                highlight_id,
                output_files_with_segments,
            )

            # HTTP 전송 결과에 관계없이 COMPLETE 이벤트는 성공으로 간주
            await _report_completion(
                job_id=job_id,         # ✅ 파라미터 이름 정확히 job_id
                member_id=member_id,
                success=True,
                total_clips=len(output_files_with_segments),
                highlight_id=highlight_id,
                output_paths=final_output_paths,
            )

            if http_success:
                final_message = "Task processing COMPLETE and SUCCESSFULLY REPORTED to Spring."
            else:
                final_message = (
                    f"Task processing COMPLETE but HTTP REPORTING FAILED after {MAX_RETRIES} retries."
                )

        logger.info(f"[{job_id}] {final_message}. Total output files: {len(output_files_with_segments)}")

    except FileNotFoundError as e:
        await _report_completion(
            job_id=job_id,
            member_id=member_id,
            success=False,
            total_clips=0,
            highlight_id=highlight_id,
            output_paths=[],
            error_message=str(e),
        )
        logger.error(f"[{job_id}] Task FAILED: Original file not found at {original_path}")
    except Exception as e:
        await _report_completion(
            job_id=job_id,
            member_id=member_id,
            success=False,
            total_clips=0,
            highlight_id=highlight_id,
            output_paths=[],
            error_message=f"Unexpected error in AI worker: {type(e).__name__}: {e}",
        )
        logger.error(f"[{job_id}] Task FAILED unexpectedly: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────
# 5. Main Worker 루프 (수신자/Consumer 로직)
# ─────────────────────────────────────────────────────────────

async def ai_worker_main():
    logger.info("--- AI Worker Simulator (Consumer) Started ---")
    logger.info(f"Monitoring queue: {settings.AI_QUEUE_NAME}")

    redis = get_redis_client()

    while True:
        try:
            result = await redis.blpop(settings.AI_QUEUE_NAME, timeout=0)
            if result:
                queue_name, json_data = result
                try:
                    task = AITaskPayload.model_validate_json(json_data)
                    logger.info(f"New job received from queue: {task.jobId}")
                    asyncio.create_task(process_task(task))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON payload received: {json_data}")
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
        except RedisConnectionError:
            logger.error("Lost connection to Redis. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            logger.info("Worker stopped by user.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in worker loop: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(ai_worker_main())
    except RedisConnectionError:
        logger.critical("AI Worker cannot start without Redis connection. Please check Redis URL.")
    except Exception as e:
        logger.error(f"Worker crashed during startup: {e}")
