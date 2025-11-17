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
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://host.docker.internal:6379/0")
    AI_QUEUE_NAME: str = os.getenv("REDIS_AI_JOB_QUEUE", "opencv-ai-job-queue")
    SPRING_API_URL: str = os.getenv("SPRING_API_URL", "http://host.docker.internal:8080/api/v1/jobs")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/data/highlights/processed")
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
# 2. 핵심 로직: AI 시뮬레이션, FFmpeg, 보고 함수 (기존 로직 유지)
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

        is_resolution_match = (input_width == plan.get("width", 0) and input_height == plan.get("height", 0))

        if is_size_match and is_duration_match and is_resolution_match:
            segments = [{"start": s, "duration": d, "label": l} for s, d, l in plan["segments"]]
            return {"name": f"Demo Plan {plan['id']}", "segments": segments}
    return None


async def _run_ai_pipeline_simulation(job_id: str, original_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    start_time = time.time()
    try:
        logger.info(f"[{job_id}] [SIMU 1/5] Starting video decoding and AI pipeline initialization...")
        cap = cv2.VideoCapture(original_path)
        if not cap.isOpened():
            logger.error(f"[{job_id}] ERROR: Could not open video file for AI analysis simulation. Path: {original_path}")
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
            logger.warning(f"[{job_id}] [SIMU 4/5] AI Analysis Completed in {analysis_time:.2f}s, but no highlights found. Metadata: {metadata}")
            return []

        logger.info(f"[{job_id}] [SIMU 4/5] AI Analysis Success! Found matching plan: {matching_plan['name']}")
        logger.info(f"[{job_id}] Starting Frame-by-Frame Inference (Time Consuming Process)...")
        analysis_time_sim = metadata.get("duration_sec", 10.0) * 0.7
        await asyncio.sleep(analysis_time_sim)

        logger.info(f"[{job_id}] [SIMU 5/5] Inference finished in {analysis_time_sim:.2f}s. {len(matching_plan['segments'])} segments identified.")
        return matching_plan["segments"]

    except Exception as e:
        logger.critical(f"[{job_id}] AI PIPELINE CRASHED: {type(e).__name__}: {e}")
        return []


def _run_ffmpeg_cut(job_id: str, src_path: str, segment: Dict[str, Any], output_path: str) -> None:
    start = max(0.0, float(segment["start"]))
    duration = max(0.1, float(segment["duration"]))

    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", src_path,
        "-t", f"{duration:.3f}", "-c", "copy",
        "-avoid_negative_ts", "make_zero", output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"[{job_id}] Segment cut success: {output_path} ({segment['label']})")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{job_id}] FFmpeg cutting failed: {e.stderr.decode()}")
        raise RuntimeError(f"FFmpeg cut failed for segment: {segment['label']}")


async def _report_progress(job_id: str, status: str, progress: float, highlight_id: Optional[str] = None):
    redis = get_redis_client()
    try:
        channel_name = f"{settings.PROGRESS_CHANNEL_PREFIX}:{job_id}"
        payload = {"jobId": job_id, "status": status, "progress": round(progress, 2)}
        if highlight_id:
            payload["highlightIdentifier"] = highlight_id
        await redis.publish(channel_name, json.dumps(payload))
        logger.info(
            f"[{job_id}] Progress published to {channel_name}: {status}, {progress:.1f}%"
            f"{' | highlightIdentifier='+highlight_id if highlight_id else ''}"
        )
    except Exception as e:
        logger.error(f"[{job_id}] Failed to report progress via Pub/Sub: {e}")


async def _report_completion(job_id: str, success: bool, output_paths: List[str] = None, highlight_id: Optional[str] = None):
    status = "COMPLETED" if success else "FAILED"
    await _report_progress(job_id, status, 100.0 if success else -1.0, highlight_id=highlight_id)

    payload = {
        "jobId": job_id,
        "status": status,
        "resultFiles": output_paths or [],
        "message": "AI analysis and video cutting finished." if success else "AI processing failed."
    }
    if highlight_id:
        payload["highlightIdentifier"] = highlight_id

    logger.info(f"[{job_id}] Final Status: {status}"
                f"{' | highlightIdentifier='+highlight_id if highlight_id else ''}")
    logger.info(f"[{job_id}] (Simulated) Final report target: {settings.SPRING_API_URL}/complete")


async def process_task(task: AITaskPayload):
    job_id = task.jobId
    original_path = task.originalFilePath
    highlight_id = (task.highlightIdentifier or "").strip()

    if not highlight_id:
        # 하이라이트 키 없으면 즉시 실패 (변경 없음)
        await _report_completion(job_id, success=False, highlight_id=None)
        logger.error(f"[{job_id}] Task FAILED: highlightKey/highlightIdentifier is missing in payload.")
        return

    # 파일 경로 + 세그먼트 정보를 함께 담는 리스트
    output_files_with_segments: List[Dict[str, Any]] = []

    try:
        logger.info(
            f"[{job_id}] Task Received and Starting for member {task.memberId}. "
            f"File: {original_path} | highlightIdentifier={highlight_id}"
        )

        await _report_progress(job_id, "JOB_RECEIVED_INIT", 1, highlight_id=highlight_id)
        logger.info(f"[{job_id}] Simulating initial video file load and decoding setup (1.0s delay)...")
        await asyncio.sleep(1.0)
        await _report_progress(job_id, "VIDEO_LOAD_INIT", 5, highlight_id=highlight_id)

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

        metadata = get_video_metadata(original_path)
        logger.info(
            f"[{job_id}] Metadata: Duration={metadata.get('duration_sec'):.2f}s, "
            f"Size={metadata.get('size_bytes')}, Resolution={metadata.get('width')}x{metadata.get('height')}"
        )
        await _report_progress(job_id, "METADATA_EXTRACTED", 15, highlight_id=highlight_id)

        highlight_segments = await _run_ai_pipeline_simulation(job_id, original_path, metadata)
        await _report_progress(job_id, "ANALYSIS_FINISHED", 70, highlight_id=highlight_id)

        # 4단계: 하이라이트 컷팅
        if not highlight_segments:
            logger.warning(f"[{job_id}] AI analysis completed successfully, but found no highlights to cut.")
        else:
            logger.info(f"[{job_id}] Starting video cutting for {len(highlight_segments)} segments.")

            # ✅ jobId별 하위 폴더 생성: /data/highlights/processed/{jobId}
            job_output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
            os.makedirs(job_output_dir, exist_ok=True)

            total_segments = len(highlight_segments)
            for i, segment in enumerate(highlight_segments):
                # 파일 이름은 기존 패턴 유지 (jobId_segment_xx.mp4) → 다른 로직 영향 최소화
                output_filename = f"{job_id}_segment_{i+1:02d}.mp4"
                # ✅ 이제는 jobId 폴더 안에 저장
                output_path = os.path.join(job_output_dir, output_filename)

                _run_ffmpeg_cut(job_id, original_path, segment, output_path)

                # 파일 경로와 세그먼트 정보를 묶어 저장 (spring_reporter로 전달)
                output_files_with_segments.append({
                    "output_path": output_path,
                    "segment": segment
                })

        # 5단계: 완료 보고 (HTTP 전송 + Pub/Sub)
        final_output_paths = [d["output_path"] for d in output_files_with_segments]

        if not output_files_with_segments:
            # 하이라이트가 없으면 HTTP 전송 없이 Pub/Sub으로만 완료 보고
            await _report_completion(job_id, success=True, highlight_id=highlight_id)
            final_message = "Task processing COMPLETE. No highlights found."
        else:
            # 하이라이트가 있으면 HTTP 전송 시도 (재시도 포함)
            http_success = await send_highlight_result_with_retry(
                task.memberId,
                highlight_id,
                output_files_with_segments
            )

            # HTTP 전송 결과에 관계없이 Pub/Sub으로 최종 완료 상태 보고 (AI 작업 성공으로 간주)
            await _report_completion(
                job_id,
                success=True,
                output_paths=final_output_paths,
                highlight_id=highlight_id
            )

            if http_success:
                final_message = "Task processing COMPLETE and SUCCESSFULLY REPORTED to Spring."
            else:
                final_message = f"Task processing COMPLETE but HTTP REPORTING FAILED after {MAX_RETRIES} retries."

        logger.info(f"[{job_id}] {final_message}. Total output files: {len(output_files_with_segments)}")

    except FileNotFoundError:
        await _report_completion(job_id, success=False, highlight_id=highlight_id)
        logger.error(f"[{job_id}] Task FAILED: Original file not found at {original_path}")
    except Exception as e:
        await _report_completion(job_id, success=False, highlight_id=highlight_id)
        logger.error(f"[{job_id}] Task FAILED unexpectedly: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────
# 4. Main Worker 루프 (수신자/Consumer 로직)
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
