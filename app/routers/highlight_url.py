# app/routers/highlight_url.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from pathlib import Path

from app.core.config import settings
from app.services.downloader import download_to_temp
from app.services.pipeline import generate_highlight_clips, save_clip_to_repo
from app.services.pubsub import publish_progress, publish_result, publish_error

router = APIRouter()

class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

class HighlightUrlReq(BaseModel):
    memberId: str
    jobId: str
    sourceUrl: str
    # timestamps는 둘 중 하나:
    # 1) 중심 시점 배열 -> [1.0, 5.2, 12.0]
    # 2) 세그먼트 배열 -> [{"start":3.2,"end":8.2}, ...] (중심 시점으로 변환해 사용)
    timestamps: Union[List[float], List[Segment]]
    pre: Optional[float] = None
    post: Optional[float] = None
    maxClips: Optional[int] = None
    jerseyNumber: Optional[int] = None  # 옵션(파일명 접두사에만 사용 가능)

def _normalize_centers(ts: Union[List[float], List[Segment]]) -> List[float]:
    centers: List[float] = []
    if not ts:
        return centers
    if isinstance(ts[0], dict) or isinstance(ts[0], Segment):
        for s in ts:
            s = s if isinstance(s, Segment) else Segment(**s)
            centers.append((float(s.start) + float(s.end)) / 2.0)
    else:
        centers = [float(x) for x in ts]
    return centers

@router.post("/highlight/url")
async def highlight_by_url(req: HighlightUrlReq):
    """
    외부/Presigned URL을 받아 임시 저장 → 하이라이트 생성 → SAVE_ROOT에 UUID 파일명으로 저장 →
    Redis 진행률(PROCESSING)과 최종 결과(COMPLETE)을 발행(PUB/SUB + KV 저장).
    상세 결과 JSON은 Redis KV key: `highlight-{jobId}` 로 확인 가능.
    """
    member_id = req.memberId
    job_id = req.jobId

    # 1) 시작 진행률
    await publish_progress(member_id, job_id, 0.01, "downloading source")

    # 2) 다운로드
    try:
        src_path = await download_to_temp(req.sourceUrl, suffix=".mp4")
    except Exception as e:
        await publish_error(member_id, job_id, f"download failed: {e}")
        raise HTTPException(status_code=400, detail=f"download failed: {e}")

    try:
        # 3) 중심 시점 정규화
        centers = _normalize_centers(req.timestamps)
        if not centers:
            raise ValueError("no valid timestamps")

        # 4) 클립 생성 (임시 디렉터리에 생성)
        results = generate_highlight_clips(
            src=Path(src_path),
            jersey_number=req.jerseyNumber if req.jerseyNumber is not None else -1,
            timestamps=centers,
            pre=req.pre,
            post=req.post,
            max_clips=req.maxClips,
        )
        tmp_clips = [p for _, p in results]
        if not tmp_clips:
            raise RuntimeError("no clips generated")

        total = len(tmp_clips)

        # 5) SAVE_ROOT/{memberId}/{jobId}/{uuid}.mp4 로 이동 + 공개 URL 생성
        urls: List[str] = []
        for i, tmp in enumerate(tmp_clips):
            _, url = save_clip_to_repo(tmp, member_id, job_id, index=i)
            urls.append(url)
            await publish_progress(member_id, job_id, (i + 1) / total, f"saved {i+1}/{total}")

        # 6) 최종 COMPLETE 발행 (+KV 저장)
        await publish_result(member_id, job_id, urls, final=True)

    except Exception as e:
        await publish_error(member_id, job_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 임시 소스 정리
        try:
            Path(src_path).unlink(missing_ok=True)
        except Exception:
            pass

    # 간단 응답 — 상세 결과는 Redis KV highlight-{jobId}에 저장됨
    return {
        "status": 200,
        "success": True,
        "message": f"{len(urls)} clips created",
        "memberId": member_id,
        "jobId": job_id,
        "count": len(urls),
        "urls": urls,  # 디버그/개발 편의상 포함(운영에서 빼도 됨)
    }
