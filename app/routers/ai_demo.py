# app/routers/ai_demo.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import glob

from app.core.config import settings
from app.routers.highlight import _save_upload, _cleanup_paths
from app.services.bh_edit import cut_and_overlay, concat_videos, ffprobe_duration
from app.services.pipeline import build_zip_in_memory, save_clip_as_uuid  # ✅ UUID 저장 유틸 사용
from app.services.plan_registry import PlanRegistry
from app.services.media_tags import annotate_inplace

router = APIRouter()

def _find_latest_original(member_id: str, job_id: str) -> Optional[Path]:
    """
    /data/highlights/<member>/<job>/original_*.mp4 중 최신 파일을 찾아 반환.
    없으면 None.
    """
    root = Path(settings.SAVE_ROOT)
    pattern = str(root / member_id / job_id / "original_*.mp4")
    candidates = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0]) if candidates else None

class AISelector:
    """겉모습만 AI처럼: 실제로는 레지스트리의 segments를 그대로 사용"""
    def __init__(self, plan: dict, tag: str = "AI-Selector"):
        self.plan = plan
        self.tag = tag
    def select(self) -> List[Tuple[float, float, str]]:
        segs = self.plan.get("segments", [])
        return [(float(a), float(b), str(lbl)) for (a, b, lbl) in segs]

@router.post("/highlight/ai-demo")
async def highlight_ai_demo(
    # 업로드 파일(옵션: useExisting=true면 없어도 됨)
    video: Optional[UploadFile] = File(default=None, description="데모 영상 1개(업로드 모드일 때만 필요)"),
    # 기존 저장된 원본을 사용할지 여부
    useExisting: bool = Form(False, description="이미 저장된 original_*.mp4를 사용"),
    # 힌트(1|2|3)
    videoCode: Optional[str] = Form(default=None, description="선택: 힌트(1|2|3) – 없어도 자동 식별"),
    overlay_model_tag: str = Form("AI-Selector"),
    merge_output: bool = Form(True, description="개별 클립들을 하나로 이어 붙일지"),
    memberId: Optional[str] = Form(default=None),
    jobId: Optional[str] = Form(default=None),
    returnZip: bool = Form(True, description="ZIP 스트림으로 반환(아니면 URL JSON)"),
):
    # 0) 입력 검증: useExisting이면 memberId/jobId 필수
    if useExisting:
        if not (memberId and jobId):
            return JSONResponse(status_code=400, content={"error": "useExisting=true requires memberId and jobId"})
        src_path = _find_latest_original(memberId, jobId)
        if not src_path:
            return JSONResponse(status_code=404, content={"error": "no original_*.mp4 found for given memberId/jobId"})
        tmp_in = src_path  # 이미 SAVE_ROOT에 존재하는 파일을 직접 사용
        cleanup_needed = False
        base_name = src_path.stem
    else:
        if not video:
            return JSONResponse(status_code=400, content={"error": "upload mode requires 'video' file"})
        tmp_in = _save_upload(video)
        cleanup_needed = True
        base_name = Path(video.filename or tmp_in.stem).stem

    clips: List[Path] = []
    merged: Optional[Path] = None

    try:
        # 1) 자동 식별 (파이썬 레지스트리만)
        reg = PlanRegistry.from_config()
        plan = reg.match(tmp_in, hint=videoCode)
        if not plan:
            return JSONResponse(
                status_code=404,
                content={"error": "no matching plan (check duration/size/sha256/phash in registry.py)"}
            )
        video_id = str(plan.get("id", "X"))

        selector = AISelector(plan, tag=overlay_model_tag)
        segments = selector.select()
        if not segments:
            return JSONResponse(status_code=400, content={"error": "plan has empty segments"})

        # 2) 컷 생성 + 메타데이터 태깅 + 집계
        duration = ffprobe_duration(str(tmp_in))
        work_dir = Path("/tmp/uploads"); work_dir.mkdir(parents=True, exist_ok=True)

        counts = {"2PT": 0, "3PT": 0}
        manifest: List[Dict[str, str | float | int]] = []
        public_urls: List[str] = []

        for i, (s, e, lbl) in enumerate(segments, start=1):
            s2 = max(0.0, s)
            e2 = min(e, duration) if duration and duration > 0 else e
            if e2 - s2 < 0.3:
                continue

            # 임시 파일명(워킹 디렉터리)
            tmp_out = work_dir / f"{base_name}_seg{i:02d}_{lbl}_s{int(s2*1000)}_e{int(e2*1000)}.mp4"

            # 오버레이 텍스트(겉모습: AI)
            txt = f"[{overlay_model_tag}] {lbl}  {int(s2//60):02d}:{int(s2%60):02d}~{int(e2//60):02d}:{int(e2%60):02d}"
            cut_and_overlay(str(tmp_in), s2, e2, txt, str(tmp_out))

            # MP4 메타데이터(임시 파일에 먼저 씌움)
            annotate_inplace(str(tmp_out), {
                "clip_type": lbl,
                "video_id": video_id,
                "segment_index": str(i),
                "start_sec": f"{s2:.3f}",
                "end_sec": f"{e2:.3f}"
            })

            # 집계(FT 제외)
            if lbl == "2PT": counts["2PT"] += 1
            if lbl == "3PT": counts["3PT"] += 1

            # ✅ 서비스 저장소로 최종 이동 + 공개 URL (UUID, shorts/ 하위)
            dst_path, url = save_clip_as_uuid(tmp_out, memberId or "local", jobId or f"ai-demo-{video_id}",
                                              subdir="shorts", prefix="short_")
            clips.append(dst_path)
            public_urls.append(url)

            manifest.append({
                "file": dst_path.name,
                "clip_type": lbl,
                "video_id": video_id,
                "segment_index": i,
                "start_sec": round(s2, 3),
                "end_sec": round(e2, 3),
                "url": url
            })

        if not clips:
            return JSONResponse(status_code=400, content={"error": "no valid segments"})

        # 3) 병합본(옵션) — 병합본도 shorts/에 UUID로 저장
        merged_url = None
        if merge_output and clips:
            merged_tmp = work_dir / f"{base_name}_merged_ai_demo.mp4"
            concat_videos([str(p) for p in clips], str(merged_tmp))
            annotate_inplace(str(merged_tmp), {
                "video_id": video_id,
                "total_2pt": str(counts["2PT"]),
                "total_3pt": str(counts["3PT"]),
                "segment_count": str(len(clips))
            })
            merged_dst, merged_url = save_clip_as_uuid(merged_tmp, memberId or "local", jobId or f"ai-demo-{video_id}",
                                                       subdir="shorts", prefix="merged_")
            clips.append(merged_dst)
            public_urls.append(merged_url)

        # 4) summary.json — SAVE_ROOT/<member>/<job>/summary.json
        save_member = memberId or "local"
        save_job = jobId or f"ai-demo-{video_id}"
        dst_dir = Path(settings.SAVE_ROOT) / save_member / save_job
        dst_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "video_id": video_id,
            "counts": counts,         # {"2PT": n, "3PT": m}
            "segments": manifest,     # 각 클립 상세(+url)
            "merged": merged_url
        }
        (dst_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        # 5) 응답
        if returnZip:
            # shorts/에 들어간 최종 파일들 중, 이번에 만든 것만 ZIP (clips 리스트)
            buf = build_zip_in_memory(clips)
            fname = f"{base_name}_{video_id}_ai_demo.zip"
            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'}
            )
        else:
            return JSONResponse(content={
                "planId": video_id,
                "counts": counts,
                "segments": manifest,
                "public_urls": public_urls,
                "summary_url": f"{settings.STATIC_BASE_URL.rstrip('/')}/{save_member}/{save_job}/summary.json"
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # 업로드 모드일 때만 임시 입력 삭제
        if cleanup_needed:
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass
